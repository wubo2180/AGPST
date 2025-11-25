"""
Enhanced Temporal Encoding Module
Combines Positional Encoding with Time Index Embeddings

Key Insight:
- Positional Encoding: Captures sequence order (1st, 2nd, 3rd timestep...)
- Time Index: Captures periodic patterns (Monday vs Sunday, Rush hour vs Night)
- Adaptive Multi-scale PE: Automatically learns periodic patterns without time index
"""

import torch
import torch.nn as nn
import math


class AdaptiveMultiScalePositionalEncoding(nn.Module):
    """
    Adaptive Multi-Scale Positional Encoding
    
    Automatically learns periodic patterns at multiple scales without time index.
    Combines:
    1. Fourier Features (for multiple frequencies)
    2. Learnable scale weights (adaptive to data)
    3. Convolutional smoothing (for local patterns)
    
    Key Innovation:
    - No need for explicit time index (hour, day, etc.)
    - Automatically discovers daily/weekly/monthly patterns
    - Works with any sequence length
    
    Args:
        embed_dim: Embedding dimension
        max_len: Maximum sequence length
        num_scales: Number of frequency scales (default 8)
        learnable: Whether to make frequencies learnable (default True)
        dropout: Dropout rate
    """
    def __init__(self, embed_dim, max_len=288, num_scales=8, learnable=True, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.num_scales = num_scales
        self.dropout = nn.Dropout(p=dropout)
        
        # Dimension allocation
        dim_per_scale = embed_dim // num_scales
        
        # Multi-scale frequency bands (automatic period discovery)
        # Scale 0: Very short (2-4 steps) - immediate patterns
        # Scale 1: Short (4-8 steps) - hourly patterns
        # Scale 2: Medium-short (12-24 steps) - half-day patterns
        # Scale 3: Medium (24-48 steps) - daily patterns  ← Most important for traffic
        # Scale 4: Long (48-96 steps) - 2-day patterns
        # Scale 5: Very long (96-192 steps) - weekly patterns
        # Scale 6-7: Ultra long - seasonal patterns
        
        if learnable:
            # Learnable frequencies (initialized near common periods)
            initial_periods = torch.tensor([
                4.0,      # 4 steps (20 min at 5-min resolution)
                12.0,     # 12 steps (1 hour)
                24.0,     # 24 steps (2 hours)
                72.0,     # 72 steps (6 hours, quarter day)
                144.0,    # 144 steps (12 hours, half day)
                288.0,    # 288 steps (1 day) ← Critical for traffic
                2016.0,   # 2016 steps (1 week)
                8064.0    # 8064 steps (1 month approximation)
            ][:num_scales])
            
            # Make periods learnable (model can adjust them)
            self.log_periods = nn.Parameter(torch.log(initial_periods))
        else:
            # Fixed frequencies
            periods = torch.logspace(
                start=math.log10(4.0),
                end=math.log10(max_len * 2),
                steps=num_scales
            )
            self.register_buffer('log_periods', torch.log(periods))
        
        # Learnable scale weights (importance of each frequency)
        self.scale_weights = nn.Parameter(torch.ones(num_scales))
        
        # Convolutional smoothing for local pattern refinement
        self.conv_refine = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        )
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(embed_dim)
        
        # Generate base encodings
        self._generate_encodings(max_len, dim_per_scale)
        
    def _generate_encodings(self, max_len, dim_per_scale):
        """Generate multi-scale sinusoidal encodings"""
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)  # (T, 1)
        
        encodings = []
        for scale_idx in range(self.num_scales):
            # For each scale, generate sin/cos pairs
            dim_this_scale = dim_per_scale
            if scale_idx == self.num_scales - 1:
                # Last scale gets remaining dimensions
                dim_this_scale = self.embed_dim - dim_per_scale * (self.num_scales - 1)
            
            # Create frequency components
            div_term = torch.exp(
                torch.arange(0, dim_this_scale, 2, dtype=torch.float) * 
                -(math.log(10000.0) / dim_this_scale)
            )
            
            pe = torch.zeros(max_len, dim_this_scale)
            pe[:, 0::2] = torch.sin(position * div_term)
            if dim_this_scale > 1:
                pe[:, 1::2] = torch.cos(position * div_term[:dim_this_scale//2])
            
            encodings.append(pe)
        
        # Concatenate and register as buffer (will be modulated by learned periods)
        base_pe = torch.cat(encodings, dim=-1)  # (T, D)
        self.register_buffer('base_pe', base_pe)
    
    def forward(self, x):
        """
        Apply adaptive multi-scale positional encoding
        
        Args:
            x: (B, N, T, D) - Input tensor
        Returns:
            encoded: (B, N, T, D) - With adaptive positional encoding
        """
        B, N, T, D = x.shape
        
        # Get learned periods (clamped to reasonable range)
        periods = torch.exp(self.log_periods).clamp(min=2.0, max=self.max_len * 4)
        
        # Compute normalized scale weights (sum to 1)
        weights = torch.softmax(self.scale_weights, dim=0)
        
        # Generate position encoding with learned periods
        position = torch.arange(T, dtype=torch.float, device=x.device).unsqueeze(1)
        
        encodings = []
        dim_per_scale = D // self.num_scales
        
        for scale_idx in range(self.num_scales):
            dim_this_scale = dim_per_scale
            if scale_idx == self.num_scales - 1:
                dim_this_scale = D - dim_per_scale * (self.num_scales - 1)
            
            # Modulate frequency by learned period
            period = periods[scale_idx]
            phase = 2 * math.pi * position / period
            
            # Generate sin/cos with learned period
            pe_scale = torch.zeros(T, dim_this_scale, device=x.device)
            
            for i in range(0, dim_this_scale, 2):
                pe_scale[:, i] = torch.sin(phase * (i // 2 + 1)).squeeze()
                if i + 1 < dim_this_scale:
                    pe_scale[:, i + 1] = torch.cos(phase * (i // 2 + 1)).squeeze()
            
            # Weight by learned importance
            pe_scale = pe_scale * weights[scale_idx]
            encodings.append(pe_scale)
        
        # Combine all scales
        pe = torch.cat(encodings, dim=-1)  # (T, D)
        
        # Apply convolutional refinement (learn local patterns)
        # (T, D) → (1, D, T) → Conv → (1, D, T) → (T, D)
        pe_conv = self.conv_refine(pe.unsqueeze(0).transpose(1, 2)).transpose(1, 2).squeeze(0)
        
        # Combine base PE with refined PE
        pe_final = pe + 0.5 * pe_conv  # Residual connection
        
        # Add batch and node dimensions: (1, 1, T, D)
        pe_final = pe_final.unsqueeze(0).unsqueeze(0)
        
        # Add to input and normalize
        x_encoded = x + pe_final
        x_encoded = self.norm(x_encoded)
        
        return self.dropout(x_encoded)
    
    def get_learned_periods(self):
        """Get the learned period values (for analysis)"""
        with torch.no_grad():
            periods = torch.exp(self.log_periods).clamp(min=2.0, max=self.max_len * 4)
            weights = torch.softmax(self.scale_weights, dim=0)
            return periods, weights


class CyclicPositionalEncoding(nn.Module):
    """
    Cyclic Positional Encoding with Multiple Periods
    
    Combines:
    1. Standard PE (long-range dependencies)
    2. Daily cycle (24 hours or 288 5-min intervals)
    3. Weekly cycle (7 days)
    
    Args:
        embed_dim: Embedding dimension
        max_len: Maximum sequence length (default 288 for 5-min resolution)
        dropout: Dropout rate
    """
    def __init__(self, embed_dim, max_len=288, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(p=dropout)
        
        # Allocate dimensions for different periods
        dim_standard = embed_dim // 2  # 50% for standard PE
        dim_daily = embed_dim // 4     # 25% for daily cycle
        dim_weekly = embed_dim // 4    # 25% for weekly cycle
        
        # 1. Standard Positional Encoding (short-term dependencies)
        pe_standard = self._get_sinusoidal_encoding(
            max_len, dim_standard, period=10000.0
        )
        
        # 2. Daily Cycle Encoding (288 intervals = 1 day at 5-min resolution)
        pe_daily = self._get_sinusoidal_encoding(
            max_len, dim_daily, period=288.0
        )
        
        # 3. Weekly Cycle Encoding (288 * 7 = 2016 intervals = 1 week)
        pe_weekly = self._get_sinusoidal_encoding(
            max_len, dim_weekly, period=288.0 * 7
        )
        
        # Concatenate all encodings
        pe = torch.cat([pe_standard, pe_daily, pe_weekly], dim=-1)
        
        # Add batch and node dimensions: (1, 1, max_len, embed_dim)
        pe = pe.unsqueeze(0).unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def _get_sinusoidal_encoding(self, max_len, dim, period):
        """
        Generate sinusoidal encoding with custom period
        
        Args:
            max_len: Sequence length
            dim: Embedding dimension (must be even)
            period: Period of the cycle
        """
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float) * 
            -(math.log(period) / dim)
        )
        
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def forward(self, x):
        """
        Add positional encoding to input
        
        Args:
            x: (B, N, T, D) - Input tensor
        Returns:
            x + pe: (B, N, T, D) - With positional encoding
        """
        B, N, T, D = x.shape
        x = x + self.pe[:, :, :T, :]
        return self.dropout(x)


class TimeIndexEmbedding(nn.Module):
    """
    Time Index Embeddings for Periodic Patterns
    
    Embeds discrete time indices:
    - Hour of day (0-23)
    - Day of week (0-6, Monday=0)
    - Day of month (1-31)
    - Month of year (1-12)
    - Holiday indicator (0 or 1)
    
    Args:
        embed_dim: Total embedding dimension
        use_hour: Use hour-of-day embedding
        use_day: Use day-of-week embedding
        use_month: Use month-of-year embedding
        use_holiday: Use holiday indicator
    """
    def __init__(self, embed_dim, use_hour=True, use_day=True, 
                 use_month=False, use_holiday=False):
        super().__init__()
        
        self.use_hour = use_hour
        self.use_day = use_day
        self.use_month = use_month
        self.use_holiday = use_holiday
        
        # Count active embeddings
        num_active = sum([use_hour, use_day, use_month, use_holiday])
        if num_active == 0:
            raise ValueError("At least one time index must be enabled")
        
        dim_per_index = embed_dim // num_active
        
        # Create embeddings
        if use_hour:
            self.hour_embed = nn.Embedding(24, dim_per_index)
        if use_day:
            self.day_embed = nn.Embedding(7, dim_per_index)
        if use_month:
            self.month_embed = nn.Embedding(12, dim_per_index)
        if use_holiday:
            self.holiday_embed = nn.Embedding(2, dim_per_index)
        
        # Projection to target dimension
        total_dim = dim_per_index * num_active
        if total_dim != embed_dim:
            self.projection = nn.Linear(total_dim, embed_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, x, time_indices):
        """
        Add time index embeddings to input
        
        Args:
            x: (B, N, T, D) - Input tensor
            time_indices: dict with keys from ['hour', 'day', 'month', 'holiday']
                Each value is (B, T) tensor with integer indices
        Returns:
            x + time_emb: (B, N, T, D)
        """
        B, N, T, D = x.shape
        embeddings = []
        
        if self.use_hour and 'hour' in time_indices:
            hour_emb = self.hour_embed(time_indices['hour'])  # (B, T, dim)
            embeddings.append(hour_emb)
        
        if self.use_day and 'day' in time_indices:
            day_emb = self.day_embed(time_indices['day'])  # (B, T, dim)
            embeddings.append(day_emb)
        
        if self.use_month and 'month' in time_indices:
            month_emb = self.month_embed(time_indices['month'])  # (B, T, dim)
            embeddings.append(month_emb)
        
        if self.use_holiday and 'holiday' in time_indices:
            holiday_emb = self.holiday_embed(time_indices['holiday'])  # (B, T, dim)
            embeddings.append(holiday_emb)
        
        if not embeddings:
            return x
        
        # Concatenate all embeddings
        time_emb = torch.cat(embeddings, dim=-1)  # (B, T, total_dim)
        time_emb = self.projection(time_emb)  # (B, T, D)
        
        # Expand to match input shape
        time_emb = time_emb.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, T, D)
        
        return x + time_emb


class EnhancedTemporalEncoding(nn.Module):
    """
    Combined Temporal Encoding
    
    Integrates both:
    1. Cyclic Positional Encoding (for sequence order + periodic patterns)
    2. Time Index Embeddings (for semantic time information)
    
    Usage:
        # Only positional encoding
        encoder = EnhancedTemporalEncoding(embed_dim=64, use_time_index=False)
        x_encoded = encoder(x)
        
        # With time indices
        encoder = EnhancedTemporalEncoding(embed_dim=64, use_time_index=True)
        x_encoded = encoder(x, time_indices={'hour': hour_tensor, 'day': day_tensor})
    """
    def __init__(self, embed_dim, max_len=288, dropout=0.0, 
                 use_time_index=False, use_hour=True, use_day=True,
                 use_month=False, use_holiday=False):
        super().__init__()
        
        self.use_time_index = use_time_index
        
        if use_time_index:
            # Split embedding dimension
            pe_dim = embed_dim // 2
            ti_dim = embed_dim - pe_dim
            
            self.pos_encoding = CyclicPositionalEncoding(pe_dim, max_len, dropout)
            self.time_index_embedding = TimeIndexEmbedding(
                ti_dim, use_hour, use_day, use_month, use_holiday
            )
            
        else:
            # Only positional encoding
            self.pos_encoding = CyclicPositionalEncoding(embed_dim, max_len, dropout)
            self.time_index_embedding = None
    
    def forward(self, x, time_indices=None):
        """
        Args:
            x: (B, N, T, D)
            time_indices: Optional dict with time index tensors
        Returns:
            encoded: (B, N, T, D)
        """
        # Apply positional encoding
        x = self.pos_encoding(x)
        
        # Apply time index embeddings if available
        if self.use_time_index and self.time_index_embedding is not None:
            if time_indices is None:
                raise ValueError("time_indices required when use_time_index=True")
            x = self.time_index_embedding(x, time_indices)
        
        return x


# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """Demonstrate how to use enhanced temporal encoding"""
    
    B, N, T, D = 32, 207, 12, 64  # METR-LA with 12 timesteps
    
    # Example 1: Only positional encoding (current approach)
    print("=" * 60)
    print("Example 1: Positional Encoding Only")
    print("=" * 60)
    
    encoder_pe_only = EnhancedTemporalEncoding(
        embed_dim=D,
        max_len=288,
        use_time_index=False
    )
    
    x = torch.randn(B, N, T, D)
    x_encoded = encoder_pe_only(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_encoded.shape}")
    print()
    
    # Example 2: Positional encoding + Time indices
    print("=" * 60)
    print("Example 2: PE + Time Index Embeddings")
    print("=" * 60)
    
    encoder_full = EnhancedTemporalEncoding(
        embed_dim=D,
        max_len=288,
        use_time_index=True,
        use_hour=True,
        use_day=True,
        use_month=False,
        use_holiday=False
    )
    
    # Create sample time indices
    time_indices = {
        'hour': torch.randint(0, 24, (B, T)),  # Hour of day
        'day': torch.randint(0, 7, (B, T))     # Day of week
    }
    
    x_encoded_full = encoder_full(x, time_indices)
    print(f"Input shape: {x.shape}")
    print(f"Time indices: {list(time_indices.keys())}")
    print(f"Output shape: {x_encoded_full.shape}")
    print()
    
    # Example 3: Adaptive Multi-Scale PE (Best for time series)
    print("=" * 60)
    print("Example 3: Adaptive Multi-Scale PE (Recommended)")
    print("=" * 60)
    
    adaptive_pe = AdaptiveMultiScalePositionalEncoding(
        embed_dim=D,
        max_len=288,
        num_scales=8,
        learnable=True
    )
    
    x_adaptive = adaptive_pe(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_adaptive.shape}")
    
    # Show learned periods
    periods, weights = adaptive_pe.get_learned_periods()
    print(f"\nLearned periods (in time steps):")
    for i, (p, w) in enumerate(zip(periods, weights)):
        hours = p.item() * 5 / 60  # Convert 5-min steps to hours
        print(f"  Scale {i}: {p.item():.1f} steps ({hours:.2f} hours) - weight: {w.item():.3f}")
    print()
    
    # Example 4: Compare all PE methods
    print("=" * 60)
    print("Example 4: Performance Comparison")
    print("=" * 60)
    
    cyclic_pe = CyclicPositionalEncoding(D, max_len=288)
    
    x_cyclic = cyclic_pe(x)
    x_adaptive = adaptive_pe(x)
    
    print(f"Cyclic PE output: {x_cyclic.shape}")
    print(f"Adaptive PE output: {x_adaptive.shape}")
    print(f"Difference norm: {torch.norm(x_cyclic - x_adaptive):.4f}")
    

if __name__ == "__main__":
    example_usage()
