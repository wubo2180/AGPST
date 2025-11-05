import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTemporalAttention(nn.Module):
    """
    Decoupled Spatial-Temporal Attention for better modeling
    Spatial attention: captures node relationships
    Temporal attention: captures time dependencies
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Spatial attention (across nodes)
        self.spatial_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.spatial_norm = nn.LayerNorm(embed_dim)
        
        # Temporal attention (across time)
        self.temporal_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.temporal_norm = nn.LayerNorm(embed_dim)
        
        # Cross attention (optional, for interaction)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (B, N, L, D) where B=batch, N=nodes, L=time, D=dim
        """
        B, N, L, D = x.shape
        
        # === Spatial Attention (across nodes at each time step) ===
        x_spatial = x.permute(0, 2, 1, 3).contiguous()  # (B, L, N, D)
        x_spatial = x_spatial.view(B * L, N, D)  # (B*L, N, D)
        
        spatial_out, _ = self.spatial_attn(x_spatial, x_spatial, x_spatial)
        spatial_out = x_spatial + self.dropout(spatial_out)
        spatial_out = self.spatial_norm(spatial_out)
        
        spatial_out = spatial_out.view(B, L, N, D).permute(0, 2, 1, 3)  # (B, N, L, D)
        
        # === Temporal Attention (across time for each node) ===
        x_temporal = spatial_out.view(B * N, L, D)  # (B*N, L, D)
        
        temporal_out, _ = self.temporal_attn(x_temporal, x_temporal, x_temporal)
        temporal_out = x_temporal + self.dropout(temporal_out)
        temporal_out = self.temporal_norm(temporal_out)
        
        temporal_out = temporal_out.view(B, N, L, D)  # (B, N, L, D)
        
        return temporal_out


class EnhancedTransformerBlock(nn.Module):
    """Transformer block with spatial-temporal attention"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.st_attn = SpatialTemporalAttention(embed_dim, num_heads, dropout)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # FFN
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Spatial-Temporal Attention
        x = x + self.st_attn(self.norm1(x))
        
        # FFN
        x = x + self.mlp(self.norm2(x))
        
        return x
