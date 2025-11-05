import torch
from torch import nn
import torch.nn.functional as F


class CrossScaleAttention(nn.Module):
    """Cross-scale attention to capture multi-scale temporal patterns"""
    def __init__(self, embed_dim, num_scales):
        super().__init__()
        self.num_scales = num_scales
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5
        
    def forward(self, scale_features):
        """
        Args:
            scale_features: List of (B, C, P, N, K) from different scales
        """
        # Concatenate all scales
        all_features = torch.cat([f.flatten(2, 4) for f in scale_features], dim=2)  # (B, C, P*N*K)
        all_features = all_features.permute(0, 2, 1)  # (B, P*N*K, C)
        
        Q = self.query(all_features)
        K = self.key(all_features)
        V = self.value(all_features)
        
        attn = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) * self.scale, dim=-1)
        out = torch.matmul(attn, V)
        
        return out.permute(0, 2, 1)  # (B, C, P*N*K)


class EnhancedMultiScalePatchEmbedding(nn.Module):
    """
    Improved multi-scale patch embedding with:
    1. Cross-scale attention
    2. Learnable scale weights
    3. Better fusion strategy
    """
    def __init__(self, patch_size, in_channel, embed_dim, num_nodes, topK, norm_layer, patch_sizes=None):
        super().__init__()
        self.output_channel = embed_dim
        self.len_patch = patch_size
        self.input_channel = in_channel
        
        if patch_sizes is None:
            self.patch_sizes = [patch_size]
            self.multi_scale = False
        else:
            self.patch_sizes = sorted(patch_sizes)  # Sort for consistency
            self.multi_scale = True
        
        num_scales = len(self.patch_sizes)
        
        # Per-scale embedding layers
        self.input_embeddings = nn.ModuleList()
        for p_size in self.patch_sizes:
            self.input_embeddings.append(
                nn.Conv3d(
                    in_channel,
                    embed_dim // num_scales if self.multi_scale else embed_dim,
                    kernel_size=(p_size, 1, topK),
                    stride=(p_size, 1, 1),
                    padding=(0, 0, 0)
                )
            )
        
        self.norm_layer = norm_layer if norm_layer is not None else nn.Identity()
        
        if self.multi_scale:
            # Learnable scale importance weights
            self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
            
            # Cross-scale attention for better fusion
            self.cross_scale_attn = CrossScaleAttention(embed_dim // num_scales, num_scales)
            
            # Final fusion layer with residual
            self.fusion = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            )
            
            # Scale-specific normalization
            self.scale_norms = nn.ModuleList([
                nn.LayerNorm(embed_dim // num_scales) for _ in range(num_scales)
            ])

    def forward(self, long_term_history):
        """
        Args:
            long_term_history: (B, L, N, K, C)
        Returns:
            output: (B, C, P, N, K) with enhanced multi-scale fusion
        """
        B, L, N, K, C = long_term_history.shape
        long_term_history = long_term_history.permute(0, 4, 1, 2, 3)  # (B, C, L, N, K)
        
        if not self.multi_scale:
            output = self.input_embeddings[0](long_term_history)
            output = self.norm_layer(output)
            return output
        
        # === Multi-scale processing ===
        multi_scale_outputs = []
        min_patches = float('inf')
        
        # Extract features at each scale
        for idx, (embedding_layer, p_size) in enumerate(zip(self.input_embeddings, self.patch_sizes)):
            scale_output = embedding_layer(long_term_history)  # (B, C/S, P, N, K)
            
            # Apply scale-specific normalization
            B, C_s, P, N, K = scale_output.shape
            scale_output = scale_output.permute(0, 2, 3, 4, 1)  # (B, P, N, K, C/S)
            scale_output = self.scale_norms[idx](scale_output)
            scale_output = scale_output.permute(0, 4, 1, 2, 3)  # (B, C/S, P, N, K)
            
            num_patches = scale_output.shape[2]
            min_patches = min(min_patches, num_patches)
            multi_scale_outputs.append(scale_output)
        
        # === Align temporal dimension across scales ===
        aligned_outputs = []
        for idx, scale_output in enumerate(multi_scale_outputs):
            if scale_output.shape[2] != min_patches:
                # Use adaptive pooling for better alignment
                B, C_s, P, N, K = scale_output.shape
                scale_output = scale_output.permute(0, 1, 3, 4, 2)  # (B, C/S, N, K, P)
                scale_output = F.adaptive_avg_pool1d(
                    scale_output.reshape(B * C_s * N * K, P),
                    min_patches
                ).reshape(B, C_s, N, K, min_patches)
                scale_output = scale_output.permute(0, 1, 4, 2, 3)  # (B, C/S, P, N, K)
            
            # Apply learnable scale weight
            scale_output = scale_output * self.scale_weights[idx]
            aligned_outputs.append(scale_output)
        
        # === Concatenate and fuse ===
        output = torch.cat(aligned_outputs, dim=1)  # (B, C, P, N, K)
        
        # Apply fusion with residual connection
        B, C_total, P, N, K = output.shape
        output_reshaped = output.permute(0, 2, 3, 4, 1)  # (B, P, N, K, C)
        
        # Residual fusion
        fused = self.fusion(output_reshaped)
        output_reshaped = output_reshaped + fused  # Residual connection
        
        output = output_reshaped.permute(0, 4, 1, 2, 3)  # (B, C, P, N, K)
        output = self.norm_layer(output)
        
        return output
