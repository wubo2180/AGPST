import math
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F


class ImprovedTransformerLayer(nn.Module):
    """Enhanced Transformer with Pre-LN, GELU, and better normalization"""
    def __init__(self, hidden_dim, num_heads, mlp_ratio, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=False)
        
        # Pre-LayerNorm (more stable than Post-LN)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # FFN with GELU (better than ReLU for transformers)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),  # Changed from ReLU
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Stochastic Depth for better regularization
        self.drop_path = nn.Identity()  # Can be replaced with DropPath
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Pre-LN: norm before attention
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)
        src = src + self.drop_path(src2)
        
        # Pre-LN: norm before FFN
        src2 = self.norm2(src)
        src2 = self.ffn(src2)
        src = src + self.drop_path(src2)
        
        return src


class ImprovedTransformerLayers(nn.Module):
    """Enhanced version with Pre-LN and better components"""
    def __init__(self, hidden_dim, nlayers, mlp_ratio, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = hidden_dim
        self.layers = nn.ModuleList([
            ImprovedTransformerLayer(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(nlayers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)  # Final norm for Pre-LN architecture

    def forward(self, src):
        B, N, L, D = src.shape
        src = src * math.sqrt(self.d_model)
        src = src.contiguous()
        src = src.view(B*N, L, D)
        src = src.transpose(0, 1)  # (L, B*N, D)
        
        for layer in self.layers:
            src = layer(src)
        
        output = self.final_norm(src)  # Final normalization
        output = output.transpose(0, 1).view(B, N, L, D)
        return output
