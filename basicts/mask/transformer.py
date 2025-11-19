"""
Transformer编码器模块
用于时序建模
"""
import math
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerLayers(nn.Module):
    """
    Transformer编码器
    输入: (B, N, P, D) - Batch, Nodes, Patches, Dimensions
    输出: (B, N, P, D)
    """
    def __init__(self, hidden_dim, nlayers, mlp_ratio, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = hidden_dim
        
        encoder_layers = TransformerEncoderLayer(
            hidden_dim, 
            num_heads, 
            hidden_dim * mlp_ratio, 
            dropout
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def forward(self, src):
        """
        Args:
            src: (B, N, P, D)
        Returns:
            output: (B, N, P, D)
        """
        B, N, P, D = src.shape
        
        # 缩放
        src = src * math.sqrt(self.d_model)
        
        # (B, N, P, D) -> (P, B*N, D)
        src = src.contiguous().view(B * N, P, D)
        src = src.transpose(0, 1)
        
        # Transformer编码
        output = self.transformer_encoder(src, mask=None)
        
        # (P, B*N, D) -> (B, N, P, D)
        output = output.transpose(0, 1).view(B, N, P, D)
        
        return output
