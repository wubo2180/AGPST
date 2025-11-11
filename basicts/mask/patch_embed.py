"""
Patch Embedding模块
将时间序列转换为patches
"""
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    时间序列Patch嵌入
    输入: (B, N, L, C) - Batch, Nodes, TimeLength, Channels
    输出: (B, N, P, D) - Batch, Nodes, Patches, EmbedDim
    """
    def __init__(self, patch_size, in_channel, embed_dim, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        
        self.input_embedding = nn.Conv2d(
            in_channel,
            embed_dim,
            kernel_size=(patch_size, 1),
            stride=(patch_size, 1)
        )
        
        self.norm_layer = norm_layer if norm_layer is not None else nn.Identity()
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重防止NaN"""
        nn.init.xavier_uniform_(self.input_embedding.weight)
        if self.input_embedding.bias is not None:
            nn.init.zeros_(self.input_embedding.bias)

    def forward(self, long_term_history):
        """
        Args:
            long_term_history: (B, N, L, C) 长期历史数据
        Returns:
            output: (B, N, P, D) patch嵌入
        """
        B, N, L, C = long_term_history.shape
        
        # (B, N, L, C) -> (B*N, C, L, 1)
        x = long_term_history.unsqueeze(-1)
        x = x.reshape(B * N, C, L, 1)
        
        # Conv2d: (B*N, C, L, 1) -> (B*N, D, P, 1)
        x = self.input_embedding(x)
        x = self.norm_layer(x)
        
        # (B*N, D, P, 1) -> (B, N, P, D)
        x = x.squeeze(-1).view(B, N, self.embed_dim, -1)
        x = x.transpose(-1, -2)
        
        return x
