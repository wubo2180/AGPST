import torch
from torch import nn


class PatchEmbedding(nn.Module):
    """Patchify time series."""

    def __init__(self, patch_size, in_channel, embed_dim, norm_layer):
        super().__init__()
        self.output_channel = embed_dim
        self.len_patch = patch_size             # the L
        self.input_channel = in_channel
        self.output_channel = embed_dim
        self.input_embedding = nn.Conv2d(
                                        in_channel,
                                        embed_dim,
                                        kernel_size=(self.len_patch, 1),
                                        stride=(self.len_patch, 1))
        self.norm_layer = norm_layer if norm_layer is not None else nn.Identity()
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights to prevent NaN"""
        nn.init.xavier_uniform_(self.input_embedding.weight)
        if self.input_embedding.bias is not None:
            nn.init.zeros_(self.input_embedding.bias)

    def forward(self, long_term_history):
        """
        Args:
            long_term_history (torch.Tensor): Very long-term historical MTS with shape [B, N, L, C],
                                                where B=batch, N=nodes, L=time_length, C=channels.

        Returns:
            torch.Tensor: patchified time series with shape [B, N, P, d]
        """
        batch_size, num_nodes, len_time_series, num_feat = long_term_history.shape
        
        # 调试信息
        if hasattr(self, '_debug_count'):
            self._debug_count += 1
        else:
            self._debug_count = 0
            
        if self._debug_count < 3:
            print(f"[PATCH EMBED DEBUG {self._debug_count}] Input shape: {long_term_history.shape}")
            print(f"  Input stats: min={long_term_history.min():.6f}, max={long_term_history.max():.6f}, mean={long_term_history.mean():.6f}")
            if torch.isnan(long_term_history).any():
                print(f"  [ERROR] Input contains NaN!")
            if torch.isinf(long_term_history).any():
                print(f"  [ERROR] Input contains Inf!")
        
        # 转换为Conv2d期望的格式: (B, N, L, C) -> (B, N, C, L)
        long_term_history = long_term_history.unsqueeze(-1) # B, N, C, L, 1
        
        if self._debug_count < 3:
            print(f"  After unsqueeze: {long_term_history.shape}, contains NaN: {torch.isnan(long_term_history).any()}")
        
        # B*N,  C, L, 1
        long_term_history = long_term_history.reshape(batch_size*num_nodes, num_feat, len_time_series, 1)
        
        if self._debug_count < 3:
            print(f"  After reshape: {long_term_history.shape}, contains NaN: {torch.isnan(long_term_history).any()}")
        
        # B*N,  d, L/P, 1
        output = self.input_embedding(long_term_history)
        
        if self._debug_count < 3:
            print(f"  After conv2d: {output.shape}, stats: min={output.min():.6f}, max={output.max():.6f}, mean={output.mean():.6f}")
            if torch.isnan(output).any():
                print(f"  [ERROR] Conv2d output contains NaN!")
                # 检查卷积层权重
                print(f"  Conv weight stats: min={self.input_embedding.weight.min():.6f}, max={self.input_embedding.weight.max():.6f}")
                if torch.isnan(self.input_embedding.weight).any():
                    print(f"  [ERROR] Conv weights contain NaN!")
        
        # norm
        output = self.norm_layer(output)
        
        if self._debug_count < 3:
            print(f"  After norm: contains NaN: {torch.isnan(output).any()}")
        
        # reshape
        output = output.squeeze(-1).view(batch_size, num_nodes, self.output_channel, -1)    # B, N, d, P
        # transpose to get (B, N, P, d) format
        output = output.transpose(-1, -2)  # B, N, P, d
        assert output.shape[2] == len_time_series / self.len_patch
        return output
