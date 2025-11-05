import torch
from torch import nn
from positional_encodings.torch_encodings import PositionalEncoding2D


class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self, num_feat=96):  # 设置默认值为96，对应embed_dim
        super().__init__()
        self.num_feat = num_feat
        # 在初始化时就创建PositionalEncoding2D，而不是在forward中动态创建
        self.tp_enc_2d = PositionalEncoding2D(num_feat)


    def forward(self, input_data, index=None, abs_idx=None):
        """Positional encoding

        Args:
            input_data (torch.tensor): input sequence with shape [B, N, P, d].
            index (list or None): add positional embedding by index.

        Returns:
            torch.tensor: output sequence
        """

        batch_size, num_nodes, num_time, num_feat = input_data.shape
        
        # 确保设备匹配
        if self.tp_enc_2d.inv_freq.device != input_data.device:
            self.tp_enc_2d = self.tp_enc_2d.to(input_data.device)
        
        # 检查特征维度是否匹配
        if self.num_feat != num_feat:
            raise ValueError(f"Expected num_feat={self.num_feat}, but got {num_feat}")
        
        pos_encoding = self.tp_enc_2d(input_data)
        input_data = input_data + pos_encoding
        return input_data, pos_encoding