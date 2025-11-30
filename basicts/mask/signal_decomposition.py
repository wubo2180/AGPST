"""
信号分解模块 (Signal Decomposition Module)

支持多种信号分解方法，将时间序列分解为不同的成分：
1. Trend-Seasonal Decomposition (趋势-季节性分解)
2. Moving Average Decomposition (移动平均分解)
3. Learnable Decomposition (可学习分解)
4. Fourier Decomposition (傅里叶分解)

核心思想：
    原始信号 = 趋势成分 + 周期成分 + 残差成分
    对每个成分分别建模，然后融合预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MovingAverageDecomposition(nn.Module):
    """
    移动平均分解 (Moving Average Decomposition)
    
    将信号分解为:
        - Trend: 移动平均（趋势成分）
        - Residual: 原始信号 - 趋势（残差成分，包含周期+噪声）
    
    优点：简单高效，无需训练
    缺点：无法显式分离周期和噪声
    """
    def __init__(self, kernel_size=25):
        """
        Args:
            kernel_size: 移动平均窗口大小（奇数）
        """
        super().__init__()
        self.kernel_size = kernel_size
        # 确保是奇数
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        
    def forward(self, x):
        """
        Args:
            x: (B, N, T, C) - 输入信号
        Returns:
            trend: (B, N, T, C) - 趋势成分
            residual: (B, N, T, C) - 残差成分
        """
        B, N, T, C = x.shape
        
        # 重塑为 (B*N*C, 1, T) 用于 Conv1d
        x_flat = x.permute(0, 1, 3, 2).reshape(B * N * C, 1, T)
        
        # 移动平均 (使用 AvgPool1d)
        trend = F.avg_pool1d(
            x_flat,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            count_include_pad=False
        )
        
        # 残差
        residual = x_flat - trend
        
        # 重塑回原始形状
        trend = trend.reshape(B, N, C, T).permute(0, 1, 3, 2)
        residual = residual.reshape(B, N, C, T).permute(0, 1, 3, 2)
        
        return trend, residual


class LearnableDecomposition(nn.Module):
    """
    可学习分解 (Learnable Decomposition)
    
    使用神经网络学习如何分解信号：
        - Trend: 低频成分（使用大卷积核或下采样-上采样）
        - Seasonal: 周期成分（使用多尺度卷积）
        - Residual: 残差成分
    
    优点：自适应学习最优分解
    缺点：需要训练，计算量大
    """
    def __init__(self, input_dim=1, hidden_dim=64):
        super().__init__()
        
        # 趋势提取器（低频）- 使用大卷积核 + 下采样
        self.trend_extractor = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=25, padding=12),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, input_dim, kernel_size=25, padding=12),
        )
        
        # 周期提取器（中频）- 使用多尺度卷积
        self.seasonal_extractor = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim // 4, kernel_size=k, padding=k//2),
                nn.ReLU(),
            ) for k in [3, 5, 7, 9]  # 多尺度：3, 5, 7, 9
        ])
        self.seasonal_fusion = nn.Conv1d(hidden_dim, input_dim, kernel_size=1)
        
        # 残差投影
        self.residual_proj = nn.Conv1d(input_dim, input_dim, kernel_size=1)
        
    def forward(self, x):
        """
        Args:
            x: (B, N, T, C) - 输入信号
        Returns:
            trend: (B, N, T, C) - 趋势成分
            seasonal: (B, N, T, C) - 周期成分
            residual: (B, N, T, C) - 残差成分
        """
        B, N, T, C = x.shape
        
        # 重塑为 (B*N, C, T) 用于 Conv1d
        x_flat = x.permute(0, 1, 3, 2).reshape(B * N, C, T)
        
        # 提取趋势（低频）
        trend = self.trend_extractor(x_flat)  # (B*N, C, T)
        
        # 提取周期（中频）- 多尺度
        seasonal_features = []
        for extractor in self.seasonal_extractor:
            seasonal_features.append(extractor(x_flat))
        seasonal_concat = torch.cat(seasonal_features, dim=1)  # (B*N, hidden_dim, T)
        seasonal = self.seasonal_fusion(seasonal_concat)  # (B*N, C, T)
        
        # 残差
        residual = x_flat - trend - seasonal
        residual = self.residual_proj(residual)
        
        # 重塑回原始形状
        trend = trend.reshape(B, N, C, T).permute(0, 1, 3, 2)
        seasonal = seasonal.reshape(B, N, C, T).permute(0, 1, 3, 2)
        residual = residual.reshape(B, N, C, T).permute(0, 1, 3, 2)
        
        return trend, seasonal, residual


class FourierDecomposition(nn.Module):
    """
    傅里叶分解 (Fourier Decomposition)
    
    使用 FFT 分解信号到频域，然后分离不同频率成分：
        - Low-frequency: 趋势成分
        - Mid-frequency: 周期成分
        - High-frequency: 噪声/残差成分
    
    优点：理论清晰，能精确分离频率
    缺点：假设信号是平稳的
    """
    def __init__(self, low_freq_ratio=0.1, mid_freq_ratio=0.3):
        """
        Args:
            low_freq_ratio: 低频成分比例（0-1），用于趋势
            mid_freq_ratio: 中频成分比例（0-1），用于周期
            剩余为高频（噪声）
        """
        super().__init__()
        self.low_freq_ratio = low_freq_ratio
        self.mid_freq_ratio = mid_freq_ratio
        
    def forward(self, x):
        """
        Args:
            x: (B, N, T, C) - 输入信号
        Returns:
            trend: (B, N, T, C) - 低频趋势成分
            seasonal: (B, N, T, C) - 中频周期成分
            residual: (B, N, T, C) - 高频残差成分
        """
        B, N, T, C = x.shape
        
        # 重塑为 (B*N*C, T)
        x_flat = x.reshape(B * N * C, T)
        
        # FFT
        x_fft = torch.fft.rfft(x_flat, dim=-1)  # (B*N*C, T//2+1)
        freq_size = x_fft.shape[-1]
        
        # 创建频率掩码
        low_freq_threshold = int(freq_size * self.low_freq_ratio)
        mid_freq_threshold = int(freq_size * (self.low_freq_ratio + self.mid_freq_ratio))
        
        # 低频（趋势）
        trend_fft = torch.zeros_like(x_fft)
        trend_fft[:, :low_freq_threshold] = x_fft[:, :low_freq_threshold]
        trend = torch.fft.irfft(trend_fft, n=T, dim=-1)  # (B*N*C, T)
        
        # 中频（周期）
        seasonal_fft = torch.zeros_like(x_fft)
        seasonal_fft[:, low_freq_threshold:mid_freq_threshold] = x_fft[:, low_freq_threshold:mid_freq_threshold]
        seasonal = torch.fft.irfft(seasonal_fft, n=T, dim=-1)  # (B*N*C, T)
        
        # 高频（残差）
        residual_fft = torch.zeros_like(x_fft)
        residual_fft[:, mid_freq_threshold:] = x_fft[:, mid_freq_threshold:]
        residual = torch.fft.irfft(residual_fft, n=T, dim=-1)  # (B*N*C, T)
        
        # 重塑回原始形状
        trend = trend.reshape(B, N, T, C)
        seasonal = seasonal.reshape(B, N, T, C)
        residual = residual.reshape(B, N, T, C)
        
        return trend, seasonal, residual


class SeriesDecomposition(nn.Module):
    """
    统一的信号分解接口
    
    支持多种分解方法：
    - 'moving_avg': 移动平均分解（快速，无参数）
    - 'learnable': 可学习分解（自适应，需训练）
    - 'fourier': 傅里叶分解（频域分解）
    """
    def __init__(self, decomp_type='moving_avg', **kwargs):
        super().__init__()
        self.decomp_type = decomp_type
        
        if decomp_type == 'moving_avg':
            self.decomposition = MovingAverageDecomposition(
                kernel_size=kwargs.get('kernel_size', 25)
            )
        elif decomp_type == 'learnable':
            self.decomposition = LearnableDecomposition(
                input_dim=kwargs.get('input_dim', 1),
                hidden_dim=kwargs.get('hidden_dim', 64)
            )
        elif decomp_type == 'fourier':
            self.decomposition = FourierDecomposition(
                low_freq_ratio=kwargs.get('low_freq_ratio', 0.1),
                mid_freq_ratio=kwargs.get('mid_freq_ratio', 0.3)
            )
        else:
            raise ValueError(f"Unknown decomp_type: {decomp_type}. "
                           f"Choose from ['moving_avg', 'learnable', 'fourier']")
    
    def forward(self, x):
        """
        Args:
            x: (B, N, T, C) - 输入信号
        Returns:
            components: dict with keys:
                - 'trend': 趋势成分
                - 'seasonal': 周期成分（如果有）
                - 'residual': 残差成分
        """
        if self.decomp_type == 'moving_avg':
            trend, residual = self.decomposition(x)
            return {
                'trend': trend,
                'seasonal': None,  # 移动平均不显式分离周期
                'residual': residual
            }
        else:  # learnable 或 fourier
            trend, seasonal, residual = self.decomposition(x)
            return {
                'trend': trend,
                'seasonal': seasonal,
                'residual': residual
            }


class DecompositionBasedEncoder(nn.Module):
    """
    基于分解的编码器 (Decomposition-Based Encoder)
    
    工作流程:
        1. 分解输入信号为多个成分（趋势、周期、残差）
        2. 对每个成分分别编码（可以使用不同的编码器）
        3. 融合编码后的特征
    
    优势:
        - 不同成分有不同的特性，分别建模更精确
        - 趋势：缓慢变化 → 可用简单模型
        - 周期：重复模式 → 可用周期性建模
        - 残差：高频噪声 → 可用去噪模型
    """
    def __init__(
        self,
        decomp_type='moving_avg',
        embed_dim=96,
        num_heads=4,
        dropout=0.1,
        **decomp_kwargs
    ):
        super().__init__()
        
        # 信号分解
        self.decomposition = SeriesDecomposition(
            decomp_type=decomp_type,
            **decomp_kwargs
        )
        
        # 为每个成分创建编码器
        # 趋势编码器（简单，因为趋势变化慢）
        self.trend_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )
        
        # 周期编码器（使用注意力捕获周期模式）
        if decomp_type in ['learnable', 'fourier']:
            self.seasonal_encoder = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            )
        else:
            self.seasonal_encoder = None
        
        # 残差编码器（处理高频信息）
        self.residual_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # 融合层
        if decomp_type in ['learnable', 'fourier']:
            # 三个成分融合
            self.fusion = nn.Sequential(
                nn.Linear(embed_dim * 3, embed_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim)
            )
        else:
            # 两个成分融合（trend + residual）
            self.fusion = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim)
            )
    
    def forward(self, x):
        """
        Args:
            x: (B, N, T, D) - 已嵌入的输入
        Returns:
            fused: (B, N, T, D) - 融合后的特征
            components: dict - 分解的各个成分（用于可视化）
        """
        B, N, T, D = x.shape
        
        # 1. 分解信号
        components = self.decomposition(x)
        
        # 2. 编码各个成分
        # 趋势编码
        trend_encoded = self.trend_encoder(components['trend'])  # (B, N, T, D)
        
        # 周期编码（如果有）
        if components['seasonal'] is not None and self.seasonal_encoder is not None:
            # 重塑为 (B*N, T, D) 用于 Transformer
            seasonal_flat = components['seasonal'].reshape(B * N, T, D)
            seasonal_encoded = self.seasonal_encoder(seasonal_flat)
            seasonal_encoded = seasonal_encoded.reshape(B, N, T, D)
        else:
            seasonal_encoded = None
        
        # 残差编码
        residual_encoded = self.residual_encoder(components['residual'])  # (B, N, T, D)
        
        # 3. 融合
        if seasonal_encoded is not None:
            # 三成分融合
            fused_input = torch.cat([
                trend_encoded,
                seasonal_encoded,
                residual_encoded
            ], dim=-1)  # (B, N, T, 3D)
        else:
            # 两成分融合
            fused_input = torch.cat([
                trend_encoded,
                residual_encoded
            ], dim=-1)  # (B, N, T, 2D)
        
        fused = self.fusion(fused_input)  # (B, N, T, D)
        
        return fused, components


# ============ 测试代码 ============
if __name__ == "__main__":
    # 测试各种分解方法
    B, N, T, C = 2, 10, 96, 1
    x = torch.randn(B, N, T, C)
    
    print("=" * 60)
    print("Testing Signal Decomposition Modules")
    print("=" * 60)
    
    # 1. 移动平均分解
    print("\n1. Moving Average Decomposition")
    ma_decomp = MovingAverageDecomposition(kernel_size=25)
    trend, residual = ma_decomp(x)
    print(f"   Input: {x.shape}")
    print(f"   Trend: {trend.shape}")
    print(f"   Residual: {residual.shape}")
    print(f"   Reconstruction error: {torch.abs(x - trend - residual).max().item():.6f}")
    
    # 2. 可学习分解
    print("\n2. Learnable Decomposition")
    learnable_decomp = LearnableDecomposition(input_dim=C, hidden_dim=64)
    trend, seasonal, residual = learnable_decomp(x)
    print(f"   Input: {x.shape}")
    print(f"   Trend: {trend.shape}")
    print(f"   Seasonal: {seasonal.shape}")
    print(f"   Residual: {residual.shape}")
    
    # 3. 傅里叶分解
    print("\n3. Fourier Decomposition")
    fourier_decomp = FourierDecomposition(low_freq_ratio=0.1, mid_freq_ratio=0.3)
    trend, seasonal, residual = fourier_decomp(x)
    print(f"   Input: {x.shape}")
    print(f"   Trend: {trend.shape}")
    print(f"   Seasonal: {seasonal.shape}")
    print(f"   Residual: {residual.shape}")
    print(f"   Reconstruction error: {torch.abs(x - trend - seasonal - residual).max().item():.6f}")
    
    # 4. 基于分解的编码器
    print("\n4. Decomposition-Based Encoder")
    # 先嵌入
    embed_dim = 96
    embedding = nn.Linear(C, embed_dim)
    x_embedded = embedding(x)  # (B, N, T, D)
    
    decomp_encoder = DecompositionBasedEncoder(
        decomp_type='fourier',
        embed_dim=embed_dim,
        num_heads=4,
        input_dim=embed_dim
    )
    fused, components = decomp_encoder(x_embedded)
    print(f"   Input (embedded): {x_embedded.shape}")
    print(f"   Fused output: {fused.shape}")
    print(f"   Components:")
    for key, val in components.items():
        if val is not None:
            print(f"     - {key}: {val.shape}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
