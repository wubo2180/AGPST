"""
交替时空编码解码架构 (Alternating Spatio-Temporal Architecture)

核心思想: 分离时间和空间的建模，通过交替编码-融合-解码-再编码的方式
逐步提取和精炼时空特征

架构流程:
    Input → Temporal Enc → Spatial Enc → Fusion → 
    Decoder → Temporal Enc2 → Spatial Enc2 → Fusion2 → Output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TemporalEncoder(nn.Module):
    """
    时间编码器 - 专门处理时间维度的依赖关系
    
    输入: (B, N, T, D)  [batch, nodes, time, features]
    输出: (B, N, T, D)
    
    使用 1D Conv 或 Transformer 捕获每个节点的时间演变模式
    """
    def __init__(self, d_model, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Temporal Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 层归一化
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Args:
            x: (B, N, T, D)
        Returns:
            out: (B, N, T, D) - 每个节点的时间特征
        """
        B, N, T, D = x.shape
        
        # 重塑: 将 N 个节点分别处理
        # (B, N, T, D) → (B*N, T, D)
        x_flat = x.reshape(B * N, T, D)
        
        # Transformer 编码时间序列
        # (B*N, T, D) → (B*N, T, D)
        temporal_features = self.encoder(x_flat)
        temporal_features = self.norm(temporal_features)
        
        # 重塑回原始形状
        # (B*N, T, D) → (B, N, T, D)
        temporal_features = temporal_features.reshape(B, N, T, D)
        
        return temporal_features


class SpatialEncoder(nn.Module):
    """
    空间编码器 - 专门处理空间维度的依赖关系
    
    输入: (B, N, T, D)  [batch, nodes, time, features]
    输出: (B, N, T, D)
    
    使用图卷积或空间注意力捕获每个时刻的节点间关系
    """
    def __init__(self, num_nodes, d_model, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        
        # 方案1: 使用 Self-Attention 作为空间编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 层归一化
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Args:
            x: (B, N, T, D)
        Returns:
            out: (B, N, T, D) - 每个时刻的空间特征
        """
        B, N, T, D = x.shape
        
        # 重塑: 将 T 个时间步分别处理
        # (B, N, T, D) → (B, T, N, D) → (B*T, N, D)
        x_transposed = x.transpose(1, 2)  # (B, T, N, D)
        x_flat = x_transposed.reshape(B * T, N, D)
        
        # Transformer 编码空间关系
        # (B*T, N, D) → (B*T, N, D)
        spatial_features = self.encoder(x_flat)
        spatial_features = self.norm(spatial_features)
        
        # 重塑回原始形状
        # (B*T, N, D) → (B, T, N, D) → (B, N, T, D)
        spatial_features = spatial_features.reshape(B, T, N, D)
        spatial_features = spatial_features.transpose(1, 2)
        
        return spatial_features


class FusionLayer(nn.Module):
    """
    时空特征融合层
    
    输入: temporal_feat (B, N, T, D), spatial_feat (B, N, T, D)
    输出: fused_feat (B, N, T, D)
    
    支持三种融合方式:
    1. 'concat': 拼接后投影
    2. 'gated': 门控融合 (类似 LSTM 的门机制)
    3. 'cross_attn': 交叉注意力融合
    """
    def __init__(self, d_model, fusion_type='gated', dropout=0.1):
        super().__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == 'concat':
            # 拼接后投影
            self.projection = nn.Sequential(
                nn.Linear(d_model * 2, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model)
            )
            
        elif fusion_type == 'gated':
            # 门控融合 (推荐)
            self.temporal_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )
            self.spatial_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )
            self.fusion_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model)
            )
            
        elif fusion_type == 'cross_attn':
            # 交叉注意力
            self.cross_attn_t2s = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            self.cross_attn_s2t = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            self.fusion_norm = nn.LayerNorm(d_model)
        
    def forward(self, temporal_feat, spatial_feat):
        """
        Args:
            temporal_feat: (B, N, T, D)
            spatial_feat: (B, N, T, D)
        Returns:
            fused: (B, N, T, D)
        """
        if self.fusion_type == 'concat':
            # 简单拼接后投影
            concat = torch.cat([temporal_feat, spatial_feat], dim=-1)  # (B, N, T, 2D)
            fused = self.projection(concat)  # (B, N, T, D)
            
        elif self.fusion_type == 'gated':
            # 门控融合 (类似 LSTM)
            concat = torch.cat([temporal_feat, spatial_feat], dim=-1)  # (B, N, T, 2D)
            
            # 计算门控权重
            temporal_weight = self.temporal_gate(concat)  # (B, N, T, D)
            spatial_weight = self.spatial_gate(concat)    # (B, N, T, D)
            
            # 归一化门控权重 (确保和为1)
            total_weight = temporal_weight + spatial_weight
            temporal_weight = temporal_weight / (total_weight + 1e-8)
            spatial_weight = spatial_weight / (total_weight + 1e-8)
            
            # 加权融合
            fused = temporal_weight * temporal_feat + spatial_weight * spatial_feat
            fused = self.fusion_proj(fused)
            
        elif self.fusion_type == 'cross_attn':
            # 交叉注意力融合
            B, N, T, D = temporal_feat.shape
            
            # 重塑为 (B*N, T, D) 用于 MultiheadAttention
            t_flat = temporal_feat.reshape(B * N, T, D)
            s_flat = spatial_feat.reshape(B * N, T, D)
            
            # Temporal attends to Spatial
            t2s, _ = self.cross_attn_t2s(t_flat, s_flat, s_flat)
            # Spatial attends to Temporal
            s2t, _ = self.cross_attn_s2t(s_flat, t_flat, t_flat)
            
            # 融合
            fused = self.fusion_norm(t2s + s2t + t_flat + s_flat)
            fused = fused.reshape(B, N, T, D)
        
        return fused


class STDecoder(nn.Module):
    """
    时空解码器 - 将融合特征解码回时间和空间分量
    
    输入: (B, N, T, D)
    输出: temporal_component (B, N, T, D), spatial_component (B, N, T, D)
    
    这一步是为了保持时空特征的可分离性，为第二阶段编码做准备
    """
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        
        # 使用 Transformer Decoder 解码
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        
        # 分离投影: 将解码结果投影到时间和空间分量
        self.temporal_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )
        self.spatial_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )
        
        # 学习查询向量 (用于解码)
        self.temporal_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.spatial_query = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 初始化
        nn.init.xavier_uniform_(self.temporal_query)
        nn.init.xavier_uniform_(self.spatial_query)
        
    def forward(self, fused_features):
        """
        Args:
            fused_features: (B, N, T, D)
        Returns:
            temporal_component: (B, N, T, D)
            spatial_component: (B, N, T, D)
        """
        B, N, T, D = fused_features.shape
        
        # 重塑为 (B*N, T, D) 用于 Transformer
        fused_flat = fused_features.reshape(B * N, T, D)
        
        # 解码 (使用可学习的 query)
        temporal_queries = self.temporal_query.expand(B * N, T, -1)
        spatial_queries = self.spatial_query.expand(B * N, T, -1)
        
        # Decoder: query attends to memory
        decoded_temporal = self.decoder(temporal_queries, fused_flat)  # (B*N, T, D)
        decoded_spatial = self.decoder(spatial_queries, fused_flat)    # (B*N, T, D)
        
        # 投影到时间和空间分量
        temporal_component = self.temporal_proj(decoded_temporal)
        spatial_component = self.spatial_proj(decoded_spatial)
        
        # 重塑回原始形状
        temporal_component = temporal_component.reshape(B, N, T, D)
        spatial_component = spatial_component.reshape(B, N, T, D)
        
        return temporal_component, spatial_component


class AlternatingSTModel(nn.Module):
    """
    完整的交替时空编码解码模型
    
    架构流程:
        Input (B, N, T, 1)
          ↓ Embedding
        (B, N, T, D)
          ↓
        ┌─── Stage 1: 初级特征提取 ───┐
        │ Temporal Encoder 1          │
        │ Spatial Encoder 1           │
        │ Fusion Layer 1              │
        └─────────────────────────────┘
          ↓
        ┌─── Decoder: 特征解构 ───┐
        │ ST Decoder                │
        │ → temporal + spatial      │
        └───────────────────────────┘
          ↓
        ┌─── Stage 2: 深层特征精炼 ───┐
        │ Temporal Encoder 2           │
        │ Spatial Encoder 2            │
        │ Fusion Layer 2               │
        └──────────────────────────────┘
          ↓
        Output Projection
          ↓
        Prediction (B, N, T_future, 1)
    """
    def __init__(
        self,
        num_nodes,
        in_steps=12,
        out_steps=12,
        input_dim=1,
        embed_dim=96,
        num_heads=4,
        temporal_depth_1=2,
        spatial_depth_1=2,
        temporal_depth_2=2,
        spatial_depth_2=2,
        fusion_type='gated',
        dropout=0.05,
        use_denoising=True,
        denoise_type='conv',
        **kwargs
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.embed_dim = embed_dim
        self.use_denoising = use_denoising
        self.denoise_type = denoise_type
        
        # ============ 输入嵌入 ============
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )
        
        # ============ 去噪模块 (可选) ============
        if use_denoising:
            if denoise_type == 'conv':
                # 卷积去噪 (快速,适合局部噪声)
                self.denoise = nn.Sequential(
                    nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
                )
            elif denoise_type == 'attention':
                # 注意力去噪 (更强大,适合复杂噪声模式)
                self.denoise = nn.MultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                )
                self.denoise_norm = nn.LayerNorm(embed_dim)
            else:
                raise ValueError(f"Unknown denoise_type: {denoise_type}. Choose 'conv' or 'attention'.")
        
        # ============ Stage 1: 初级特征提取 ============
        self.temporal_encoder_1 = TemporalEncoder(
            d_model=embed_dim,
            num_heads=num_heads,
            num_layers=temporal_depth_1,
            dropout=dropout
        )
        
        self.spatial_encoder_1 = SpatialEncoder(
            num_nodes=num_nodes,
            d_model=embed_dim,
            num_heads=num_heads,
            num_layers=spatial_depth_1,
            dropout=dropout
        )
        
        self.fusion_1 = FusionLayer(
            d_model=embed_dim,
            fusion_type=fusion_type,
            dropout=dropout
        )
        
        # ============ Decoder: 特征解构 ============
        self.st_decoder = STDecoder(
            d_model=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # ============ Stage 2: 深层特征精炼 ============
        self.temporal_encoder_2 = TemporalEncoder(
            d_model=embed_dim,
            num_heads=num_heads,
            num_layers=temporal_depth_2,
            dropout=dropout
        )
        
        self.spatial_encoder_2 = SpatialEncoder(
            num_nodes=num_nodes,
            d_model=embed_dim,
            num_heads=num_heads,
            num_layers=spatial_depth_2,
            dropout=dropout
        )
        
        self.fusion_2 = FusionLayer(
            d_model=embed_dim,
            fusion_type=fusion_type,
            dropout=dropout
        )
        
        # ============ 输出投影 ============
        # 首先调整时间维度: in_steps → out_steps
        self.temporal_projection = nn.Linear(in_steps, out_steps)
        
        # 然后将特征维度投影到输出
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, input_dim)
        )
        
        # ============ 位置编码 ============
        self.register_buffer(
            'positional_encoding',
            self._get_sinusoidal_encoding(in_steps, embed_dim)
        )
        
    def _get_sinusoidal_encoding(self, seq_len, d_model):
        """生成固定的 sin/cos 位置编码"""
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * 
                            -(math.log(10000.0) / d_model))
        
        pos_encoding = torch.zeros(seq_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
    
    def forward(self, history_data, **kwargs):
        """
        Args:
            history_data: (B, T, N, C) 或 (B, N, T, C)
        Returns:
            prediction: (B, T_future, N, 1)
        """
        # 统一输入格式: (B, N, T, C)
        if history_data.dim() == 4:
            if history_data.shape[1] == self.in_steps:  # (B, T, N, C)
                x = history_data.transpose(1, 2)  # → (B, N, T, C)
            else:  # (B, N, T, C)
                x = history_data
        else:
            raise ValueError(f"Unexpected input shape: {history_data.shape}")
        
        B, N, T, C = x.shape
        
        # ============ 输入嵌入 ============
        # (B, N, T, C) → (B, N, T, D)
        x = self.input_embedding(x)
        
        # 添加位置编码
        x = x + self.positional_encoding[:, :, :T, :]
        
        # ============ 去噪 (可选) ============
        if self.use_denoising:
            if self.denoise_type == 'conv':
                # 卷积去噪 (快速)
                # (B, N, T, D) → (B*N, D, T)
                x_denoise = x.reshape(B * N, T, self.embed_dim).transpose(1, 2)
                # Conv1d 去噪
                x_denoise = self.denoise(x_denoise)
                # 残差连接: (B*N, D, T) → (B*N, T, D) → (B, N, T, D)
                x_denoise = x_denoise.transpose(1, 2).reshape(B, N, T, self.embed_dim)
                x = x + x_denoise
                
            elif self.denoise_type == 'attention':
                # 注意力去噪 (更强大)
                # (B, N, T, D) → (B*N, T, D)
                x_denoise = x.reshape(B * N, T, self.embed_dim)
                # Self-Attention 去噪
                x_denoise_attn, _ = self.denoise(x_denoise, x_denoise, x_denoise)
                # 残差连接 + 归一化
                x_denoise = self.denoise_norm(x_denoise + x_denoise_attn)
                # (B*N, T, D) → (B, N, T, D)
                x_denoise = x_denoise.reshape(B, N, T, self.embed_dim)
                x = x + x_denoise
        
        # ============ Stage 1: 初级特征提取 ============
        # 时间编码
        temporal_feat_1 = self.temporal_encoder_1(x)  # (B, N, T, D)
        
        # 空间编码
        spatial_feat_1 = self.spatial_encoder_1(x)  # (B, N, T, D)
        
        # 融合
        fused_1 = self.fusion_1(temporal_feat_1, spatial_feat_1)  # (B, N, T, D)
        
        # ============ Decoder: 特征解构 ============
        temporal_component, spatial_component = self.st_decoder(fused_1)
        
        # ============ Stage 2: 深层特征精炼 ============
        # 时间再编码 (使用解构的时间分量)
        temporal_feat_2 = self.temporal_encoder_2(temporal_component)  # (B, N, T, D)
        
        # 空间再编码 (使用解构的空间分量)
        spatial_feat_2 = self.spatial_encoder_2(spatial_component)  # (B, N, T, D)
        
        # 最终融合
        fused_2 = self.fusion_2(temporal_feat_2, spatial_feat_2)  # (B, N, T, D)
        
        # ============ 输出投影 ============
        # 调整时间维度: (B, N, T, D) → (B, N, T_future, D)
        # 先转置: (B, N, T, D) → (B, N, D, T)
        x_out = fused_2.transpose(2, 3)
        # 投影: (B, N, D, T) → (B, N, D, T_future)
        x_out = self.temporal_projection(x_out)
        # 转回: (B, N, D, T_future) → (B, N, T_future, D)
        x_out = x_out.transpose(2, 3)
        
        # 特征投影: (B, N, T_future, D) → (B, N, T_future, 1)
        prediction = self.output_projection(x_out)
        
        # 转换为输出格式: (B, N, T_future, 1) → (B, T_future, N, 1)
        prediction = prediction.transpose(1, 2)
        
        return prediction
