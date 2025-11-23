"""
交替时空架构 - HimNet 启发版本 (Alternating ST with HimNet Insights)

核心改进:
1. 异质性节点嵌入 (Heterogeneity-Aware Node Embedding)
2. GCN + Transformer 混合空间编码
3. 更鲁棒的特征融合

基于 HimNet (KDD'24) 的设计理念,同时保持我们交替架构的优势
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TemporalEncoder(nn.Module):
    """
    时间编码器 - 与 Phase 1 相同
    使用 Transformer 捕获时间依赖
    """
    def __init__(self, d_model, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        B, N, T, D = x.shape
        x_flat = x.reshape(B * N, T, D)
        temporal_features = self.encoder(x_flat)
        temporal_features = self.norm(temporal_features)
        return temporal_features.reshape(B, N, T, D)


class GraphConvLayer(nn.Module):
    """
    图卷积层 - 借鉴 HimNet 的 GCN 设计
    使用邻接矩阵聚合邻居信息
    """
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, x, adj):
        """
        Args:
            x: (B*T, N, D) - 节点特征
            adj: (N, N) - 归一化邻接矩阵
        Returns:
            out: (B*T, N, D) - 图卷积后的特征
        """
        # 1. 特征变换: X @ W
        support = torch.matmul(x, self.weight)  # (B*T, N, D_out)
        
        # 2. 邻居聚合: A @ (X @ W)
        output = torch.matmul(adj, support)  # (N, N) @ (B*T, N, D_out)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class HeterogeneousSpatialEncoder(nn.Module):
    """
    异质性感知的空间编码器 - 核心创新!
    
    设计理念 (借鉴 HimNet):
    1. 每个节点有独立的元嵌入 (捕获节点异质性)
    2. 混合 GCN + Transformer (物理先验 + 语义学习)
    3. 节点嵌入动态调制注意力权重
    
    输入: (B, N, T, D)
    输出: (B, N, T, D)
    """
    def __init__(self, num_nodes, d_model, adj_mx=None, d_meta=64, 
                 num_heads=4, num_layers=2, dropout=0.1, use_gcn=True):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.use_gcn = use_gcn
        
        # 🔥 创新 1: 节点异质性嵌入
        self.node_emb = nn.Parameter(torch.randn(num_nodes, d_meta))
        
        # 元嵌入 → Query/Key 偏置 (为不同节点生成不同注意力模式)
        self.meta_q = nn.Linear(d_meta, d_model)
        self.meta_k = nn.Linear(d_meta, d_model)
        
        # 🔥 创新 2: GCN 分支 (利用物理邻接关系)
        if use_gcn and adj_mx is not None:
            self.gcn_layers = nn.ModuleList([
                GraphConvLayer(d_model, d_model) for _ in range(2)
            ])
            self.gcn_norm = nn.LayerNorm(d_model)
            
            # 归一化邻接矩阵 (只做一次)
            self.register_buffer('adj_mx', self._normalize_adj(adj_mx))
        else:
            self.gcn_layers = None
        
        # 🔥 创新 3: Transformer 分支 (学习隐式语义关系)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.trans_norm = nn.LayerNorm(d_model)
        
        # 融合 GCN + Transformer
        if use_gcn and adj_mx is not None:
            self.fusion = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        
    def _normalize_adj(self, adj_mx):
        """
        对称归一化: D^(-0.5) @ A @ D^(-0.5)
        """
        adj_mx = adj_mx + torch.eye(adj_mx.size(0), device=adj_mx.device)  # 添加自环
        rowsum = adj_mx.sum(1)
        d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return d_mat_inv_sqrt @ adj_mx @ d_mat_inv_sqrt
    
    def forward(self, x):
        """
        Args:
            x: (B, N, T, D)
        Returns:
            out: (B, N, T, D) - 空间编码后的特征
        """
        B, N, T, D = x.shape
        
        # 生成节点特定的 Q/K 偏置
        node_q_bias = self.meta_q(self.node_emb)  # (N, D)
        node_k_bias = self.meta_k(self.node_emb)  # (N, D)
        
        # 重塑: (B, N, T, D) → (B*T, N, D)
        x_flat = x.permute(0, 2, 1, 3).reshape(B * T, N, D)
        
        # === Transformer 分支 ===
        # 添加节点特定的偏置
        x_trans = x_flat + node_q_bias.unsqueeze(0)  # 广播到 (B*T, N, D)
        x_trans = self.transformer(x_trans)
        x_trans = self.trans_norm(x_trans)
        
        # === GCN 分支 (如果启用) ===
        if self.use_gcn and self.gcn_layers is not None:
            x_gcn = x_flat
            for gcn_layer in self.gcn_layers:
                x_gcn = F.relu(gcn_layer(x_gcn, self.adj_mx))
            x_gcn = self.gcn_norm(x_gcn)
            
            # 融合两条路径
            x_fused = torch.cat([x_trans, x_gcn], dim=-1)  # (B*T, N, 2D)
            spatial_features = self.fusion(x_fused)  # (B*T, N, D)
        else:
            spatial_features = x_trans
        
        # 重塑回原始形状: (B*T, N, D) → (B, N, T, D)
        spatial_features = spatial_features.reshape(B, T, N, D).permute(0, 2, 1, 3)
        
        return spatial_features


class ImprovedFusionLayer(nn.Module):
    """
    改进的融合层 - 更鲁棒的特征融合
    
    借鉴 HimNet 的门控机制 + Cross-Attention
    """
    def __init__(self, d_model, num_heads=4, dropout=0.1, fusion_type='gated_cross_attn'):
        super().__init__()
        self.d_model = d_model
        self.fusion_type = fusion_type
        
        if fusion_type == 'gated_cross_attn':
            # Cross-Attention: 时间特征 attend to 空间特征
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            
            # 门控机制
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )
            
            self.fusion_proj = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            
        elif fusion_type == 'cross_attn':
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.norm = nn.LayerNorm(d_model)
            
        else:  # 'gated'
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )
            self.fusion_proj = nn.Linear(d_model * 2, d_model)
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, temporal_feat, spatial_feat):
        """
        Args:
            temporal_feat: (B, N, T, D)
            spatial_feat: (B, N, T, D)
        Returns:
            fused: (B, N, T, D)
        """
        B, N, T, D = temporal_feat.shape
        
        if self.fusion_type == 'gated_cross_attn':
            # 重塑用于 Cross-Attention
            temp_flat = temporal_feat.reshape(B * N, T, D)
            spat_flat = spatial_feat.reshape(B * N, T, D)
            
            # Cross-Attention: temporal as Query, spatial as Key/Value
            attn_out, _ = self.cross_attn(temp_flat, spat_flat, spat_flat)
            attn_out = attn_out.reshape(B, N, T, D)
            
            # 门控融合
            concat = torch.cat([temporal_feat, attn_out], dim=-1)
            gate = self.gate(concat)
            fused = self.fusion_proj(concat) * gate + temporal_feat * (1 - gate)
            
        elif self.fusion_type == 'cross_attn':
            temp_flat = temporal_feat.reshape(B * N, T, D)
            spat_flat = spatial_feat.reshape(B * N, T, D)
            
            attn_out, _ = self.cross_attn(temp_flat, spat_flat, spat_flat)
            fused = self.norm(temp_flat + attn_out)
            fused = fused.reshape(B, N, T, D)
            
        else:  # 'gated'
            concat = torch.cat([temporal_feat, spatial_feat], dim=-1)
            gate = self.gate(concat)
            fused = self.fusion_proj(concat) * gate + temporal_feat * (1 - gate)
        
        return self.norm(fused)


class STDecoder(nn.Module):
    """
    时空解码器 - 与 Phase 1 相同
    将融合特征解码回时间和空间维度
    """
    def __init__(self, d_model, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 时空特征分离投影
        self.temporal_proj = nn.Linear(d_model, d_model)
        self.spatial_proj = nn.Linear(d_model, d_model)
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, fused_features, num_queries=24):
        """
        Args:
            fused_features: (B, N, T, D) - 融合后的时空特征
            num_queries: int - 解码器查询数量
        Returns:
            temporal_decoded: (B, N, T, D)
            spatial_decoded: (B, N, T, D)
        """
        B, N, T, D = fused_features.shape
        
        # 创建可学习的查询向量
        queries = fused_features.mean(dim=(1, 2), keepdim=True).expand(B, N, T, D)
        
        # 重塑用于 Transformer Decoder
        queries_flat = queries.reshape(B * N, T, D)
        memory_flat = fused_features.reshape(B * N, T, D)
        
        # 解码
        decoded = self.decoder(queries_flat, memory_flat)
        decoded = self.norm(decoded)
        decoded = decoded.reshape(B, N, T, D)
        
        # 分离时空特征
        temporal_decoded = self.temporal_proj(decoded)
        spatial_decoded = self.spatial_proj(decoded)
        
        return temporal_decoded, spatial_decoded


class AlternatingSTModel_HimNet(nn.Module):
    """
    交替时空模型 - HimNet 启发版本
    
    核心改进:
    1. ✅ 异质性节点嵌入 (每个节点有独立元嵌入)
    2. ✅ GCN + Transformer 混合空间编码
    3. ✅ 改进的门控融合机制
    4. ✅ 保持交替架构的优势 (信息流动)
    
    架构:
        Input → Temporal Enc1 → Spatial Enc1 (Heterogeneous + GCN) → Fusion1 →
        Decoder → Temporal Enc2 → Spatial Enc2 (Heterogeneous + GCN) → Fusion2 → Output
    """
    def __init__(
        self,
        num_nodes,
        in_channel,
        embed_dim,
        output_len=12,
        input_len=12,
        adj_mx=None,
        num_heads=4,
        temporal_depth_1=1,
        spatial_depth_1=1,
        temporal_depth_2=3,
        spatial_depth_2=3,
        decoder_depth=2,
        dropout=0.1,
        fusion_type='gated_cross_attn',
        use_gcn=True,
        d_meta=64
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.embed_dim = embed_dim
        
        # 输入嵌入
        self.input_proj = nn.Linear(in_channel, embed_dim)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, num_nodes, input_len, embed_dim))
        
        # === Stage 1: 初步特征提取 ===
        self.temporal_encoder_1 = TemporalEncoder(
            embed_dim, num_heads, temporal_depth_1, dropout
        )
        
        self.spatial_encoder_1 = HeterogeneousSpatialEncoder(
            num_nodes, embed_dim, adj_mx, d_meta, 
            num_heads, spatial_depth_1, dropout, use_gcn
        )
        
        self.fusion_1 = ImprovedFusionLayer(
            embed_dim, num_heads, dropout, fusion_type
        )
        
        # 解码器
        self.decoder = STDecoder(embed_dim, num_heads, decoder_depth, dropout)
        
        # === Stage 2: 精细化建模 ===
        self.temporal_encoder_2 = TemporalEncoder(
            embed_dim, num_heads, temporal_depth_2, dropout
        )
        
        self.spatial_encoder_2 = HeterogeneousSpatialEncoder(
            num_nodes, embed_dim, adj_mx, d_meta,
            num_heads, spatial_depth_2, dropout, use_gcn
        )
        
        self.fusion_2 = ImprovedFusionLayer(
            embed_dim, num_heads, dropout, fusion_type
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, in_channel)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, history_data, **kwargs):
        """
        Args:
            history_data: (B, T, N, C) - 历史数据
        Returns:
            prediction: (B, T_pred, N, C)
        """
        # 转换维度: (B, T, N, C) → (B, N, T, C)
        x = history_data.permute(0, 2, 1, 3)
        B, N, T, C = x.shape
        
        # 输入嵌入
        x = self.input_proj(x)  # (B, N, T, D)
        x = x + self.pos_encoding[:, :, :T, :]  # 位置编码
        
        # === Stage 1: 初步编码 ===
        temp_out_1 = self.temporal_encoder_1(x)
        spat_out_1 = self.spatial_encoder_1(temp_out_1)
        fused_1 = self.fusion_1(temp_out_1, spat_out_1)
        
        # 解码
        temporal_decoded, spatial_decoded = self.decoder(fused_1)
        
        # === Stage 2: 精细化编码 ===
        # 使用解码结果作为输入
        stage2_input = temporal_decoded + spatial_decoded
        
        temp_out_2 = self.temporal_encoder_2(stage2_input)
        spat_out_2 = self.spatial_encoder_2(temp_out_2)
        final_features = self.fusion_2(temp_out_2, spat_out_2)
        
        # 输出投影
        output = self.output_proj(final_features)  # (B, N, T, C)
        
        # 转换回原始维度: (B, N, T, C) → (B, T, N, C)
        output = output.permute(0, 2, 1, 3)
        
        return output
