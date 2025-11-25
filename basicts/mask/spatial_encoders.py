"""
多种空间编码器实现 (Spatial Encoders)

支持:
=== 基于注意力 ===
1. TransformerSpatialEncoder - 基于 Self-Attention (原始版本)
4. GATSpatialEncoder - 基于图注意力网络 (GAT)
10. SpatialAttentionEncoder - 纯空间注意力 (无图结构)

=== 基于图神经网络 ===
2. GCNSpatialEncoder - 基于图卷积网络 (GCN)
3. ChebNetSpatialEncoder - 基于 Chebyshev 图卷积
11. SpectralGCNEncoder - 谱图卷积 (频域)
12. GraphWaveletEncoder - 图小波变换

=== 基于卷积 ===
6. CNNSpatialEncoder - 基于 1D/2D 卷积

=== 基于循环神经网络 ===
7. RNNSpatialEncoder - 基于 LSTM/GRU (跨节点)

=== 自适应/动态 ===
8. AdaptiveGraphEncoder - 自适应图学习 (无需预定义邻接矩阵)

=== 混合编码器 ===
5. HybridSpatialEncoder - GNN + Transformer 混合 (推荐)
9. MultiScaleSpatialEncoder - 多尺度空间编码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 1. TransformerSpatialEncoder (原始版本)
# ============================================================
class TransformerSpatialEncoder(nn.Module):
    """
    基于 Transformer 的空间编码器
    
    优点: 可以捕获全局依赖
    缺点: 忽略图结构,计算复杂度高 O(N²)
    """
    def __init__(self, num_nodes, d_model, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
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
        
    def forward(self, x, adj_mx=None):
        """
        Args:
            x: (B, N, T, D)
            adj_mx: 不使用 (为了接口统一)
        Returns:
            out: (B, N, T, D)
        """
        B, N, T, D = x.shape
        x_transposed = x.transpose(1, 2)  # (B, T, N, D)
        x_flat = x_transposed.reshape(B * T, N, D)
        
        spatial_features = self.encoder(x_flat)
        spatial_features = self.norm(spatial_features)
        
        spatial_features = spatial_features.reshape(B, T, N, D)
        spatial_features = spatial_features.transpose(1, 2)
        
        return spatial_features


# ============================================================
# 2. GCNSpatialEncoder (图卷积网络)
# ============================================================
class GraphConvLayer(nn.Module):
    """
    单层图卷积 (GCN Layer)
    
    公式: H' = σ(D^(-1/2) A D^(-1/2) H W)
    其中: A 是邻接矩阵, D 是度矩阵, W 是权重矩阵
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        Args:
            x: (B, N, D_in) - 节点特征
            adj: (N, N) - 归一化邻接矩阵
        Returns:
            out: (B, N, D_out)
        """
        # 线性变换: (B, N, D_in) @ (D_in, D_out) = (B, N, D_out)
        support = torch.matmul(x, self.weight)
        
        # 图卷积: (N, N) @ (B, N, D_out) = (B, N, D_out)
        # 先转置: (B, N, D_out) → (B, D_out, N)
        support = support.transpose(1, 2)
        # 矩阵乘法: (B, D_out, N) @ (N, N) = (B, D_out, N)
        output = torch.matmul(support, adj)
        # 转回: (B, D_out, N) → (B, N, D_out)
        output = output.transpose(1, 2)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class GCNSpatialEncoder(nn.Module):
    """
    基于 GCN 的空间编码器
    
    优点: 
    - 显式利用图结构
    - 计算效率高 O(E)
    - 物理意义明确
    
    输入:
    - x: (B, N, T, D)
    - adj_mx: (N, N) 归一化邻接矩阵
    """
    def __init__(self, num_nodes, d_model, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.num_layers = num_layers
        
        # 多层 GCN
        self.gcn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gcn_layers.append(
                GraphConvLayer(d_model, d_model)
            )
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
    def forward(self, x, adj_mx):
        """
        Args:
            x: (B, N, T, D)
            adj_mx: (N, N) - 归一化邻接矩阵
        Returns:
            out: (B, N, T, D)
        """
        B, N, T, D = x.shape
        
        # 逐时间步处理 (每个时间步独立进行图卷积)
        outputs = []
        for t in range(T):
            x_t = x[:, :, t, :]  # (B, N, D)
            
            # 多层 GCN + 残差连接
            for i, gcn_layer in enumerate(self.gcn_layers):
                x_residual = x_t
                x_t = gcn_layer(x_t, adj_mx)  # (B, N, D)
                x_t = self.activation(x_t)
                x_t = self.dropout(x_t)
                x_t = self.layer_norms[i](x_t + x_residual)  # 残差连接
            
            outputs.append(x_t)
        
        # 拼接: [(B, N, D)] * T → (B, N, T, D)
        output = torch.stack(outputs, dim=2)
        
        return output


# ============================================================
# 3. ChebNetSpatialEncoder (Chebyshev 图卷积)
# ============================================================
class ChebConvLayer(nn.Module):
    """
    Chebyshev 图卷积层
    
    使用 Chebyshev 多项式近似图卷积,效率更高
    公式: H' = Σ_{k=0}^{K} T_k(L_norm) H W_k
    """
    def __init__(self, in_features, out_features, K=3):
        super().__init__()
        self.K = K
        self.in_features = in_features
        self.out_features = out_features
        
        # K 个 Chebyshev 系数对应的权重
        self.weights = nn.Parameter(torch.FloatTensor(K, in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, laplacian):
        """
        Args:
            x: (B, N, D_in)
            laplacian: (N, N) - 归一化拉普拉斯矩阵
        Returns:
            out: (B, N, D_out)
        """
        B, N, D_in = x.shape
        
        # Chebyshev 递归: T_0 = I, T_1 = L, T_k = 2L*T_{k-1} - T_{k-2}
        cheb_polynomials = [x]  # T_0 = x
        if self.K > 1:
            # T_1 = L @ x
            x1 = torch.matmul(laplacian, x.transpose(1, 2)).transpose(1, 2)
            cheb_polynomials.append(x1)
            
            for k in range(2, self.K):
                # T_k = 2 * L @ T_{k-1} - T_{k-2}
                x_k = 2 * torch.matmul(laplacian, cheb_polynomials[-1].transpose(1, 2)).transpose(1, 2) - cheb_polynomials[-2]
                cheb_polynomials.append(x_k)
        
        # 加权求和: Σ T_k @ W_k
        output = torch.zeros(B, N, self.out_features, device=x.device)
        for k in range(self.K):
            # (B, N, D_in) @ (D_in, D_out) = (B, N, D_out)
            output += torch.matmul(cheb_polynomials[k], self.weights[k])
        
        output += self.bias
        
        return output


class ChebNetSpatialEncoder(nn.Module):
    """
    基于 Chebyshev 图卷积的空间编码器
    
    优点: 比 GCN 更高效,K-hop 邻域聚合
    """
    def __init__(self, num_nodes, d_model, num_layers=2, K=3, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        
        self.cheb_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.cheb_layers.append(ChebConvLayer(d_model, d_model, K=K))
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
    
    def forward(self, x, laplacian):
        """
        Args:
            x: (B, N, T, D)
            laplacian: (N, N) - 归一化拉普拉斯矩阵
        Returns:
            out: (B, N, T, D)
        """
        B, N, T, D = x.shape
        
        outputs = []
        for t in range(T):
            x_t = x[:, :, t, :]  # (B, N, D)
            
            for i, cheb_layer in enumerate(self.cheb_layers):
                x_residual = x_t
                x_t = cheb_layer(x_t, laplacian)
                x_t = self.activation(x_t)
                x_t = self.dropout(x_t)
                x_t = self.layer_norms[i](x_t + x_residual)
            
            outputs.append(x_t)
        
        output = torch.stack(outputs, dim=2)
        return output


# ============================================================
# 4. GATSpatialEncoder (图注意力网络)
# ============================================================
class GATLayer(nn.Module):
    """
    图注意力层 (Graph Attention Layer)
    
    动态学习节点间的注意力权重,不依赖预定义的邻接矩阵
    """
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.1, concat=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        
        # 每个 head 的输出维度
        self.head_dim = out_features // num_heads if concat else out_features
        
        # 线性变换
        self.W = nn.Linear(in_features, self.head_dim * num_heads, bias=False)
        
        # 注意力系数
        self.a = nn.Parameter(torch.FloatTensor(num_heads, 2 * self.head_dim, 1))
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, x, adj_mx=None):
        """
        Args:
            x: (B, N, D_in)
            adj_mx: (N, N) - 可选,用于 mask (0 表示无连接)
        Returns:
            out: (B, N, D_out)
        """
        B, N, D_in = x.shape
        
        # 线性变换: (B, N, D_in) → (B, N, num_heads * head_dim)
        h = self.W(x)  # (B, N, num_heads * head_dim)
        
        # 重塑: (B, N, num_heads, head_dim)
        h = h.view(B, N, self.num_heads, self.head_dim)
        
        # 计算注意力分数
        # 拼接: h_i || h_j for all pairs (i, j)
        # (B, N, num_heads, head_dim) → (B, N, 1, num_heads, head_dim)
        h_i = h.unsqueeze(2)  # (B, N, 1, num_heads, head_dim)
        h_j = h.unsqueeze(1)  # (B, 1, N, num_heads, head_dim)
        
        # 广播拼接: (B, N, N, num_heads, 2*head_dim)
        h_cat = torch.cat([
            h_i.expand(B, N, N, self.num_heads, self.head_dim),
            h_j.expand(B, N, N, self.num_heads, self.head_dim)
        ], dim=-1)
        
        # 注意力分数: (B, N, N, num_heads, 2*head_dim) @ (num_heads, 2*head_dim, 1)
        # = (B, N, N, num_heads)
        e = torch.matmul(h_cat, self.a.unsqueeze(0)).squeeze(-1)  # (B, N, N, num_heads)
        e = self.leaky_relu(e)
        
        # Mask (如果提供邻接矩阵)
        if adj_mx is not None:
            # (N, N) → (1, N, N, 1)
            mask = (adj_mx == 0).unsqueeze(0).unsqueeze(-1)
            e = e.masked_fill(mask, float('-inf'))
        
        # Softmax 归一化
        attention = F.softmax(e, dim=2)  # (B, N, N, num_heads)
        attention = self.dropout(attention)
        
        # 加权聚合: (B, N, N, num_heads) @ (B, N, num_heads, head_dim)
        # = (B, N, num_heads, head_dim)
        # 先转置 h: (B, N, num_heads, head_dim) → (B, num_heads, N, head_dim)
        h_transposed = h.transpose(1, 2)  # (B, num_heads, N, head_dim)
        
        # 注意力加权: 对每个 head 分别计算
        output_heads = []
        for head in range(self.num_heads):
            # (B, N, N) @ (B, N, head_dim) = (B, N, head_dim)
            out_h = torch.matmul(attention[:, :, :, head], h_transposed[:, head, :, :])
            output_heads.append(out_h)
        
        # 合并 heads
        if self.concat:
            # 拼接: (B, N, num_heads * head_dim)
            output = torch.cat(output_heads, dim=-1)
        else:
            # 平均: (B, N, head_dim)
            output = torch.mean(torch.stack(output_heads, dim=0), dim=0)
        
        return output


class GATSpatialEncoder(nn.Module):
    """
    基于图注意力网络的空间编码器
    
    优点: 动态学习邻居重要性,适应性强
    """
    def __init__(self, num_nodes, d_model, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            concat = (i < num_layers - 1)  # 最后一层不拼接
            self.gat_layers.append(
                GATLayer(d_model, d_model, num_heads=num_heads, dropout=dropout, concat=concat)
            )
        
        self.activation = nn.GELU()
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
    
    def forward(self, x, adj_mx=None):
        """
        Args:
            x: (B, N, T, D)
            adj_mx: (N, N) - 可选邻接矩阵
        Returns:
            out: (B, N, T, D)
        """
        B, N, T, D = x.shape
        
        outputs = []
        for t in range(T):
            x_t = x[:, :, t, :]  # (B, N, D)
            
            for i, gat_layer in enumerate(self.gat_layers):
                x_residual = x_t
                x_t = gat_layer(x_t, adj_mx)
                x_t = self.activation(x_t)
                x_t = self.layer_norms[i](x_t + x_residual)
            
            outputs.append(x_t)
        
        output = torch.stack(outputs, dim=2)
        return output


# ============================================================
# 5. HybridSpatialEncoder (GNN + Transformer 混合) - 推荐!
# ============================================================
class HybridSpatialEncoder(nn.Module):
    """
    混合空间编码器: GNN + Transformer
    
    设计思想:
    1. GNN 层: 捕获局部邻域结构 (1-2 hop)
    2. Transformer 层: 捕获全局长程依赖
    
    优点: 结合两者优势,性能最强
    
    推荐用于交通预测!
    """
    def __init__(self, num_nodes, d_model, num_gnn_layers=1, num_transformer_layers=1, 
                 num_heads=4, dropout=0.1, gnn_type='gcn'):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.gnn_type = gnn_type
        
        # ========== GNN 层 (局部结构) ==========
        if gnn_type == 'gcn':
            self.gnn_layers = nn.ModuleList([
                GraphConvLayer(d_model, d_model) for _ in range(num_gnn_layers)
            ])
        elif gnn_type == 'gat':
            self.gnn_layers = nn.ModuleList([
                GATLayer(d_model, d_model, num_heads=num_heads, dropout=dropout)
                for _ in range(num_gnn_layers)
            ])
        else:
            raise ValueError(f"Unknown gnn_type: {gnn_type}")
        
        self.gnn_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_gnn_layers)
        ])
        
        # ========== Transformer 层 (全局依赖) ==========
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.transformer_norm = nn.LayerNorm(d_model)
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, adj_mx=None):
        """
        Args:
            x: (B, N, T, D)
            adj_mx: (N, N) - GNN 需要的邻接矩阵
        Returns:
            out: (B, N, T, D)
        """
        B, N, T, D = x.shape
        
        outputs = []
        for t in range(T):
            x_t = x[:, :, t, :]  # (B, N, D)
            
            # ========== Stage 1: GNN (局部) ==========
            for i, gnn_layer in enumerate(self.gnn_layers):
                x_residual = x_t
                
                if self.gnn_type == 'gcn':
                    x_t = gnn_layer(x_t, adj_mx)
                else:  # gat
                    x_t = gnn_layer(x_t, adj_mx)
                
                x_t = self.activation(x_t)
                x_t = self.dropout(x_t)
                x_t = self.gnn_norms[i](x_t + x_residual)
            
            # ========== Stage 2: Transformer (全局) ==========
            x_global = self.transformer(x_t.unsqueeze(0)).squeeze(0)  # (B, N, D)
            x_t = self.transformer_norm(x_t + x_global)
            
            outputs.append(x_t)
        
        output = torch.stack(outputs, dim=2)
        return output
