"""
Advanced Adaptive Graph Construction Methods for Spatial-Temporal Modeling

提供多种先进的动态邻接矩阵构建方法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimpleAdaptiveGraph(nn.Module):
    """原始方法: 简单的矩阵乘法 + ReLU + Softmax"""
    
    def __init__(self, num_nodes, embed_dim):
        super().__init__()
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(embed_dim, num_nodes), requires_grad=True)
    
    def forward(self, x=None):
        """
        Args:
            x: 输入特征 (可选), shape: [B, N, D]
        Returns:
            adp: 邻接矩阵, shape: [N, N]
        """
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        return adp


class MultiHeadAdaptiveGraph(nn.Module):
    """方法1: 多头注意力机制的自适应图
    
    优点:
    - 能够捕获多种不同类型的节点关系
    - 类似于 Transformer 的多头注意力
    - 表达能力更强
    """
    
    def __init__(self, num_nodes, embed_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        
        # 每个头有独立的节点嵌入
        self.node_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(num_nodes, self.head_dim), requires_grad=True)
            for _ in range(num_heads)
        ])
        
        # 可选: 头融合层
        self.fusion = nn.Linear(num_heads, 1)
    
    def forward(self, x=None):
        """
        Returns:
            adp: 融合后的邻接矩阵, shape: [N, N]
        """
        adp_list = []
        for i in range(self.num_heads):
            # 每个头计算自己的邻接矩阵
            adp_i = torch.mm(self.node_embeddings[i], self.node_embeddings[i].T)
            adp_i = F.softmax(F.relu(adp_i), dim=1)
            adp_list.append(adp_i)
        
        # 堆叠所有头: [N, N, num_heads]
        adp_stacked = torch.stack(adp_list, dim=-1)
        
        # 融合多个头
        adp = self.fusion(adp_stacked).squeeze(-1)
        adp = F.softmax(adp, dim=1)
        
        return adp


class DynamicAdaptiveGraph(nn.Module):
    """方法2: 动态自适应图 (基于输入特征)
    
    优点:
    - 根据当前输入动态调整图结构
    - 更适应数据的时变特性
    - 能够捕获动态的空间依赖
    """
    
    def __init__(self, num_nodes, embed_dim, feature_dim):
        super().__init__()
        # 静态节点嵌入
        self.static_nodevec1 = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
        self.static_nodevec2 = nn.Parameter(torch.randn(embed_dim, num_nodes), requires_grad=True)
        
        # 动态嵌入生成器
        self.dynamic_encoder = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # 融合权重
        self.alpha = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x):
        """
        Args:
            x: 输入特征, shape: [B, N, D] 或 [B, T, N, D]
        Returns:
            adp: 动态邻接矩阵, shape: [B, N, N] 或 [N, N]
        """
        # 静态图
        static_adp = F.softmax(F.relu(torch.mm(self.static_nodevec1, self.static_nodevec2)), dim=1)
        
        if x is None:
            return static_adp
        
        # 处理输入维度
        if x.dim() == 4:  # [B, T, N, D]
            x = x.mean(dim=1)  # 平均时间维度: [B, N, D]
        
        # 动态节点嵌入
        dynamic_emb = self.dynamic_encoder(x)  # [B, N, embed_dim]
        
        # 计算动态图 (批量版本)
        dynamic_adp = torch.bmm(dynamic_emb, dynamic_emb.transpose(1, 2))  # [B, N, N]
        dynamic_adp = F.softmax(F.relu(dynamic_adp), dim=-1)
        
        # 融合静态和动态图
        alpha = torch.sigmoid(self.alpha)
        adp = alpha * static_adp.unsqueeze(0) + (1 - alpha) * dynamic_adp
        
        # 如果需要单个图，取平均
        if adp.size(0) > 1:
            adp = adp.mean(dim=0)
        else:
            adp = adp.squeeze(0)
        
        return adp


class GaussianAdaptiveGraph(nn.Module):
    """方法3: 高斯核自适应图
    
    优点:
    - 平滑的相似度度量
    - 自动学习最优带宽
    - 更鲁棒的距离度量
    """
    
    def __init__(self, num_nodes, embed_dim):
        super().__init__()
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
        # 可学习的带宽参数
        self.bandwidth = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x=None):
        """
        Returns:
            adp: 高斯核邻接矩阵, shape: [N, N]
        """
        # 计算节点间的欧氏距离
        diff = self.node_embeddings.unsqueeze(1) - self.node_embeddings.unsqueeze(0)  # [N, N, D]
        dist_sq = (diff ** 2).sum(dim=-1)  # [N, N]
        
        # 高斯核
        bandwidth = F.softplus(self.bandwidth)  # 确保 > 0
        adp = torch.exp(-dist_sq / (2 * bandwidth ** 2))
        
        # 归一化
        adp = adp / (adp.sum(dim=1, keepdim=True) + 1e-8)
        
        return adp


class HyperbolicAdaptiveGraph(nn.Module):
    """方法4: 双曲空间自适应图
    
    优点:
    - 适合层次结构数据（如道路网络）
    - 在低维空间中捕获复杂的层次关系
    - 特别适合交通网络这种有层次的图结构
    """
    
    def __init__(self, num_nodes, embed_dim, curv=1.0):
        super().__init__()
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
        self.curv = curv  # 曲率参数
    
    def forward(self, x=None):
        """
        Returns:
            adp: 双曲空间邻接矩阵, shape: [N, N]
        """
        # 双曲距离计算
        emb = self.node_embeddings
        
        # Poincaré球模型中的距离
        norm_sq = (emb ** 2).sum(dim=-1, keepdim=True)  # [N, 1]
        
        # 计算两两距离
        dot_product = torch.mm(emb, emb.T)  # [N, N]
        norm_i = norm_sq
        norm_j = norm_sq.T
        
        # 双曲距离公式
        numerator = (emb.unsqueeze(1) - emb.unsqueeze(0)).pow(2).sum(dim=-1)
        denominator = (1 - norm_i) * (1 - norm_j).T + 1e-8
        
        dist = torch.acosh(1 + 2 * numerator / denominator + 1e-8)
        
        # 转换为相似度
        adp = torch.exp(-dist * self.curv)
        adp = adp / (adp.sum(dim=1, keepdim=True) + 1e-8)
        
        return adp


class SparseAdaptiveGraph(nn.Module):
    """方法5: 稀疏自适应图 (带 Top-K)
    
    优点:
    - 自动学习稀疏结构
    - 减少计算和内存开销
    - 只保留最重要的连接
    """
    
    def __init__(self, num_nodes, embed_dim, topk=10):
        super().__init__()
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(embed_dim, num_nodes), requires_grad=True)
        self.topk = topk
    
    def forward(self, x=None):
        """
        Returns:
            adp: 稀疏邻接矩阵, shape: [N, N]
        """
        # 计算原始邻接矩阵
        adp = torch.mm(self.nodevec1, self.nodevec2)
        adp = F.relu(adp)
        
        # Top-K 稀疏化
        if self.topk < adp.size(1):
            # 保留每行的 top-k 值
            topk_values, topk_indices = torch.topk(adp, self.topk, dim=1)
            
            # 创建稀疏矩阵
            adp_sparse = torch.zeros_like(adp)
            adp_sparse.scatter_(1, topk_indices, topk_values)
            adp = adp_sparse
        
        # Softmax 归一化
        adp = F.softmax(adp, dim=1)
        
        return adp


class TemporalAdaptiveGraph(nn.Module):
    """方法6: 时序自适应图
    
    优点:
    - 显式建模时间依赖
    - 图结构随时间演化
    - 适合长期预测
    """
    
    def __init__(self, num_nodes, embed_dim, num_time_steps):
        super().__init__()
        self.num_time_steps = num_time_steps
        
        # 空间嵌入
        self.spatial_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
        
        # 时间嵌入
        self.temporal_embeddings = nn.Parameter(torch.randn(num_time_steps, embed_dim), requires_grad=True)
        
        # 融合层
        self.fusion = nn.Linear(embed_dim * 2, embed_dim)
    
    def forward(self, x=None, time_idx=None):
        """
        Args:
            time_idx: 时间索引, shape: [B] 或 int
        Returns:
            adp: 时序自适应邻接矩阵, shape: [N, N]
        """
        if time_idx is None:
            time_idx = 0
        
        # 获取时间嵌入
        if isinstance(time_idx, int):
            temp_emb = self.temporal_embeddings[time_idx]  # [embed_dim]
        else:
            temp_emb = self.temporal_embeddings[time_idx].mean(dim=0)  # [embed_dim]
        
        # 融合空间和时间信息
        spatial_emb = self.spatial_embeddings  # [N, embed_dim]
        temp_emb_expanded = temp_emb.unsqueeze(0).expand(spatial_emb.size(0), -1)  # [N, embed_dim]
        
        fused_emb = torch.cat([spatial_emb, temp_emb_expanded], dim=-1)  # [N, 2*embed_dim]
        fused_emb = self.fusion(fused_emb)  # [N, embed_dim]
        
        # 计算邻接矩阵
        adp = torch.mm(fused_emb, fused_emb.T)
        adp = F.softmax(F.relu(adp), dim=1)
        
        return adp


class AdaptiveGraphFactory:
    """自适应图工厂类 - 方便创建和切换不同的图构建方法"""
    
    @staticmethod
    def create(graph_type, num_nodes, embed_dim, **kwargs):
        """
        创建自适应图模块
        
        Args:
            graph_type: 图类型, 可选:
                - 'simple': 简单矩阵乘法 (原始方法)
                - 'multihead': 多头注意力
                - 'dynamic': 动态自适应 (需要输入特征)
                - 'gaussian': 高斯核
                - 'hyperbolic': 双曲空间
                - 'sparse': 稀疏图 (Top-K)
                - 'temporal': 时序自适应
            num_nodes: 节点数量
            embed_dim: 嵌入维度
            **kwargs: 其他参数
        
        Returns:
            adaptive_graph: 自适应图模块
        """
        if graph_type == 'simple':
            return SimpleAdaptiveGraph(num_nodes, embed_dim)
        
        elif graph_type == 'multihead':
            num_heads = kwargs.get('num_heads', 4)
            return MultiHeadAdaptiveGraph(num_nodes, embed_dim, num_heads)
        
        elif graph_type == 'dynamic':
            feature_dim = kwargs.get('feature_dim', embed_dim)
            return DynamicAdaptiveGraph(num_nodes, embed_dim, feature_dim)
        
        elif graph_type == 'gaussian':
            return GaussianAdaptiveGraph(num_nodes, embed_dim)
        
        elif graph_type == 'hyperbolic':
            curv = kwargs.get('curv', 1.0)
            return HyperbolicAdaptiveGraph(num_nodes, embed_dim, curv)
        
        elif graph_type == 'sparse':
            topk = kwargs.get('topk', 10)
            return SparseAdaptiveGraph(num_nodes, embed_dim, topk)
        
        elif graph_type == 'temporal':
            num_time_steps = kwargs.get('num_time_steps', 288)
            return TemporalAdaptiveGraph(num_nodes, embed_dim, num_time_steps)
        
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")


# 使用示例
if __name__ == "__main__":
    num_nodes = 358
    embed_dim = 10
    batch_size = 8
    
    print("=" * 60)
    print("自适应图构建方法对比")
    print("=" * 60)
    
    # 1. 简单方法
    print("\n1. Simple Adaptive Graph")
    simple_graph = SimpleAdaptiveGraph(num_nodes, embed_dim)
    adp = simple_graph()
    print(f"   输出形状: {adp.shape}")
    print(f"   参数量: {sum(p.numel() for p in simple_graph.parameters())}")
    
    # 2. 多头注意力
    print("\n2. Multi-Head Adaptive Graph")
    multihead_graph = MultiHeadAdaptiveGraph(num_nodes, embed_dim, num_heads=4)
    adp = multihead_graph()
    print(f"   输出形状: {adp.shape}")
    print(f"   参数量: {sum(p.numel() for p in multihead_graph.parameters())}")
    
    # 3. 动态自适应
    print("\n3. Dynamic Adaptive Graph")
    dynamic_graph = DynamicAdaptiveGraph(num_nodes, embed_dim, feature_dim=64)
    x = torch.randn(batch_size, num_nodes, 64)
    adp = dynamic_graph(x)
    print(f"   输出形状: {adp.shape}")
    print(f"   参数量: {sum(p.numel() for p in dynamic_graph.parameters())}")
    
    # 4. 高斯核
    print("\n4. Gaussian Adaptive Graph")
    gaussian_graph = GaussianAdaptiveGraph(num_nodes, embed_dim)
    adp = gaussian_graph()
    print(f"   输出形状: {adp.shape}")
    print(f"   参数量: {sum(p.numel() for p in gaussian_graph.parameters())}")
    
    # 5. 双曲空间
    print("\n5. Hyperbolic Adaptive Graph")
    hyperbolic_graph = HyperbolicAdaptiveGraph(num_nodes, embed_dim)
    adp = hyperbolic_graph()
    print(f"   输出形状: {adp.shape}")
    print(f"   参数量: {sum(p.numel() for p in hyperbolic_graph.parameters())}")
    
    # 6. 稀疏图
    print("\n6. Sparse Adaptive Graph")
    sparse_graph = SparseAdaptiveGraph(num_nodes, embed_dim, topk=10)
    adp = sparse_graph()
    print(f"   输出形状: {adp.shape}")
    print(f"   稀疏度: {(adp == 0).sum().item() / adp.numel() * 100:.2f}%")
    print(f"   参数量: {sum(p.numel() for p in sparse_graph.parameters())}")
    
    # 7. 时序自适应
    print("\n7. Temporal Adaptive Graph")
    temporal_graph = TemporalAdaptiveGraph(num_nodes, embed_dim, num_time_steps=288)
    adp = temporal_graph(time_idx=0)
    print(f"   输出形状: {adp.shape}")
    print(f"   参数量: {sum(p.numel() for p in temporal_graph.parameters())}")
    
    print("\n" + "=" * 60)
    
    # 使用工厂模式
    print("\n使用工厂模式创建:")
    graph = AdaptiveGraphFactory.create('multihead', num_nodes, embed_dim, num_heads=4)
    print(f"创建的图类型: {type(graph).__name__}")
