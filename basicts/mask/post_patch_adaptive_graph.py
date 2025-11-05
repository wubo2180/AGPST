import torch
import torch.nn as nn
import torch.nn.functional as F


class PostPatchAdaptiveGraphLearner(nn.Module):
    """
    用于patch embedding后的增强自适应图学习
    
    输入数据格式: (B, N, P, D) 其中
    - B: batch size
    - N: 节点数 (358)
    - P: patch数量 (L/patch_size, 如 864/12=72)
    - D: embedding维度 (如96)
    """
    def __init__(self, num_nodes, node_dim, embed_dim, graph_heads=4, topk=10, dropout=0.1, use_temporal_info=True):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.embed_dim = embed_dim
        self.graph_heads = graph_heads
        self.topk = topk
        self.use_temporal_info = use_temporal_info
        
        # 基础节点嵌入 (静态部分)
        self.static_node_embeddings1 = nn.Parameter(torch.randn(graph_heads, num_nodes, node_dim))
        self.static_node_embeddings2 = nn.Parameter(torch.randn(graph_heads, node_dim, num_nodes))
        
        # 动态特征编码器 (利用patch embedding后的特征)
        if use_temporal_info:
            self.dynamic_encoder = nn.Sequential(
                nn.Linear(embed_dim, node_dim),
                nn.LayerNorm(node_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(node_dim, node_dim)
            )
        
        # 可学习温度参数
        self.temperature = nn.Parameter(torch.ones(graph_heads) * 0.5)
        
        # 时空融合权重
        if use_temporal_info:
            self.fusion_weight = nn.Parameter(torch.tensor(0.5))  # 静态vs动态权重
        
        # 多头融合编码器
        self.edge_encoder = nn.Sequential(
            nn.Linear(graph_heads, graph_heads * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(graph_heads * 2, 1)
        )
        
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        nn.init.xavier_uniform_(self.static_node_embeddings1)
        nn.init.xavier_uniform_(self.static_node_embeddings2)
        
    def compute_static_graphs(self):
        """计算基于静态嵌入的多头图"""
        static_adjs = []
        
        for h in range(self.graph_heads):
            # 静态相似度计算
            adj = torch.mm(self.static_node_embeddings1[h], self.static_node_embeddings2[h])
            adj = F.relu(adj)  # 确保非负
            adj = F.softmax(adj / self.temperature[h], dim=1)
            static_adjs.append(adj)
            
        return torch.stack(static_adjs, dim=0)  # (H, N, N)
    
    def compute_dynamic_graphs(self, patch_features):
        """
        基于patch特征计算动态图
        Args:
            patch_features: (B, N, P, D) patch embedding后的特征
        """
        B, N, P, D = patch_features.shape
        
        if not self.use_temporal_info:
            return None
            
        # 时间聚合: 将所有patch的信息聚合为节点表示
        # 方式1: 平均池化 - 沿着patch维度(P)聚合
        node_repr = patch_features.mean(dim=2)  # (B, N, D)
        
        # 方式2: 注意力加权 (可选)
        # attention_weights = F.softmax(patch_features.mean(dim=1), dim=1)  # (B, P, D)
        # node_repr = torch.sum(patch_features * attention_weights.unsqueeze(1), dim=2)  # (B, N, D)
        
        # 编码为节点嵌入
        dynamic_embeds = self.dynamic_encoder(node_repr)  # (B, N, node_dim)
        
        # 计算批次内动态图
        dynamic_adjs = []
        for b in range(B):
            batch_adjs = []
            for h in range(self.graph_heads):
                # 动态相似度 = 当前特征 × 静态嵌入
                dynamic_sim = torch.mm(
                    dynamic_embeds[b],  # (N, node_dim)
                    self.static_node_embeddings2[h]  # (node_dim, N)
                )
                dynamic_sim = F.relu(dynamic_sim)
                dynamic_sim = F.softmax(dynamic_sim / self.temperature[h], dim=1)
                batch_adjs.append(dynamic_sim)
            
            dynamic_adjs.append(torch.stack(batch_adjs, dim=0))  # (H, N, N)
        
        return torch.stack(dynamic_adjs, dim=0)  # (B, H, N, N)
    
    def apply_topk_sparsification(self, adj_matrices):
        """对邻接矩阵进行Top-K稀疏化"""
        if self.topk >= self.num_nodes:
            return adj_matrices
        
        # 创建副本避免inplace操作
        adj_matrices = adj_matrices.clone()
            
        # 确定输入维度
        if adj_matrices.dim() == 3:  # (H, N, N) - 静态图
            H, N, N = adj_matrices.shape
            sparsified_adjs = []
            for h in range(H):
                adj = adj_matrices[h]
                topk_values, topk_indices = torch.topk(adj, self.topk, dim=1)
                mask = torch.zeros_like(adj)
                mask.scatter_(1, topk_indices, 1)
                sparse_adj = adj * mask
                # 重新归一化
                row_sum = sparse_adj.sum(1, keepdim=True)
                sparse_adj = sparse_adj / (row_sum + 1e-8)
                sparsified_adjs.append(sparse_adj)
            return torch.stack(sparsified_adjs, dim=0)
                
        elif adj_matrices.dim() == 4:  # (B, H, N, N) - 动态图
            B, H, N, N = adj_matrices.shape
            sparsified_adjs = []
            for b in range(B):
                batch_adjs = []
                for h in range(H):
                    adj = adj_matrices[b, h]
                    topk_values, topk_indices = torch.topk(adj, self.topk, dim=1)
                    mask = torch.zeros_like(adj)
                    mask.scatter_(1, topk_indices, 1)
                    sparse_adj = adj * mask
                    # 重新归一化
                    row_sum = sparse_adj.sum(1, keepdim=True)
                    sparse_adj = sparse_adj / (row_sum + 1e-8)
                    batch_adjs.append(sparse_adj)
                sparsified_adjs.append(torch.stack(batch_adjs, dim=0))
            return torch.stack(sparsified_adjs, dim=0)
        
        return adj_matrices
    
    def forward(self, patch_features):
        """
        Args:
            patch_features: (B, N, P, D) patch embedding后的特征
        Returns:
            final_adj: (B, N, N) 或 (N, N) 最终邻接矩阵
            static_adjs: (H, N, N) 静态多头图
            dynamic_adjs: (B, H, N, N) 动态多头图 (如果使用)
        """
        B, N, P, D = patch_features.shape
        
        # 1. 计算静态图 (基于预学习的节点嵌入)
        static_adjs = self.compute_static_graphs()  # (H, N, N)
        
        # 2. 计算动态图 (基于当前batch的patch特征)
        dynamic_adjs = None
        if self.use_temporal_info:
            dynamic_adjs = self.compute_dynamic_graphs(patch_features)  # (B, H, N, N)
        
        # 3. Top-K稀疏化
        static_adjs = self.apply_topk_sparsification(static_adjs)
        if dynamic_adjs is not None:
            dynamic_adjs = self.apply_topk_sparsification(dynamic_adjs)
        
        # 4. 融合静态和动态图
        if dynamic_adjs is not None:
            # 扩展静态图到batch维度
            static_expanded = static_adjs.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, N, N)
            
            # 加权融合
            fused_adjs = (1 - self.fusion_weight) * static_expanded + self.fusion_weight * dynamic_adjs
        else:
            fused_adjs = static_adjs.unsqueeze(0).expand(B, -1, -1, -1)
        
        # 5. 多头融合
        final_adjs = []
        for b in range(B):
            # 转换维度用于edge_encoder: (H, N, N) -> (N, N, H)
            multi_head = fused_adjs[b].permute(1, 2, 0)  # (N, N, H)
            edge_weights = self.edge_encoder(multi_head.unsqueeze(0)).squeeze(0).squeeze(-1)  # (N, N)
            
            # 最终邻接矩阵
            final_adj = torch.sigmoid(edge_weights) * fused_adjs[b].mean(0)  # (N, N)
            final_adjs.append(final_adj)
        
        final_adjs = torch.stack(final_adjs, dim=0)  # (B, N, N)
        
        return final_adjs, static_adjs, dynamic_adjs


class PostPatchDynamicGraphConv(nn.Module):
    """用于patch embedding后的动态图卷积"""
    def __init__(self, embed_dim, num_nodes, node_dim, graph_heads=4, topk=10, dropout=0.1):
        super().__init__()
        self.graph_learner = PostPatchAdaptiveGraphLearner(
            num_nodes=num_nodes,
            node_dim=node_dim, 
            embed_dim=embed_dim,
            graph_heads=graph_heads,
            topk=topk,
            dropout=dropout
        )
        
        # 图卷积权重
        self.weight = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.bias = nn.Parameter(torch.zeros(embed_dim))
        
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, patch_features):
        """
        Args:
            patch_features: (B, N, P, D) patch embedding后的特征
        Returns:
            output: (B, N, P, D) 图卷积后的特征
            learned_adjs: (B, N, N) 学习到的邻接矩阵
        """
        B, N, P, D = patch_features.shape
        
        # 学习动态图
        learned_adjs, static_adjs, dynamic_adjs = self.graph_learner(patch_features)  # (B, N, N)
        
        # 对每个时间patch进行图卷积
        output_patches = []
        for p in range(P):
            patch_p = patch_features[:, :, p, :]  # (B, N, D)
            
            # 批量图卷积
            batch_outputs = []
            for b in range(B):
                # 特征变换
                support = torch.mm(patch_p[b], self.weight)  # (N, D)
                # 图卷积
                graph_output = torch.mm(learned_adjs[b].t(), support)  # (N, D)
                graph_output = graph_output + self.bias
                batch_outputs.append(graph_output)
            
            output_patches.append(torch.stack(batch_outputs, dim=0))  # (B, N, D)
        
        output = torch.stack(output_patches, dim=2)  # (B, N, P, D)
        
        return output, learned_adjs