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
            # 时间注意力机制
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim, num_heads=4, dropout=dropout, batch_first=True
            )
            self.temporal_norm = nn.LayerNorm(embed_dim)
            
            # 增强的动态编码器
            self.dynamic_encoder = nn.Sequential(
                nn.Linear(embed_dim, node_dim * 2),
                nn.LayerNorm(node_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(node_dim * 2, node_dim),
                nn.LayerNorm(node_dim)
            )
            
            # 位置编码用于时间感知
            self.position_encoding = nn.Parameter(torch.randn(1, 72, embed_dim))  # 72个patch
            
            # 多层GNN用于增强节点表示学习
            self.node_gnn_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(node_dim, node_dim),
                    nn.LayerNorm(node_dim),
                    nn.GELU(),
                    nn.Dropout(dropout)
                ) for _ in range(2)
            ])
        
        # 可学习温度参数（避免过小值）
        self.temperature = nn.Parameter(torch.ones(graph_heads) * 0.5)
        # 限制温度参数范围
        self.temperature.data.clamp_(min=0.1, max=2.0)
        
        # 多尺度图学习：不同感受野的图学习分支
        self.local_graph_heads = graph_heads // 2  # 局部图头数
        self.global_graph_heads = graph_heads - self.local_graph_heads  # 全局图头数
        
        # 局部图学习（小感受野）
        self.local_node_embeddings1 = nn.Parameter(torch.randn(self.local_graph_heads, num_nodes, node_dim // 2))
        self.local_node_embeddings2 = nn.Parameter(torch.randn(self.local_graph_heads, node_dim // 2, num_nodes))
        
        # 全局图学习（大感受野）  
        self.global_node_embeddings1 = nn.Parameter(torch.randn(self.global_graph_heads, num_nodes, node_dim))
        self.global_node_embeddings2 = nn.Parameter(torch.randn(self.global_graph_heads, node_dim, num_nodes))
        
        # 自适应图融合网络
        if use_temporal_info:
            self.graph_fusion_attention = nn.Sequential(
                nn.Linear(embed_dim, graph_heads),
                nn.Sigmoid()
            )
        
        # 增强的多头融合编码器
        self.edge_encoder = nn.Sequential(
            nn.Linear(graph_heads, graph_heads * 4),
            nn.LayerNorm(graph_heads * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(graph_heads * 4, graph_heads * 2),
            nn.GELU(),
            nn.Linear(graph_heads * 2, 1)
        )
        
        # 对比学习组件
        if use_temporal_info:
            self.contrastive_projection = nn.Sequential(
                nn.Linear(node_dim, node_dim // 2),
                nn.ReLU(),
                nn.Linear(node_dim // 2, node_dim // 4)
            )
        
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        nn.init.xavier_uniform_(self.static_node_embeddings1)
        nn.init.xavier_uniform_(self.static_node_embeddings2)
        nn.init.xavier_uniform_(self.local_node_embeddings1)
        nn.init.xavier_uniform_(self.local_node_embeddings2)
        nn.init.xavier_uniform_(self.global_node_embeddings1)
        nn.init.xavier_uniform_(self.global_node_embeddings2)
        
    def compute_static_graphs(self):
        """计算基于静态嵌入的多尺度多头图"""
        static_adjs = []
        
        # 局部图学习（捕获近邻关系）
        for h in range(self.local_graph_heads):
            adj = torch.mm(self.local_node_embeddings1[h], self.local_node_embeddings2[h])
            adj = F.relu(adj)
            # 局部图使用更高的温度，关注近邻，并限制温度范围
            temp = torch.clamp(self.temperature[h] * 2, min=0.1, max=5.0)
            adj = F.softmax(adj / temp, dim=1)
            static_adjs.append(adj)
        
        # 全局图学习（捕获长距离关系）
        for h in range(self.global_graph_heads):
            adj = torch.mm(self.global_node_embeddings1[h], self.global_node_embeddings2[h])
            adj = F.relu(adj)
            # 全局图使用较低的温度，关注全局模式，并限制温度范围
            temp_idx = h + self.local_graph_heads
            temp = torch.clamp(self.temperature[temp_idx] * 0.5, min=0.1, max=2.0)
            adj = F.softmax(adj / temp, dim=1)
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
            
        # 时间聚合: 使用注意力机制聚合patch信息
        # 重塑为 (B*N, P, D) 进行时间注意力
        BN, P, D = B * N, patch_features.size(2), patch_features.size(3)
        patch_reshaped = patch_features.view(BN, P, D)  # (B*N, P, D)
        
        # 添加位置编码
        pos_encoding = self.position_encoding[:, :P, :].expand(BN, -1, -1)
        patch_with_pos = patch_reshaped + pos_encoding
        
        # 时间注意力：每个节点学习不同时间patch的重要性
        attended_patches, attention_weights = self.temporal_attention(
            patch_with_pos, patch_with_pos, patch_with_pos
        )  # (B*N, P, D)
        
        # 加权聚合：使用注意力权重而不是简单平均
        # 计算每个patch的重要性权重
        patch_importance = F.softmax(attended_patches.mean(dim=-1), dim=1)  # (B*N, P)
        node_repr = torch.sum(attended_patches * patch_importance.unsqueeze(-1), dim=1)  # (B*N, D)
        
        # 残差连接和层归一化
        node_repr = self.temporal_norm(node_repr + patch_reshaped.mean(dim=1))
        
        # 重塑回 (B, N, D)
        node_repr = node_repr.view(B, N, D)
        
        # 编码为节点嵌入
        dynamic_embeds = self.dynamic_encoder(node_repr)  # (B, N, node_dim)
        
        # 使用多层GNN增强节点表示 - 向量化优化版本  
        for gnn_layer in self.node_gnn_layers:
            # 向量化处理所有batch
            # 计算批量节点相似性矩阵
            similarities = torch.bmm(dynamic_embeds, dynamic_embeds.transpose(1, 2))  # (B, N, N)
            similarities = F.softmax(similarities / 0.2, dim=2)  # 温度缩放
            
            # 批量图卷积：聚合邻居信息
            aggregated = torch.bmm(similarities, dynamic_embeds)  # (B, N, node_dim)
            
            # 通过GNN层
            enhanced = gnn_layer(aggregated)  # (B, N, node_dim)
            
            # 残差连接
            dynamic_embeds = enhanced + dynamic_embeds
        
        # 计算批次内动态图 - 向量化优化版本
        # 预计算所有头的静态嵌入矩阵 (H, node_dim, N)
        static_embeds_stack = torch.stack([self.static_node_embeddings2[h] for h in range(self.graph_heads)], dim=0)
        
        # 向量化计算: (B, N, node_dim) × (H, node_dim, N) -> (B, H, N, N)
        # 使用更清晰的矩阵乘法替代einsum
        B, N, D = dynamic_embeds.shape
        H = self.graph_heads
        
        # 扩展维度用于批量计算
        dynamic_expanded = dynamic_embeds.unsqueeze(1).expand(-1, H, -1, -1)  # (B, H, N, D)
        static_expanded = static_embeds_stack.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, D, N)
        
        # 批量矩阵乘法
        dynamic_sims = torch.matmul(dynamic_expanded, static_expanded)  # (B, H, N, N)
        dynamic_sims = F.relu(dynamic_sims)
        
        # 向量化温度缩放和softmax
        temps = torch.clamp(self.temperature, min=0.1, max=2.0)  # (H,)
        temps = temps.view(1, -1, 1, 1)  # (1, H, 1, 1) 用于广播
        dynamic_sims = F.softmax(dynamic_sims / temps, dim=3)  # (B, H, N, N)
        
        return dynamic_sims  # (B, H, N, N)
    
    def apply_topk_sparsification(self, adj_matrices):
        """对邻接矩阵进行Top-K稀疏化 - 完全向量化版本"""
        if self.topk >= self.num_nodes:
            return adj_matrices
        
        # 创建副本避免inplace操作
        adj_matrices = adj_matrices.clone()
            
        # 向量化Top-K处理
        if adj_matrices.dim() == 3:  # (H, N, N) - 静态图
            # 向量化topk和mask操作
            topk_values, topk_indices = torch.topk(adj_matrices, self.topk, dim=2)  # (H, N, K)
            
            # 创建mask
            H, N, _ = adj_matrices.shape
            mask = torch.zeros_like(adj_matrices)  # (H, N, N)
            
            # 批量scatter操作
            batch_idx = torch.arange(H).view(H, 1, 1).expand(-1, N, self.topk)
            row_idx = torch.arange(N).view(1, N, 1).expand(H, -1, self.topk)
            mask[batch_idx, row_idx, topk_indices] = 1
            
            # 应用mask和归一化
            sparse_adj = adj_matrices * mask
            row_sum = sparse_adj.sum(2, keepdim=True)
            sparse_adj = sparse_adj / (row_sum + 1e-8)
            
            return sparse_adj
                
        elif adj_matrices.dim() == 4:  # (B, H, N, N) - 动态图
            # 向量化4D tensor的topk处理
            topk_values, topk_indices = torch.topk(adj_matrices, self.topk, dim=3)  # (B, H, N, K)
            
            # 创建mask
            B, H, N, _ = adj_matrices.shape
            mask = torch.zeros_like(adj_matrices)  # (B, H, N, N)
            
            # 批量scatter操作
            batch_idx = torch.arange(B).view(B, 1, 1, 1).expand(-1, H, N, self.topk)
            head_idx = torch.arange(H).view(1, H, 1, 1).expand(B, -1, N, self.topk)
            row_idx = torch.arange(N).view(1, 1, N, 1).expand(B, H, -1, self.topk)
            mask[batch_idx, head_idx, row_idx, topk_indices] = 1
            
            # 应用mask和归一化
            sparse_adj = adj_matrices * mask
            row_sum = sparse_adj.sum(3, keepdim=True)
            sparse_adj = sparse_adj / (row_sum + 1e-8)
            
            return sparse_adj
        
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
        
        # 4. 自适应融合静态和动态图
        if dynamic_adjs is not None:
            # 扩展静态图到batch维度
            static_expanded = static_adjs.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, N, N)
            
            # 基于节点特征计算融合权重
            node_repr_for_fusion = patch_features.mean(dim=2)  # (B, N, D)
            fusion_weights = self.graph_fusion_attention(node_repr_for_fusion)  # (B, N, H)
            
            # 扩展融合权重到邻接矩阵维度
            fusion_weights = fusion_weights.unsqueeze(-1).expand(B, N, self.graph_heads, N)
            fusion_weights = fusion_weights.permute(0, 2, 1, 3)  # (B, H, N, N)
            
            # 自适应加权融合
            fused_adjs = (1 - fusion_weights) * static_expanded + fusion_weights * dynamic_adjs
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
        
        # 6. 对比学习（可选，用于增强图表示质量）
        contrastive_loss = None
        if self.use_temporal_info and self.training and dynamic_adjs is not None:
            # 重新计算节点嵌入用于对比学习
            node_repr = patch_features.mean(dim=2)  # (B, N, D)
            node_embeddings = self.dynamic_encoder(node_repr)  # (B, N, node_dim)
            contrastive_loss = self.compute_contrastive_loss(node_embeddings)
        
        return final_adjs, static_adjs, dynamic_adjs, contrastive_loss
    
    def compute_contrastive_loss(self, node_embeddings):
        """计算时空对比学习损失 - GPU优化向量化版本"""
        B, N, D = node_embeddings.shape
        
        # 投影到对比学习空间
        projected = self.contrastive_projection(node_embeddings)  # (B, N, D//4)
        
        # L2归一化确保数值稳定性
        projected = F.normalize(projected, p=2, dim=-1)
        
        # 向量化计算所有batch的相似度矩阵
        # projected: (B, N, D//4) -> (B, N, N) similarity matrices
        similarity_matrices = torch.bmm(projected, projected.transpose(1, 2))  # (B, N, N)
        
        # 限制相似度范围避免极值
        similarity_matrices = torch.clamp(similarity_matrices, -0.99, 0.99)
        
        # 温度缩放
        temperature = 0.2
        scaled_similarities = similarity_matrices / temperature  # (B, N, N)
        
        # 向量化InfoNCE损失计算
        # 提取对角线（正样本）
        pos_similarities = torch.diagonal(scaled_similarities, dim1=1, dim2=2)  # (B, N)
        
        # 创建mask排除对角线元素（负样本）
        mask = ~torch.eye(N, dtype=torch.bool, device=node_embeddings.device)  # (N, N)
        mask = mask.unsqueeze(0).expand(B, -1, -1)  # (B, N, N)
        
        # 向量化log-sum-exp计算
        # 为数值稳定性，先找到每行的最大值
        max_sims = torch.max(scaled_similarities, dim=2, keepdim=True)[0]  # (B, N, 1)
        
        # 稳定的exp计算
        stable_similarities = scaled_similarities - max_sims  # (B, N, N)
        exp_similarities = torch.exp(stable_similarities)  # (B, N, N)
        
        # 正样本exp值
        pos_exp = torch.diagonal(exp_similarities, dim1=1, dim2=2)  # (B, N)
        
        # 负样本exp值之和（排除对角线）
        neg_exp_sum = torch.sum(exp_similarities * mask.float(), dim=2)  # (B, N)
        
        # InfoNCE损失：-log(pos_exp / (pos_exp + neg_exp_sum))
        eps = 1e-8
        contrastive_losses = -torch.log(pos_exp / (pos_exp + neg_exp_sum + eps))  # (B, N)
        
        # 检查并处理异常值（向量化）
        valid_mask = torch.isfinite(contrastive_losses)  # (B, N)
        if not valid_mask.all():
            contrastive_losses = torch.where(valid_mask, contrastive_losses, torch.zeros_like(contrastive_losses))
        
        # 返回所有batch和节点的平均损失
        return contrastive_losses.mean()


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
        graph_output = self.graph_learner(patch_features)
        if len(graph_output) == 4:
            learned_adjs, static_adjs, dynamic_adjs, contrastive_loss = graph_output
        else:
            learned_adjs, static_adjs, dynamic_adjs = graph_output
            contrastive_loss = None
        
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
        
        return output, learned_adjs, contrastive_loss