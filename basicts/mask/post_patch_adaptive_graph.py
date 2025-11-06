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
        
        # 可学习温度参数
        self.temperature = nn.Parameter(torch.ones(graph_heads) * 0.5)
        
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
            # 局部图使用更高的温度，关注近邻
            adj = F.softmax(adj / (self.temperature[h] * 2), dim=1)
            static_adjs.append(adj)
        
        # 全局图学习（捕获长距离关系）
        for h in range(self.global_graph_heads):
            adj = torch.mm(self.global_node_embeddings1[h], self.global_node_embeddings2[h])
            adj = F.relu(adj)
            # 全局图使用较低的温度，关注全局模式
            temp_idx = h + self.local_graph_heads
            adj = F.softmax(adj / (self.temperature[temp_idx] * 0.5), dim=1)
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
        
        # 使用多层GNN增强节点表示
        for gnn_layer in self.node_gnn_layers:
            # 对每个batch独立进行GNN操作
            enhanced_embeds = []
            for b in range(B):
                # 构建简单的全连接图进行消息传递
                node_features = dynamic_embeds[b]  # (N, node_dim)
                
                # 计算节点相似性作为临时邻接矩阵
                similarity = torch.mm(node_features, node_features.t())  # (N, N)
                similarity = F.softmax(similarity / 0.1, dim=1)  # 温度缩放
                
                # 图卷积：聚合邻居信息
                aggregated = torch.mm(similarity, node_features)  # (N, node_dim)
                
                # 通过GNN层
                enhanced = gnn_layer(aggregated)
                
                # 残差连接
                enhanced = enhanced + node_features
                enhanced_embeds.append(enhanced)
            
            dynamic_embeds = torch.stack(enhanced_embeds, dim=0)  # (B, N, node_dim)
        
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
        """计算时空对比学习损失"""
        B, N, D = node_embeddings.shape
        
        # 投影到对比学习空间
        projected = self.contrastive_projection(node_embeddings)  # (B, N, D//4)
        
        # 构造正负样本对
        # 正样本：同一节点在不同时间的表示应该相似
        # 负样本：不同节点的表示应该不同
        
        loss = 0.0
        temperature = 0.1
        
        for b in range(B):
            node_features = projected[b]  # (N, D//4)
            
            # 计算相似度矩阵
            similarity_matrix = F.cosine_similarity(
                node_features.unsqueeze(1), 
                node_features.unsqueeze(0), 
                dim=2
            )  # (N, N)
            
            # 对比损失：同一节点自相似度最大化，不同节点相似度最小化
            positive_loss = -torch.log(torch.exp(similarity_matrix.diag() / temperature) + 1e-8).mean()
            
            # 负样本损失
            mask = torch.eye(N, device=node_features.device).bool()
            negative_similarities = similarity_matrix.masked_select(~mask)
            negative_loss = torch.log(torch.exp(negative_similarities / temperature) + 1e-8).mean()
            
            loss += positive_loss + 0.1 * negative_loss
        
        return loss / B


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