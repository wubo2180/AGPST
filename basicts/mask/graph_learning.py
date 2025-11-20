"""
自适应图学习模块
Multi-scale adaptive graph learning with contrastive loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGraphLearner(nn.Module):
    """
    多尺度自适应图学习器
    
    输入: (B, N, P, D) - Batch, Nodes, Patches, Dimensions
    输出: (B, N, N) - 学习到的邻接矩阵
    """
    def __init__(self, num_nodes, node_dim, embed_dim, graph_heads=4, topk=10, dropout=0.1, use_temporal_info=True):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.embed_dim = embed_dim
        self.graph_heads = graph_heads
        self.topk = topk
        self.use_temporal_info = use_temporal_info
        
        # 静态节点嵌入
        self.static_node_embeddings1 = nn.Parameter(torch.randn(graph_heads, num_nodes, node_dim))
        self.static_node_embeddings2 = nn.Parameter(torch.randn(graph_heads, node_dim, num_nodes))
        
        # 动态特征编码
        if use_temporal_info:
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim, num_heads=4, dropout=dropout, batch_first=True
            )
            self.temporal_norm = nn.LayerNorm(embed_dim)
            
            self.dynamic_encoder = nn.Sequential(
                nn.Linear(embed_dim, node_dim * 2),
                nn.LayerNorm(node_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(node_dim * 2, node_dim),
                nn.LayerNorm(node_dim)
            )
            
            self.position_encoding = nn.Parameter(torch.randn(1, 72, embed_dim))
            
            self.node_gnn_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(node_dim, node_dim),
                    nn.LayerNorm(node_dim),
                    nn.GELU(),
                    nn.Dropout(dropout)
                ) for _ in range(2)
            ])
        
        # 温度参数
        self.temperature = nn.Parameter(torch.ones(graph_heads) * 0.5)
        
        # 多尺度图学习
        self.local_graph_heads = graph_heads // 2
        self.global_graph_heads = graph_heads - self.local_graph_heads
        
        self.local_node_embeddings1 = nn.Parameter(torch.randn(self.local_graph_heads, num_nodes, node_dim // 2))
        self.local_node_embeddings2 = nn.Parameter(torch.randn(self.local_graph_heads, node_dim // 2, num_nodes))
        
        self.global_node_embeddings1 = nn.Parameter(torch.randn(self.global_graph_heads, num_nodes, node_dim))
        self.global_node_embeddings2 = nn.Parameter(torch.randn(self.global_graph_heads, node_dim, num_nodes))
        
        # 图融合
        if use_temporal_info:
            self.graph_fusion_attention = nn.Sequential(
                nn.Linear(embed_dim, graph_heads),
                nn.Sigmoid()
            )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(graph_heads, graph_heads * 4),
            nn.LayerNorm(graph_heads * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(graph_heads * 4, graph_heads * 2),
            nn.GELU(),
            nn.Linear(graph_heads * 2, 1)
        )
        
        # 对比学习
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
        """计算静态多头图"""
        static_adjs = []
        
        # 局部图
        for h in range(self.local_graph_heads):
            adj = torch.mm(self.local_node_embeddings1[h], self.local_node_embeddings2[h])
            adj = F.relu(adj)
            temp = torch.clamp(self.temperature[h] * 2, min=0.1, max=5.0)
            adj = F.softmax(adj / temp, dim=1)
            static_adjs.append(adj)
        
        # 全局图
        for h in range(self.global_graph_heads):
            adj = torch.mm(self.global_node_embeddings1[h], self.global_node_embeddings2[h])
            adj = F.relu(adj)
            temp_idx = h + self.local_graph_heads
            temp = torch.clamp(self.temperature[temp_idx] * 0.5, min=0.1, max=2.0)
            adj = F.softmax(adj / temp, dim=1)
            static_adjs.append(adj)
            
        return torch.stack(static_adjs, dim=0)  # (H, N, N)
    
    def compute_dynamic_graphs(self, patch_features):
        """基于patch特征计算动态图"""
        if not self.use_temporal_info:
            return None
            
        B, N, P, D = patch_features.shape
        BN = B * N
        
        # 时间聚合
        patch_reshaped = patch_features.view(BN, P, D)
        pos_encoding = self.position_encoding[:, :P, :].expand(BN, -1, -1)
        patch_with_pos = patch_reshaped + pos_encoding
        
        attended_patches, _ = self.temporal_attention(
            patch_with_pos, patch_with_pos, patch_with_pos
        )
        
        patch_importance = F.softmax(attended_patches.mean(dim=-1), dim=1)
        node_repr = torch.sum(attended_patches * patch_importance.unsqueeze(-1), dim=1)
        node_repr = self.temporal_norm(node_repr + patch_reshaped.mean(dim=1))
        node_repr = node_repr.view(B, N, D)
        
        # 编码为节点嵌入
        dynamic_embeds = self.dynamic_encoder(node_repr)
        
        # GNN增强
        for gnn_layer in self.node_gnn_layers:
            similarities = torch.bmm(dynamic_embeds, dynamic_embeds.transpose(1, 2))
            similarities = F.softmax(similarities / 0.2, dim=2)
            aggregated = torch.bmm(similarities, dynamic_embeds)
            enhanced = gnn_layer(aggregated)
            dynamic_embeds = enhanced + dynamic_embeds
        
        # 计算动态图
        static_embeds_stack = torch.stack([self.static_node_embeddings2[h] for h in range(self.graph_heads)], dim=0)
        H = self.graph_heads
        
        dynamic_expanded = dynamic_embeds.unsqueeze(1).expand(-1, H, -1, -1)
        static_expanded = static_embeds_stack.unsqueeze(0).expand(B, -1, -1, -1)
        
        dynamic_sims = torch.matmul(dynamic_expanded, static_expanded)
        dynamic_sims = F.relu(dynamic_sims)
        
        temps = torch.clamp(self.temperature, min=0.1, max=2.0).view(1, -1, 1, 1)
        dynamic_sims = F.softmax(dynamic_sims / temps, dim=3)
        
        return dynamic_sims  # (B, H, N, N)
    
    def apply_topk_sparsification(self, adj_matrices):
        """Top-K稀疏化"""
        if self.topk >= self.num_nodes:
            return adj_matrices
        
        adj_matrices = adj_matrices.clone()
            
        if adj_matrices.dim() == 3:  # (H, N, N)
            topk_values, topk_indices = torch.topk(adj_matrices, self.topk, dim=2)
            H, N, _ = adj_matrices.shape
            mask = torch.zeros_like(adj_matrices)
            
            batch_idx = torch.arange(H).view(H, 1, 1).expand(-1, N, self.topk)
            row_idx = torch.arange(N).view(1, N, 1).expand(H, -1, self.topk)
            mask[batch_idx, row_idx, topk_indices] = 1
            
            sparse_adj = adj_matrices * mask
            row_sum = sparse_adj.sum(2, keepdim=True)
            sparse_adj = sparse_adj / (row_sum + 1e-8)
            
            return sparse_adj
                
        elif adj_matrices.dim() == 4:  # (B, H, N, N)
            topk_values, topk_indices = torch.topk(adj_matrices, self.topk, dim=3)
            B, H, N, _ = adj_matrices.shape
            mask = torch.zeros_like(adj_matrices)
            
            batch_idx = torch.arange(B).view(B, 1, 1, 1).expand(-1, H, N, self.topk)
            head_idx = torch.arange(H).view(1, H, 1, 1).expand(B, -1, N, self.topk)
            row_idx = torch.arange(N).view(1, 1, N, 1).expand(B, H, -1, self.topk)
            mask[batch_idx, head_idx, row_idx, topk_indices] = 1
            
            sparse_adj = adj_matrices * mask
            row_sum = sparse_adj.sum(3, keepdim=True)
            sparse_adj = sparse_adj / (row_sum + 1e-8)
            
            return sparse_adj
        
        return adj_matrices
    
    def compute_contrastive_loss(self, node_embeddings):
        """InfoNCE对比学习损失"""
        B, N, D = node_embeddings.shape
        
        projected = self.contrastive_projection(node_embeddings)
        projected = F.normalize(projected, p=2, dim=-1)
        
        similarity_matrices = torch.bmm(projected, projected.transpose(1, 2))
        similarity_matrices = torch.clamp(similarity_matrices, -0.99, 0.99)
        
        temperature = 0.2
        scaled_similarities = similarity_matrices / temperature
        
        pos_similarities = torch.diagonal(scaled_similarities, dim1=1, dim2=2)
        
        mask = ~torch.eye(N, dtype=torch.bool, device=node_embeddings.device)
        mask = mask.unsqueeze(0).expand(B, -1, -1)
        
        max_sims = torch.max(scaled_similarities, dim=2, keepdim=True)[0]
        stable_similarities = scaled_similarities - max_sims
        exp_similarities = torch.exp(stable_similarities)
        
        pos_exp = torch.diagonal(exp_similarities, dim1=1, dim2=2)
        neg_exp_sum = torch.sum(exp_similarities * mask.float(), dim=2)
        
        eps = 1e-8
        contrastive_losses = -torch.log(pos_exp / (pos_exp + neg_exp_sum + eps))
        
        valid_mask = torch.isfinite(contrastive_losses)
        if not valid_mask.all():
            contrastive_losses = torch.where(valid_mask, contrastive_losses, torch.zeros_like(contrastive_losses))
        
        return contrastive_losses.mean()
    
    def forward(self, patch_features):
        """
        Args:
            patch_features: (B, N, P, D)
        Returns:
            final_adj: (B, N, N)
            contrastive_loss: scalar
        """
        B, N, P, D = patch_features.shape
        
        # 静态图
        static_adjs = self.compute_static_graphs()
        
        # 动态图
        dynamic_adjs = None
        if self.use_temporal_info:
            dynamic_adjs = self.compute_dynamic_graphs(patch_features)
        
        # Top-K稀疏化
        static_adjs = self.apply_topk_sparsification(static_adjs)
        if dynamic_adjs is not None:
            dynamic_adjs = self.apply_topk_sparsification(dynamic_adjs)
        
        # 融合
        if dynamic_adjs is not None:
            static_expanded = static_adjs.unsqueeze(0).expand(B, -1, -1, -1)
            
            node_repr_for_fusion = patch_features.mean(dim=2)
            fusion_weights = self.graph_fusion_attention(node_repr_for_fusion)
            fusion_weights = fusion_weights.unsqueeze(-1).expand(B, N, self.graph_heads, N)
            fusion_weights = fusion_weights.permute(0, 2, 1, 3)
            
            fused_adjs = (1 - fusion_weights) * static_expanded + fusion_weights * dynamic_adjs
        else:
            fused_adjs = static_adjs.unsqueeze(0).expand(B, -1, -1, -1)
        
        # 多头融合
        final_adjs = []
        for b in range(B):
            multi_head = fused_adjs[b].permute(1, 2, 0)
            edge_weights = self.edge_encoder(multi_head.unsqueeze(0)).squeeze(0).squeeze(-1)
            final_adj = torch.sigmoid(edge_weights) * fused_adjs[b].mean(0)
            final_adjs.append(final_adj)
        
        final_adjs = torch.stack(final_adjs, dim=0)
        
        # 对比学习
        contrastive_loss = None
        if self.use_temporal_info and self.training and dynamic_adjs is not None:
            node_repr = patch_features.mean(dim=2)
            node_embeddings = self.dynamic_encoder(node_repr)
            contrastive_loss = self.compute_contrastive_loss(node_embeddings)
        
        return final_adjs, contrastive_loss


class DynamicGraphConv(nn.Module):
    """动态图卷积模块"""
    def __init__(self, embed_dim, num_nodes, node_dim, graph_heads=4, topk=10, dropout=0.1):
        super().__init__()
        

        self.graph_learner = AdaptiveGraphLearner(
                num_nodes=num_nodes,
                node_dim=node_dim, 
                embed_dim=embed_dim,
                graph_heads=graph_heads,
                topk=topk,
                dropout=dropout
            )
        
        self.weight = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.bias = nn.Parameter(torch.zeros(embed_dim))
        
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, patch_features):
        """
        Args:
            patch_features: (B, N, P, D)
        Returns:
            output: (B, N, P, D)
            learned_adjs: (B, N, N)
            contrastive_loss: scalar
        """
        B, N, P, D = patch_features.shape
        
        # 学习图结构
        learned_adjs, contrastive_loss = self.graph_learner(patch_features)
        
        # 对每个patch做图卷积
        output_patches = []
        for p in range(P):
            patch_p = patch_features[:, :, p, :]
            
            batch_outputs = []
            for b in range(B):
                support = torch.mm(patch_p[b], self.weight)
                graph_output = torch.mm(learned_adjs[b].t(), support)
                graph_output = graph_output + self.bias
                batch_outputs.append(graph_output)
            
            output_patches.append(torch.stack(batch_outputs, dim=0))
        
        output = torch.stack(output_patches, dim=2)
        
        return output, learned_adjs, contrastive_loss
