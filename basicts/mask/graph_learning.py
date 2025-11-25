"""
Simplified Adaptive Graph Learning Module
Core functionality: Learn adaptive graph structure from node embeddings
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGraphLearner(nn.Module):
    """
    Simplified Adaptive Graph Learner
    
    Input: (B, N, P, D) or (B, N, D) - Batch, Nodes, [Patches], Dimensions
    Output: (B, N, N) - Learned adjacency matrix
    """
    def __init__(self, num_nodes, node_dim, embed_dim=None, graph_heads=4, topk=10, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.embed_dim = embed_dim if embed_dim is not None else node_dim
        self.graph_heads = graph_heads
        self.topk = topk
        
        # Static node embeddings (learnable)
        self.node_embeddings1 = nn.Parameter(torch.randn(graph_heads, num_nodes, node_dim))
        self.node_embeddings2 = nn.Parameter(torch.randn(graph_heads, node_dim, num_nodes))
        
        # Temperature parameter for softmax
        self.temperature = nn.Parameter(torch.ones(graph_heads) * 0.5)
        
        # Multi-head fusion
        self.fusion_weights = nn.Parameter(torch.ones(graph_heads) / graph_heads)
        
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        """Initialize parameters with Xavier uniform"""
        nn.init.xavier_uniform_(self.node_embeddings1)
        nn.init.xavier_uniform_(self.node_embeddings2)
        
    def compute_adjacency_matrix(self):
        """
        Compute multi-head adjacency matrices from static embeddings
        
        Returns:
            adjs: (H, N, N) - Multi-head adjacency matrices
        """
        adjs = []
        for h in range(self.graph_heads):
            # Compute similarity: (N, node_dim) @ (node_dim, N) = (N, N)
            adj = torch.mm(self.node_embeddings1[h], self.node_embeddings2[h])
            
            # Apply ReLU to ensure non-negativity
            adj = F.relu(adj)
            
            # Apply temperature scaling and softmax normalization
            temp = torch.clamp(self.temperature[h], min=0.1, max=2.0)
            adj = F.softmax(adj / temp, dim=1)
            
            adjs.append(adj)
            
        return torch.stack(adjs, dim=0)  # (H, N, N)
    
    def apply_topk_sparsification(self, adj_matrices):
        """
        Apply Top-K sparsification to adjacency matrices
        Keep only top-K strongest connections for each node
        
        Args:
            adj_matrices: (H, N, N) or (B, H, N, N)
        Returns:
            sparse_adj: Same shape as input, but sparsified
        """
        if self.topk >= self.num_nodes:
            return adj_matrices
        
        adj_matrices = adj_matrices.clone()
        
        if adj_matrices.dim() == 3:  # (H, N, N)
            # Get top-K values and indices for each row
            topk_values, topk_indices = torch.topk(adj_matrices, self.topk, dim=2)
            H, N, _ = adj_matrices.shape
            
            # Create mask
            mask = torch.zeros_like(adj_matrices)
            head_idx = torch.arange(H, device=adj_matrices.device).view(H, 1, 1).expand(-1, N, self.topk)
            row_idx = torch.arange(N, device=adj_matrices.device).view(1, N, 1).expand(H, -1, self.topk)
            mask[head_idx, row_idx, topk_indices] = 1
            
            # Apply mask and re-normalize
            sparse_adj = adj_matrices * mask
            row_sum = sparse_adj.sum(2, keepdim=True)
            sparse_adj = sparse_adj / (row_sum + 1e-8)
            
            return sparse_adj
        
        elif adj_matrices.dim() == 4:  # (B, H, N, N)
            topk_values, topk_indices = torch.topk(adj_matrices, self.topk, dim=3)
            B, H, N, _ = adj_matrices.shape
            
            mask = torch.zeros_like(adj_matrices)
            batch_idx = torch.arange(B, device=adj_matrices.device).view(B, 1, 1, 1).expand(-1, H, N, self.topk)
            head_idx = torch.arange(H, device=adj_matrices.device).view(1, H, 1, 1).expand(B, -1, N, self.topk)
            row_idx = torch.arange(N, device=adj_matrices.device).view(1, 1, N, 1).expand(B, H, -1, self.topk)
            mask[batch_idx, head_idx, row_idx, topk_indices] = 1
            
            sparse_adj = adj_matrices * mask
            row_sum = sparse_adj.sum(3, keepdim=True)
            sparse_adj = sparse_adj / (row_sum + 1e-8)
            
            return sparse_adj
        
        return adj_matrices
    
    def forward(self, x=None):
        """
        Forward pass
        
        Args:
            x: Optional input features (not used in simplified version)
        Returns:
            final_adj: (N, N) or (B, N, N) - Final adjacency matrix
        """
        # Compute multi-head adjacency matrices
        multi_head_adjs = self.compute_adjacency_matrix()  # (H, N, N)
        
        # Apply Top-K sparsification
        multi_head_adjs = self.apply_topk_sparsification(multi_head_adjs)
        
        # Fuse multi-head graphs using learned weights
        fusion_weights = F.softmax(self.fusion_weights, dim=0)  # (H,)
        final_adj = torch.sum(multi_head_adjs * fusion_weights.view(-1, 1, 1), dim=0)  # (N, N)
        
        return final_adj
