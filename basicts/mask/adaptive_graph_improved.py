import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGraphLearner(nn.Module):
    """
    Enhanced adaptive graph learning with:
    1. Multi-head graph attention
    2. Dynamic edge pruning
    3. Learnable temperature
    """
    def __init__(self, num_nodes, node_dim, num_heads=4, topk=10, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.num_heads = num_heads
        self.topk = topk
        
        # Multi-head node embeddings
        self.node_embeddings1 = nn.Parameter(torch.randn(num_heads, num_nodes, node_dim))
        self.node_embeddings2 = nn.Parameter(torch.randn(num_heads, node_dim, num_nodes))
        
        # Learnable temperature for softmax
        self.temperature = nn.Parameter(torch.ones(num_heads))
        
        # Edge feature learning
        self.edge_encoder = nn.Sequential(
            nn.Linear(num_heads, num_heads * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_heads * 2, 1)
        )
        
        nn.init.xavier_uniform_(self.node_embeddings1)
        nn.init.xavier_uniform_(self.node_embeddings2)
        
    def forward(self, x=None):
        """
        Args:
            x: Optional input features (B, N, L, D) to guide graph construction
        Returns:
            adj: Adjacency matrix (N, N) or multi-head (H, N, N)
        """
        # Multi-head adjacency matrices
        adjs = []
        for h in range(self.num_heads):
            # Compute similarity
            adj = F.softmax(
                F.relu(torch.mm(self.node_embeddings1[h], self.node_embeddings2[h])) / self.temperature[h],
                dim=1
            )
            
            # Top-K sparsification for each node
            if self.topk < self.num_nodes:
                topk_values, topk_indices = torch.topk(adj, self.topk, dim=1)
                mask = torch.zeros_like(adj)
                mask.scatter_(1, topk_indices, 1)
                adj = adj * mask
                
                # Re-normalize after sparsification
                row_sum = adj.sum(1, keepdim=True)
                adj = adj / (row_sum + 1e-8)
            
            adjs.append(adj)
        
        # Stack multi-head adjacencies
        adjs = torch.stack(adjs, dim=0)  # (H, N, N)
        
        # Aggregate multi-head graphs with learned weights
        # Shape: (H, N, N) -> (1, N, N, H) -> (N, N, 1)
        adjs_transposed = adjs.permute(1, 2, 0).unsqueeze(0)  # (1, N, N, H)
        edge_weights = self.edge_encoder(adjs_transposed).squeeze(-1).squeeze(0)  # (N, N)
        
        # Final adjacency with learned aggregation
        final_adj = torch.sigmoid(edge_weights) * adjs.mean(0)
        
        return final_adj, adjs


class DynamicGraphConv(nn.Module):
    """Graph convolution with dynamic adjacency"""
    def __init__(self, in_dim, out_dim, num_nodes, node_dim, num_heads=4, topk=10):
        super().__init__()
        self.graph_learner = AdaptiveGraphLearner(num_nodes, node_dim, num_heads, topk)
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, x):
        """
        Args:
            x: (B, N, L, D)
        Returns:
            out: (B, N, L, D_out)
        """
        B, N, L, D = x.shape
        
        # Learn dynamic graph
        adj, multi_head_adjs = self.graph_learner(x)  # (N, N)
        
        # Reshape for graph convolution
        x_reshape = x.permute(0, 2, 1, 3).contiguous()  # (B, L, N, D)
        x_reshape = x_reshape.view(B * L, N, D)
        
        # Graph convolution: (B*L, N, D) @ (N, N).T @ (D, D_out)
        support = torch.matmul(x_reshape, self.weight)  # (B*L, N, D_out)
        out = torch.matmul(adj.t(), support)  # (B*L, N, D_out)
        out = out + self.bias
        
        # Reshape back
        out = out.view(B, L, N, -1).permute(0, 2, 1, 3)  # (B, N, L, D_out)
        
        return out, adj
