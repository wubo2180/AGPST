import torch
import torch.nn as nn
from ..graphwavenet import GraphWaveNet


class AGPSTModel(nn.Module):
    """
    简化的AGPST模型 - 不使用patch embedding
    直接处理短期时间序列 (B, 12, N, 1)
    
    架构:
    1. 简单的时间嵌入 (Linear)
    2. 自适应图学习
    3. 图卷积 + Transformer
    4. GraphWaveNet预测
    """
    def __init__(self, num_nodes, dim, topK, patch_size, in_channel, embed_dim, 
                 num_heads, graph_heads, mlp_ratio, dropout, encoder_depth, backend_args):
        super().__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.seq_len = 12  # 固定的短期历史长度
        
        # 1. 时间特征嵌入 (替代patch embedding)
        # (B, N, T, C) -> (B, N, T, D)
        self.time_embedding = nn.Sequential(
            nn.Linear(in_channel, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # 2. 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, 1, self.seq_len, embed_dim))
        
        # 3. 自适应图学习
        self.node_embeddings1 = nn.Parameter(torch.randn(num_nodes, dim))
        self.node_embeddings2 = nn.Parameter(torch.randn(dim, num_nodes))
        self.topK = topK
        
        # 4. 图卷积层
        self.graph_conv = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(2)
        ])
        
        # 5. Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)
        
        # 6. 后端预测
        self.backend = GraphWaveNet(**backend_args)
        
        self.contrastive_loss = None
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.node_embeddings1)
        nn.init.xavier_uniform_(self.node_embeddings2)
        nn.init.normal_(self.pos_embed, std=0.02)
        
    def learn_graph(self):
        """学习自适应图结构"""
        # 计算节点相似度
        adj = torch.mm(self.node_embeddings1, self.node_embeddings2)  # (N, N)
        adj = torch.relu(adj)
        
        # Top-K稀疏化
        if self.topK < self.num_nodes:
            topk_values, topk_indices = torch.topk(adj, self.topK, dim=1)
            mask = torch.zeros_like(adj)
            mask.scatter_(1, topk_indices, 1)
            adj = adj * mask
        
        # 归一化
        adj = adj / (adj.sum(1, keepdim=True) + 1e-8)
        
        return adj
        
    def graph_convolution(self, x, adj):
        """
        图卷积
        Args:
            x: (B, N, T, D)
            adj: (N, N)
        Returns:
            x: (B, N, T, D)
        """
        B, N, T, D = x.shape
        
        for conv in self.graph_conv:
            # 对每个时间步做图卷积
            out = []
            for t in range(T):
                xt = x[:, :, t, :]  # (B, N, D)
                # 特征变换
                h = conv(xt)  # (B, N, D)
                # 图聚合
                h = torch.matmul(adj.t().unsqueeze(0), h)  # (1, N, N) @ (B, N, D) -> (B, N, D)
                out.append(h)
            x = torch.stack(out, dim=2)  # (B, N, T, D)
            x = torch.relu(x)
        
        return x
        
    def forward(self, history_data, long_history_data=None, future_data=None, batch_size=None, epoch=None):
        """
        Args:
            history_data: (B, 12, N, 1) 短期历史
            long_history_data: 不使用（保持接口兼容）
            
        Returns:
            prediction: (B, 12, N, 1) 预测结果
        """
        # 使用短期历史数据
        B, T, N, C = history_data.shape
        
        # 转换格式: (B, T, N, C) -> (B, N, T, C)
        x = history_data.permute(0, 2, 1, 3)  # (B, N, T, C)
        
        # Step 1: 时间特征嵌入
        x = self.time_embedding(x)  # (B, N, T, D)
        
        # Step 2: 添加位置编码
        x = x + self.pos_embed  # (B, N, T, D)
        
        # Step 3: 自适应图学习
        adj = self.learn_graph()  # (N, N)
        
        # Step 4: 图卷积
        x = self.graph_convolution(x, adj)  # (B, N, T, D)
        
        # Step 5: Transformer时序建模
        # (B, N, T, D) -> (B*N, T, D)
        BN, T, D = B * N, x.size(2), x.size(3)
        x_flat = x.reshape(BN, T, D)
        x_flat = self.transformer(x_flat)  # (B*N, T, D)
        x = x_flat.reshape(B, N, T, D)  # (B, N, T, D)
        
        # Step 6: 时间聚合
        x_agg = x.mean(dim=2)  # (B, N, D)
        
        # Step 7: 后端预测
        # GraphWaveNet需要: (B, D, N, 1)
        backend_input = x_agg.permute(0, 2, 1).unsqueeze(-1)  # (B, D, N, 1)
        prediction = self.backend(backend_input)  # (B, N, 12, 1)
        
        # 转换输出格式: (B, N, 12, 1) -> (B, 12, N, 1)
        prediction = prediction.permute(0, 2, 1, 3)
        
        return prediction


ForecastingWithAdaptiveGraph = AGPSTModel
