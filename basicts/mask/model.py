import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph_learning import DynamicGraphConv


class DenoiseAttention(nn.Module):
    """
    基于自注意力的去噪模块
    对时间序列进行自注意力处理，降低噪声影响
    """
    def __init__(self, in_channels, hidden_dim, dropout=0.1):
        super().__init__()
        self.query = nn.Linear(in_channels, hidden_dim)
        self.key = nn.Linear(in_channels, hidden_dim)
        self.value = nn.Linear(in_channels, hidden_dim)
        self.output = nn.Linear(hidden_dim, in_channels)
        self.dropout = nn.Dropout(dropout)
        self.scale = hidden_dim ** -0.5
        
    def forward(self, x):
        """
        Args:
            x: (B, T, N, C)
        Returns:
            denoised: (B, T, N, C)
        """
        B, T, N, C = x.shape
        # 重塑为 (B*N, T, C) 以便在时间维度上应用注意力
        x_flat = x.reshape(B * N, T, C)
        
        # 计算 Q, K, V
        Q = self.query(x_flat)  # (B*N, T, H)
        K = self.key(x_flat)    # (B*N, T, H)
        V = self.value(x_flat)  # (B*N, T, H)
        
        # 注意力分数
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B*N, T, T)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 加权求和
        out = torch.matmul(attn, V)  # (B*N, T, H)
        out = self.output(out)  # (B*N, T, C)
        
        # 重塑回原始形状
        out = out.reshape(B, T, N, C)
        
        return out


class AGPSTModel(nn.Module):
    """
    AGPST模型 - Encoder-Decoder 架构
    集成自适应图学习和时空去噪
    
    架构:
    Encoder:
        0. 去噪模块 (可选)
        1. 时间特征嵌入
        2. 自适应图学习 + 动态图卷积
        3. Transformer 编码器
    
    Decoder:
        4. 可学习的未来查询向量
        5. Transformer 解码器 (交叉注意力)
        6. 输出投影层
    """
    def __init__(self, num_nodes, dim, topK, in_channel, embed_dim, 
                 num_heads, mlp_ratio, dropout, encoder_depth,
                 use_denoising=True, denoise_type='conv',
                 use_advanced_graph=True, graph_heads=4,
                 pred_len=12, decoder_depth=2):
        super().__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.seq_len = 12  # 固定的短期历史长度
        self.pred_len = pred_len  # 预测长度
        self.use_denoising = use_denoising
        self.denoise_type = denoise_type
        self.use_advanced_graph = use_advanced_graph
        
        # ============ Encoder 部分 ============
        
        # 0. 去噪模块
        if use_denoising:
            if denoise_type == 'conv':
                # 基于卷积的去噪：时间维度平滑
                self.denoiser = nn.Sequential(
                    # 1D卷积用于时间维度去噪
                    nn.Conv1d(in_channel, 16, kernel_size=3, padding=1),
                    nn.BatchNorm1d(16),
                    nn.ReLU(),
                    nn.Conv1d(16, in_channel, kernel_size=3, padding=1),
                    nn.Tanh()  # 输出范围 [-1, 1]，适合残差连接
                )
            elif denoise_type == 'attention':
                # 基于注意力的去噪
                self.denoiser = DenoiseAttention(in_channel, embed_dim // 4, dropout)
            else:
                raise ValueError(f"Unknown denoise_type: {denoise_type}")
        
        # 1. 时间特征嵌入
        # (B, N, T, C) -> (B, N, T, D)
        self.time_embedding = nn.Sequential(
            nn.Linear(in_channel, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # 2. 位置编码 (编码器)
        self.encoder_pos_embed = nn.Parameter(torch.randn(1, 1, self.seq_len, embed_dim))
        
        # 3. 自适应图学习
        if use_advanced_graph:
            self.dynamic_graph_conv = DynamicGraphConv(
                embed_dim=embed_dim,
                num_nodes=num_nodes,
                node_dim=dim,
                graph_heads=graph_heads,
                topk=topK,
                dropout=dropout
            )
        else:
            # 使用简单图学习（原版）
            self.node_embeddings1 = nn.Parameter(torch.randn(num_nodes, dim))
            self.node_embeddings2 = nn.Parameter(torch.randn(dim, num_nodes))
            self.topK = topK
            
            # 简单图卷积层
            self.graph_conv = nn.ModuleList([
                nn.Linear(embed_dim, embed_dim) for _ in range(2)
            ])
        
        # 4. Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)
        
        # ============ Decoder 部分 ============
        
        # 5. 可学习的未来查询向量 (代表未来 pred_len 个时间步)
        self.future_queries = nn.Parameter(torch.randn(1, pred_len, embed_dim))
        
        # 6. 位置编码 (解码器)
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, 1, pred_len, embed_dim))
        
        # 7. Transformer 解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_depth)
        
        # 8. 输出投影层 (embed_dim -> 1)
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),      # 保持维度
            nn.LayerNorm(embed_dim),              # 归一化
            nn.GELU(),                            # 更平滑的激活
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2), # 逐步降维
            nn.GELU(),
            nn.Dropout(dropout / 2),              # 减少 dropout
            nn.Linear(embed_dim // 2, 1)          # 最终输出
        )
        
        self.contrastive_loss = None
        self._init_weights()
        
    def _init_weights(self):
        if not self.use_advanced_graph:
            nn.init.xavier_uniform_(self.node_embeddings1)
            nn.init.xavier_uniform_(self.node_embeddings2)
        nn.init.normal_(self.encoder_pos_embed, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, std=0.02)
        # 未来查询使用更大的初始化范围
        nn.init.xavier_normal_(self.future_queries)
        
    def learn_graph(self):
        """学习自适应图结构（简单版本）"""
        if self.use_advanced_graph:
            raise NotImplementedError("Use graph_learner.forward() for advanced graph learning")
        
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
        
    def forward(self, history_data):
        """
        Encoder-Decoder 前向传播
        
        Args:
            history_data: (B, seq_len, N, 1) 短期历史数据
            
        Returns:
            prediction: (B, pred_len, N, 1) 预测结果
        """
        
        # ============ Step 0: 去噪（如果启用）============
        if self.use_denoising:
            if self.denoise_type == 'conv':
                # 卷积去噪：在时间维度上处理
                B, T, N, C = history_data.shape
                # 重塑为 (B*N, C, T) 以便使用Conv1d
                x_denoise = history_data.permute(0, 2, 3, 1).reshape(B * N, C, T)
                # 去噪
                noise = self.denoiser(x_denoise)  # (B*N, C, T)
                # 残差连接：原始数据 - 噪声
                x_denoise = x_denoise - noise
                # 重塑回 (B, T, N, C)
                history_data_clean = x_denoise.reshape(B, N, C, T).permute(0, 3, 1, 2)
            elif self.denoise_type == 'attention':
                # 注意力去噪
                history_data_clean = self.denoiser(history_data)  # (B, T, N, C)
            else:
                history_data_clean = history_data
        else:
            history_data_clean = history_data
        
        B, T, N, C = history_data_clean.shape
        
        # ============ ENCODER 部分 ============
        
        # 转换格式: (B, T, N, C) -> (B, N, T, C)
        x = history_data_clean.permute(0, 2, 1, 3)  # (B, N, T, C)

        # Step 1: 时间特征嵌入
        x = self.time_embedding(x)  # (B, N, T, D)
        
        # Step 2: 添加位置编码 (编码器)
        x = x + self.encoder_pos_embed  # (B, N, T, D)
        
        # Step 3: 自适应图学习 + 图卷积
        if self.use_advanced_graph:
            # 使用高级图学习模块
            x, learned_adjs, contrastive_loss = self.dynamic_graph_conv(x)  # (B, N, T, D)
            self.contrastive_loss = contrastive_loss
            x = F.relu(x)
        else:
            # 使用简单图学习（原版）
            adj = self.learn_graph()  # (N, N)
            x = self.graph_convolution(x, adj)  # (B, N, T, D)
        
        # Step 4: Transformer 编码器
        # (B, N, T, D) -> (B*N, T, D)
        x_flat = x.reshape(B * N, T, self.embed_dim)
        encoder_output = self.encoder(x_flat)  # (B*N, T, D)
        # (B*N, T, D) -> (B, N, T, D)
        encoder_output = encoder_output.reshape(B, N, T, self.embed_dim)
        
        # ============ DECODER 部分 ============
        
        # Step 5: 准备解码器查询向量
        # future_queries: (1, pred_len, D) -> (B*N, pred_len, D)
        queries = self.future_queries.expand(B * N, -1, -1)  # (B*N, pred_len, D)
        
        # Step 6: 添加位置编码 (解码器)
        # decoder_pos_embed: (1, 1, pred_len, D) -> (B*N, pred_len, D)
        queries = queries + self.decoder_pos_embed.squeeze(1)  # (B*N, pred_len, D)
        
        # Step 7: Transformer 解码器 (交叉注意力)
        # 准备编码器输出作为 memory: (B, N, T, D) -> (B*N, T, D)
        memory = encoder_output.reshape(B * N, T, self.embed_dim)
        
        # 解码
        decoder_output = self.decoder(queries, memory)  # (B*N, pred_len, D)
        
        # Step 8: 输出投影
        # (B*N, pred_len, D) -> (B*N, pred_len, 1)
        prediction_flat = self.output_projection(decoder_output)
        
        # Step 9: 重塑为最终输出格式
        # (B*N, pred_len, 1) -> (B, N, pred_len, 1) -> (B, pred_len, N, 1)
        prediction = prediction_flat.reshape(B, N, self.pred_len, 1)
        prediction = prediction.permute(0, 2, 1, 3)  # (B, pred_len, N, 1)
        
        return prediction


ForecastingWithAdaptiveGraph = AGPSTModel
