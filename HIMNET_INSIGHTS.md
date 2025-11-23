# HimNet 设计理念借鉴分析

## 📊 HimNet vs 我们的架构对比

| 维度 | HimNet (KDD'24) | 我们的 AlternatingST | 借鉴价值 |
|------|-----------------|---------------------|---------|
| **编码方式** | 双编码器并行 | 交替编码 | ✅ 我们更优 (有信息流) |
| **节点建模** | 元学习节点嵌入 | 统一参数 | ⭐⭐⭐⭐⭐ 高价值 |
| **图卷积** | HimGCN (节点特定参数) | Self-Attention | ⭐⭐⭐⭐ 可尝试 |
| **训练策略** | 计划采样 + 教师强制 | 标准训练 | ⭐⭐⭐ 中等价值 |
| **损失函数** | Huber/MaskedMAE | MAE | ⭐⭐⭐⭐ 值得尝试 |
| **复杂度** | 3.2M 参数 | 1.23M 参数 | ✅ 我们更轻量 |

---

## 🎯 核心借鉴点详解

### 1. 异质性节点嵌入 (Heterogeneity-Aware Node Embedding)

**HimNet 的实现**:
```python
class HimGCN(nn.Module):
    def __init__(self, num_nodes, d_meta=64):
        # 每个节点有独立的元嵌入
        self.meta_node_emb = nn.Parameter(torch.randn(num_nodes, d_meta))
        self.meta_fc = nn.Linear(d_meta, d_model * d_model)
    
    def forward(self, x, adj):
        # 生成节点特定的卷积权重
        W_spatial = self.meta_fc(self.meta_node_emb)  # (N, D*D)
        W_spatial = W_spatial.reshape(num_nodes, d_model, d_model)
        
        # 每个节点使用自己的权重
        out = []
        for i in range(num_nodes):
            out.append(torch.matmul(x[:, i], W_spatial[i]))  # (B, T, D) @ (D, D)
        return torch.stack(out, dim=1)  # (B, N, T, D)
```

**对我们的应用**:
```python
class HeterogeneousSpatialEncoder(nn.Module):
    """
    异质性感知的空间编码器
    为不同节点生成不同的注意力权重
    """
    def __init__(self, num_nodes, d_model, d_meta=64):
        super().__init__()
        # 节点元嵌入
        self.node_emb = nn.Parameter(torch.randn(num_nodes, d_meta))
        
        # 生成节点特定的 Query/Key 偏置
        self.meta_q = nn.Linear(d_meta, d_model)
        self.meta_k = nn.Linear(d_meta, d_model)
        
        # 标准 Transformer
        self.encoder = nn.TransformerEncoder(...)
        
    def forward(self, x):
        B, N, T, D = x.shape
        
        # 为每个节点生成特定的偏置
        node_q_bias = self.meta_q(self.node_emb)  # (N, D)
        node_k_bias = self.meta_k(self.node_emb)  # (N, D)
        
        # 重塑并添加节点偏置
        x_flat = x.reshape(B*T, N, D)
        x_flat = x_flat + node_q_bias.unsqueeze(0)  # 广播到 (B*T, N, D)
        
        # Transformer 编码
        spatial_features = self.encoder(x_flat)
        return spatial_features.reshape(B, N, T, D)
```

**优势**:
- ✅ 捕获节点异质性 (高速公路 vs 城市道路)
- ✅ 参数增加少: `N × d_meta` (358 × 64 = 22k 参数)
- ✅ 与我们的 Transformer 架构兼容

---

### 2. 引入图卷积层 (Graph Convolution)

**HimNet 的 HimGCN**:
```python
class HimGCN(nn.Module):
    def forward(self, x, adj):
        # 1. 邻接矩阵归一化
        D = torch.diag(adj.sum(1))
        A_norm = D^(-0.5) @ adj @ D^(-0.5)
        
        # 2. 图卷积: X' = A_norm @ X @ W
        # 每个节点 i 聚合邻居信息
        support = torch.matmul(A_norm, x)  # (N, N) @ (B, N, T, D)
        out = torch.matmul(support, W_spatial)  # (B, N, T, D) @ (D, D)
        return out
```

**对我们的混合方案**:
```python
class HybridSpatialEncoder(nn.Module):
    """
    混合空间编码器: GCN + Transformer
    - GCN: 利用物理邻接关系
    - Transformer: 学习语义关系
    """
    def __init__(self, num_nodes, d_model, adj_mx):
        super().__init__()
        # GCN 分支
        self.gcn = nn.ModuleList([
            GraphConv(d_model, d_model) for _ in range(2)
        ])
        
        # Transformer 分支
        self.transformer = nn.TransformerEncoder(...)
        
        # 融合
        self.fusion = nn.Linear(d_model * 2, d_model)
        
        # 邻接矩阵归一化
        self.adj_mx = self._normalize_adj(adj_mx)
        
    def forward(self, x):
        B, N, T, D = x.shape
        
        # GCN 路径
        x_gcn = x.reshape(B*T, N, D)
        for gcn_layer in self.gcn:
            x_gcn = gcn_layer(x_gcn, self.adj_mx)  # (B*T, N, D)
        x_gcn = x_gcn.reshape(B, N, T, D)
        
        # Transformer 路径
        x_trans = x.reshape(B*T, N, D)
        x_trans = self.transformer(x_trans).reshape(B, N, T, D)
        
        # 融合两条路径
        x_fused = torch.cat([x_gcn, x_trans], dim=-1)  # (B, N, T, 2D)
        return self.fusion(x_fused)  # (B, N, T, D)
```

**优势**:
- ✅ GCN 利用先验知识 (邻接矩阵)
- ✅ Transformer 学习隐式关系
- ✅ 双路径互补

---

### 3. 计划采样 (Scheduled Sampling)

**HimNet 的训练策略**:
```python
# 训练时逐步减少教师强制比例
teacher_forcing_ratio = 0.5  # 初始 50% 使用真实标签

for epoch in range(epochs):
    if epoch > 10:  # 预热期后启用
        # 以一定概率使用模型预测而非真实标签
        use_gt = random.random() < teacher_forcing_ratio
        
        if use_gt:
            decoder_input = ground_truth[:, t-1]
        else:
            decoder_input = model_prediction[:, t-1]
        
        # 逐步降低教师强制比例
        teacher_forcing_ratio *= 0.999  # 衰减
```

**对我们的应用** (在交替架构中):
```python
class AlternatingSTModelWithSampling(nn.Module):
    def forward(self, x, teacher_forcing_ratio=0.0):
        # Stage 1
        temp_out = self.temporal_encoder_1(x)
        spat_out = self.spatial_encoder_1(temp_out)
        fused = self.fusion_1(temp_out, spat_out)
        decoded = self.decoder(fused)
        
        # 计划采样: 以概率 p 使用解码结果,否则使用原始输入
        if self.training and random.random() > teacher_forcing_ratio:
            stage2_input = decoded  # 使用模型解码的结果
        else:
            stage2_input = x  # 使用真实输入 (教师强制)
        
        # Stage 2
        temp_out_2 = self.temporal_encoder_2(stage2_input)
        spat_out_2 = self.spatial_encoder_2(temp_out_2)
        final_out = self.fusion_2(temp_out_2, spat_out_2)
        
        return final_out
```

**优势**:
- ✅ 提高模型鲁棒性 (训练时见过自己的错误)
- ✅ 减少训练-测试差异
- ⚠️ 但可能增加训练不稳定性

---

### 4. 更鲁棒的损失函数

**HimNet 的损失设计**:
```python
class HuberLoss(nn.Module):
    """
    Huber Loss: 结合 MAE 和 MSE 的优点
    - 小误差: 使用 L2 (平滑梯度)
    - 大误差: 使用 L1 (对异常值鲁棒)
    """
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred, true):
        error = torch.abs(pred - true)
        
        # 小误差: 0.5 * error^2
        quadratic = 0.5 * error ** 2
        
        # 大误差: delta * (error - 0.5*delta)
        linear = self.delta * (error - 0.5 * self.delta)
        
        # 分段函数
        loss = torch.where(error <= self.delta, quadratic, linear)
        return loss.mean()
```

**混合损失方案**:
```python
class HybridLoss(nn.Module):
    """
    混合损失: Huber + MAE + MAPE
    """
    def __init__(self, alpha=0.6, beta=0.3, gamma=0.1):
        super().__init__()
        self.huber = HuberLoss(delta=1.0)
        self.alpha = alpha  # Huber 权重
        self.beta = beta    # MAE 权重
        self.gamma = gamma  # MAPE 权重
    
    def forward(self, pred, true, null_val=0.0):
        # 掩码
        mask = (true != null_val).float()
        
        # Huber Loss
        huber = self.huber(pred * mask, true * mask)
        
        # MAE
        mae = torch.abs(pred - true) * mask
        mae = mae.sum() / mask.sum()
        
        # MAPE
        mape = torch.abs((pred - true) / (true + 1e-5)) * mask
        mape = mape.sum() / mask.sum()
        
        return self.alpha * huber + self.beta * mae + self.gamma * mape
```

---

## 🔥 推荐的改进优先级

### ⭐⭐⭐⭐⭐ 最高优先级: 异质性节点嵌入
**实施难度**: ⚡ 低 (只需修改 SpatialEncoder)  
**预期收益**: 📈 5-10% MAE 降低  
**风险**: ⚠️ 低 (参数增加少)

**行动**: 创建 `alternating_st_heterogeneous.py`

---

### ⭐⭐⭐⭐ 高优先级: Huber Loss
**实施难度**: ⚡⚡ 低 (只需修改损失函数)  
**预期收益**: 📈 3-5% 鲁棒性提升  
**风险**: ⚠️ 极低

**行动**: 在 `basicts/losses/losses.py` 添加 `HuberLoss`

---

### ⭐⭐⭐ 中等优先级: GCN + Transformer 混合
**实施难度**: ⚡⚡⚡ 中 (需要实现 GCN 层)  
**预期收益**: 📈 5-8% (利用邻接矩阵先验)  
**风险**: ⚠️⚠️ 中 (可能过拟合)

**行动**: 创建 `HybridSpatialEncoder`

---

### ⭐⭐ 低优先级: 计划采样
**实施难度**: ⚡⚡⚡⚡ 高 (需要修改训练逻辑)  
**预期收益**: 📈 2-5% (减少训练-测试差异)  
**风险**: ⚠️⚠️⚠️ 高 (可能训练不稳定)

**行动**: 最后尝试,需要大量调参

---

## 📝 实施建议

### 快速验证方案 (1-2 天)
1. ✅ **添加 Huber Loss** (1 小时)
2. ✅ **异质性节点嵌入** (3-4 小时)
3. ✅ **对比实验** (PEMS03, 10 epochs)

**预期**: MAE 从 5.4 降至 **4.8-5.0**

### 完整改进方案 (1 周)
1. ✅ Huber Loss
2. ✅ 异质性节点嵌入
3. ✅ GCN + Transformer 混合空间编码
4. ✅ 完整训练 (150 epochs, 4 个数据集)

**预期**: MAE 从 5.4 降至 **4.2-4.5** (达到 SOTA 水平)

---

## 🎯 核心结论

### HimNet 给我们的启示:
1. **简洁架构 + 关键创新** > 复杂架构
   - HimNet 也是双编码器 (简单)
   - 但引入节点异质性 (关键创新)

2. **利用领域知识**
   - 邻接矩阵 (GCN)
   - 节点类型差异 (元嵌入)

3. **训练技巧很重要**
   - 损失函数选择 (Huber)
   - 采样策略 (Scheduled Sampling)

### 我们的优势:
- ✅ **交替架构理论上优于并行** (有信息流动)
- ✅ **参数更少** (1.23M vs 3.2M)
- ✅ **已验证的基线** (MAE 5.4)

### 下一步:
**不需要重新设计架构!**  
只需在 Phase 1 基础上增加:
1. 异质性节点嵌入 (小修改)
2. Huber Loss (小修改)
3. (可选) GCN 混合 (中等修改)

预期: **从 MAE 5.4 → 4.2-4.5** (20-25% 提升)
