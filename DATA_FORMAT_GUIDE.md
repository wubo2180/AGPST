# 数据格式说明文档

## AGPST Direct Forecasting 数据流

### 📊 输入数据格式

所有数据集遵循统一格式：**`(B, T, N, C)`**

- **B**: Batch size（批次大小）
- **T**: Time steps（时间步数）
- **N**: Number of nodes（节点数量）= 358（PEMS03数据集）
- **C**: Number of channels/features（通道/特征数）= 1

### 🔄 数据流转换

#### 1. DataLoader 输出

```python
future_data, history_data, long_history_data = data

# 数据形状
history_data:      (B, 12, 358, 1)   # 短期历史
long_history_data: (B, 864, 358, 1)  # 长期历史
future_data:       (B, 12, 358, 1)   # 未来真实值（用于计算损失）
```

#### 2. ForecastingWithAdaptiveGraph 内部转换

```python
def forward(self, history_data, long_history_data, future_data, batch_seen, epoch):
    # Step 1: 转换长期历史数据用于 Patch Embedding
    # 从 (B, T, N, C) -> (B, N, T, C)
    long_history_data = long_history_data.transpose(1, 2)
    # 结果: (B, 358, 864, 1)
    
    # Step 2: Patch Embedding
    # 输入: (B, N, T, C) = (B, 358, 864, 1)
    # 输出: (B, N, P, D) = (B, 358, 72, 96)
    # 其中 P = 864/12 = 72 个patch，D = embed_dim = 96
    patches = self.patch_embedding(long_history_data)
    
    # Step 3: Dynamic Graph Learning
    # 输入/输出: (B, N, P, D) = (B, 358, 72, 96)
    patches, learned_adj, contrastive_loss = self.dynamic_graph_conv(patches)
    
    # Step 4: Positional Encoding + Transformer
    # 输入/输出: (B, N, P, D) = (B, 358, 72, 96)
    patches, _ = self.positional_encoding(patches)
    hidden_states = self.encoder(patches)
    hidden_states = self.encoder_norm(hidden_states)
    
    # Step 5: 提取节点特征
    # 输入: (B, N, P, D) = (B, 358, 72, 96)
    # 输出: (B, N, D) = (B, 358, 96)
    node_features = hidden_states[:, :, -1, :]  # 取最后一个patch
    node_features = self.output_adapter(node_features)
    
    # Step 6: GraphWaveNet 预测
    # 输入1: history_data (B, T, N, C) = (B, 12, 358, 1)  [未转换，保持原格式]
    # 输入2: node_features (B, N, D) = (B, 358, 96)
    # 输出: (B, N, L) = (B, 358, 12)
    y_hat = self.backend(history_data, hidden_states=node_features)
    
    # Step 7: 调整输出格式
    # 从 (B, N, L) -> (B, L, N) -> (B, N, L, 1)
    y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
    # 最终输出: (B, 358, 12, 1)
```

#### 3. 损失计算

```python
# 预测值和真实值格式
preds:  (B, 358, 12, 1)  # 模型输出
labels: (B, 12, 358, 1)  # 需要转换为 (B, 358, 12, 1) 或调整loss计算

# 注意：当前代码中 labels = future_data 保持 (B, 12, 358, 1) 格式
# 需要确保 SCALER 和 metric 函数能正确处理这两种格式
```

### ⚠️ 重要注意事项

#### 格式不一致问题

当前存在一个**潜在的格式不匹配**：

```python
# 在 direct_forecasting() 函数中
labels = future_data.to(args.device)  # (B, 12, 358, 1)
preds = model(...)                     # (B, 358, 12, 1)

# 这两个格式不一致！
```

#### 解决方案

有两种方案：

**方案 1: 在模型输出后转换**
```python
# 在 ForecastingWithAdaptiveGraph.forward() 最后
y_hat = y_hat.transpose(1, 2).unsqueeze(-1)  # (B, N, L, 1)
# 改为
y_hat = y_hat.unsqueeze(-1)  # (B, N, L, 1) 
# 然后在外部转换为 (B, L, N, 1)
```

**方案 2: 在损失计算前转换标签**
```python
# 在 direct_forecasting() 中
labels = future_data.transpose(1, 2).to(args.device)  # (B, 12, 358, 1) -> (B, 358, 12, 1)
```

### 🎯 推荐的标准化格式

建议统一使用 **`(B, L, N, C)`** 作为预测输出格式，与数据集格式保持一致：

```python
class ForecastingWithAdaptiveGraph:
    def forward(...):
        # ... 前面的处理 ...
        
        # GraphWaveNet 输出 (B, N, L)
        y_hat = self.backend(history_data, hidden_states=node_features)
        
        # 转换为 (B, L, N, 1) 与输入格式一致
        y_hat = y_hat.permute(0, 2, 1).unsqueeze(-1)
        
        return y_hat  # (B, L, N, C)
```

### 📝 各模块期望的输入格式总结

| 模块 | 输入格式 | 输出格式 |
|------|---------|---------|
| DataLoader | - | `(B, T, N, C)` |
| PatchEmbedding | `(B, N, T, C)` | `(B, N, P, D)` |
| DynamicGraphConv | `(B, N, P, D)` | `(B, N, P, D)` |
| Transformer | `(B, N, P, D)` | `(B, N, P, D)` |
| GraphWaveNet | `(B, T, N, C)` | `(B, N, L)` |
| 损失函数 | `(B, L, N, C)` & `(B, L, N, C)` | scalar |

### 🔧 调试检查清单

运行模型时，打印以下形状进行验证：

```python
print(f"history_data: {history_data.shape}")           # 应为 (B, 12, 358, 1)
print(f"long_history_data: {long_history_data.shape}") # 应为 (B, 864, 358, 1)
print(f"future_data: {future_data.shape}")             # 应为 (B, 12, 358, 1)
print(f"preds: {preds.shape}")                         # 应为 (B, 12, 358, 1) 或 (B, 358, 12, 1)
```

### ✅ 验证代码

```python
# 在训练循环中添加
if idx == 0 and epoch == 0:
    print("=" * 50)
    print("Data Shape Verification:")
    print(f"  history_data: {history_data.shape}")
    print(f"  long_history_data (before): {long_history_data_orig.shape}")
    print(f"  long_history_data (after transpose): {long_history_data.shape}")
    print(f"  future_data (labels): {labels.shape}")
    print(f"  predictions: {preds.shape}")
    print("=" * 50)
```

---

**注意**: 确保所有维度转换都经过仔细验证，避免出现维度不匹配导致的训练错误！
