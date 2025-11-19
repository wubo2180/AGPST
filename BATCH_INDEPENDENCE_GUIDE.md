# 🔒 避免Batch间信息泄露 - 最佳实践指南

## 🚨 问题说明

### 错误示例（信息泄露）

```python
# ❌ 错误：跨batch信息泄露
node_embeddings = ...  # (B, N, C)

# 这个einsum会混合不同batch的样本
supports = F.softmax(
    F.relu(torch.einsum('bnc,bmc->nm', node_embeddings, node_embeddings)), 
    dim=1
)
# 输出: (N, M) - 只有一个图，混合了所有batch样本的信息！
```

**问题**:
1. **信息泄露**: 预测样本A时使用了样本B的信息
2. **Batch依赖**: 同一数据在不同batch_size下结果不同
3. **不可复现**: 即使设置随机种子，shuffle也会影响结果
4. **测试集污染**: 训练时可能泄露测试集信息

---

## ✅ 正确实现

### 方案1: 使用正确的 einsum

```python
# ✅ 正确：每个样本独立计算
node_embeddings = ...  # (B, N, C)

# 保留batch维度
supports = F.softmax(
    F.relu(torch.einsum('bnc,bmc->bnm', node_embeddings, node_embeddings)), 
    dim=-1
)
# 输出: (B, N, M) - 每个样本有独立的图
```

**解释**:
```python
# 'bnc,bmc->bnm' 的含义：
# 对于每个batch b:
#   result[b, n, m] = sum_c (embeddings[b,n,c] * embeddings[b,m,c])
# 不同batch之间完全独立
```

---

### 方案2: 使用 bmm (批量矩阵乘法)

```python
# ✅ 正确：等价于上面的einsum
supports = F.softmax(
    F.relu(torch.bmm(
        node_embeddings, 
        node_embeddings.transpose(1, 2)
    )),
    dim=-1
)
# 输出: (B, N, N)
```

---

### 方案3: 使用 matmul (自动广播)

```python
# ✅ 正确：自动处理批量维度
supports = F.softmax(
    F.relu(torch.matmul(
        node_embeddings,  # (B, N, C)
        node_embeddings.transpose(-2, -1)  # (B, C, N)
    )),
    dim=-1
)
# 输出: (B, N, N)
```

---

## 📊 AGPST 代码检查结果

### ✅ 当前实现是正确的！

我检查了 `basicts/mask/graph_learning.py`，所有的图学习操作都正确地保持了batch独立性：

#### 1. 静态图学习（无batch维度）
```python
# compute_static_graphs() - 正确
# 静态图不依赖batch数据，每次都一样
adj = torch.mm(self.local_node_embeddings1[h], self.local_node_embeddings2[h])
# 输出: (N, N) - 静态参数，不涉及batch
```

#### 2. 动态图学习（保持batch独立）
```python
# compute_dynamic_graphs() - 正确
# 使用 bmm 保持batch独立
similarities = torch.bmm(dynamic_embeds, dynamic_embeds.transpose(1, 2))
# 输出: (B, N, N) - 每个样本独立

# 使用 matmul 也是正确的
dynamic_sims = torch.matmul(dynamic_expanded, static_expanded)
# (B, H, N, C) @ (B, H, C, N) -> (B, H, N, N)
# 每个样本独立计算
```

#### 3. 对比学习（batch内独立）
```python
# compute_contrastive_loss() - 正确
similarity_matrices = torch.bmm(projected, projected.transpose(1, 2))
# 输出: (B, N, N) - 每个样本的节点间相似度
# 不跨样本计算
```

---

## 🔍 如何检测信息泄露

### 测试方法

```python
import torch

def test_batch_independence():
    """测试batch独立性"""
    
    # 模拟数据
    B, N, C = 4, 10, 16
    embeddings = torch.randn(B, N, C)
    
    # 方法1：错误的实现（会泄露）
    def wrong_method(emb):
        return torch.einsum('bnc,bmc->nm', emb, emb)
    
    # 方法2：正确的实现（不泄露）
    def correct_method(emb):
        return torch.einsum('bnc,bmc->bnm', emb, emb)
    
    # 测试1：单样本 vs 多样本
    single_result = correct_method(embeddings[0:1])  # (1, N, N)
    multi_result = correct_method(embeddings)[0]     # (N, N)
    
    # 正确实现：单样本结果应该与多样本中第一个样本一致
    print(f"单样本vs多样本差异: {torch.abs(single_result[0] - multi_result).max().item():.6f}")
    # 应该接近 0
    
    # 错误实现：会有差异
    wrong_single = wrong_method(embeddings[0:1])
    wrong_multi = wrong_method(embeddings)
    print(f"错误实现差异: {torch.abs(wrong_single - wrong_multi).max().item():.6f}")
    # 会很大，因为混合了其他样本
    
    # 测试2：不同batch size
    bs_2 = correct_method(embeddings[0:2])[0]
    bs_4 = correct_method(embeddings[0:4])[0]
    
    print(f"不同batch_size差异: {torch.abs(bs_2 - bs_4).max().item():.6f}")
    # 应该接近 0（正确实现）

if __name__ == '__main__':
    test_batch_independence()
```

**预期输出**:
```
单样本vs多样本差异: 0.000000  ✅ 正确
错误实现差异: 2.345678       ❌ 错误
不同batch_size差异: 0.000000  ✅ 正确
```

---

## 🛡️ 防护检查清单

在实现图学习模块时，检查以下几点：

### ✅ 正确实现检查

- [ ] 所有矩阵运算保留batch维度
- [ ] 使用 `bmm` 或 `matmul` 而非 `mm`
- [ ] einsum 输出包含batch维度 (如 'bnc,bmc->bnm')
- [ ] 没有在batch维度上求和/平均
- [ ] 不同batch_size下同一样本结果一致

### ❌ 常见错误模式

- [ ] `torch.mm()` 用于batch数据
- [ ] einsum 丢失batch维度 (如 'bnc,bmc->nm')
- [ ] `tensor.mean(dim=0)` 在batch维度求均值
- [ ] `tensor.sum(dim=0)` 在batch维度求和
- [ ] 全局池化操作跨batch

---

## 📝 代码修复示例

### 场景1: 节点嵌入相似度

```python
# ❌ 错误
node_emb = get_embeddings(x)  # (B, N, C)
sim = torch.einsum('bnc,bmc->nm', node_emb, node_emb)

# ✅ 修复
node_emb = get_embeddings(x)  # (B, N, C)
sim = torch.einsum('bnc,bmc->bnm', node_emb, node_emb)
# 或
sim = torch.bmm(node_emb, node_emb.transpose(1, 2))
```

### 场景2: 图卷积

```python
# ❌ 错误：使用全局聚合的邻接矩阵
adj = compute_adj(all_batches)  # (N, N) 跨batch
output = torch.mm(adj, features)  # 所有样本用同一个图

# ✅ 修复：每个样本独立计算
adj = compute_adj_per_sample(batch)  # (B, N, N)
output = torch.bmm(adj, features)  # (B, N, C)
```

### 场景3: 注意力机制

```python
# ❌ 错误：跨batch计算注意力
attn = torch.einsum('bnc,bmc->nm', Q, K)  # 丢失batch维度

# ✅ 修复：保持batch独立
attn = torch.einsum('bnc,bmc->bnm', Q, K)
# 或
attn = torch.bmm(Q, K.transpose(1, 2)) / sqrt(d_k)
```

---

## 🎯 AGPST 具体建议

### 当前状态

✅ **好消息**: AGPST的图学习模块已经正确实现，不存在信息泄露！

所有关键操作都使用了正确的方法：
- `torch.bmm()` 用于动态图学习
- `torch.matmul()` 用于批量操作
- 所有操作保持 `(B, N, N)` 或 `(B, H, N, N)` 格式

### 未来开发注意

如果添加新的图学习功能，确保：

1. **使用模板代码**:
```python
# 节点相似度计算模板
def compute_node_similarity(node_features):
    """
    Args:
        node_features: (B, N, C)
    Returns:
        similarity: (B, N, N)  # 注意保留B
    """
    return torch.bmm(
        node_features, 
        node_features.transpose(1, 2)
    )
```

2. **添加断言检查**:
```python
def compute_adaptive_graph(embeddings):
    B, N, C = embeddings.shape
    graph = compute_graph(embeddings)
    
    # 断言：输出必须包含batch维度
    assert graph.shape[0] == B, f"Expected batch dim {B}, got {graph.shape[0]}"
    assert graph.shape == (B, N, N), f"Expected (B,N,N), got {graph.shape}"
    
    return graph
```

3. **单元测试**:
```python
def test_no_batch_leakage():
    model = YourGraphModel()
    x1 = torch.randn(1, 12, 358, 1)
    x2 = torch.randn(2, 12, 358, 1)
    
    # 第一个样本在不同batch中结果应该一致
    out1 = model(x1)[0]
    out2 = model(torch.cat([x1, x2]))[0]
    
    assert torch.allclose(out1, out2, atol=1e-6)
```

---

## 📚 参考资料

### PyTorch 批量操作对比

| 操作 | 输入 | 输出 | 用途 | Batch安全 |
|------|------|------|------|----------|
| `torch.mm(A, B)` | (N,M), (M,P) | (N,P) | 单个矩阵乘法 | ❌ 不支持batch |
| `torch.bmm(A, B)` | (B,N,M), (B,M,P) | (B,N,P) | 批量矩阵乘法 | ✅ 安全 |
| `torch.matmul(A, B)` | 任意维度 | 广播结果 | 通用乘法 | ✅ 安全（自动广播）|
| `einsum('nm,mp->np')` | (N,M), (M,P) | (N,P) | 灵活张量操作 | ❌ 取决于下标 |
| `einsum('bnm,bmp->bnp')` | (B,N,M), (B,M,P) | (B,N,P) | 批量操作 | ✅ 安全 |

### 安全的Batch操作原则

1. **保留batch维度**: 所有操作输出应包含batch维度
2. **使用batch-aware函数**: `bmm`, `matmul`, 正确的einsum
3. **避免全局聚合**: 不要跨batch求和/平均
4. **独立性测试**: 验证单样本结果不受其他样本影响

---

## 🎓 总结

### 核心原则

> **每个样本的图结构学习必须完全独立于其他样本**

### 检测口诀

```
看到 mm → 改 bmm
看到 einsum → 检查下标
看到 mean(0) → 警惕
看到 sum(0) → 警惕
输出无 B → 错误
```

### AGPST现状

✅ **当前实现完全正确**，无需修改  
✅ 所有图学习操作都保持batch独立性  
✅ 可以放心使用和训练

---

**版本**: v1.0  
**更新时间**: 2025-11-16  
**状态**: AGPST代码已验证安全 ✅
