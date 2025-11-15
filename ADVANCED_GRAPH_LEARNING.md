# 🔗 高级自适应图学习模块集成指南

## 📋 概述

AGPST 模型现在集成了**高级多尺度自适应图学习模块** (`AdaptiveGraphLearner`)，提供比简单图学习更强大的图结构学习能力。

---

## 🆚 两种图学习模式对比

### 模式1: 简单图学习 (Simple Graph Learning)

**特点**：
- 基于静态节点嵌入
- 单一尺度图结构
- 轻量级，计算快速

**架构**：
```python
adj = MM(node_embed1, node_embed2)  # (N, N)
adj = ReLU(adj)
adj = TopK(adj, k)
adj = Normalize(adj)
```

**适用场景**：
- 快速原型验证
- 计算资源受限
- 图结构相对静态

---

### 模式2: 高级图学习 (Advanced Graph Learning) ✨ **推荐**

**特点**：
- 多头注意力机制
- 动态+静态图融合
- 多尺度图学习（局部+全局）
- 基于时序信息的动态图
- InfoNCE对比学习

**架构**：
```python
# 静态图（多头）
static_graphs = [
    LocalGraph_1, LocalGraph_2, ...,  # 捕捉局部结构
    GlobalGraph_1, GlobalGraph_2, ... # 捕捉全局模式
]

# 动态图（基于时序特征）
temporal_features = TemporalAttention(patches)
dynamic_embeds = DynamicEncoder(temporal_features)
dynamic_graphs = Learn(dynamic_embeds)

# 融合
final_graph = Fusion(static_graphs, dynamic_graphs)

# 对比学习
contrastive_loss = InfoNCE(node_embeddings)
```

**适用场景**：
- 追求最佳性能
- 复杂时空依赖
- 图结构动态变化
- 有充足计算资源

---

## 🚀 使用方法

### 方式1: 启用高级图学习（推荐）

编辑 `parameters/PEMS03_v3.yaml`:

```yaml
# Adaptive graph learning
use_advanced_graph: True   # 启用高级图学习
graph_heads: 4             # 多头注意力数量
dim: 10                    # 节点嵌入维度
topK: 10                   # Top-K稀疏化
```

### 方式2: 使用简单图学习（快速）

```yaml
# Adaptive graph learning
use_advanced_graph: False  # 使用简单图学习
dim: 10
topK: 10
```

---

## 📊 参数详解

### 核心参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_advanced_graph` | bool | `True` | 是否使用高级图学习 |
| `graph_heads` | int | `4` | 图注意力头数（建议2-8） |
| `dim` | int | `10` | 节点嵌入维度 |
| `topK` | int | `10` | Top-K邻居数量 |

### 高级参数（在 graph_learning.py 中）

```python
AdaptiveGraphLearner(
    num_nodes=358,           # 节点数量
    node_dim=10,             # 节点嵌入维度
    embed_dim=96,            # 特征维度
    graph_heads=4,           # 多头数量
    topk=10,                 # Top-K稀疏化
    dropout=0.1,             # Dropout率
    use_temporal_info=True   # 是否使用时序信息
)
```

---

## 🔬 技术细节

### 多尺度图学习

**局部图** (Local Graphs):
- 数量: `graph_heads // 2`
- 维度: `node_dim // 2`
- 温度: `temperature * 2`
- 目标: 捕捉近邻关系

**全局图** (Global Graphs):
- 数量: `graph_heads - local_heads`
- 维度: `node_dim`
- 温度: `temperature * 0.5`
- 目标: 捕捉长程依赖

### 动态图学习流程

```
输入: patch_features (B, N, P, D)
  ↓
时序注意力聚合
  ↓
动态节点嵌入编码
  ↓
GNN增强 (2层)
  ↓
动态相似度计算
  ↓
动态图 (B, H, N, N)
```

### 图融合策略

```python
# 自适应融合权重
fusion_weights = Sigmoid(Linear(node_features))

# 融合
fused_graph = (1 - α) * static_graph + α * dynamic_graph

# 多头聚合
final_graph = EdgeEncoder(MultiHeadGraphs)
```

### 对比学习

**InfoNCE Loss**:
```python
# 节点嵌入投影
z = Projection(node_embeddings)  # (B, N, D')
z = Normalize(z)

# 相似度矩阵
sim = MM(z, z^T) / temperature

# 对比损失
loss = -log(exp(sim_pos) / sum(exp(sim_all)))
```

---

## 💡 性能对比

### 计算开销

| 模式 | 参数量 | 前向时间 | GPU内存 |
|------|--------|----------|---------|
| Simple | ~7K | 1.0x | 1.0x |
| Advanced | ~50K | 1.5-2.0x | 1.3-1.5x |

### 预测精度（预期提升）

| 数据集 | Simple MAE | Advanced MAE | 提升 |
|--------|------------|--------------|------|
| PEMS03 | X.XX | X.XX - 0.5 | ~5-10% |
| PEMS04 | X.XX | X.XX - 0.3 | ~3-8% |
| PEMS07 | X.XX | X.XX - 0.4 | ~4-9% |
| PEMS08 | X.XX | X.XX - 0.6 | ~6-12% |

*注: 实际效果需要实验验证*

---

## 🧪 实验建议

### 对比实验

**实验1: 图学习模式对比**
```yaml
# Baseline
use_advanced_graph: False

# Advanced
use_advanced_graph: True
graph_heads: 4
```

**实验2: 图头数消融**
```yaml
use_advanced_graph: True
graph_heads: [2, 4, 6, 8]  # 分别测试
```

**实验3: Top-K 敏感性**
```yaml
use_advanced_graph: True
topK: [5, 10, 15, 20]  # 分别测试
```

### 可视化建议

1. **学习到的图结构**
   ```python
   # 在 forward 中保存
   learned_adjs, _ = self.graph_learner(patch_features)
   torch.save(learned_adjs, 'learned_graphs.pt')
   
   # 可视化
   import networkx as nx
   import matplotlib.pyplot as plt
   adj = learned_adjs[0].detach().cpu().numpy()
   G = nx.from_numpy_array(adj)
   nx.draw(G)
   ```

2. **对比学习效果**
   ```python
   # 记录对比损失
   if self.contrastive_loss is not None:
       swanlab.log({"train/contrastive_loss": self.contrastive_loss})
   ```

---

## 🔧 调试与优化

### 常见问题

**Q1: 显存不足？**
```yaml
# 减少图头数
graph_heads: 2

# 或使用简单模式
use_advanced_graph: False
```

**Q2: 训练过慢？**
```yaml
# 关闭时序信息（在代码中）
use_temporal_info: False

# 或减少 topK
topK: 5
```

**Q3: 对比损失为NaN？**
- 检查温度参数是否合理
- 增加数值稳定性处理（已在代码中实现）

### 性能优化

**技巧1: 梯度累积**
```python
# 在 main.py 中
accumulation_steps = 2
for i, batch in enumerate(dataloader):
    loss = loss / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**技巧2: 混合精度训练**
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## 📈 进阶使用

### 自定义图头配置

修改 `basicts/mask/graph_learning.py`:

```python
# 调整局部/全局图比例
self.local_graph_heads = graph_heads // 3  # 1/3局部
self.global_graph_heads = graph_heads - self.local_graph_heads  # 2/3全局
```

### 自定义融合策略

```python
# 在 AdaptiveGraphLearner.forward 中
# 改为固定权重融合
alpha = 0.5  # 50% static, 50% dynamic
fused_adjs = (1 - alpha) * static_expanded + alpha * dynamic_adjs
```

### 添加额外的图正则化

```python
# 在模型训练循环中
graph_reg = torch.norm(learned_adjs, p='fro') * 0.001
total_loss = prediction_loss + contrastive_loss + graph_reg
```

---

## 📚 相关文档

- [去噪模块指南](./DENOISING_MODULE.md)
- [快速开始](./ADAPTIVE_GRAPH_QUICKSTART.md)
- [完整教程](./ADAPTIVE_GRAPH_GUIDE.md)

---

## 🎯 快速开始示例

```bash
# 1. 测试简单图学习
python main.py --config=parameters/PEMS03_v3.yaml --test_mode=1 \
    --device=cuda

# 2. 测试高级图学习（需要先在配置中设置 use_advanced_graph: True）
python main.py --config=parameters/PEMS03_v3.yaml --test_mode=1 \
    --device=cuda

# 3. 完整训练
python main.py --config=parameters/PEMS03_v3.yaml --device=cuda
```

---

## ✅ 检查清单

在使用高级图学习前，确保：

- [ ] 配置文件中设置 `use_advanced_graph: True`
- [ ] 设置合适的 `graph_heads` (建议2-8)
- [ ] GPU 显存充足（至少8GB）
- [ ] 已导入 `graph_learning.py` 模块
- [ ] 理解对比学习损失的作用

---

**版本**: v2.0  
**更新时间**: 2025-11-14  
**作者**: AGPST Team

---

## 🔗 引用

如果高级图学习模块对您的研究有帮助，请考虑引用：

```bibtex
@article{agpst2025,
  title={Adaptive Graph-based Probabilistic Spatial-Temporal Network},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```
