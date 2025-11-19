# basicts/mask 模块架构说明

## 📁 文件结构

```
basicts/mask/
├── __init__.py              # 模块导出
├── model.py                 # 🎯 主模型文件
├── graph_learning.py        # 📊 自适应图学习模块
├── patch_embed.py           # 🔲 Patch嵌入模块
├── transformer.py           # 🔄 Transformer编码器
└── positional_encoding.py   # 📍 位置编码
```

---

## 🎯 model.py - 主模型文件

**类**: `AGPSTModel` (别名: `ForecastingWithAdaptiveGraph`)

**功能**: 端到端的自适应图时空预测模型

**架构流程**:
```
输入数据
  ↓
PatchEmbedding (patch_embed.py)
  ↓
PositionalEncoding (positional_encoding.py)
  ↓
DynamicGraphConv (graph_learning.py)
  ↓
TransformerLayers (transformer.py)
  ↓
GraphWaveNet (后端预测)
  ↓
输出预测
```

**导入方式**:
```python
from basicts.mask import AGPSTModel
# 或
from basicts.mask.model import AGPSTModel
```

---

## 📊 graph_learning.py - 图学习模块

### 类1: `AdaptiveGraphLearner`
- **功能**: 多尺度自适应图学习
- **输入**: (B, N, P, D) - patch features
- **输出**: (B, N, N) - 邻接矩阵 + 对比学习损失

**特性**:
- ✅ 静态图 (基于预训练嵌入)
- ✅ 动态图 (基于当前batch特征)
- ✅ 多尺度 (局部图 + 全局图)
- ✅ Top-K稀疏化
- ✅ InfoNCE对比学习

### 类2: `DynamicGraphConv`
- **功能**: 动态图卷积
- **输入**: (B, N, P, D) - patch features
- **输出**: (B, N, P, D) - 图卷积后的特征

**导入方式**:
```python
from basicts.mask import DynamicGraphConv, AdaptiveGraphLearner
# 或
from basicts.mask.graph_learning import DynamicGraphConv
```

---

## 🔲 patch_embed.py - Patch嵌入

### 类: `PatchEmbedding`
- **功能**: 将时间序列转换为patches
- **输入**: (B, N, L, C) - 长期历史数据
- **输出**: (B, N, P, D) - patch嵌入

**实现**:
- Conv2d进行patch分割
- Xavier初始化防止NaN

**导入方式**:
```python
from basicts.mask import PatchEmbedding
# 或
from basicts.mask.patch_embed import PatchEmbedding
```

---

## 🔄 transformer.py - Transformer编码器

### 类: `TransformerLayers`
- **功能**: 时序建模
- **输入**: (B, N, P, D)
- **输出**: (B, N, P, D)

**特性**:
- PyTorch原生TransformerEncoder
- 多层编码
- 位置缩放

**导入方式**:
```python
from basicts.mask import TransformerLayers
# 或
from basicts.mask.transformer import TransformerLayers
```

---

## 📍 positional_encoding.py - 位置编码

### 类: `PositionalEncoding`
- **功能**: 为patches添加位置信息
- **输入**: (B, N, P, D)
- **输出**: (B, N, P, D)

**导入方式**:
```python
from basicts.mask import PositionalEncoding
# 或
from basicts.mask.positional_encoding import PositionalEncoding
```

---

## 🔄 模块依赖关系

```
model.py
  ├─ import graph_learning.DynamicGraphConv
  ├─ import patch_embed.PatchEmbedding
  ├─ import positional_encoding.PositionalEncoding
  ├─ import transformer.TransformerLayers
  └─ import ..graphwavenet.GraphWaveNet

graph_learning.py
  └─ import torch, torch.nn (无内部依赖)

patch_embed.py
  └─ import torch, torch.nn (无内部依赖)

transformer.py
  └─ import torch.nn.TransformerEncoder

positional_encoding.py
  └─ import torch, torch.nn (无内部依赖)
```

---

## 🚀 使用示例

### 完整模型使用
```python
from basicts.mask import AGPSTModel

model = AGPSTModel(
    num_nodes=358,
    dim=10,
    topK=10,
    patch_size=12,
    in_channel=1,
    embed_dim=96,
    num_heads=4,
    graph_heads=4,
    mlp_ratio=4,
    dropout=0.1,
    encoder_depth=4,
    backend_args={...}
)

# 前向传播
prediction = model(
    history_data,      # (B, 12, 358, 1)
    long_history_data, # (B, 864, 358, 1)
)
```

### 单独使用组件
```python
from basicts.mask import PatchEmbedding, DynamicGraphConv, TransformerLayers

# Patch嵌入
patch_embed = PatchEmbedding(patch_size=12, in_channel=1, embed_dim=96)
patches = patch_embed(long_history)  # (B, N, 864, 1) -> (B, N, 72, 96)

# 图学习
graph_conv = DynamicGraphConv(embed_dim=96, num_nodes=358, node_dim=10)
graph_features, adj, loss = graph_conv(patches)

# Transformer
transformer = TransformerLayers(hidden_dim=96, nlayers=4, mlp_ratio=4)
temporal_features = transformer(graph_features)
```

---

## ✅ 优化点

### 相比旧版本的改进:
1. **文件精简**: 从13个文件减少到5个核心文件
2. **命名清晰**: model.py作为主入口，其他文件功能明确
3. **模块化**: 每个文件负责单一功能
4. **无冗余**: 删除了所有_improved, integration_example等文件
5. **易维护**: 清晰的依赖关系，便于调试和扩展

### 代码优化:
- ✅ 完全向量化 (无Python循环)
- ✅ GPU优化 (batch操作)
- ✅ 数值稳定性 (温度限制、归一化)
- ✅ 内存高效 (inplace操作、梯度裁剪)

---

## 📝 文件对应关系

| 旧文件 | 新文件 | 说明 |
|--------|--------|------|
| `forecasting_with_adaptive_graph.py` | `model.py` | 主模型 |
| `post_patch_adaptive_graph.py` | `graph_learning.py` | 图学习 |
| `patch.py` | `patch_embed.py` | Patch嵌入 |
| `transformer_layers.py` | `transformer.py` | Transformer |
| `positional_encoding.py` | ✅ 保持不变 | 位置编码 |
| `model.py` (旧) | ❌ 删除 | 预训练模型 |
| `maskgenerator.py` | ❌ 删除 | 预训练用 |
| `*_improved.py` | ❌ 删除 | 未使用的改进版 |
| `integration_example.py` | ❌ 删除 | 示例代码 |
| `adaptive_graph.py` | ❌ 删除 | 旧版图学习 |
| `contrastive_loss.py` | ❌ 删除 | 已集成到graph_learning |
| `GIN.py` | ❌ 删除 | 未使用 |
| `spatial_temporal_attention.py` | ❌ 删除 | 未使用 |

---

## 🔧 维护建议

1. **添加新功能**: 
   - 如果是新的图学习方法 → 扩展 `graph_learning.py`
   - 如果是新的嵌入方式 → 扩展 `patch_embed.py`
   - 如果是新的架构 → 修改 `model.py`

2. **调试**:
   - 图结构问题 → 检查 `graph_learning.py`
   - 数据格式问题 → 检查 `patch_embed.py`
   - 时序建模问题 → 检查 `transformer.py`

3. **性能优化**:
   - GPU利用率 → 检查batch操作是否向量化
   - 内存占用 → 检查中间变量是否及时释放
   - 训练速度 → 检查是否有不必要的CPU-GPU传输

---

**Last Updated**: 2025-01-11
**Version**: 2.0 (精简版)
