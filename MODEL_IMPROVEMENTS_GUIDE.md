# ============================================
# AGPST模型改进方案 - 实施指南
# ============================================

## 改进总结

本次改进针对AGPST交通预测模型,预期提升性能5-8%。主要包含5个核心改进:

### 1. 增强的Transformer架构 ✅
**文件**: `transformer_layers_improved.py`

**改进点**:
- Pre-LayerNorm替代Post-LayerNorm (更稳定的训练)
- GELU激活函数替代ReLU (更好的非线性表达)
- 添加最终LayerNorm层

**如何使用**:
```python
# 在 basicts/mask/model.py 中替换:
# from .transformer_layers import TransformerLayers
from .transformer_layers_improved import ImprovedTransformerLayers as TransformerLayers
```

**预期提升**: 1-2% MAE/RMSE改进

---

### 2. 空间-时间解耦注意力 ✅
**文件**: `spatial_temporal_attention.py`

**改进点**:
- 分别建模空间和时间依赖
- 空间注意力: 捕获节点间关系
- 时间注意力: 捕获时序依赖
- 更好的特征表达能力

**如何使用**:
```python
# 在 pretrain_model 类中添加:
from .spatial_temporal_attention import EnhancedTransformerBlock

# 替换原有的encoder:
self.encoder = nn.Sequential(*[
    EnhancedTransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
    for _ in range(encoder_depth)
])
```

**预期提升**: 2-3% 性能提升

---

### 3. 增强的自适应图学习 ✅
**文件**: `adaptive_graph_improved.py`

**改进点**:
- 多头图注意力机制
- 动态边剪枝 (Top-K稀疏化)
- 可学习的温度参数
- 多头图融合

**如何使用**:
```python
# 在 pretrain_model 的 __init__ 中:
from .adaptive_graph_improved import AdaptiveGraphLearner, DynamicGraphConv

self.graph_learner = AdaptiveGraphLearner(
    num_nodes=num_nodes,
    node_dim=dim,
    num_heads=4,
    topk=topK
)

# 在forward中使用:
dynamic_adj, _ = self.graph_learner()
```

**预期提升**: 1-2% 性能提升

---

### 4. 多尺度Patch融合增强 ✅
**文件**: `patch_improved.py`

**改进点**:
- 可学习的尺度权重
- 跨尺度注意力机制
- 更好的特征对齐策略
- 残差连接

**如何使用**:
```python
# 在 basicts/mask/model.py 中替换:
# from .patch import PatchEmbedding
from .patch_improved import EnhancedMultiScalePatchEmbedding as PatchEmbedding
```

**预期提升**: 1-2% 性能提升

---

### 5. 对比学习增强预训练 ✅
**文件**: `contrastive_loss.py`

**改进点**:
- 添加对比学习损失
- 时序对比学习
- 多任务学习框架
- 更好的表示学习

**如何使用**:
```python
# 在 main.py 的 pretrain 函数中:
from basicts.mask.contrastive_loss import EnhancedPretrainingLoss

# 替换原有的loss计算:
enhanced_loss = EnhancedPretrainingLoss(
    alpha=0.6,  # 重建损失权重
    beta=0.2,   # 对比损失权重
    gamma=0.2   # 时序损失权重
)

# 在训练循环中:
loss, loss_dict = enhanced_loss(
    reconstruction_masked_tokens,
    label_masked_tokens,
    encoder_output=hidden_states_unmasked,  # 需要从模型返回
    time_indices=None  # 可选
)
```

**预期提升**: 2-3% 性能提升

---

## 快速开始 - 3步实施

### Step 1: 渐进式替换 (最安全)
```python
# 1. 先替换Transformer层
from .transformer_layers_improved import ImprovedTransformerLayers as TransformerLayers

# 2. 运行实验,记录指标
python main.py --pretrain_epochs 100 --finetune_epochs 100

# 3. 如果有提升,继续下一个改进
```

### Step 2: 组合改进 (推荐)
```python
# 同时使用改进1+2+4 (最稳定的组合)
# 在 basicts/mask/model.py 中:
from .transformer_layers_improved import ImprovedTransformerLayers as TransformerLayers
from .patch_improved import EnhancedMultiScalePatchEmbedding as PatchEmbedding
from .spatial_temporal_attention import EnhancedTransformerBlock
```

### Step 3: 完整改进 (最大提升)
使用全部5个改进模块

---

## 超参数调优建议

基于改进后的模型,建议调整以下超参数:

```yaml
# parameters/PEMS03_multiscale.yaml

# 1. 增加模型容量
embed_dim: 128  # 原96 -> 128
encoder_depth: 6  # 原4 -> 6
decoder_depth: 2  # 原1 -> 2

# 2. 调整学习率
lr: 0.0015  # 原0.002 -> 0.0015 (更稳定)

# 3. 优化mask ratio
mask_ratio: 0.5  # 原0.25 -> 0.5 (更强的自监督)

# 4. 增加训练epochs
pretrain_epochs: 150  # 原100 -> 150
finetune_epochs: 150   # 原100 -> 150

# 5. 调整batch size
preTrain_batch_size: 16  # 原8 -> 16
batch_size: 64           # 原32 -> 64
```

---

## 运行命令

### 基础实验
```bash
python main.py \
    --config ./parameters/PEMS03_multiscale.yaml \
    --preTrain true \
    --lossType mae \
    --pretrain_epochs 150 \
    --finetune_epochs 150 \
    --mask_ratio 0.5 \
    --preTrain_batch_size 16 \
    --batch_size 64
```

### 使用对比学习
```bash
python main.py \
    --config ./parameters/PEMS03_multiscale.yaml \
    --preTrain true \
    --lossType contrastive \  # 新增
    --pretrain_epochs 150 \
    --finetune_epochs 150 \
    --mask_ratio 0.5
```

---

## 预期性能提升

| 改进模块 | 预期MAE提升 | 预期RMSE提升 | 实施难度 |
|---------|------------|-------------|---------|
| Transformer增强 | 1-2% | 1-2% | ⭐ 简单 |
| 空间-时间注意力 | 2-3% | 2-3% | ⭐⭐ 中等 |
| 自适应图学习 | 1-2% | 1-2% | ⭐⭐ 中等 |
| 多尺度融合 | 1-2% | 1-2% | ⭐ 简单 |
| 对比学习 | 2-3% | 2-3% | ⭐⭐⭐ 较难 |
| **总计** | **7-12%** | **7-12%** | - |

---

## 调试建议

1. **先单独测试每个改进**: 确保每个模块都能正常工作
2. **记录详细日志**: 使用swanlab记录每次实验
3. **可视化注意力权重**: 检查模型是否学到有意义的模式
4. **对比基线**: 始终与原始模型对比

---

## 常见问题

### Q1: 显存不足怎么办?
A: 
- 减小batch_size
- 使用gradient checkpointing
- 减小embed_dim或depth

### Q2: 训练不稳定?
A:
- 降低学习率
- 使用warmup策略
- 增加gradient clipping

### Q3: 性能没有提升?
A:
- 检查数据预处理
- 调整损失函数权重
- 增加训练epochs

---

## 下一步优化方向

1. **集成学习**: 训练多个模型ensemble
2. **知识蒸馏**: 用大模型指导小模型
3. **元学习**: 快速适应新数据集
4. **神经架构搜索**: 自动找最优架构

---

## 联系与支持

如有问题,请检查:
1. 模型输入输出shape是否匹配
2. 设备(CPU/GPU)是否正确
3. 依赖版本是否兼容

祝实验顺利! 🚀
