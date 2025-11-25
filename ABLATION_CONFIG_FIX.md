# 消融实验配置文件修复说明

## 🐛 问题诊断

### 错误信息
```python
File "/root/miniconda3/lib/python3.12/site-packages/torch/optim/lr_scheduler.py", line 1381
new_lr = max(old_lr * self.factor, self.min_lrs[i])
TypeError: '>' not supported between instances of 'str' and 'float'
```

### 根本原因
`min_lr: 1e-6` 在 YAML 文件中被解析为**字符串** `"1e-6"` 而不是浮点数 `0.000001`

**为什么?**
- YAML 解析器对科学记数法的处理不一致
- `1e-6` 可能被当作字符串 (取决于 YAML 库版本)
- PyTorch 的 `ReduceLROnPlateau` 需要 `min_lrs` 是浮点数列表

---

## ✅ 修复方案

### 方法 1: 使用十进制格式 (已采用)
```yaml
# ❌ 错误: 可能被解析为字符串
min_lr: 1e-6

# ✅ 正确: 明确的浮点数
min_lr: 0.000001
```

### 方法 2: 强制浮点数 (备选)
```yaml
# 使用 !! 强制类型
min_lr: !!float 1e-6
```

### 方法 3: 在代码中转换 (最安全)
```python
# main.py 中
config['min_lr'] = float(config.get('min_lr', 1e-6))
```

---

## 📝 所有消融配置文件已修复

### 修复内容总结

| 配置文件 | 主要修改 | 状态 |
|---------|---------|------|
| `full_model.yaml` | ✅ 对齐 METR-LA 参数 + 修复 min_lr | 完成 |
| `wo_temporal.yaml` | ✅ 对齐 METR-LA 参数 + 修复 min_lr | 完成 |
| `wo_spatial.yaml` | ✅ 对齐 METR-LA 参数 + 修复 min_lr | 完成 |
| `wo_stage2.yaml` | ✅ 对齐 METR-LA 参数 + 修复 min_lr | 完成 |
| `embedding_only.yaml` | ✅ 对齐 METR-LA 参数 + 修复 min_lr | 完成 |
| `wo_denoising.yaml` | ✅ 对齐 METR-LA 参数 + 修复 min_lr | 完成 |

### 关键对齐的参数

与 `METR-LA_alternating.yaml` 保持一致:

```yaml
# Dataset
dataset_name: "METR-LA"
num_nodes: 207
input_len: 12
output_len: 12

# Training
batch_size: 32  # ← 从 64 改为 32
lr: 0.001
weight_decay: 0.0005
lr_patience: 10
lr_decay_factor: 0.5
min_lr: 0.000001  # ← 修复!从 "1e-6" 改为数字

# Architecture
embed_dim: 64  # ← 从 96 改为 64
dropout: 0.1   # ← 从 0.05 改为 0.1
num_heads: 4
mlp_ratio: 4

# Depths
temporal_depth_1: 2
spatial_depth_1: 2
temporal_depth_2: 2
spatial_depth_2: 2

# Fusion
fusion_type: 'cross_attn'  # ← 从 'gated' 改为 'cross_attn'

# Gradient clipping
train:
    clip_grad_param:
        max_norm: 5.0  # ← 添加梯度裁剪配置
    null_val: 0.0

# Metrics
metrics:
    MAE: "masked_mae"
    RMSE: "masked_rmse"
    MAPE: "masked_mape"
```

### 唯一不同的参数 (消融开关)

每个配置文件**仅**在以下开关上有差异:

| 配置 | use_temporal | use_spatial | use_stage2 | use_denoising |
|------|-------------|-------------|-----------|---------------|
| full_model | ✅ True | ✅ True | ✅ True | ✅ True |
| wo_temporal | ❌ **False** | ✅ True | ✅ True | ✅ True |
| wo_spatial | ✅ True | ❌ **False** | ✅ True | ✅ True |
| wo_stage2 | ✅ True | ✅ True | ❌ **False** | ✅ True |
| embedding_only | ❌ **False** | ❌ **False** | ❌ **False** | ✅ True |
| wo_denoising | ✅ True | ✅ True | ✅ True | ❌ **False** |

---

## 🧪 验证步骤

### 1. 快速语法检查
```bash
# 使用 Python 验证 YAML 格式
python -c "
import yaml
files = [
    'parameters/ablation/full_model.yaml',
    'parameters/ablation/wo_temporal.yaml',
    'parameters/ablation/wo_spatial.yaml',
    'parameters/ablation/wo_stage2.yaml',
    'parameters/ablation/embedding_only.yaml',
    'parameters/ablation/wo_denoising.yaml'
]
for f in files:
    with open(f) as file:
        config = yaml.safe_load(file)
        print(f'✅ {f}: min_lr={config[\"min_lr\"]} (type: {type(config[\"min_lr\"]).__name__})')
"
```

**预期输出**:
```
✅ parameters/ablation/full_model.yaml: min_lr=1e-06 (type: float)
✅ parameters/ablation/wo_temporal.yaml: min_lr=1e-06 (type: float)
✅ parameters/ablation/wo_spatial.yaml: min_lr=1e-06 (type: float)
✅ parameters/ablation/wo_stage2.yaml: min_lr=1e-06 (type: float)
✅ parameters/ablation/embedding_only.yaml: min_lr=1e-06 (type: float)
✅ parameters/ablation/wo_denoising.yaml: min_lr=1e-06 (type: float)
```

### 2. 运行单个消融实验测试
```bash
# 测试 wo_spatial (之前报错的配置)
python main.py --cfg parameters/ablation/wo_spatial.yaml --epochs 5

# 如果成功,应该不再报错
```

### 3. 批量运行所有消融实验
```bash
# 完整运行
run_ablation.bat
```

---

## 📊 配置对比表

| 参数 | METR-LA_alternating.yaml | 之前的消融配置 | 修复后的消融配置 |
|------|-------------------------|--------------|----------------|
| dataset_name | METR-LA | PEMS03 | ✅ METR-LA |
| num_nodes | 207 | 358 | ✅ 207 |
| batch_size | 32 | 64 | ✅ 32 |
| embed_dim | 64 | 96 | ✅ 64 |
| dropout | 0.1 | 0.05 | ✅ 0.1 |
| spatial_depth_1 | 2 | 1 | ✅ 2 |
| spatial_depth_2 | 2 | 1 | ✅ 2 |
| fusion_type | cross_attn | gated | ✅ cross_attn |
| min_lr | 0.000001 | "1e-6" (字符串) | ✅ 0.000001 |
| clip_grad | 5.0 (in train.clip_grad_param) | 1.0 | ✅ 5.0 |

---

## 🎯 重要提醒

### 1. min_lr 格式问题
**永远使用十进制格式**,不要使用科学记数法:
```yaml
# ❌ 危险 (可能被解析为字符串)
min_lr: 1e-6
min_lr: 1.0e-6

# ✅ 安全 (明确的浮点数)
min_lr: 0.000001
min_lr: 0.0000001
```

### 2. 消融实验的关键原则
> **除了消融目标开关外,所有其他参数必须与 baseline 完全一致!**

否则无法确定性能差异是由于:
- 消融组件的缺失 (正确)
- 其他参数不同 (错误,混淆因素)

### 3. 检查清单
运行消融实验前,确认:
- [ ] 所有配置文件的 `dataset_name` 一致
- [ ] 所有配置文件的 `num_nodes` 一致
- [ ] 所有配置文件的 `batch_size` 一致
- [ ] 所有配置文件的 `lr`, `weight_decay` 等训练参数一致
- [ ] 所有配置文件的 `embed_dim`, `num_heads` 等架构参数一致
- [ ] **唯一不同**: 消融开关 (`use_temporal_encoder`, `use_spatial_encoder`, etc.)
- [ ] `min_lr` 使用十进制格式 (0.000001),不用科学记数法

---

## 🚀 现在可以运行了!

### 快速测试
```bash
# 测试 5 epochs 验证修复
python main.py --cfg parameters/ablation/wo_spatial.yaml --epochs 5
```

### 完整消融实验
```bash
# 运行所有 6 个消融实验 (100 epochs)
run_ablation.bat
```

### 预期结果
所有实验应该能够正常运行,不再报 `TypeError: '>' not supported` 错误!

---

## 📚 相关文档
- `ABLATION_STUDY_GUIDE.md` - 消融实验完整指南
- `METR-LA_alternating.yaml` - Baseline 配置参考
- `run_ablation.bat` - 批量运行脚本
- `analyze_ablation.py` - 结果分析脚本
