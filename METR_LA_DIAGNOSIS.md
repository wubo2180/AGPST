# 🔍 METR-LA 问题诊断报告

## 📊 问题描述

**现象**: METR-LA 数据集上 MAE 不下降
**对比**: PEMS03/04/07/08 与 baseline 差距仅 ~2

## 🎯 可能原因分析

### 1. **数据特性差异** ⭐⭐⭐⭐⭐

METR-LA 与 PEMS 系列数据集的显著区别:

| 特性 | METR-LA | PEMS 系列 |
|------|---------|----------|
| 区域 | 洛杉矶高速公路 | 加州湾区 |
| 节点数 | 207 | 170-883 |
| 数据源 | 环形检测器 (loop detectors) | PeMS 系统 |
| 噪声水平 | **较高** | 较低 |
| 缺失值 | **较多** | 较少 |
| 时间模式 | **复杂** (城市交通) | 较规律 (高速公路) |

### 2. **学习率问题** ⭐⭐⭐⭐

```yaml
# 当前配置
lr: 0.001

# 可能问题:
- 对 METR-LA 来说学习率可能过大,导致震荡
- 或过小,导致收敛慢
```

### 3. **Batch Size 不匹配** ⭐⭐⭐

```yaml
# 当前配置
batch_size: 32

# METR-LA 特点:
- 207 nodes (中等规模)
- 但数据噪声大,可能需要更小的 batch size 获得更好的梯度估计
```

### 4. **模型容量问题** ⭐⭐⭐

```yaml
# 当前配置
embed_dim: 96
temporal_depth_1: 2
spatial_depth_1: 2

# 可能问题:
- METR-LA 的复杂性可能需要更大的模型容量
- 或相反,过拟合需要减小容量
```

### 5. **去噪模块问题** ⭐⭐⭐⭐⭐

```yaml
use_denoising: True

# METR-LA 噪声特别多!
# 当前去噪模块可能不够强
```

## 🔧 诊断方案

### 方案 A: 快速诊断 (2-3小时) - **推荐优先**

运行诊断脚本,测试多种配置:

```bash
chmod +x diagnose_metr_la.sh
bash diagnose_metr_la.sh
```

**测试内容**:
1. ✅ 学习率: 0.0005 / 0.001 / 0.002
2. ✅ Batch size: 16 / 32 / 64
3. ✅ 使用 AGPST 原始架构对比
4. ✅ 每个配置只跑 10 epochs,快速看趋势

**预期结果**:
- 如果某个配置 MAE 明显下降 → 找到问题,用该配置完整训练
- 如果所有配置都不行 → 说明是架构问题,需要 Phase 2/3

### 方案 B: 针对性优化 (半天)

基于 METR-LA 特点的专门配置:

```yaml
# parameters/METR-LA_optimized.yaml

# 更小的学习率 + 更多 warmup
lr: 0.0005
warmup_epochs: 10

# 更小的 batch size (更精确的梯度)
batch_size: 16

# 更强的正则化
dropout: 0.15  # 增加 dropout
weight_decay: 0.0005  # 增加 weight decay

# 更强的去噪
use_denoising: True
denoise_type: 'attention'  # 使用注意力去噪,而非 conv

# 更深的时间编码器 (METR-LA 时间模式复杂)
temporal_depth_1: 3  # 2 → 3
temporal_depth_2: 3  # 2 → 3

# 更多训练 epochs
epochs: 150  # METR-LA 可能收敛慢
```

### 方案 C: Phase 2/3 深度优化 (1-2天)

如果快速诊断显示是架构问题,再考虑:

**Phase 2 优化** (3-4小时):
- ✅ 添加跳跃连接 (Skip Connections)
- ✅ 批处理优化
- ✅ 参数共享

**Phase 3 高级优化** (1天):
- ✅ 多次循环编码
- ✅ 交叉注意力融合
- ✅ 自适应融合权重

## 📈 决策流程图

```
开始
  ↓
运行快速诊断 (diagnose_metr_la.sh)
  ↓
查看 10 epochs 后的 MAE
  ↓
┌─────────────┬─────────────┐
│ MAE 下降?   │ MAE 不下降? │
│ (找到配置)  │ (架构问题)  │
└─────────────┴─────────────┘
      ↓                ↓
完整训练最佳配置    Phase 2/3 优化
  (100 epochs)      (深度改进)
      ↓                ↓
  ✅ 解决          需要更多时间
```

## 💡 我的建议

### **优先级 1: 快速诊断** ⭐⭐⭐⭐⭐

```bash
# 只需 2-3 小时
bash diagnose_metr_la.sh

# 然后查看结果
cat diagnosis/metr_la_*/test*.log | grep "Val MAE"
```

**理由**:
- ✅ 时间成本低 (2-3小时 vs 1-2天)
- ✅ 可以快速定位是配置问题还是架构问题
- ✅ PEMS 系列已经证明架构可行,METR-LA 很可能是参数问题

### **优先级 2: 如果诊断无效,再 Phase 2**

只有在快速诊断显示**所有配置**都不理想时,才考虑 Phase 2/3

## 🎯 具体行动方案

### 立即执行 (现在):

```bash
# 1. 赋予权限
chmod +x diagnose_metr_la.sh

# 2. 运行诊断 (自动测试 4 种配置)
bash diagnose_metr_la.sh

# 3. 2-3 小时后查看结果
# 诊断脚本会自动总结最佳配置
```

### 等待结果后 (2-3小时后):

**情况 A: 找到好配置 (MAE 下降)**
```bash
# 用最佳配置完整训练
python main.py --config diagnosis/metr_la_*/best_config.yaml \
  --device cuda --swanlab_mode online
```

**情况 B: 所有配置都不行 (MAE 仍不下降)**
```bash
# 开始 Phase 2 优化
# 实现跳跃连接、批处理优化等
```

## 📊 参考数据

### METR-LA 典型 Baseline 性能

| 模型 | MAE | RMSE |
|------|-----|------|
| DCRNN | 2.77 | 5.38 |
| GraphWaveNet | 2.69 | 5.15 |
| STGCN | 2.88 | 5.74 |
| AGCRN | 2.87 | 5.58 |

**你的目标**: MAE < 3.0 即算成功

### PEMS03 vs METR-LA 对比

| 指标 | PEMS03 | METR-LA |
|------|--------|---------|
| Baseline MAE | ~14-15 | ~2.7-2.9 |
| 你的结果 | ~16-17 (差距2) | ??? |
| 归一化方式 | Z-score | Z-score |
| 数据规模 | 26,208 samples | 34,272 samples |

## ✅ 总结

**建议**: 
1. **先快速诊断** (2-3小时) ← **从这里开始!**
2. 如果诊断有效 → 完整训练
3. 如果诊断无效 → Phase 2/3 优化

**理由**:
- PEMS 系列已证明架构可行
- METR-LA 很可能是超参数/数据特性问题
- 快速诊断成本低,回报高

---

**下一步行动**: 运行 `bash diagnose_metr_la.sh`,等 2-3 小时后查看结果! 🚀
