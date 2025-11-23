# Phase 3 显存优化指南

## 🔥 问题：CUDA Out of Memory

**错误信息**:
```
torch.OutOfMemoryError: CUDA out of memory. 
Tried to allocate 5.88 GiB. 
GPU 0 has a total capacity of 31.36 GiB of which 5.61 GiB is free.
Process has 24.93 GiB allocated by PyTorch.
```

**硬件**: RTX 5090 32GB  
**实际可用**: ~5.6 GB (已用 25 GB)

---

## 📊 Phase 3 显存占用分析

### 原始配置 (OOM ❌)
```yaml
batch_size: 32
embed_dim: 96
num_heads: 4
num_layers: 2
num_stages: 3
patch_sizes: [1, 2, 4]  # 3 scales
use_cross_stage_attention: True
```

**显存占用估算**:
- Batch data: 32 × 207 × 12 × 96 = ~7.6 MB (negligible)
- **Encoders**: 3 stages × 3 scales × 2 types × 2 layers = **36 编码器**
  - 每个编码器: ~700 MB
  - 总计: 36 × 700 MB = **~25 GB** ⬅️ 主要开销
- Cross-stage attention: ~2 GB
- Activations: ~3 GB

**总计**: ~30 GB ❌ 超过 32 GB

---

## ✅ 解决方案

### 策略 A: 轻量级配置 (推荐)

已创建: `METR-LA_alternating_phase3_lite.yaml`

```yaml
batch_size: 16        # ↓50% memory
embed_dim: 64         # ↓33% memory  
num_layers: 1         # ↓50% memory per encoder
num_stages: 2         # ↓33% stages
patch_sizes: [1, 2]   # ↓33% scales
use_cross_stage_attention: False  # ↓15% memory
```

**显存占用估算**:
- Encoders: 2 stages × 2 scales × 2 types × 1 layer = **8 编码器**
  - 每个编码器: ~400 MB (embed_dim 64)
  - 总计: 8 × 400 MB = **~3.2 GB**
- Activations (batch 16): ~1.5 GB
- 总计: **~5-6 GB** ✅ 大量余量

**运行**:
```bash
python main.py --cfg parameters/METR-LA_alternating_phase3_lite.yaml --epochs 2
```

---

### 策略 B: 极简配置 (如果 A 还 OOM)

```yaml
batch_size: 8         # Further reduce
embed_dim: 48         # Minimal dimension
num_layers: 1
num_stages: 2
patch_sizes: [1]      # Single scale (degrades to Phase 1)
use_cross_stage_attention: False
```

**显存**: ~2-3 GB ✅ 最保守

---

### 策略 C: 使用混合精度训练

```yaml
use_amp: True  # Automatic Mixed Precision
```

**优势**:
- FP16 训练，显存占用减少 **~40%**
- 训练速度提升 **~30%**

**风险**:
- 可能影响数值稳定性
- 需要仔细调整学习率

**修改 main.py**:
```python
# 确保启用 AMP
if config.get('use_amp', False):
    from torch.amp import autocast, GradScaler
    scaler = GradScaler('cuda')
```

---

### 策略 D: 梯度累积 (保持大 batch size 效果)

如果需要 batch_size=32 的效果，但显存不够：

```python
# 在 main.py 中实现梯度累积
accumulation_steps = 2  # 累积 2 步 = 等效 batch_size 32

for i, batch in enumerate(dataloader):
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**配置**:
```yaml
batch_size: 16
gradient_accumulation_steps: 2  # 等效 batch_size=32
```

---

## 🎯 渐进式扩展策略

### Step 1: 验证轻量级配置 ✅
```bash
python main.py --cfg parameters/METR-LA_alternating_phase3_lite.yaml --epochs 2
```

**观察**:
- 显存占用 < 10 GB? ✅ 继续
- 初始 MAE < 5.4? ✅ 有潜力

### Step 2: 逐步扩展

如果 Step 1 成功，**逐个**增加复杂度：

#### 2.1 增加阶段数
```yaml
num_stages: 2 → 3  # 预计 +2 GB
```

#### 2.2 增加尺度
```yaml
patch_sizes: [1, 2] → [1, 2, 4]  # 预计 +3 GB
```

#### 2.3 增加维度
```yaml
embed_dim: 64 → 96  # 预计 +4 GB
```

#### 2.4 启用注意力
```yaml
use_cross_stage_attention: True  # 预计 +2 GB
```

#### 2.5 增加批次
```yaml
batch_size: 16 → 24 → 32  # 预计 +1-2 GB per step
```

**重要**: **每次只改一个参数**，确认不 OOM 后再继续。

---

## 📈 显存-性能权衡

| 配置 | 显存 | 初始 MAE (估计) | 最终 MAE (估计) |
|------|------|----------------|----------------|
| **Phase 3 Full** | ~30 GB ❌ | < 5.0 | < 3.5 |
| **Phase 3 Lite** | ~6 GB ✅ | < 5.5 | < 4.0 |
| **Phase 3 Minimal** | ~3 GB ✅ | ~5.5 | ~4.5 |
| Phase 1 | ~4 GB ✅ | ~5.4 | ~4.5 |

**结论**: 
- Phase 3 Lite 是**最佳平衡点**
- 保留核心创新 (多阶段 + 多尺度)
- 显存可控 (~6 GB)
- 性能预期优于 Phase 1

---

## 🛠️ 显存监控

### 训练前检查
```bash
# 查看 GPU 显存
nvidia-smi
```

### 训练中监控
```python
# 在 main.py 中添加
import torch

def log_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

# 每个 epoch 后调用
log_gpu_memory()
```

### 清理显存
```python
# 在训练循环中定期清理
import gc
torch.cuda.empty_cache()
gc.collect()
```

---

## 🚨 紧急降级方案

如果 Phase 3 Lite 还是 OOM：

### 方案 1: 回退到 Phase 1
```bash
python main.py --cfg parameters/METR-LA_alternating.yaml
```
- 已验证可行 (MAE 5.4)
- 显存 ~4 GB
- 稳定可靠

### 方案 2: 使用单尺度 Phase 3
```yaml
num_stages: 2
patch_sizes: [1]  # 退化为简单的 2 阶段
embed_dim: 64
batch_size: 32
```
- 显存 ~3 GB
- 失去多尺度优势
- 但保留多阶段循环

---

## 💡 最佳实践

### ✅ 推荐
1. **从 Phase 3 Lite 开始**
2. **监控显存占用**
3. **渐进式扩展**
4. **每次改一个参数**
5. **记录每个配置的 MAE**

### ❌ 避免
1. ❌ 直接用 Full 配置 (会 OOM)
2. ❌ 同时改多个参数
3. ❌ 忽略显存监控
4. ❌ 盲目增加复杂度

---

## 📋 配置对照表

| 参数 | Full | Lite | Minimal |
|------|------|------|---------|
| batch_size | 32 | 16 | 8 |
| embed_dim | 96 | 64 | 48 |
| num_layers | 2 | 1 | 1 |
| num_stages | 3 | 2 | 2 |
| patch_sizes | [1,2,4] | [1,2] | [1] |
| cross_attn | True | False | False |
| **显存** | ~30GB | ~6GB | ~3GB |
| **状态** | ❌ OOM | ✅ OK | ✅ OK |

---

## 🎯 立即行动

### 现在就运行
```bash
python main.py --cfg parameters/METR-LA_alternating_phase3_lite.yaml --epochs 2
```

### 观察关键指标
1. **显存**: 应该 < 10 GB ✅
2. **初始 MAE**: 应该 < 5.5
3. **是否能训练**: 不 OOM

### 根据结果决定
- ✅ 如果成功: 完整训练 150 epochs
- ⚠️ 如果还 OOM: 用 Minimal 配置
- ❌ 如果初始 MAE > 5.5: 回退 Phase 1

---

## 总结

**Phase 3 Full 配置太重** (30 GB)，**必须优化**。

**推荐路径**:
1. 先用 **Phase 3 Lite** (6 GB) ✅
2. 如果成功，逐步扩展
3. 如果失败，回退 Phase 1

**现在就试试**: `python main.py --cfg parameters/METR-LA_alternating_phase3_lite.yaml --epochs 2`
