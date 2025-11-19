# 🔧 AGPST 去噪模块使用指南

## 📋 概述

为了提高模型对噪声数据的鲁棒性，AGPST 模型新增了**可选的去噪模块**。该模块在数据进入主网络之前对输入进行预处理，减少噪声干扰。

---

## 🎯 去噪模块类型

### 1. 卷积去噪 (Conv Denoising) ✅ **推荐**

**原理**：
- 使用 1D 卷积在时间维度上进行平滑
- 通过残差连接：`clean_data = raw_data - noise`
- 轻量级，计算高效

**优点**：
- ✅ 计算开销小
- ✅ 适合实时应用
- ✅ 对周期性噪声效果好

**适用场景**：
- 数据包含高频噪声
- 计算资源有限
- 需要快速训练

**配置**：
```yaml
use_denoising: True
denoise_type: 'conv'
```

---

### 2. 注意力去噪 (Attention Denoising)

**原理**：
- 使用自注意力机制识别和过滤噪声
- 学习时间步之间的依赖关系
- 自适应地降权噪声较大的时间点

**优点**：
- ✅ 更强大的去噪能力
- ✅ 自适应学习噪声模式
- ✅ 适合复杂噪声

**缺点**：
- ❌ 计算开销较大
- ❌ 需要更多训练时间

**适用场景**：
- 噪声模式复杂
- 计算资源充足
- 追求最佳性能

**配置**：
```yaml
use_denoising: True
denoise_type: 'attention'
```

---

## 🚀 使用方法

### 配置文件设置

在 `parameters/PEMS03_v3.yaml` 中添加/修改以下参数：

```yaml
# --------------------- Model Architecture ---------------------
# Denoising module
use_denoising: True      # 启用/禁用去噪
denoise_type: 'conv'     # 去噪类型: 'conv' 或 'attention'
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_denoising` | bool | `True` | 是否启用去噪模块 |
| `denoise_type` | str | `'conv'` | 去噪类型：`'conv'` (卷积) 或 `'attention'` (注意力) |

---

## 📊 效果对比

### 实验建议

**对比实验**：
1. **Baseline**：`use_denoising: False`
2. **Conv Denoising**：`use_denoising: True, denoise_type: 'conv'`
3. **Attention Denoising**：`use_denoising: True, denoise_type: 'attention'`

**评估指标**：
- MAE, RMSE, MAPE（预测精度）
- 训练时间（效率）
- GPU 内存占用（资源消耗）

---

## 🔍 技术细节

### 卷积去噪架构

```python
Conv1d(in=1, out=16, kernel=3, padding=1)
  ↓
BatchNorm1d(16)
  ↓
ReLU
  ↓
Conv1d(in=16, out=1, kernel=3, padding=1)
  ↓
Tanh (输出噪声估计)
  ↓
clean_data = raw_data - noise (残差连接)
```

**数据流**：
- 输入: `(B, T, N, C)` → `(B*N, C, T)`
- 卷积处理: 时间维度平滑
- 输出: `(B*N, C, T)` → `(B, T, N, C)`

### 注意力去噪架构

```python
Input: (B, T, N, C)
  ↓
Reshape: (B*N, T, C)
  ↓
Q = Linear(C, H)
K = Linear(C, H)  
V = Linear(C, H)
  ↓
Attention = Softmax(QK^T / √H)
  ↓
Output = Attention @ V
  ↓
Linear(H, C)
  ↓
Reshape: (B, T, N, C)
```

**特点**：
- 在时间维度上应用自注意力
- 每个节点独立处理
- 学习时间步之间的关联

---

## 🛠️ 调试与优化

### 检查去噪效果

添加打印语句查看去噪前后的统计信息：

```python
# 在 model.py 的 forward 方法中
print(f"Before denoising - Mean: {history_data.mean():.4f}, Std: {history_data.std():.4f}")
print(f"After denoising - Mean: {history_data_clean.mean():.4f}, Std: {history_data_clean.std():.4f}")
```

### 性能优化建议

1. **数据集较小**：使用 `denoise_type: 'conv'`
2. **数据集较大**：可以尝试 `denoise_type: 'attention'`
3. **过拟合**：降低去噪模块的复杂度或禁用去噪
4. **欠拟合**：增强去噪能力或调整其他超参数

### 常见问题

**Q: 去噪模块会增加多少计算时间？**
- Conv: 约 5-10% 增加
- Attention: 约 15-25% 增加

**Q: 如何判断是否需要去噪？**
- 可视化原始数据，观察是否有明显高频噪声
- 对比有/无去噪的训练损失曲线
- 对比验证集性能

**Q: 可以同时使用多种去噪方式吗？**
- 目前不支持，需要选择一种
- 未来可以考虑级联多个去噪模块

---

## 📈 实验建议

### 推荐实验流程

1. **Baseline**: 不使用去噪，建立基准
   ```yaml
   use_denoising: False
   ```

2. **快速测试**: 使用卷积去噪
   ```yaml
   use_denoising: True
   denoise_type: 'conv'
   ```

3. **深度优化**: 如果效果不理想，尝试注意力去噪
   ```yaml
   use_denoising: True
   denoise_type: 'attention'
   ```

### 超参数调优

如果需要进一步调整去噪强度，可以修改 `basicts/mask/model.py`:

```python
# 卷积去噪 - 调整通道数
self.denoiser = nn.Sequential(
    nn.Conv1d(in_channel, 32, kernel_size=3, padding=1),  # 16 → 32 (更强去噪)
    ...
)

# 注意力去噪 - 调整隐藏维度
self.denoiser = DenoiseAttention(in_channel, embed_dim // 2, dropout)  # //4 → //2
```

---

## 📝 引用

如果去噪模块对您的研究有帮助，欢迎引用相关工作：

- **卷积去噪**: 基于信号处理中的平滑滤波思想
- **注意力去噪**: 受 Denoising Transformer 启发

---

## 🔗 相关资源

- [AGPST 模型架构](./ADAPTIVE_GRAPH_GUIDE.md)
- [快速开始指南](./ADAPTIVE_GRAPH_QUICKSTART.md)
- [主要 README](./readme)

---

**更新时间**: 2025-11-14
**版本**: v1.0
