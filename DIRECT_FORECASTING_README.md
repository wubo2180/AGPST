# AGPST Direct Forecasting 模式

## 概述

这是一个**简化版本**的AGPST模型，**跳过预训练阶段**，直接进行端到端的forecasting训练。该模式集成了自适应图学习（Adaptive Graph Learning）和对比学习（Contrastive Learning），提供了更简洁高效的训练流程。

## 主要特点

### ✅ 优势
1. **无需预训练**：省去预训练阶段，直接端到端训练
2. **自适应图学习**：使用PostPatchDynamicGraphConv进行动态图学习
3. **对比学习增强**：通过对比学习提升节点表示质量
4. **多尺度图学习**：局部图（近邻关系）+ 全局图（长距离关系）
5. **简化流程**：一步到位，减少训练复杂度

### 🎯 核心组件
- **PatchEmbedding**: 将长期时间序列转换为patch表示
- **PostPatchDynamicGraphConv**: 动态图学习模块
  - 时间注意力机制
  - 多头图学习（局部+全局）
  - 自适应图融合
  - 对比学习损失
- **Transformer Encoder**: 编码patch特征
- **GraphWaveNet Backend**: 最终预测模块

## 模型架构

```
长期历史数据 (B, 864, N, C)
    ↓
Patch Embedding (B, N, 72, D)
    ↓
Dynamic Graph Learning
    ├─ Static Graph (多尺度)
    ├─ Dynamic Graph (基于patch特征)
    ├─ Adaptive Fusion
    └─ Contrastive Loss
    ↓
Positional Encoding
    ↓
Transformer Encoder
    ↓
Node Features (B, N, D)
    ↓
GraphWaveNet + 短期历史 (B, 12, N, C)
    ↓
预测结果 (B, N, 12, 1)
```

## 快速开始

### 1. 配置文件
使用 `parameters/PEMS03_direct_forecasting.yaml`

关键参数：
```yaml
mode: direct_forecasting
finetune_epochs: 100
batch_size: 16
lr: 0.001
patch_size: 12
embed_dim: 96
encoder_depth: 4
graph_heads: 4
topK: 10
contrastive_weight: 0.05
```

### 2. 运行训练

**Windows**:
```bash
run_direct_forecasting.bat
```

**Linux/Mac**:
```bash
python main.py \
    --config ./parameters/PEMS03_direct_forecasting.yaml \
    --device cuda \
    --mode direct_forecasting \
    --swanlab_mode online
```

### 3. 命令行参数

```bash
--config: 配置文件路径
--device: 设备 (cuda/cpu)
--mode: 训练模式 (direct_forecasting)
--swanlab_mode: SwanLab模式 (online/disabled)
--test_mode: 测试模式 (0/1)
```

## 与原版AGPST对比

| 特性 | 原版AGPST | Direct Forecasting |
|------|-----------|-------------------|
| 预训练阶段 | ✅ 需要 | ❌ 不需要 |
| 训练复杂度 | 高（两阶段） | 低（单阶段） |
| 自适应图学习 | 在预训练中 | 直接集成 |
| 对比学习 | 在预训练中 | 直接集成 |
| 训练时间 | 长 | 短 |
| 适用场景 | 数据充足，追求极致性能 | 快速实验，端到端优化 |

## 配置参数说明

### 核心参数

**数据参数**
- `seq_len: 864`: 长期历史数据长度
- `dataset_input_len: 12`: 短期历史数据长度
- `dataset_output_len: 12`: 预测长度
- `num_nodes: 358`: 节点数量

**Patch Embedding**
- `patch_size: 12`: 每个patch的时间步数（864/12=72个patch）
- `in_channel: 1`: 输入通道数（通常是流量/速度）
- `embed_dim: 96`: Patch embedding维度

**Adaptive Graph**
- `dim: 10`: 节点嵌入维度
- `graph_heads: 4`: 多头图学习的头数
- `topK: 10`: Top-K稀疏化参数（保留每个节点的top10邻居）

**Transformer**
- `encoder_depth: 4`: Transformer编码器层数
- `num_heads: 4`: 注意力头数
- `mlp_ratio: 4`: MLP扩展比例
- `dropout: 0.1`: Dropout率

**训练**
- `finetune_epochs: 100`: 训练轮数
- `batch_size: 16`: 批次大小
- `lr: 0.001`: 学习率
- `contrastive_weight: 0.05`: 对比学习损失权重

## 损失函数

总损失 = 预测损失 + 对比学习损失

```python
total_loss = mae_loss + contrastive_weight * contrastive_loss
```

- **预测损失**: MAE (Mean Absolute Error)
- **对比学习损失**: InfoNCE Loss
  - 增强节点表示的区分性
  - 学习更好的图结构

## 性能优化

### 已实现的优化
1. ✅ **完全向量化**：所有主要计算使用批量矩阵运算
2. ✅ **GPU加速**：充分利用GPU并行计算能力
3. ✅ **梯度裁剪**：防止梯度爆炸（max_norm=5.0）
4. ✅ **学习率调度**：ReduceLROnPlateau自适应调整
5. ✅ **权重初始化**：Xavier uniform初始化

### GPU利用率优化
- DataLoader: `num_workers=8, pin_memory=True`
- 批量矩阵乘法: `torch.bmm`, `torch.matmul`
- 向量化Top-K稀疏化
- 无Python循环

## 监控与日志

### SwanLab实时监控
训练过程中会记录：
- `train/loss`: 训练损失
- `train/contrastive_loss`: 对比学习损失
- `train/lr`: 学习率
- `val/MAE`: 验证MAE
- `val/RMSE`: 验证RMSE
- `val/MAPE`: 验证MAPE
- `test/MAE`: 测试MAE
- `test/RMSE`: 测试RMSE
- `test/MAPE`: 测试MAPE

### 模型保存
最佳模型自动保存到：
```
checkpoints/PEMS03_DirectForecasting/best_model.pt
```

## 调参建议

### 学习率
- 起始值: `0.001` (推荐)
- 范围: `0.0005 - 0.01`
- 策略: ReduceLROnPlateau (patience=5)

### 图学习参数
- `topK`: 
  - 小图（<200节点）: 5-10
  - 中图（200-500节点）: 10-20
  - 大图（>500节点）: 20-30
  
- `graph_heads`: 
  - 推荐: 4-8
  - 更多头数 = 更丰富的图结构，但计算量增加

### 对比学习
- `contrastive_weight`: 
  - 推荐: 0.05-0.1
  - 太大会影响主任务
  - 太小则效果不明显

### Patch Size
- 必须能整除`seq_len`
- 推荐: 12, 24, 48
- 更小的patch = 更细粒度的时间建模

## 故障排除

### 问题1: 损失为NaN
**解决方案**:
- ✅ 已修复：添加了权重初始化
- ✅ 已修复：添加了梯度裁剪
- 检查学习率是否过大

### 问题2: GPU利用率低
**解决方案**:
- 增加batch_size
- 检查DataLoader的num_workers
- 确保使用GPU版本的PyTorch

### 问题3: 内存不足
**解决方案**:
- 减小batch_size
- 减小embed_dim
- 减小encoder_depth

## 下一步计划

### 可能的改进
1. [ ] 添加多数据集支持（PEMS04, PEMS07, PEMS08）
2. [ ] 实验不同的图学习策略
3. [ ] 尝试不同的Transformer架构
4. [ ] 添加早停机制
5. [ ] 超参数自动搜索

### 实验建议
1. 先用小epoch数（10-20）快速验证模型
2. 观察对比学习损失的影响
3. 对比有/无自适应图学习的性能
4. 不同topK值的消融实验

## 引用

如果使用此代码，请引用原始AGPST论文及相关工作。

## 联系方式

有问题或建议请提Issue或联系开发者。

---

**Happy Forecasting! 🚀**
