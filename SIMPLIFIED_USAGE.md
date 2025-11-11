# AGPST 简化版使用指南

## 📌 主要变化

### ✅ 简化后的架构
- **移除**: 预训练（pretrain）和微调（finetune）两阶段训练
- **新增**: 端到端训练（train），自适应图学习直接集成在forecasting中
- **模式**: 只有 train / val / test 三个阶段

### 🔧 代码清理
- 移除所有调试打印语句
- 移除 pretrain_model 和 finetune_model 类
- 移除 pretrain()、finetune()、preTrain_test() 函数
- 简化为单一 train() 函数

---

## 🚀 快速开始

### 1. 直接运行（推荐）

```bash
# Windows
run_direct_forecasting.bat

# Linux/Mac
bash run_experiments_swanlab.sh
```

### 2. 命令行运行

```bash
python main.py \
    --config parameters/PEMS03_direct_forecasting.yaml \
    --mode train \
    --device cuda \
    --swanlab_mode online
```

### 3. 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | `parameters/PEMS03_direct_forecasting.yaml` | 配置文件路径 |
| `--mode` | `train` | 训练模式（仅支持 train） |
| `--device` | `cuda` | 设备（cuda/cpu） |
| `--swanlab_mode` | `disabled` | SwanLab模式（online/disabled） |
| `--test_mode` | `0` | 测试模式（1=只处理一个batch） |

---

## 📂 核心文件

### 模型文件
```
basicts/mask/
├── forecasting_with_adaptive_graph.py  # 主模型（端到端训练）
├── post_patch_adaptive_graph.py        # 自适应图学习模块
├── patch.py                             # Patch embedding
└── transformer_layers.py                # Transformer编码器
```

### 配置文件
```
parameters/
└── PEMS03_direct_forecasting.yaml      # 直接forecasting配置
```

### 训练入口
```
main.py                                  # 主程序（已简化）
run_direct_forecasting.bat              # Windows启动脚本
```

---

## 🎯 模型架构

```
输入数据 (B, T, N, C)
    ├─ short_data:  (B, 12, 358, 1)   # 短期历史
    └─ long_data:   (B, 864, 358, 1)  # 长期历史
           ↓
    [1] PatchEmbedding
        long_data → patches (B, N, 72, 96)
           ↓
    [2] PostPatchDynamicGraphConv
        ├─ Multi-scale adaptive graph learning
        ├─ Local graph (node-level)
        ├─ Global graph (patch-level)
        ├─ Adaptive fusion
        └─ InfoNCE contrastive loss
           ↓
    [3] Transformer Encoder (4 layers)
        Temporal modeling
           ↓
    [4] GraphWaveNet Backend
        Final prediction
           ↓
输出预测 (B, 12, 358, 1)
```

---

## ⚙️ 关键参数

### 数据格式
```yaml
dataset_input_len: 12      # 短期历史长度
dataset_output_len: 12     # 预测长度
seq_len: 864               # 长期历史长度
num_nodes: 358             # 节点数
```

### 训练参数
```yaml
epochs: 100                # 训练轮数
batch_size: 16             # 批次大小
lr: 0.001                  # 学习率
```

### 模型参数
```yaml
patch_size: 12             # 864/12 = 72个patches
embed_dim: 96              # Patch embedding维度
encoder_depth: 4           # Transformer层数
topK: 10                   # 图稀疏化Top-K
graph_heads: 4             # 图学习多头数量
contrastive_weight: 0.05   # 对比学习权重
```

---

## 📊 训练流程

### 1. 数据加载
```python
train_dataset = ForecastingDataset(...)
val_dataset = ForecastingDataset(...)
test_dataset = ForecastingDataset(...)
```

### 2. 模型训练
```python
for epoch in range(epochs):
    # 训练阶段
    model.train()
    for batch in train_loader:
        preds = model(short_data, long_data, ...)
        loss = MAE(preds, labels)
        loss += contrastive_weight * contrastive_loss
        loss.backward()
        optimizer.step()
    
    # 验证阶段
    val_loss = validate(val_loader, model, ...)
    
    # 测试阶段
    test(test_loader, model, ...)
```

### 3. 保存最佳模型
```python
if val_loss < best_val_loss:
    torch.save(model.state_dict(), "best_model.pt")
```

---

## 📈 监控指标

### SwanLab 记录
```python
swanlab.log({
    "train/loss": train_loss,
    "train/contrastive_loss": contrastive_loss,
    "train/lr": learning_rate,
    "val/MAE": val_mae,
    "val/RMSE": val_rmse,
    "val/MAPE": val_mape,
    "test/MAE": test_mae,
    "test/RMSE": test_rmse,
    "test/MAPE": test_mape
})
```

### 控制台输出
```
============ Epoch 0/100 ============
📊 Data Shape Verification:
  history_data (short-term):     torch.Size([16, 12, 358, 1])
  long_history_data (long-term): torch.Size([16, 864, 358, 1])
  future_data (labels):          torch.Size([16, 12, 358, 1])
  predictions (model output):    torch.Size([16, 12, 358, 1])
============================================
Epoch 0 - Train Loss: 3.5421, Contrastive Loss: 0.1234
============ Validation ============
Val MAE: 2.8765, Val RMSE: 4.3210, Val MAPE: 0.1234
✅ Best model saved with val loss: 2.8765
============ Test ============
Test MAE: 2.9123, Test RMSE: 4.3876, Test MAPE: 0.1289
```

---

## 🔍 调试模式

### 测试模式（只处理一个batch）
```bash
python main.py \
    --config parameters/PEMS03_direct_forecasting.yaml \
    --mode train \
    --test_mode 1
```

### 数据格式验证
- 第一个epoch的第一个batch会自动打印所有数据形状
- 确保所有tensor都是 (B, T, N, C) 格式

---

## ❓ 常见问题

### Q1: 如何修改数据集？
A: 修改配置文件中的路径：
```yaml
dataset_dir: 'datasets/PEMS04/data_in12_out12.pkl'
dataset_index_dir: 'datasets/PEMS04/index_in12_out12.pkl'
scaler_dir: 'datasets/PEMS04/scaler_in12_out12.pkl'
adj_dir: "datasets/PEMS04/adj_mx.pkl"
num_nodes: 307  # PEMS04有307个节点
```

### Q2: 如何调整patch大小？
A: 修改 `patch_size` 参数：
```yaml
patch_size: 24  # 864/24 = 36个patches
```
注意：864 必须能被 patch_size 整除

### Q3: 如何关闭SwanLab记录？
A: 使用 `--swanlab_mode disabled`：
```bash
python main.py --swanlab_mode disabled
```

### Q4: 如何调整对比学习权重？
A: 修改配置文件：
```yaml
contrastive_weight: 0.1  # 范围：0.01-0.5
```

### Q5: 训练速度慢怎么办？
A: 
1. 减小batch_size
2. 减少encoder_depth
3. 减小embed_dim
4. 使用更少的num_heads

---

## 📝 版本历史

### v2.0 (当前版本)
- ✅ 移除预训练机制
- ✅ 端到端训练
- ✅ 简化代码结构
- ✅ 移除所有调试语句
- ✅ 统一数据格式为 (B, T, N, C)

### v1.0 (旧版本)
- ❌ 预训练 + 微调两阶段
- ❌ 复杂的模型切换逻辑
- ❌ 大量调试输出

---

## 📚 相关文档

- [完整README](DIRECT_FORECASTING_README.md)
- [数据格式指南](DATA_FORMAT_GUIDE.md)
- [自适应图指南](ADAPTIVE_GRAPH_GUIDE.md)
- [快速开始](ADAPTIVE_GRAPH_QUICKSTART.md)

---

## 💡 最佳实践

1. **首次运行**: 使用默认配置测试
2. **数据验证**: 检查第一个epoch的数据形状输出
3. **性能调优**: 根据验证集调整学习率和权重
4. **模型保存**: 启用 `save_model: True` 保存最佳模型
5. **监控训练**: 使用 SwanLab 在线监控实验

---

**Last Updated**: 2024
**Author**: AGPST Team
