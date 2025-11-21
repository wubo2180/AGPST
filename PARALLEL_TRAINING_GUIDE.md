# 🚀 6x RTX 5090 并行训练指南

## 📋 概述

使用 6 张 RTX 5090 GPU 同时训练 6 个数据集，每张卡负责一个数据集。每个数据集在 SwanLab 中是独立的实验。

## 🎯 GPU 分配方案

| GPU ID | 数据集 | 节点数 | 配置文件 |
|--------|--------|--------|----------|
| GPU 0  | PEMS03 | 358    | `parameters/PEMS03_alternating.yaml` |
| GPU 1  | PEMS04 | 307    | `parameters/PEMS04_alternating.yaml` |
| GPU 2  | PEMS07 | 883    | `parameters/PEMS07_alternating.yaml` |
| GPU 3  | PEMS08 | 170    | `parameters/PEMS08_alternating.yaml` |
| GPU 4  | METR-LA | 207   | `parameters/METR-LA_alternating.yaml` |
| GPU 5  | PEMS-BAY | 325  | `parameters/PEMS-BAY_alternating.yaml` |

## 🚀 快速开始

### 1. 赋予执行权限
```bash
chmod +x train_all_parallel.sh
chmod +x monitor_training.sh
chmod +x stop_training.sh
```

### 2. 启动并行训练
```bash
# 前台运行（推荐用于测试）
bash train_all_parallel.sh

# 后台运行（推荐用于长时间训练）
nohup bash train_all_parallel.sh > parallel_train.log 2>&1 &
```

### 3. 实时监控训练
```bash
# 打开新终端，运行监控脚本
bash monitor_training.sh
```

### 4. 停止训练（如需要）
```bash
bash stop_training.sh
```

## 📊 SwanLab 实验管理

### 实验命名规则
每个数据集会创建独立的 SwanLab 实验：
- `PEMS03_AlternatingST_gated_lr0.001_bs32`
- `PEMS04_AlternatingST_gated_lr0.001_bs32`
- `PEMS07_AlternatingST_gated_lr0.001_bs16`
- `PEMS08_AlternatingST_gated_lr0.001_bs32`
- `METR-LA_AlternatingST_gated_lr0.001_bs32`
- `PEMS-BAY_AlternatingST_gated_lr0.001_bs32`

### 查看实验
```bash
# 登录 SwanLab 网页端
https://swanlab.cn

# 项目名称: AGPST-forecasting
# 可以看到 6 个并行运行的实验
```

## 📁 文件结构

训练会生成以下文件：

```
results/parallel_20250120_143022/
├── parallel_training.log        # 总日志
├── PEMS03_gpu0.log              # GPU 0 - PEMS03
├── PEMS04_gpu1.log              # GPU 1 - PEMS04
├── PEMS07_gpu2.log              # GPU 2 - PEMS07
├── PEMS08_gpu3.log              # GPU 3 - PEMS08
├── METR-LA_gpu4.log             # GPU 4 - METR-LA
└── PEMS-BAY_gpu5.log            # GPU 5 - PEMS-BAY

checkpoints/
├── PEMS03_AlternatingST/
│   └── best_model.pt
├── PEMS04_AlternatingST/
│   └── best_model.pt
├── PEMS07_AlternatingST/
│   └── best_model.pt
├── PEMS08_AlternatingST/
│   └── best_model.pt
├── METR-LA_AlternatingST/
│   └── best_model.pt
└── PEMS-BAY_AlternatingST/
    └── best_model.pt
```

## 🔍 监控命令

### GPU 使用情况
```bash
# 实时监控所有 GPU
watch -n 1 nvidia-smi

# 查看特定 GPU
nvidia-smi -i 0,1,2,3,4,5
```

### 查看训练日志
```bash
# 查看特定数据集日志
tail -f results/parallel_*/PEMS03_gpu0.log

# 查看所有日志
tail -f results/parallel_*/parallel_training.log
```

### 查看进程状态
```bash
# 查看所有训练进程
ps aux | grep "python main.py"

# 查看进程和 GPU 绑定
nvidia-smi pmon -i 0,1,2,3,4,5
```

## ⚡ 性能优化

### 当前配置
- **Workers**: 8 per GPU (总共 48 workers)
- **Batch Size**: 
  - PEMS07: 16 (最大图，883 nodes)
  - 其他: 32
- **Mixed Precision**: 默认关闭（可在 yaml 中设置 `use_amp: True`）

### 启用混合精度训练
编辑配置文件，设置：
```yaml
use_amp: True  # 可提速 30-50%
```

### 调整并发 workers
编辑 `train_all_parallel.sh`：
```bash
WORKERS="4"  # 减少 workers 如果内存不足
WORKERS="16" # 增加 workers 如果 I/O 瓶颈
```

## 🔧 常见问题

### Q1: 显存不足 (OOM)
```bash
# 解决方案 1: 减少 batch size
# 编辑对应的 yaml 文件
batch_size: 16  # 改为 16 或 8

# 解决方案 2: 使用混合精度
use_amp: True
```

### Q2: 某个数据集失败了
```bash
# 查看失败日志
cat results/parallel_*/PEMS03_gpu0.log | grep -i error

# 单独重新训练该数据集
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config parameters/PEMS03_alternating.yaml \
  --device cuda \
  --swanlab_mode online
```

### Q3: 如何暂停和恢复训练？
```bash
# 当前不支持 checkpoint 恢复
# 建议: 使用 screen 或 tmux 保持会话

# 使用 screen
screen -S training
bash train_all_parallel.sh
# Ctrl+A+D 离开
# screen -r training 恢复
```

### Q4: 修改自动关机时间
编辑 `train_all_parallel.sh`：
```bash
# 修改倒计时为 10 分钟
for i in {600..1}; do

# 修改为立即关机
for i in {0..1}; do
```

## 📈 预期性能

### 训练时间（估算）
- **单个数据集**: 约 2-4 小时 (100 epochs, RTX 5090)
- **并行训练总时间**: 约 2-4 小时（所有 6 个数据集同时完成）
- **顺序训练总时间**: 约 12-24 小时（一个接一个）

**速度提升**: 并行训练比顺序训练快 **6 倍** ⚡

### 资源使用
- **总显存**: 约 6-8 GB per GPU (取决于数据集大小)
- **系统内存**: 约 64-96 GB (48 workers × 2GB)
- **磁盘 I/O**: 高（建议使用 SSD）

## ✅ 完整工作流程

```bash
# 1. 检查 GPU 状态
nvidia-smi

# 2. 启动并行训练（后台）
nohup bash train_all_parallel.sh > train.log 2>&1 &

# 3. 新开终端，启动监控
bash monitor_training.sh

# 4. 查看 SwanLab 实时指标
# 打开浏览器: https://swanlab.cn

# 5. 等待训练完成（自动关机）
# 或手动停止: bash stop_training.sh

# 6. 查看结果
cat results/parallel_*/parallel_training.log
```

## 🎯 AutoDL 特定配置

### 自动关机
- 所有 6 个数据集成功完成 → 5 分钟倒计时后自动关机
- 任何数据集失败 → 不会自动关机（方便检查错误）

### 取消自动关机
在倒计时期间按 `Ctrl+C`

### 手动关机
```bash
/usr/bin/shutdown
```

## 📞 联系和支持

如有问题，检查：
1. 日志文件: `results/parallel_*/parallel_training.log`
2. 个别数据集日志: `results/parallel_*/PEMS03_gpu0.log`
3. GPU 状态: `nvidia-smi`
4. 进程状态: `ps aux | grep python`

---

**祝训练顺利！6x RTX 5090 火力全开！** 🚀🔥
