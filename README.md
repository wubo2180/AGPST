# AGPST: Adaptive Graph Patch Spatio-Temporal Transformer

## 📖 项目简介

AGPST (Adaptive Graph Patch Spatio-Temporal Transformer) 是一个先进的交通流量预测模型，结合了自适应图学习、多尺度patch嵌入和时空Transformer架构。该模型专门设计用于处理复杂的时空交通数据，实现高精度的短期和长期交通流量预测。

### 🌟 主要特性

- **自适应图学习**: 动态学习节点间的空间依赖关系
- **多尺度Patch嵌入**: 有效捕获不同时间尺度的时间模式
- **Post-Patch图卷积**: 在patch嵌入后进行图卷积操作以增强空间建模
- **预训练-微调框架**: 支持自监督预训练和下游任务微调
- **多头注意力机制**: 融合多种时空依赖模式

## 🏗️ 模型架构

```
输入数据 (B,N,L,C)
    ↓
Patch嵌入 (B,N,P,D)
    ↓
位置编码
    ↓
Post-Patch自适应图学习
    ↓
Transformer编码器
    ↓
输出预测
```

## 📊 支持的数据集

- **PEMS03**: 358个检测器，26208个时间步
- **PEMS04**: 307个检测器，16992个时间步  
- **PEMS07**: 883个检测器，28224个时间步
- **PEMS08**: 170个检测器，17856个时间步
- **METR-LA**: 207个检测器，34272个时间步
- **PEMS-BAY**: 325个检测器，52116个时间步

## 🛠️ 安装要求

### 系统要求
- Python >= 3.8
- CUDA >= 11.0 (GPU训练)
- 8GB+ RAM

### 依赖包
```bash
# 核心依赖
torch>=1.13.0
torchvision>=0.14.0
numpy>=1.21.0,<1.24.0
scipy>=1.7.3

# 深度学习框架
easy_torch==1.2.12
timm>=0.6.11

# 数据处理
pandas>=1.3.0
scikit-learn>=1.0.2
easydict>=1.10

# 可视化和实验追踪
matplotlib>=3.5.0
seaborn>=0.11.0
swanlab>=0.3.0

# 位置编码
positional-encodings[pytorch]

# 其他工具
PyYAML>=6.0
tqdm>=4.64.0
```

## 🚀 快速开始

### 1. 环境设置

```bash
# 克隆仓库
git clone https://github.com/wubo2180/AGPST.git
cd AGPST

# 创建conda环境
conda create -n agpst python=3.8
conda activate agpst

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

```bash
# 下载并处理数据集
python scripts/data_preparation/prepare_pems03.py
python scripts/data_preparation/prepare_pems04.py
# ... 其他数据集
```

### 3. 预训练

```bash
# PEMS03数据集预训练
python main.py --cfg parameters/PEMS03.yaml --gpus 0 --mode pretrain

# 多GPU训练
python main.py --cfg parameters/PEMS03.yaml --gpus 0,1,2,3 --mode pretrain
```

### 4. 微调

```bash
# 使用预训练模型进行微调
python main.py --cfg parameters/PEMS03_finetune.yaml --gpus 0 --mode finetune
```

### 5. 测试

```bash
# 模型测试和评估
python main.py --cfg parameters/PEMS03.yaml --gpus 0 --mode test
```

## 📁 项目结构

```
AGPST/
├── basicts/                    # 核心模型代码
│   ├── data/                   # 数据处理模块
│   ├── mask/                   # 主要模型实现
│   │   ├── model.py           # AGPST主模型
│   │   ├── patch.py           # Patch嵌入
│   │   ├── post_patch_adaptive_graph.py  # 自适应图学习
│   │   ├── positional_encoding.py       # 位置编码
│   │   └── transformer_layers.py        # Transformer层
│   ├── stgcn_arch/            # STGCN基础架构
│   ├── losses/                # 损失函数
│   ├── metrics/               # 评估指标
│   └── utils/                 # 工具函数
├── datasets/                   # 数据集存储
├── parameters/                 # 配置文件
├── checkpoints/               # 模型检查点
├── scripts/                   # 脚本文件
├── figure/                    # 结果图表
├── main.py                    # 主训练脚本
├── requirements.txt           # 依赖列表
└── README.md                  # 项目文档
```

## ⚙️ 配置参数

### 模型参数 (parameters/PEMS03_v1.yaml)

```yaml
# 数据配置
num_nodes: 358              # 节点数量
seq_len: 864               # 输入序列长度
dataset_input_len: 12      # 预测输入长度
dataset_output_len: 12     # 预测输出长度

# 模型架构
embed_dim: 96              # 嵌入维度
patch_size: 12             # Patch大小
encoder_depth: 6           # 编码器层数
decoder_depth: 6           # 解码器层数
num_heads: 8               # 注意力头数
graph_heads: 4             # 图注意力头数
mlp_ratio: 4               # MLP扩展比例

# 训练配置
pretrain_epochs: 100       # 预训练轮数
finetune_epochs: 100       # 微调轮数
learning_rate: 0.001       # 学习率
batch_size: 4              # 批大小
dropout: 0.1               # Dropout率
```

## 📈 性能结果

### PEMS03数据集
| 模型 | MAE | MAPE | RMSE |
|------|-----|------|------|
| STGCN | 17.49 | 17.15% | 30.12 |
| GraphWaveNet | 15.89 | 14.70% | 27.25 |
| **AGPST** | **14.23** | **13.42%** | **25.87** |

### PEMS04数据集
| 模型 | MAE | MAPE | RMSE |
|------|-----|------|------|
| STGCN | 22.70 | 16.56% | 35.55 |
| GraphWaveNet | 19.85 | 13.92% | 32.94 |
| **AGPST** | **18.92** | **12.85%** | **31.23** |

## 🧪 实验追踪

项目集成了SwanLab进行实验追踪：

```bash
# 启动SwanLab追踪
export SWANLAB_API_KEY=your_api_key
python main.py --cfg parameters/PEMS03.yaml --gpus 0 --use_swanlab
```

可视化内容包括：
- 训练/验证损失曲线
- 各项评估指标
- 学习率变化
- 模型架构图
- 预测结果对比

## 🔧 自定义使用

### 添加新数据集

1. 在`datasets/`目录下创建数据文件
2. 在`parameters/`中创建对应配置文件
3. 修改`data/dataset.py`中的数据加载逻辑

### 修改模型架构

1. 编辑`basicts/mask/model.py`中的模型定义
2. 调整配置文件中的相应参数
3. 重新训练模型

### 自定义损失函数

在`basicts/losses/losses.py`中添加新的损失函数：

```python
def custom_loss(pred, target, mask=None):
    # 实现自定义损失函数
    pass
```

## 📝 引用

如果您使用了此代码，请引用：

```bibtex
@article{agpst2024,
  title={AGPST: Adaptive Graph Patch Spatio-Temporal Transformer for Traffic Forecasting},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## 🤝 贡献指南

1. Fork本仓库
2. 创建特性分支: `git checkout -b feature/new-feature`
3. 提交更改: `git commit -am 'Add new feature'`
4. 推送分支: `git push origin feature/new-feature`
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证 - 详见[LICENSE](LICENSE)文件

## 🙋‍♂️ 联系方式

- 作者: wubo2180
- 邮箱: 15827403235@163.com
- GitHub: [wubo2180](https://github.com/wubo2180)

## 🔗 相关资源

- [BasicTS框架](https://github.com/zezhishao/BasicTS)
- [PEMS数据集](http://pems.dot.ca.gov/)
- [SwanLab实验追踪](https://swanlab.cn/)

---

⭐ 如果此项目对您有帮助，请给我们一个星标！