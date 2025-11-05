# 如何将新的自适应图方法集成到您的模型中

## 🎯 快速集成（3 步完成）

### 步骤 1: 修改 `model.py` 的 `__init__` 方法

在 `basicts/mask/model.py` 文件中，找到 `pretrain_model` 的 `__init__` 方法：

**原来的代码** (第 18-46 行):
```python
class pretrain_model(nn.Module):
    def __init__(self, num_nodes, dim, topK, adaptive, epochs, patch_size, 
                 in_channel, embed_dim, num_heads, mlp_ratio, dropout, mask_ratio, 
                 encoder_depth, decoder_depth, patch_sizes=None, mode="pre-train") -> None:
        super().__init__()
        # ...
        
        # 原来的方法: 简单的矩阵乘法
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, dim), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(dim, num_nodes), requires_grad=True)
        
        # ...
```

**新的代码** (推荐 - Multi-Head 方法):
```python
class pretrain_model(nn.Module):
    def __init__(self, num_nodes, dim, topK, adaptive, epochs, patch_size, 
                 in_channel, embed_dim, num_heads, mlp_ratio, dropout, mask_ratio, 
                 encoder_depth, decoder_depth, patch_sizes=None, mode="pre-train",
                 graph_type='multihead', graph_num_heads=4) -> None:  # 新增参数
        super().__init__()
        # ...
        
        # 导入新的自适应图模块
        from .adaptive_graph import AdaptiveGraphFactory
        
        # 创建自适应图
        self.adaptive_graph = AdaptiveGraphFactory.create(
            graph_type=graph_type,        # 'multihead', 'dynamic', 'hyperbolic', etc.
            num_nodes=num_nodes,
            embed_dim=dim,
            num_heads=graph_num_heads     # 仅 multihead 使用
        )
        
        # 如果需要兼容旧的 checkpoint，保留原来的参数
        # self.nodevec1 = nn.Parameter(torch.randn(num_nodes, dim), requires_grad=True)
        # self.nodevec2 = nn.Parameter(torch.randn(dim, num_nodes), requires_grad=True)
        
        # ...
```

---

### 步骤 2: 修改 `forward` 方法

在同一个文件中，找到 `forward` 方法（第 130-150 行）:

**原来的代码**:
```python
def forward(self, history_data: torch.Tensor, epoch):
    # 原来的方法
    adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
    
    values, indices = torch.topk(adp, self.topK)
    # ...
```

**新的代码**:
```python
def forward(self, history_data: torch.Tensor, epoch):
    # 新方法: 使用自适应图模块
    adp = self.adaptive_graph()  # 自动调用对应的图构建方法
    
    values, indices = torch.topk(adp, self.topK)
    # 其余代码完全不变
    # ...
```

**如果使用 Dynamic 方法** (需要输入特征):
```python
def forward(self, history_data: torch.Tensor, epoch):
    # Dynamic 方法需要传入输入特征
    if hasattr(self.adaptive_graph, 'dynamic_encoder'):  # 判断是否是 Dynamic
        adp = self.adaptive_graph(history_data)
    else:
        adp = self.adaptive_graph()
    
    values, indices = torch.topk(adp, self.topK)
    # ...
```

---

### 步骤 3: 更新配置文件

在 `parameters/PEMS03_multiscale.yaml` 中添加新的配置项:

```yaml
# ... 原有配置 ...

# 自适应图配置 (新增)
graph_type: 'multihead'      # 可选: simple, multihead, dynamic, hyperbolic, sparse
graph_num_heads: 4           # 仅 multihead 使用
graph_topk: 10               # 仅 sparse 使用
graph_feature_dim: 64        # 仅 dynamic 使用

# mask_args 中添加这些参数
mask_args:
  num_nodes: 358
  dim: 10
  topK: 6
  # ... 其他原有参数 ...
  
  # 新增
  graph_type: ${graph_type}
  graph_num_heads: ${graph_num_heads}
```

---

## 🔄 完整的修改示例

### 示例 1: 使用 Multi-Head 方法（推荐）

**文件**: `basicts/mask/model.py`

```python
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .adaptive_graph import MultiHeadAdaptiveGraph  # 新增导入

class pretrain_model(nn.Module):
    def __init__(self, num_nodes, dim, topK, adaptive, epochs, patch_size, 
                 in_channel, embed_dim, num_heads, mlp_ratio, dropout, mask_ratio, 
                 encoder_depth, decoder_depth, patch_sizes=None, mode="pre-train",
                 graph_type='multihead', graph_num_heads=4) -> None:
        super().__init__()
        assert topK < num_nodes
        
        # ... 其他初始化代码 ...
        
        # ============ 修改这里 ============
        # 旧方法:
        # self.nodevec1 = nn.Parameter(torch.randn(num_nodes, dim), requires_grad=True)
        # self.nodevec2 = nn.Parameter(torch.randn(dim, num_nodes), requires_grad=True)
        
        # 新方法: Multi-Head Adaptive Graph
        self.adaptive_graph = MultiHeadAdaptiveGraph(
            num_nodes=num_nodes,
            embed_dim=dim,
            num_heads=graph_num_heads
        )
        # ===============================
        
        # ... 其他初始化代码保持不变 ...
    
    def forward(self, history_data: torch.Tensor, epoch):
        # ============ 修改这里 ============
        # 旧方法:
        # adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        
        # 新方法:
        adp = self.adaptive_graph()
        # ===============================
        
        values, indices = torch.topk(adp, self.topK)
        
        # 其余代码完全不变
        K = self.topK
        B, L, N, C = history_data.shape
        history_data_khop = history_data.transpose(1, 2).reshape((B, N, -1))
        history_data_khop = history_data_khop[:, indices, :]
        history_data_khop = history_data_khop.reshape((B, N, K, L, C))
        history_data_khop = history_data_khop.permute(0, 3, 1, 2, 4)
        
        if self.mode == "pre-train":
            hidden_states_unmasked, unmasked_token_index, masked_token_index = self.encoding(history_data_khop, epoch, adp)
            reconstruction_full = self.decoding(hidden_states_unmasked, masked_token_index, adp)
            reconstruction_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_tokens(
                reconstruction_full, history_data.permute(0, 2, 3, 1), 
                unmasked_token_index, masked_token_index
            )
            return reconstruction_masked_tokens, label_masked_tokens
        else:
            hidden_states_full, _, _ = self.encoding(history_data_khop, epoch, adp, mask=False)
            return hidden_states_full
```

---

### 示例 2: 使用 Dynamic 方法（性能最佳）

```python
from .adaptive_graph import DynamicAdaptiveGraph  # 新增导入

class pretrain_model(nn.Module):
    def __init__(self, num_nodes, dim, topK, adaptive, epochs, patch_size, 
                 in_channel, embed_dim, num_heads, mlp_ratio, dropout, mask_ratio, 
                 encoder_depth, decoder_depth, patch_sizes=None, mode="pre-train",
                 graph_feature_dim=64) -> None:
        super().__init__()
        
        # Dynamic Adaptive Graph
        self.adaptive_graph = DynamicAdaptiveGraph(
            num_nodes=num_nodes,
            embed_dim=dim,
            feature_dim=graph_feature_dim  # 通常设为 in_channel 或 embed_dim
        )
    
    def forward(self, history_data: torch.Tensor, epoch):
        # Dynamic 方法需要传入输入特征
        adp = self.adaptive_graph(history_data)  # 传入 history_data
        
        values, indices = torch.topk(adp, self.topK)
        # ... 其余代码不变 ...
```

---

### 示例 3: 使用 Hyperbolic 方法（层次网络）

```python
from .adaptive_graph import HyperbolicAdaptiveGraph  # 新增导入

class pretrain_model(nn.Module):
    def __init__(self, num_nodes, dim, topK, adaptive, epochs, patch_size, 
                 in_channel, embed_dim, num_heads, mlp_ratio, dropout, mask_ratio, 
                 encoder_depth, decoder_depth, patch_sizes=None, mode="pre-train",
                 graph_curv=1.0) -> None:
        super().__init__()
        
        # Hyperbolic Adaptive Graph
        self.adaptive_graph = HyperbolicAdaptiveGraph(
            num_nodes=num_nodes,
            embed_dim=dim,
            curv=graph_curv  # 曲率参数
        )
    
    def forward(self, history_data: torch.Tensor, epoch):
        adp = self.adaptive_graph()
        
        values, indices = torch.topk(adp, self.topK)
        # ... 其余代码不变 ...
```

---

## 📋 更新 `main.py`

在 `main.py` 中，更新 `pretrain` 函数以传递新参数:

```python
def pretrain(config, args):
    print('### start pre-training ... ###')
    # ...
    
    model = pretrain_model(
        config['num_nodes'], 
        config['dim'], 
        config['topK'], 
        config['adaptive'], 
        config['pretrain_epochs'], 
        config['patch_size'], 
        config['in_channel'], 
        config['embed_dim'], 
        config['num_heads'], 
        config['mlp_ratio'], 
        config['dropout'], 
        config['mask_ratio'], 
        config['encoder_depth'], 
        config['decoder_depth'],
        
        # 新增参数
        graph_type=config.get('graph_type', 'multihead'),
        graph_num_heads=config.get('graph_num_heads', 4)
    )
    
    # ...
```

---

## ⚙️ 配置文件示例

### `parameters/PEMS03_multihead.yaml` (推荐)

```yaml
description: 'Multi-Head Adaptive Graph'
model_name: 'AGPST-MultiHead'
dataset_name: "PEMS03"

# ... 其他配置 ...

# 自适应图配置
graph_type: 'multihead'
graph_num_heads: 4

num_nodes: 358
dim: 10
topK: 6
# ...
```

### `parameters/PEMS03_dynamic.yaml` (高性能)

```yaml
description: 'Dynamic Adaptive Graph'
model_name: 'AGPST-Dynamic'
dataset_name: "PEMS03"

# 自适应图配置
graph_type: 'dynamic'
graph_feature_dim: 64

num_nodes: 358
dim: 10
# ...
```

### `parameters/PEMS03_hyperbolic.yaml` (层次网络)

```yaml
description: 'Hyperbolic Adaptive Graph'
model_name: 'AGPST-Hyperbolic'
dataset_name: "PEMS03"

# 自适应图配置
graph_type: 'hyperbolic'
graph_curv: 1.0

num_nodes: 358
dim: 10
# ...
```

---

## 🧪 测试不同方法

创建一个实验脚本 `test_adaptive_graphs.sh`:

```bash
#!/bin/bash

# 测试原始方法 (baseline)
echo "Testing Simple (baseline)..."
python main.py --config parameters/PEMS03_multiscale.yaml \
    --pretrain_epochs 10 --finetune_epochs 10

# 测试 Multi-Head
echo "Testing Multi-Head..."
python main.py --config parameters/PEMS03_multihead.yaml \
    --pretrain_epochs 10 --finetune_epochs 10

# 测试 Dynamic
echo "Testing Dynamic..."
python main.py --config parameters/PEMS03_dynamic.yaml \
    --pretrain_epochs 10 --finetune_epochs 10

# 测试 Hyperbolic
echo "Testing Hyperbolic..."
python main.py --config parameters/PEMS03_hyperbolic.yaml \
    --pretrain_epochs 10 --finetune_epochs 10

echo "All tests completed! Check SwanLab for results."
```

---

## 📊 在 SwanLab 中对比结果

运行不同方法后，在 SwanLab Dashboard 中:

1. **对比 MAE/RMSE/MAPE**
   - 查看不同方法的性能差异
   
2. **分析训练曲线**
   - 观察收敛速度
   - 检查稳定性

3. **可视化邻接矩阵**
   ```python
   # 在 main.py 中添加
   if epoch == 0:
       adp_vis = model.adaptive_graph().detach().cpu().numpy()
       swanlab.log({"adaptive_graph": swanlab.Image(adp_vis)})
   ```

---

## ✅ 检查清单

在集成新方法后，确保:

- [ ] 导入了正确的模块 (`from .adaptive_graph import ...`)
- [ ] 在 `__init__` 中创建了 `self.adaptive_graph`
- [ ] 在 `forward` 中替换了 `adp` 的计算
- [ ] 更新了配置文件
- [ ] 更新了 `main.py` 传递新参数
- [ ] 代码没有语法错误 (`python -m py_compile basicts/mask/model.py`)
- [ ] 运行一个小实验测试 (`--pretrain_epochs 1`)

---

## 🚀 开始实验！

```bash
# 1. 检查代码
python -m py_compile basicts/mask/model.py
python -m py_compile basicts/mask/adaptive_graph.py

# 2. 快速测试
python main.py --config parameters/PEMS03_multihead.yaml \
    --pretrain_epochs 1 --finetune_epochs 1

# 3. 完整训练
python main.py --config parameters/PEMS03_multihead.yaml \
    --pretrain_epochs 100 --finetune_epochs 100

# 4. 查看结果
swanlab watch
```

---

## 💡 故障排除

### 问题 1: 导入错误
```
ModuleNotFoundError: No module named 'adaptive_graph'
```
**解决**: 确保 `adaptive_graph.py` 在 `basicts/mask/` 目录下

### 问题 2: 形状不匹配
```
RuntimeError: size mismatch
```
**解决**: 检查 `embed_dim` 是否能被 `num_heads` 整除（仅 Multi-Head）

### 问题 3: 内存溢出 (Dynamic 方法)
```
CUDA out of memory
```
**解决**: 减小 batch size 或使用 `sparse` 方法

---

好了！现在您有了 **7 种先进的动态邻接矩阵构建方法**，可以根据您的需求选择使用。

**推荐顺序**:
1. 先试 **Multi-Head** (简单+有效)
2. 再试 **Hyperbolic** (适合交通网络)
3. 最后试 **Dynamic** (性能最佳但开销大)

祝实验顺利！🎉
