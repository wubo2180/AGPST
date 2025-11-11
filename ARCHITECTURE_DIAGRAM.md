# AGPST模型架构可视化

## 🏗️ 整体架构

```
┌──────────────────────────────────────────────────────────────────┐
│                         AGPST Model                              │
│                    (basicts/mask/model.py)                       │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌────────────────────────────────────────────┐
        │  输入数据                                   │
        │  • short_history:  (B, 12, 358, 1)        │
        │  • long_history:   (B, 864, 358, 1)       │
        └────────────────────────────────────────────┘
                              │
                              ▼
        ┌────────────────────────────────────────────┐
        │  Step 1: Patch Embedding                   │
        │  (basicts/mask/patch_embed.py)            │
        │  ────────────────────────────────          │
        │  Conv2d(patch_size=12)                     │
        │  (B, 864, 358, 1) → (B, 358, 72, 96)      │
        └────────────────────────────────────────────┘
                              │
                              ▼
        ┌────────────────────────────────────────────┐
        │  Step 2: Positional Encoding               │
        │  (basicts/mask/positional_encoding.py)    │
        │  ────────────────────────────────          │
        │  Add position info to patches              │
        │  (B, 358, 72, 96) → (B, 358, 72, 96)      │
        └────────────────────────────────────────────┘
                              │
                              ▼
        ┌────────────────────────────────────────────┐
        │  Step 3: Adaptive Graph Learning           │
        │  (basicts/mask/graph_learning.py)         │
        │  ────────────────────────────────          │
        │  ┌──────────────────────────────┐          │
        │  │ AdaptiveGraphLearner         │          │
        │  │  • Static Graph (358×358)    │          │
        │  │  • Dynamic Graph (358×358)   │          │
        │  │  • Multi-scale (Local+Global)│          │
        │  │  • Top-K Sparsification      │          │
        │  │  • InfoNCE Contrastive Loss  │          │
        │  └──────────────────────────────┘          │
        │  ┌──────────────────────────────┐          │
        │  │ DynamicGraphConv             │          │
        │  │  Graph convolution on patches│          │
        │  └──────────────────────────────┘          │
        │  (B, 358, 72, 96) → (B, 358, 72, 96)      │
        │  + Adjacency Matrix (B, 358, 358)         │
        │  + Contrastive Loss                        │
        └────────────────────────────────────────────┘
                              │
                              ▼
        ┌────────────────────────────────────────────┐
        │  Step 4: Transformer Encoding              │
        │  (basicts/mask/transformer.py)            │
        │  ────────────────────────────────          │
        │  4-layer Transformer encoder               │
        │  Temporal modeling across patches          │
        │  (B, 358, 72, 96) → (B, 358, 72, 96)      │
        └────────────────────────────────────────────┘
                              │
                              ▼
        ┌────────────────────────────────────────────┐
        │  Step 5: Patch Aggregation                 │
        │  ────────────────────────────────          │
        │  Mean pooling over patches                 │
        │  (B, 358, 72, 96) → (B, 358, 96)          │
        └────────────────────────────────────────────┘
                              │
                              ▼
        ┌────────────────────────────────────────────┐
        │  Step 6: GraphWaveNet Backend              │
        │  (basicts/graphwavenet)                   │
        │  ────────────────────────────────          │
        │  Final prediction layer                    │
        │  (B, 358, 96) → (B, 12, 358, 1)           │
        └────────────────────────────────────────────┘
                              │
                              ▼
        ┌────────────────────────────────────────────┐
        │  输出预测                                   │
        │  prediction: (B, 12, 358, 1)              │
        └────────────────────────────────────────────┘
```

---

## 📁 文件组织结构

```
basicts/mask/
│
├── __init__.py                 # 模块导出
│   └── exports: AGPSTModel, DynamicGraphConv, etc.
│
├── model.py                    # 🎯 主模型
│   │
│   ├── class AGPSTModel
│   │   ├── __init__()
│   │   │   ├── PatchEmbedding
│   │   │   ├── PositionalEncoding
│   │   │   ├── DynamicGraphConv
│   │   │   ├── TransformerLayers
│   │   │   └── GraphWaveNet
│   │   │
│   │   └── forward(history, long_history)
│   │       ├── patch_embed()
│   │       ├── pos_encode()
│   │       ├── graph_conv()
│   │       ├── transformer()
│   │       ├── aggregate()
│   │       └── backend()
│   │
│   └── alias: ForecastingWithAdaptiveGraph = AGPSTModel
│
├── graph_learning.py           # 📊 图学习
│   │
│   ├── class AdaptiveGraphLearner
│   │   ├── compute_static_graphs()
│   │   │   ├── local_graphs (近邻)
│   │   │   └── global_graphs (长距离)
│   │   │
│   │   ├── compute_dynamic_graphs(patches)
│   │   │   ├── temporal_attention()
│   │   │   ├── dynamic_encoder()
│   │   │   └── gnn_enhancement()
│   │   │
│   │   ├── apply_topk_sparsification()
│   │   │
│   │   ├── compute_contrastive_loss()
│   │   │   └── InfoNCE loss
│   │   │
│   │   └── forward(patches)
│   │       ├── static_graphs
│   │       ├── dynamic_graphs
│   │       ├── fusion
│   │       └── returns: adj_matrix, loss
│   │
│   └── class DynamicGraphConv
│       ├── graph_learner: AdaptiveGraphLearner
│       ├── weight: nn.Parameter
│       │
│       └── forward(patches)
│           ├── learn_graph()
│           ├── graph_conv()
│           └── returns: features, adj, loss
│
├── patch_embed.py              # 🔲 Patch嵌入
│   │
│   └── class PatchEmbedding
│       ├── input_embedding: Conv2d
│       ├── _init_weights()
│       │
│       └── forward(long_history)
│           ├── reshape to (B*N, C, L, 1)
│           ├── conv2d → (B*N, D, P, 1)
│           ├── reshape to (B, N, P, D)
│           └── returns: patches
│
├── transformer.py              # 🔄 Transformer
│   │
│   └── class TransformerLayers
│       ├── transformer_encoder: TransformerEncoder
│       │   ├── num_layers: 4
│       │   ├── num_heads: 4
│       │   └── mlp_ratio: 4
│       │
│       └── forward(src)
│           ├── scale by sqrt(d_model)
│           ├── reshape to (P, B*N, D)
│           ├── encode
│           ├── reshape to (B, N, P, D)
│           └── returns: encoded
│
└── positional_encoding.py      # 📍 位置编码
    │
    └── class PositionalEncoding
        ├── learnable position embeddings
        │
        └── forward(x)
            └── returns: x + pos_embed
```

---

## 🔄 数据流转换

```
输入阶段:
─────────
short_history:     [B, 12, 358, 1]    ┐
long_history:      [B, 864, 358, 1]   ┘ → 仅使用long_history

格式转换:
─────────
(B, 864, 358, 1) → transpose → (B, 358, 864, 1)

Patch Embedding:
────────────────
(B, 358, 864, 1) 
    → unsqueeze → (B, 358, 864, 1, 1)
    → reshape → (B*358, 1, 864, 1)
    → Conv2d(kernel=12, stride=12) → (B*358, 96, 72, 1)
    → squeeze & reshape → (B, 358, 72, 96)
    → transpose → (B, 358, 72, 96)
                  ├─ B: batch_size (16)
                  ├─ N: num_nodes (358)
                  ├─ P: num_patches (72 = 864/12)
                  └─ D: embed_dim (96)

Positional Encoding:
────────────────────
(B, 358, 72, 96) + pos_embed → (B, 358, 72, 96)

Graph Learning:
───────────────
(B, 358, 72, 96) → AdaptiveGraphLearner → (B, 358, 358) adjacency
                                         + contrastive_loss

Graph Convolution:
──────────────────
For each patch p in [0, 72):
    (B, 358, 96) @ weight → (B, 358, 96)
    (B, 358, 358) @ (B, 358, 96) → (B, 358, 96)
Stack → (B, 358, 72, 96)

Transformer:
────────────
(B, 358, 72, 96)
    → reshape → (B*358, 72, 96)
    → transpose → (72, B*358, 96)  # (seq_len, batch, dim)
    → TransformerEncoder(4 layers) → (72, B*358, 96)
    → transpose → (B*358, 72, 96)
    → reshape → (B, 358, 72, 96)

Aggregation:
────────────
(B, 358, 72, 96) → mean(dim=2) → (B, 358, 96)

Backend:
────────
(B, 358, 96) 
    → permute → (B, 96, 358, 1)
    → GraphWaveNet → (B, 358, 12, 1)
    → permute → (B, 12, 358, 1)

输出:
─────
prediction: [B, 12, 358, 1]
```

---

## 📊 参数统计

### 模型组件参数量
```
1. PatchEmbedding
   └── Conv2d(1, 96, kernel=(12,1))
       Parameters: 1 × 96 × 12 × 1 = 1,152

2. PositionalEncoding
   └── Learnable embeddings
       Parameters: ~7,000

3. AdaptiveGraphLearner
   ├── Static embeddings: 358×10×4×2 = ~28,640
   ├── Local embeddings: 358×5×2×2 = ~7,160
   ├── Global embeddings: 358×10×2×2 = ~14,320
   ├── Temporal attention: ~40,000
   ├── Dynamic encoder: ~20,000
   └── Fusion networks: ~10,000
       Subtotal: ~120,000

4. DynamicGraphConv
   └── Weight matrix: 96×96 = 9,216

5. TransformerLayers (4 layers)
   └── Each layer: ~150,000
       Subtotal: ~600,000

6. GraphWaveNet
   └── Backend prediction: ~500,000

──────────────────────────────
Total: ~1,270,000 parameters
```

---

## 🎯 关键特性

### 1. Multi-scale Graph Learning
```
Local Graph (2 heads)
  ├── Small receptive field
  ├── Captures nearby relationships
  └── Higher temperature (2×)

Global Graph (2 heads)
  ├── Large receptive field
  ├── Captures long-range dependencies
  └── Lower temperature (0.5×)

Fusion
  └── Adaptive attention-based weighting
```

### 2. Dynamic + Static Fusion
```
Static Graph
  ├── Pre-learned node embeddings
  ├── Captures fixed topology
  └── Shape: (H, N, N)

Dynamic Graph
  ├── Computed from current batch
  ├── Adapts to input patterns
  └── Shape: (B, H, N, N)

Fusion Weight
  ├── Learned from features
  └── α×static + (1-α)×dynamic
```

### 3. Contrastive Learning
```
InfoNCE Loss
  ├── Positive pairs: same node, different time
  ├── Negative pairs: different nodes
  ├── Temperature: 0.2
  └── Improves graph representation quality
```

---

## 🔧 配置参数映射

```yaml
PEMS03_direct_forecasting.yaml
──────────────────────────────
num_nodes: 358          → AGPSTModel(num_nodes=358)
dim: 10                 → AdaptiveGraphLearner(node_dim=10)
topK: 10                → AdaptiveGraphLearner(topk=10)
patch_size: 12          → PatchEmbedding(patch_size=12)
in_channel: 1           → PatchEmbedding(in_channel=1)
embed_dim: 96           → PatchEmbedding(embed_dim=96)
num_heads: 4            → TransformerLayers(num_heads=4)
graph_heads: 4          → AdaptiveGraphLearner(graph_heads=4)
mlp_ratio: 4            → TransformerLayers(mlp_ratio=4)
dropout: 0.1            → All modules
encoder_depth: 4        → TransformerLayers(nlayers=4)
contrastive_weight: 0.05 → Loss weighting
```

---

**Version**: 2.0  
**Last Updated**: 2025-01-11
