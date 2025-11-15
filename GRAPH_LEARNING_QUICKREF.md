# 🎯 图学习模块快速参考

## ⚡ 一键切换

### 配置文件 (parameters/PEMS03_v3.yaml)

```yaml
# 🔵 简单模式（快速）
use_advanced_graph: False
dim: 10
topK: 10

# 🟢 高级模式（强大）
use_advanced_graph: True
graph_heads: 4
dim: 10
topK: 10
```

---

## 📊 快速对比

| 特性 | Simple | Advanced |
|------|--------|----------|
| 参数量 | ~7K | ~50K |
| 速度 | 1.0x | 1.5x |
| 精度提升 | Baseline | +5-10% |
| 图类型 | 静态 | 动态+静态 |
| 多尺度 | ❌ | ✅ |
| 对比学习 | ❌ | ✅ |

---

## 🚀 测试命令

```bash
# 测试集成
python test_graph_integration.py

# 简单模式训练
python main.py --config=parameters/PEMS03_v3.yaml --test_mode=1

# 高级模式训练（先在配置中设置 use_advanced_graph: True）
python main.py --config=parameters/PEMS03_v3.yaml --device=cuda
```

---

## 🔧 推荐配置

### 初学者
```yaml
use_advanced_graph: False  # 快速上手
```

### 追求性能
```yaml
use_advanced_graph: True
graph_heads: 4
topK: 10
```

### 资源受限
```yaml
use_advanced_graph: True
graph_heads: 2  # 减少头数
topK: 5         # 减少邻居
```
ssh -p 20158 root@connect.bjb2.seetacloud.com
---

## 📖 详细文档

- [完整指南](./ADVANCED_GRAPH_LEARNING.md)
- [去噪模块](./DENOISING_MODULE.md)

---

**快速决策树**:

```
需要最佳性能？
├─ YES → use_advanced_graph: True
└─ NO → 计算资源充足？
    ├─ YES → use_advanced_graph: True, graph_heads: 4
    └─ NO  → use_advanced_graph: False
```
