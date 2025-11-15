# 🎯 去噪模块快速使用指南

## ⚡ 快速开始

### 1️⃣ 启用去噪（推荐配置）

编辑 `parameters/PEMS03_v3.yaml`:

```yaml
# 卷积去噪（轻量级，推荐）
use_denoising: True
denoise_type: 'conv'
```

### 2️⃣ 运行测试

```bash
# 测试去噪模块
python test_denoising.py

# 运行训练
python main.py --config=parameters/PEMS03_v3.yaml --device=cuda --test_mode=1
```

### 3️⃣ 对比实验

| 配置 | use_denoising | denoise_type | 说明 |
|------|---------------|--------------|------|
| Baseline | `False` | - | 不使用去噪 |
| Conv | `True` | `'conv'` | 卷积去噪（快速） |
| Attention | `True` | `'attention'` | 注意力去噪（强大） |

---

## 📊 去噪类型对比

### 🔹 卷积去噪 (Conv)
✅ **优点**: 快速、轻量、适合实时
❌ **缺点**: 去噪能力中等

**适用**: 高频噪声、计算资源有限

### 🔹 注意力去噪 (Attention)
✅ **优点**: 强大、自适应、效果好
❌ **缺点**: 计算开销大

**适用**: 复杂噪声、追求最佳性能

---

## 🔧 常见问题

**Q: 如何知道是否需要去噪？**
- 可视化原始数据
- 对比训练曲线
- 验证集性能提升

**Q: 计算开销多大？**
- Conv: +5-10% 时间
- Attention: +15-25% 时间

**Q: 推荐哪种？**
- 默认使用 `conv`
- 效果不好时尝试 `attention`

---

## 📚 详细文档

查看 [DENOISING_MODULE.md](./DENOISING_MODULE.md) 获取完整技术细节。

---

**版本**: v1.0 | **更新**: 2025-11-14
