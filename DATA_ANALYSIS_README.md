# 📊 数据可视化工具快速参考

## 🚀 一键分析

```bash
# Windows
run_data_analysis.bat

# Linux/Mac
bash run_data_analysis.sh
```

## 📝 单独运行

### 1. 基础可视化
```bash
python visualize_raw_data.py
```
**输出**: 时间序列、分布、相关性图

### 2. 深度分析
```bash
python analyze_noise.py
```
**输出**: SNR、频谱、异常值、自相关分析

## 📊 输出文件

所有图表保存在 `figure/` 目录：

```
figure/
├── raw_data_time_series.png      # 时间序列
├── raw_data_distribution.png     # 数据分布
├── raw_data_correlation.png      # 相关性
└── noise_analysis_report.png     # 综合报告（4合1）
```

## 🎯 判断标准

| 指标 | 优秀 | 良好 | 中等 | 差 |
|------|------|------|------|-----|
| **SNR** | >25dB | 20-25 | 15-20 | <15 |
| **高频%** | <5% | 5-10% | 10-20% | >20% |
| **异常值** | <0.5% | 0.5-1% | 1-3% | >3% |

## 💡 推荐配置

根据分析结果：

| 数据质量 | 配置 |
|---------|------|
| **优秀** | `use_denoising: False` |
| **良好** | `denoise_type: 'conv'` (可选) |
| **中等** | `denoise_type: 'conv'` |
| **差** | `denoise_type: 'attention'` |

## 📚 详细文档

查看 [DATA_VISUALIZATION_GUIDE.md](./DATA_VISUALIZATION_GUIDE.md)

---

**快速决策**: 运行 → 查看报告 → 根据建议配置 → 开始训练
