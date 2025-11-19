# 🌐 时空噪声分析指南

## 📊 时间 vs 空间噪声分析

### 核心区别

| 维度 | 分析对象 | 噪声特征 | 检测方法 |
|------|---------|---------|---------|
| **时间维度** | 单个节点的时间序列 | 时间突变、周期性异常 | 时间序列分析 |
| **空间维度** | 同一时刻的节点分布 | 空间孤立点、区域异常 | 空间相关性分析 |

---

## 1️⃣ 时间维度噪声（当前实现）

### 分析内容
```python
# 数据形状: (T, N)
# 对每个节点 n ∈ [0, N-1]:
#   分析其时间序列: data[:, n]  # (T,)
```

### 检测目标
```
节点i的流量变化
    ↑
 80 |     *              ← 时间异常值（突然飙升）
    |    * *
 60 |   *   *
    | *       *
 40 |*         **
    |            **
 20 |              ***
    |─────────────────→ 时间
```

### 噪声类型
1. **突变噪声**: 流量突然大幅变化
2. **高频噪声**: 快速振荡
3. **趋势异常**: 与正常模式偏离
4. **周期异常**: 破坏正常的日/周周期

### 应用场景
- ✅ 检测传感器故障
- ✅ 发现特殊事件（事故、施工）
- ✅ 评估数据采集质量

---

## 2️⃣ 空间维度噪声（新增分析）

### 分析内容
```python
# 数据形状: (T, N)
# 对每个时刻 t ∈ [0, T-1]:
#   分析空间分布: data[t, :]  # (N,)
```

### 检测目标
```
某时刻的空间分布（地图视图）
    
    节点布局:
    1 - 2 - 3 - 4
    |   |   |   |
    5 - 6 - 7 - 8
    |   |   |   |
    9 -10 -11 -12
    
    流量值:
   50  52  48 120  ← 节点4是空间异常值（周围都是50左右，它是120）
   51  49  51  50
   48  50  52  49
```

### 空间噪声特征

#### 特征1: 空间孤立点
```
正常情况（空间连续）:
50 → 52 → 55 → 58  (平滑变化)

异常情况（空间孤立）:
50 → 52 → 120 → 58  (节点3突然很高)
```

#### 特征2: 区域异常
```
正常区域:
50  52  48  51
49  51  50  52

异常区域:
50  52  48  51
49   5   3  52  ← 中间区域异常低
```

#### 特征3: 空间不连续
```
理论上相邻节点应该流量相似
如果 |flow[i] - flow[neighbor(i)]| 很大 → 空间噪声
```

### 噪声类型

1. **空间孤立噪声**
   - 单个节点与周围节点差异大
   - 可能原因: 传感器故障、数据传输错误

2. **空间聚类噪声**
   - 某个区域整体异常
   - 可能原因: 区域性事件（施工、封路）

3. **空间不连续噪声**
   - 相邻节点流量差异过大
   - 违反空间平滑性假设

---

## 3️⃣ 空间噪声检测方法

### 方法1: 空间自相关（Moran's I）

```python
def compute_spatial_autocorrelation(data, adj_matrix):
    """
    计算空间自相关系数
    
    Args:
        data: (T, N) 时空数据
        adj_matrix: (N, N) 邻接矩阵
    
    Returns:
        moran_i: 空间自相关系数
        - 接近1: 正相关（相邻节点相似）✅ 正常
        - 接近0: 无相关 ⚠️ 可疑
        - 接近-1: 负相关（相邻节点相反）❌ 异常
    """
    T, N = data.shape
    moran_values = []
    
    for t in range(T):
        snapshot = data[t, :]  # (N,)
        mean_val = np.mean(snapshot)
        
        # 标准化
        deviations = snapshot - mean_val
        
        # 计算Moran's I
        numerator = 0
        denominator = np.sum(deviations ** 2)
        W = np.sum(adj_matrix)  # 总权重
        
        for i in range(N):
            for j in range(N):
                numerator += adj_matrix[i, j] * deviations[i] * deviations[j]
        
        moran_i = (N / W) * (numerator / denominator)
        moran_values.append(moran_i)
    
    return np.array(moran_values)
```

**解释**:
- **高Moran's I** (>0.5): 空间平滑，相邻节点相似 → 数据质量好
- **低Moran's I** (<0.2): 空间混乱，随机分布 → 可能有噪声

---

### 方法2: 空间梯度异常检测

```python
def detect_spatial_gradient_outliers(data, adj_matrix, threshold=3.0):
    """
    检测空间梯度异常
    
    原理: 相邻节点流量差异不应太大
    """
    T, N = data.shape
    spatial_outliers = np.zeros((T, N), dtype=bool)
    
    for t in range(T):
        snapshot = data[t, :]  # (N,)
        
        for i in range(N):
            # 找到节点i的邻居
            neighbors = np.where(adj_matrix[i, :] > 0)[0]
            
            if len(neighbors) == 0:
                continue
            
            # 计算与邻居的平均差异
            neighbor_vals = snapshot[neighbors]
            avg_neighbor = np.mean(neighbor_vals)
            
            # 计算Z-score
            std_neighbor = np.std(neighbor_vals)
            if std_neighbor > 0:
                z_score = abs(snapshot[i] - avg_neighbor) / std_neighbor
                
                if z_score > threshold:
                    spatial_outliers[t, i] = True
    
    return spatial_outliers
```

**示例**:
```
节点5及其邻居:
  2 (flow=50)
  |
4-5-6  (flow=48, ?, 52)
  |
  8 (flow=51)

如果 flow[5] = 120:
  邻居平均 = (50+48+52+51)/4 = 50.25
  差异 = |120 - 50.25| = 69.75
  标准差 = 1.7
  Z-score = 69.75/1.7 = 41 >> 3 ❌ 空间异常！
```

---

### 方法3: 局部离群因子（LOF）

```python
from sklearn.neighbors import LocalOutlierFactor

def detect_spatial_lof(data, n_neighbors=5):
    """
    使用LOF检测空间异常值
    
    原理: 比较节点密度与其邻居的密度
    """
    T, N = data.shape
    lof_scores = np.zeros((T, N))
    
    for t in range(T):
        snapshot = data[t, :].reshape(-1, 1)  # (N, 1)
        
        lof = LocalOutlierFactor(n_neighbors=n_neighbors)
        scores = lof.fit_predict(snapshot)
        
        # -1表示异常值
        lof_scores[t, :] = lof.negative_outlier_factor_
    
    # 阈值: 通常 < -1.5 认为是异常
    spatial_outliers = lof_scores < -1.5
    
    return spatial_outliers, lof_scores
```

---

## 4️⃣ 时空联合噪声分析

### 综合指标

```python
def comprehensive_noise_analysis(data, adj_matrix):
    """
    时空联合噪声分析
    
    Returns:
        metrics: 包含时间和空间噪声指标
    """
    T, N = data.shape
    
    # 1. 时间维度分析（原有）
    temporal_outliers = detect_temporal_outliers(data)  # (T, N)
    temporal_ratio = temporal_outliers.sum() / temporal_outliers.size * 100
    
    # 2. 空间维度分析（新增）
    spatial_outliers = detect_spatial_gradient_outliers(data, adj_matrix)  # (T, N)
    spatial_ratio = spatial_outliers.sum() / spatial_outliers.size * 100
    
    # 3. 时空交叉异常（同时是时间和空间异常）
    spatiotemporal_outliers = temporal_outliers & spatial_outliers
    st_ratio = spatiotemporal_outliers.sum() / spatiotemporal_outliers.size * 100
    
    # 4. 空间自相关
    moran_i = compute_spatial_autocorrelation(data, adj_matrix)
    avg_moran = np.mean(moran_i)
    
    metrics = {
        'temporal_outlier_ratio': temporal_ratio,
        'spatial_outlier_ratio': spatial_ratio,
        'spatiotemporal_outlier_ratio': st_ratio,
        'avg_spatial_autocorrelation': avg_moran,
        'temporal_outliers': temporal_outliers,
        'spatial_outliers': spatial_outliers,
        'spatiotemporal_outliers': spatiotemporal_outliers
    }
    
    return metrics
```

### 可视化对比

```python
def plot_temporal_vs_spatial_noise(metrics, dataset_name):
    """对比时间和空间噪声"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. 时间异常值热图
    ax1 = axes[0, 0]
    im1 = ax1.imshow(metrics['temporal_outliers'].T, 
                     aspect='auto', cmap='Reds', interpolation='nearest')
    ax1.set_title('时间维度异常值')
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('节点')
    plt.colorbar(im1, ax=ax1)
    
    # 2. 空间异常值热图
    ax2 = axes[0, 1]
    im2 = ax2.imshow(metrics['spatial_outliers'].T, 
                     aspect='auto', cmap='Blues', interpolation='nearest')
    ax2.set_title('空间维度异常值')
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('节点')
    plt.colorbar(im2, ax=ax2)
    
    # 3. 时空交叉异常值
    ax3 = axes[0, 2]
    im3 = ax3.imshow(metrics['spatiotemporal_outliers'].T, 
                     aspect='auto', cmap='Purples', interpolation='nearest')
    ax3.set_title('时空交叉异常值')
    ax3.set_xlabel('时间步')
    ax3.set_ylabel('节点')
    plt.colorbar(im3, ax=ax3)
    
    # 4. 异常值比例对比
    ax4 = axes[1, 0]
    categories = ['时间异常', '空间异常', '时空交叉']
    ratios = [
        metrics['temporal_outlier_ratio'],
        metrics['spatial_outlier_ratio'],
        metrics['spatiotemporal_outlier_ratio']
    ]
    bars = ax4.bar(categories, ratios, color=['red', 'blue', 'purple'], alpha=0.7)
    ax4.set_ylabel('异常值比例 (%)')
    ax4.set_title('异常值类型对比')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, ratio in zip(bars, ratios):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{ratio:.2f}%', ha='center', va='bottom')
    
    # 5. 每个节点的时间vs空间异常数
    ax5 = axes[1, 1]
    temporal_counts = metrics['temporal_outliers'].sum(axis=0)
    spatial_counts = metrics['spatial_outliers'].sum(axis=0)
    
    ax5.scatter(temporal_counts, spatial_counts, alpha=0.6, s=50)
    ax5.set_xlabel('时间异常值数量')
    ax5.set_ylabel('空间异常值数量')
    ax5.set_title('节点异常值分布')
    ax5.grid(True, alpha=0.3)
    
    # 添加对角线
    max_val = max(temporal_counts.max(), spatial_counts.max())
    ax5.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='相等线')
    ax5.legend()
    
    # 6. 空间自相关时间序列
    ax6 = axes[1, 2]
    # 这里需要实际的moran_i时间序列
    ax6.set_title('空间自相关指标')
    ax6.set_xlabel('时间步')
    ax6.set_ylabel("Moran's I")
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'figure/spatiotemporal_noise_{dataset_name}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
```

---

## 5️⃣ 实际应用建议

### 场景1: 纯时间噪声
```
时间异常: 5.2%
空间异常: 0.8%
时空交叉: 0.3%
Moran's I: 0.82
```
**解读**: 
- ✅ 空间结构良好（高Moran's I）
- ⚠️ 时间序列有噪声
- **建议**: 使用**时间去噪**（conv/attention）

### 场景2: 纯空间噪声
```
时间异常: 1.2%
空间异常: 6.5%
时空交叉: 0.5%
Moran's I: 0.35
```
**解读**:
- ❌ 空间结构混乱（低Moran's I）
- ✅ 时间序列稳定
- **建议**: 
  - 检查传感器位置是否正确
  - 使用**空间平滑**或**图卷积**
  - 考虑重新学习邻接矩阵

### 场景3: 时空耦合噪声
```
时间异常: 4.8%
空间异常: 5.1%
时空交叉: 3.2%  ← 高交叉比例
Moran's I: 0.45
```
**解读**:
- ❌ 时空都有问题
- ⚠️ 高交叉比例说明噪声在时空上传播
- **建议**: 
  - 使用**时空联合去噪**
  - 结合attention去噪 + 动态图学习
  - 考虑数据清洗

---

## 6️⃣ 模型设计启示

### 对AGPST的启示

基于时空噪声分析结果，优化模型设计：

```python
# 根据噪声类型选择模块
if temporal_noise_dominant:
    # 时间噪声为主 → 强化时间去噪
    use_denoising = True
    denoise_type = 'attention'  # 或 'conv'
    use_advanced_graph = False  # 静态图即可

elif spatial_noise_dominant:
    # 空间噪声为主 → 强化图学习
    use_denoising = False  # 或轻量级
    use_advanced_graph = True  # 动态学习邻接关系
    graph_heads = 4  # 多头学习不同空间模式

elif spatiotemporal_noise:
    # 时空耦合噪声 → 全力以赴
    use_denoising = True
    denoise_type = 'attention'
    use_advanced_graph = True
    graph_heads = 4
    # 可能还需要额外的时空联合去噪模块
```

### 新模块设计思路

```python
class SpatioTemporalDenoiser(nn.Module):
    """时空联合去噪模块"""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        
        # 时间去噪
        self.temporal_denoise = DenoiseAttention(d_model)
        
        # 空间去噪（基于图结构）
        self.spatial_denoise = DynamicGraphConv(d_model, num_heads)
        
        # 融合
        self.fusion = nn.Linear(d_model * 2, d_model)
    
    def forward(self, x, adj_matrix):
        """
        Args:
            x: (B, T, N, C)
            adj_matrix: (N, N) or (B, N, N)
        """
        B, T, N, C = x.shape
        
        # 时间去噪：对每个节点的时间序列
        x_temp = rearrange(x, 'b t n c -> (b n) t c')
        x_temp_denoised = self.temporal_denoise(x_temp)
        x_temp_denoised = rearrange(x_temp_denoised, '(b n) t c -> b t n c', b=B, n=N)
        
        # 空间去噪：对每个时刻的空间分布
        x_spat = rearrange(x, 'b t n c -> (b t) n c')
        x_spat_denoised = self.spatial_denoise(x_spat, adj_matrix)
        x_spat_denoised = rearrange(x_spat_denoised, '(b t) n c -> b t n c', b=B, t=T)
        
        # 融合时间和空间去噪结果
        x_fused = torch.cat([x_temp_denoised, x_spat_denoised], dim=-1)
        x_out = self.fusion(x_fused)
        
        return x_out + x  # 残差连接
```

---

## 7️⃣ 总结对比表

| 特性 | 时间维度噪声 | 空间维度噪声 | 时空联合 |
|------|-------------|-------------|---------|
| **检测对象** | 单节点时间序列 | 单时刻空间分布 | 时空矩阵 |
| **主要方法** | IQR, SNR, FFT | Moran's I, 空间梯度 | 交叉分析 |
| **常见原因** | 传感器故障、特殊事件 | 位置错误、区域事件 | 系统性问题 |
| **去噪策略** | 时间平滑、去噪模块 | 空间平滑、图卷积 | 时空联合去噪 |
| **模型组件** | Denoising Module | Graph Learning | 两者结合 |

---

## 🎯 快速决策流程

```
噪声分析
    │
    ├─ 时间异常 > 空间异常 × 2
    │   → 时间噪声主导
    │   → 使用时间去噪模块
    │
    ├─ 空间异常 > 时间异常 × 2
    │   → 空间噪声主导
    │   → 使用动态图学习
    │
    └─ 时间异常 ≈ 空间异常
        → 时空耦合噪声
        → 使用时空联合去噪
```

---

**下一步**: 实现空间维度噪声分析脚本 `analyze_spatial_noise.py`

