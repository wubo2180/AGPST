"""
时空联合噪声分析 - 同时分析时间和空间维度的噪声
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from scipy.spatial.distance import pdist, squareform
import os
import pickle
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_dataset(dataset_name, mode='train'):
    """加载数据集"""
    data_path = f'datasets/{dataset_name}/{mode}_data.npy'
    
    if not os.path.exists(data_path):
        print(f"  ⚠️  数据文件不存在: {data_path}")
        return None
    
    data = np.load(data_path)
    print(f"  ✅ 加载数据: {data_path}")
    print(f"     形状: {data.shape} (T={data.shape[0]}, N={data.shape[1]})")
    
    return data


def load_adjacency_matrix(dataset_name):
    """加载邻接矩阵"""
    adj_path = f'datasets/{dataset_name}/adj_mx.pkl'
    
    if not os.path.exists(adj_path):
        print(f"  ⚠️  邻接矩阵不存在: {adj_path}")
        print(f"  💡 将使用数据生成邻接矩阵")
        return None
    
    try:
        # 尝试使用latin1编码加载（兼容Python 2的pickle文件）
        with open(adj_path, 'rb') as f:
            pkl_data = pickle.load(f, encoding='latin1')
        
        # 处理不同的pickle文件格式
        if isinstance(pkl_data, np.ndarray):
            # 直接是邻接矩阵（PEMS数据集的格式）
            adj_mx = pkl_data
        elif isinstance(pkl_data, tuple):
            # 元组格式，尝试找到邻接矩阵
            # 找到第一个2D数组
            adj_mx = None
            for item in pkl_data:
                if isinstance(item, np.ndarray) and len(item.shape) == 2:
                    adj_mx = item
                    break
            
            if adj_mx is None:
                print(f"  ⚠️  未能从元组中提取邻接矩阵")
                return None
        else:
            print(f"  ⚠️  未知的pickle数据类型: {type(pkl_data)}")
            return None
        
        print(f"  ✅ 加载邻接矩阵: {adj_path}")
        print(f"     形状: {adj_mx.shape}, dtype: {adj_mx.dtype}")
        
        # 确保是二进制邻接矩阵（0或1）或归一化的权重矩阵
        # 如果是距离矩阵，转换为邻接矩阵
        if adj_mx.max() > 1.0:
            print(f"     💡 检测到距离矩阵，转换为二进制邻接矩阵")
            # 使用阈值或k-近邻转换
            threshold = np.percentile(adj_mx[adj_mx > 0], 25)  # 使用25%分位数作为阈值
            adj_mx = (adj_mx > 0) & (adj_mx <= threshold)
            adj_mx = adj_mx.astype(np.float32)
        
        return adj_mx
    
    except Exception as e:
        print(f"  ⚠️  加载邻接矩阵失败: {e}")
        print(f"  💡 将使用数据生成邻接矩阵")
        return None


def generate_adjacency_from_data(data, k_neighbors=5):
    """
    从数据生成邻接矩阵（基于流量相似度）
    
    Args:
        data: (T, N) 时空数据
        k_neighbors: 每个节点保留的最近邻数量
    """
    T, N = data.shape
    
    # 计算节点间的皮尔逊相关系数（时间序列相似度）
    corr_matrix = np.corrcoef(data.T)  # (N, N)
    
    # 转换为距离（1 - 相关系数）
    dist_matrix = 1 - np.abs(corr_matrix)
    
    # 构建k-近邻邻接矩阵
    adj_matrix = np.zeros((N, N))
    
    for i in range(N):
        # 找到k个最近邻（排除自己）
        neighbors = np.argsort(dist_matrix[i, :])[1:k_neighbors+1]
        adj_matrix[i, neighbors] = 1
        adj_matrix[neighbors, i] = 1  # 对称
    
    print(f"  💡 生成k-近邻邻接矩阵 (k={k_neighbors})")
    
    return adj_matrix


def detect_temporal_outliers(data, method='iqr'):
    """
    检测时间维度异常值（沿时间轴）
    
    对每个节点的时间序列进行异常检测
    """
    T, N = data.shape
    
    if method == 'iqr':
        # IQR方法（沿时间维度）
        q1 = np.percentile(data, 25, axis=0)  # (N,)
        q3 = np.percentile(data, 75, axis=0)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = (data < lower_bound) | (data > upper_bound)
    
    elif method == 'zscore':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        z_scores = np.abs((data - mean) / (std + 1e-8))
        
        outliers = z_scores > 3
    
    return outliers  # (T, N)


def detect_spatial_outliers(data, adj_matrix, threshold=3.0):
    """
    检测空间维度异常值（沿空间维度）
    
    对每个时刻的空间分布进行异常检测
    基于空间梯度：节点值与其邻居的差异
    """
    T, N = data.shape
    spatial_outliers = np.zeros((T, N), dtype=bool)
    
    for t in range(T):
        snapshot = data[t, :]  # (N,) 当前时刻的空间分布
        
        for i in range(N):
            # 找到节点i的邻居
            neighbors = np.where(adj_matrix[i, :] > 0)[0]
            
            if len(neighbors) == 0:
                continue
            
            # 计算与邻居的平均值和标准差
            neighbor_vals = snapshot[neighbors]
            avg_neighbor = np.mean(neighbor_vals)
            std_neighbor = np.std(neighbor_vals)
            
            # 如果邻居值变化很小，使用全局标准差
            if std_neighbor < 1e-6:
                std_neighbor = np.std(snapshot)
            
            # 计算Z-score
            if std_neighbor > 0:
                z_score = abs(snapshot[i] - avg_neighbor) / std_neighbor
                
                if z_score > threshold:
                    spatial_outliers[t, i] = True
    
    return spatial_outliers  # (T, N)


def compute_spatial_autocorrelation(data, adj_matrix):
    """
    计算Moran's I空间自相关系数
    
    衡量空间相似性：相邻节点的值是否相似
    
    Returns:
        moran_i: 每个时刻的Moran's I值 (T,)
    """
    T, N = data.shape
    moran_values = []
    
    # 预计算权重矩阵的总和
    W = np.sum(adj_matrix)
    
    if W == 0:
        print("  ⚠️  邻接矩阵全为0，无法计算Moran's I")
        return np.zeros(T)
    
    for t in range(T):
        snapshot = data[t, :]  # (N,)
        mean_val = np.mean(snapshot)
        
        # 标准化
        deviations = snapshot - mean_val
        
        # 计算Moran's I
        numerator = 0
        for i in range(N):
            for j in range(N):
                if adj_matrix[i, j] > 0:
                    numerator += adj_matrix[i, j] * deviations[i] * deviations[j]
        
        denominator = np.sum(deviations ** 2)
        
        if denominator > 0:
            moran_i = (N / W) * (numerator / denominator)
        else:
            moran_i = 0
        
        moran_values.append(moran_i)
    
    return np.array(moran_values)


def plot_spatiotemporal_analysis(data, adj_matrix, dataset_name, save_dir='figure'):
    """
    生成时空联合噪声分析报告
    """
    T, N = data.shape
    
    # 1. 检测时间异常
    print("  🔍 检测时间维度异常值...")
    temporal_outliers = detect_temporal_outliers(data, method='zscore')
    temporal_ratio = temporal_outliers.sum() / temporal_outliers.size * 100
    
    # 2. 检测空间异常
    print("  🔍 检测空间维度异常值...")
    spatial_outliers = detect_spatial_outliers(data, adj_matrix, threshold=3.0)
    spatial_ratio = spatial_outliers.sum() / spatial_outliers.size * 100
    
    # 3. 时空交叉异常
    spatiotemporal_outliers = temporal_outliers & spatial_outliers
    st_ratio = spatiotemporal_outliers.sum() / spatiotemporal_outliers.size * 100
    
    # 4. 空间自相关
    print("  🔍 计算空间自相关...")
    moran_i = compute_spatial_autocorrelation(data, adj_matrix)
    avg_moran = np.mean(moran_i)
    
    # 创建可视化
    fig = plt.figure(figsize=(18, 12))
    
    # 子图1: 时间异常值热图
    ax1 = plt.subplot(3, 3, 1)
    im1 = ax1.imshow(temporal_outliers.T, aspect='auto', cmap='Reds', 
                     interpolation='nearest', vmin=0, vmax=1)
    ax1.set_title(f'{dataset_name} - 时间维度异常值', fontsize=11, fontweight='bold')
    ax1.set_xlabel('时间步', fontsize=9)
    ax1.set_ylabel('节点索引', fontsize=9)
    plt.colorbar(im1, ax=ax1, label='异常(1)/正常(0)')
    
    ax1.text(0.02, 0.98, f'比例: {temporal_ratio:.2f}%', 
             transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # 子图2: 空间异常值热图
    ax2 = plt.subplot(3, 3, 2)
    im2 = ax2.imshow(spatial_outliers.T, aspect='auto', cmap='Blues', 
                     interpolation='nearest', vmin=0, vmax=1)
    ax2.set_title(f'{dataset_name} - 空间维度异常值', fontsize=11, fontweight='bold')
    ax2.set_xlabel('时间步', fontsize=9)
    ax2.set_ylabel('节点索引', fontsize=9)
    plt.colorbar(im2, ax=ax2, label='异常(1)/正常(0)')
    
    ax2.text(0.02, 0.98, f'比例: {spatial_ratio:.2f}%', 
             transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 子图3: 时空交叉异常值
    ax3 = plt.subplot(3, 3, 3)
    im3 = ax3.imshow(spatiotemporal_outliers.T, aspect='auto', cmap='Purples', 
                     interpolation='nearest', vmin=0, vmax=1)
    ax3.set_title(f'{dataset_name} - 时空交叉异常值', fontsize=11, fontweight='bold')
    ax3.set_xlabel('时间步', fontsize=9)
    ax3.set_ylabel('节点索引', fontsize=9)
    plt.colorbar(im3, ax=ax3, label='异常(1)/正常(0)')
    
    ax3.text(0.02, 0.98, f'比例: {st_ratio:.2f}%', 
             transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='plum', alpha=0.8))
    
    # 子图4: 异常值比例对比
    ax4 = plt.subplot(3, 3, 4)
    categories = ['时间异常', '空间异常', '时空交叉']
    ratios = [temporal_ratio, spatial_ratio, st_ratio]
    colors_bar = ['red', 'blue', 'purple']
    bars = ax4.bar(categories, ratios, color=colors_bar, alpha=0.7, edgecolor='black')
    
    for bar, ratio in zip(bars, ratios):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{ratio:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax4.set_ylabel('异常值比例 (%)', fontsize=10)
    ax4.set_title('异常值类型对比', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.tick_params(axis='x', rotation=15)
    
    # 子图5: 每个节点的时间vs空间异常数
    ax5 = plt.subplot(3, 3, 5)
    temporal_counts = temporal_outliers.sum(axis=0)  # (N,)
    spatial_counts = spatial_outliers.sum(axis=0)    # (N,)
    
    ax5.scatter(temporal_counts, spatial_counts, alpha=0.6, s=30, c='steelblue')
    ax5.set_xlabel('时间异常值数量', fontsize=10)
    ax5.set_ylabel('空间异常值数量', fontsize=10)
    ax5.set_title('各节点异常值分布', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 添加对角线
    max_val = max(temporal_counts.max(), spatial_counts.max()) if temporal_counts.max() > 0 else 1
    ax5.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, linewidth=1.5, label='相等线')
    ax5.legend(fontsize=8)
    
    # 子图6: 空间自相关时间序列
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(range(len(moran_i)), moran_i, linewidth=1.5, color='green', alpha=0.7)
    ax6.axhline(float(avg_moran), color='red', linestyle='--', linewidth=2, 
                label=f'平均值: {avg_moran:.3f}')
    ax6.axhline(0.5, color='orange', linestyle=':', linewidth=1.5, 
                alpha=0.5, label='中等相关阈值')
    ax6.set_xlabel('时间步', fontsize=10)
    ax6.set_ylabel("Moran's I", fontsize=10)
    ax6.set_title('空间自相关系数变化', fontsize=11, fontweight='bold')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    # 子图7: 时间异常值数量分布
    ax7 = plt.subplot(3, 3, 7)
    ax7.bar(range(N), temporal_counts, color='coral', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax7.set_xlabel('节点索引', fontsize=10)
    ax7.set_ylabel('时间异常值数量', fontsize=10)
    ax7.set_title('各节点时间异常值数量', fontsize=11, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 子图8: 空间异常值数量分布
    ax8 = plt.subplot(3, 3, 8)
    ax8.bar(range(N), spatial_counts, color='skyblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax8.set_xlabel('节点索引', fontsize=10)
    ax8.set_ylabel('空间异常值数量', fontsize=10)
    ax8.set_title('各节点空间异常值数量', fontsize=11, fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='y')
    
    # 子图9: 邻接矩阵可视化
    ax9 = plt.subplot(3, 3, 9)
    im9 = ax9.imshow(adj_matrix, cmap='binary', aspect='auto', interpolation='nearest')
    ax9.set_title('邻接矩阵结构', fontsize=11, fontweight='bold')
    ax9.set_xlabel('节点索引', fontsize=9)
    ax9.set_ylabel('节点索引', fontsize=9)
    plt.colorbar(im9, ax=ax9, label='连接权重')
    
    edge_count = (adj_matrix > 0).sum() / 2  # 除以2因为对称
    density = edge_count / (N * (N-1) / 2) * 100
    ax9.text(0.02, 0.98, f'边数: {int(edge_count)}\n密度: {density:.1f}%', 
             transform=ax9.transAxes, fontsize=8,
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    # 保存
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'spatiotemporal_noise_{dataset_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n  📊 时空分析报告已保存: {save_path}")
    plt.close()
    
    return {
        'dataset': dataset_name,
        'temporal_outlier_ratio': temporal_ratio,
        'spatial_outlier_ratio': spatial_ratio,
        'spatiotemporal_outlier_ratio': st_ratio,
        'avg_spatial_autocorrelation': avg_moran,
        'temporal_counts': temporal_counts,
        'spatial_counts': spatial_counts
    }


def generate_spatiotemporal_recommendation(metrics):
    """根据时空指标生成建议"""
    dataset = metrics['dataset']
    temp_ratio = metrics['temporal_outlier_ratio']
    spat_ratio = metrics['spatial_outlier_ratio']
    st_ratio = metrics['spatiotemporal_outlier_ratio']
    moran = metrics['avg_spatial_autocorrelation']
    
    print(f"\n  📋 【{dataset}】时空噪声分析报告:")
    print(f"     • 时间异常值比例: {temp_ratio:.2f}%")
    print(f"     • 空间异常值比例: {spat_ratio:.2f}%")
    print(f"     • 时空交叉比例: {st_ratio:.2f}%")
    print(f"     • 空间自相关 (Moran's I): {moran:.3f}")
    
    # 判断噪声类型
    print(f"\n  🎯 噪声类型诊断:")
    
    if temp_ratio > spat_ratio * 2:
        print(f"     ✅ 【时间噪声主导】 ({temp_ratio:.1f}% vs {spat_ratio:.1f}%)")
        print(f"     💡 建议:")
        print(f"        - 使用时间去噪模块 (denoise_type='attention' 或 'conv')")
        print(f"        - 静态图即可 (use_advanced_graph=False)")
        denoise_config = {
            'use_denoising': True,
            'denoise_type': 'attention' if temp_ratio > 5 else 'conv',
            'use_advanced_graph': False
        }
    
    elif spat_ratio > temp_ratio * 2:
        print(f"     ⚠️  【空间噪声主导】 ({spat_ratio:.1f}% vs {temp_ratio:.1f}%)")
        print(f"     💡 建议:")
        print(f"        - 使用动态图学习 (use_advanced_graph=True)")
        print(f"        - 可选轻量级时间去噪 (denoise_type='conv')")
        if moran < 0.5:
            print(f"        - ⚠️ 空间自相关低 ({moran:.3f})，建议重新学习邻接矩阵")
        denoise_config = {
            'use_denoising': temp_ratio > 2,
            'denoise_type': 'conv',
            'use_advanced_graph': True,
            'graph_heads': 4
        }
    
    else:
        print(f"     🔄 【时空耦合噪声】 (时间{temp_ratio:.1f}% ≈ 空间{spat_ratio:.1f}%)")
        print(f"     💡 建议:")
        print(f"        - 同时使用时间去噪和动态图学习")
        print(f"        - 考虑时空联合去噪模块")
        denoise_config = {
            'use_denoising': True,
            'denoise_type': 'attention',
            'use_advanced_graph': True,
            'graph_heads': 4
        }
    
    # 空间结构评估
    print(f"\n  🌐 空间结构评估:")
    if moran > 0.7:
        print(f"     ✅ 空间自相关强 ({moran:.3f}) - 空间结构良好")
    elif moran > 0.4:
        print(f"     🟡 空间自相关中等 ({moran:.3f}) - 空间结构可接受")
    else:
        print(f"     🔴 空间自相关弱 ({moran:.3f}) - 空间结构混乱")
        print(f"        建议检查邻接矩阵或使用自适应图学习")
    
    return denoise_config


def find_all_datasets():
    """查找所有可用的数据集"""
    datasets_dir = 'datasets'
    
    if not os.path.exists(datasets_dir):
        return []
    
    dataset_names = []
    for item in os.listdir(datasets_dir):
        item_path = os.path.join(datasets_dir, item)
        if os.path.isdir(item_path):
            train_file = os.path.join(item_path, 'train_data.npy')
            if os.path.exists(train_file):
                dataset_names.append(item)
    
    return sorted(dataset_names)


def create_summary_comparison(all_metrics, save_dir='figure'):
    """创建汇总对比报告"""
    if not all_metrics:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    datasets = [m['dataset'] for m in all_metrics]
    temp_ratios = [m['temporal_outlier_ratio'] for m in all_metrics]
    spat_ratios = [m['spatial_outlier_ratio'] for m in all_metrics]
    st_ratios = [m['spatiotemporal_outlier_ratio'] for m in all_metrics]
    morans = [m['avg_spatial_autocorrelation'] for m in all_metrics]
    
    # 子图1: 时间vs空间异常比例
    ax1 = axes[0, 0]
    x = np.arange(len(datasets))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, temp_ratios, width, label='时间异常', 
                    color='coral', alpha=0.7, edgecolor='black')
    bars2 = ax1.bar(x + width/2, spat_ratios, width, label='空间异常', 
                    color='skyblue', alpha=0.7, edgecolor='black')
    
    ax1.set_ylabel('异常值比例 (%)', fontsize=11)
    ax1.set_title('时间 vs 空间异常值对比', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=45)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 子图2: 空间自相关对比
    ax2 = axes[0, 1]
    bars3 = ax2.bar(datasets, morans, color='green', alpha=0.7, edgecolor='black')
    ax2.axhline(0.7, color='green', linestyle='--', alpha=0.5, label='强相关阈值')
    ax2.axhline(0.4, color='orange', linestyle='--', alpha=0.5, label='中等相关阈值')
    
    for bar, moran in zip(bars3, morans):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{moran:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_ylabel("Moran's I", fontsize=11)
    ax2.set_title('空间自相关对比', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 子图3: 时空交叉异常
    ax3 = axes[1, 0]
    bars4 = ax3.bar(datasets, st_ratios, color='purple', alpha=0.7, edgecolor='black')
    
    for bar, ratio in zip(bars4, st_ratios):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{ratio:.2f}%', ha='center', va='bottom', fontsize=9)
    
    ax3.set_ylabel('交叉异常比例 (%)', fontsize=11)
    ax3.set_title('时空交叉异常值对比', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 子图4: 噪声类型散点图
    ax4 = axes[1, 1]
    
    for i, dataset in enumerate(datasets):
        temp = temp_ratios[i]
        spat = spat_ratios[i]
        
        # 根据噪声类型着色
        if temp > spat * 2:
            color = 'red'
            marker = 'o'
            label = '时间主导' if i == 0 else ''
        elif spat > temp * 2:
            color = 'blue'
            marker = 's'
            label = '空间主导' if i == 0 else ''
        else:
            color = 'purple'
            marker = '^'
            label = '时空耦合' if i == 0 else ''
        
        ax4.scatter(temp, spat, c=color, marker=marker, s=150, 
                   alpha=0.7, edgecolors='black', linewidths=1.5, label=label)
        ax4.text(temp, spat, dataset, fontsize=8, ha='right', va='bottom')
    
    # 添加对角线
    max_val = max(max(temp_ratios), max(spat_ratios))
    ax4.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1.5, label='相等线')
    ax4.plot([0, max_val], [0, max_val*2], 'r:', alpha=0.3, label='时间2倍线')
    ax4.plot([0, max_val*2], [0, max_val], 'b:', alpha=0.3, label='空间2倍线')
    
    ax4.set_xlabel('时间异常值比例 (%)', fontsize=11)
    ax4.set_ylabel('空间异常值比例 (%)', fontsize=11)
    ax4.set_title('噪声类型分类', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    summary_path = os.path.join(save_dir, 'spatiotemporal_noise_summary.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 汇总对比报告已保存: {summary_path}")
    plt.close()


def main():
    """主函数"""
    print("\n" + "🌐 " + "="*58 + " 🌐")
    print("🌐  时空联合噪声分析工具")
    print("🌐 " + "="*58 + " 🌐\n")
    
    # 查找数据集
    print("🔍 查找可用数据集...")
    dataset_names = find_all_datasets()
    
    if not dataset_names:
        print("\n❌ 未找到任何数据集！")
        return
    
    print(f"\n✅ 找到 {len(dataset_names)} 个数据集:")
    for i, name in enumerate(dataset_names, 1):
        print(f"   {i}. {name}")
    
    # 分析每个数据集
    all_metrics = []
    
    print("\n" + "="*60)
    print("📊 开始时空联合分析...")
    print("="*60)
    
    for i, dataset_name in enumerate(dataset_names, 1):
        print(f"\n[{i}/{len(dataset_names)}] 分析数据集: {dataset_name}")
        print("-" * 60)
        
        # 加载数据
        data = load_dataset(dataset_name, mode='train')
        if data is None:
            continue
        
        # 加载或生成邻接矩阵
        adj_matrix = load_adjacency_matrix(dataset_name)
        if adj_matrix is None:
            adj_matrix = generate_adjacency_from_data(data, k_neighbors=5)
        
        # 生成分析报告
        metrics = plot_spatiotemporal_analysis(data, adj_matrix, dataset_name)
        
        # 生成建议
        config = generate_spatiotemporal_recommendation(metrics)
        
        all_metrics.append(metrics)
    
    # 创建汇总报告
    if all_metrics:
        print("\n" + "="*60)
        print("📊 生成汇总对比报告...")
        print("="*60)
        create_summary_comparison(all_metrics)
    
    # 打印总结
    print("\n" + "="*60)
    print("✅ 时空联合分析完成！")
    print("="*60)
    
    print(f"\n📂 生成的文件:")
    print(f"   • 个别数据集: figure/spatiotemporal_noise_<数据集名>.png")
    print(f"   • 汇总报告: figure/spatiotemporal_noise_summary.png")
    
    print("\n💡 配置建议总结:")
    print("-" * 60)
    for metrics in all_metrics:
        dataset = metrics['dataset']
        temp_ratio = metrics['temporal_outlier_ratio']
        spat_ratio = metrics['spatial_outlier_ratio']
        
        if temp_ratio > spat_ratio * 2:
            noise_type = "时间主导"
            icon = "🔴"
        elif spat_ratio > temp_ratio * 2:
            noise_type = "空间主导"
            icon = "🔵"
        else:
            noise_type = "时空耦合"
            icon = "🟣"
        
        print(f"  {icon} {dataset:15s} → {noise_type:10s} (T:{temp_ratio:.1f}% S:{spat_ratio:.1f}%)")
    
    print("\n💡 下一步:")
    print("   1. 查看各数据集的时空分析详细报告")
    print("   2. 查看汇总报告了解数据集间的差异")
    print("   3. 根据噪声类型配置模型参数")
    print("   4. 对比时间去噪 vs 图学习的效果\n")


if __name__ == '__main__':
    main()
