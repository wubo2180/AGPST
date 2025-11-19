"""
时空噪声分析（文本版本） - 不需要matplotlib
"""
import numpy as np
import pickle
import os


def load_dataset(dataset_name, mode='train'):
    """加载数据集"""
    data_path = f'datasets/{dataset_name}/{mode}_data.npy'
    
    if not os.path.exists(data_path):
        return None
    
    data = np.load(data_path)
    return data


def load_adjacency_matrix(dataset_name):
    """加载邻接矩阵"""
    adj_path = f'datasets/{dataset_name}/adj_mx.pkl'
    
    if not os.path.exists(adj_path):
        return None
    
    try:
        with open(adj_path, 'rb') as f:
            pkl_data = pickle.load(f, encoding='latin1')
        
        if isinstance(pkl_data, np.ndarray):
            adj_mx = pkl_data
        else:
            return None
        
        # 如果是距离矩阵，转换为二进制邻接矩阵
        if adj_mx.max() > 1.0:
            threshold = np.percentile(adj_mx[adj_mx > 0], 25)
            adj_mx = (adj_mx > 0) & (adj_mx <= threshold)
            adj_mx = adj_mx.astype(np.float32)
        
        return adj_mx
    
    except Exception as e:
        return None


def generate_adjacency_from_data(data, k_neighbors=5):
    """从数据生成邻接矩阵"""
    T, N = data.shape
    corr_matrix = np.corrcoef(data.T)
    dist_matrix = 1 - np.abs(corr_matrix)
    
    adj_matrix = np.zeros((N, N))
    for i in range(N):
        neighbors = np.argsort(dist_matrix[i, :])[1:k_neighbors+1]
        adj_matrix[i, neighbors] = 1
        adj_matrix[neighbors, i] = 1
    
    return adj_matrix


def detect_temporal_outliers(data, method='zscore'):
    """检测时间维度异常值"""
    T, N = data.shape
    
    if method == 'zscore':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        z_scores = np.abs((data - mean) / (std + 1e-8))
        outliers = z_scores > 3
    
    return outliers


def detect_spatial_outliers(data, adj_matrix, threshold=3.0):
    """检测空间维度异常值"""
    T, N = data.shape
    spatial_outliers = np.zeros((T, N), dtype=bool)
    
    for t in range(T):
        snapshot = data[t, :]
        
        for i in range(N):
            neighbors = np.where(adj_matrix[i, :] > 0)[0]
            
            if len(neighbors) == 0:
                continue
            
            neighbor_vals = snapshot[neighbors]
            avg_neighbor = np.mean(neighbor_vals)
            std_neighbor = np.std(neighbor_vals)
            
            if std_neighbor < 1e-6:
                std_neighbor = np.std(snapshot)
            
            if std_neighbor > 0:
                z_score = abs(snapshot[i] - avg_neighbor) / std_neighbor
                if z_score > threshold:
                    spatial_outliers[t, i] = True
    
    return spatial_outliers


def compute_spatial_autocorrelation(data, adj_matrix):
    """计算Moran's I空间自相关系数"""
    T, N = data.shape
    moran_values = []
    
    W = np.sum(adj_matrix)
    if W == 0:
        return np.zeros(T)
    
    for t in range(T):
        snapshot = data[t, :]
        mean_val = np.mean(snapshot)
        deviations = snapshot - mean_val
        
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


def analyze_dataset(dataset_name):
    """分析单个数据集"""
    print(f"\n{'='*70}")
    print(f"📊 分析数据集: {dataset_name}")
    print(f"{'='*70}")
    
    # 加载数据
    data = load_dataset(dataset_name, mode='train')
    if data is None:
        print(f"❌ 无法加载数据")
        return None
    
    T, N = data.shape
    print(f"✅ 数据形状: (T={T}, N={N})")
    
    # 加载或生成邻接矩阵
    adj_matrix = load_adjacency_matrix(dataset_name)
    if adj_matrix is None:
        print(f"💡 使用数据生成k-近邻邻接矩阵 (k=5)")
        adj_matrix = generate_adjacency_from_data(data, k_neighbors=5)
    else:
        print(f"✅ 加载邻接矩阵: {adj_matrix.shape}")
    
    edge_count = (adj_matrix > 0).sum() / 2
    density = edge_count / (N * (N-1) / 2) * 100 if N > 1 else 0
    print(f"   边数: {int(edge_count)}, 密度: {density:.2f}%")
    
    # 1. 时间维度异常值
    print(f"\n🔍 检测时间维度异常值...")
    temporal_outliers = detect_temporal_outliers(data, method='zscore')
    temporal_ratio = temporal_outliers.sum() / temporal_outliers.size * 100
    temporal_counts = temporal_outliers.sum(axis=0)
    
    print(f"   异常值比例: {temporal_ratio:.2f}%")
    print(f"   平均每节点: {temporal_counts.mean():.1f}个")
    print(f"   最多节点: {temporal_counts.max()}个, 最少: {temporal_counts.min()}个")
    
    # 2. 空间维度异常值
    print(f"\n🌐 检测空间维度异常值...")
    spatial_outliers = detect_spatial_outliers(data, adj_matrix, threshold=3.0)
    spatial_ratio = spatial_outliers.sum() / spatial_outliers.size * 100
    spatial_counts = spatial_outliers.sum(axis=0)
    
    print(f"   异常值比例: {spatial_ratio:.2f}%")
    print(f"   平均每节点: {spatial_counts.mean():.1f}个")
    print(f"   最多节点: {spatial_counts.max()}个, 最少: {spatial_counts.min()}个")
    
    # 3. 时空交叉异常值
    print(f"\n🔄 分析时空交叉异常值...")
    spatiotemporal_outliers = temporal_outliers & spatial_outliers
    st_ratio = spatiotemporal_outliers.sum() / spatiotemporal_outliers.size * 100
    
    print(f"   交叉异常比例: {st_ratio:.2f}%")
    print(f"   占时间异常的比例: {st_ratio/temporal_ratio*100:.1f}%" if temporal_ratio > 0 else "   N/A")
    print(f"   占空间异常的比例: {st_ratio/spatial_ratio*100:.1f}%" if spatial_ratio > 0 else "   N/A")
    
    # 4. 空间自相关
    print(f"\n📈 计算空间自相关...")
    moran_i = compute_spatial_autocorrelation(data, adj_matrix)
    avg_moran = np.mean(moran_i)
    std_moran = np.std(moran_i)
    
    print(f"   平均 Moran's I: {avg_moran:.3f} ± {std_moran:.3f}")
    print(f"   最小值: {moran_i.min():.3f}, 最大值: {moran_i.max():.3f}")
    
    # 5. 诊断和建议
    print(f"\n{'='*70}")
    print(f"🎯 噪声类型诊断")
    print(f"{'='*70}")
    
    if temporal_ratio > spatial_ratio * 2:
        noise_type = "时间噪声主导"
        icon = "🔴"
        recommendation = {
            'type': 'temporal',
            'use_denoising': True,
            'denoise_type': 'attention' if temporal_ratio > 5 else 'conv',
            'use_advanced_graph': False
        }
    elif spatial_ratio > temporal_ratio * 2:
        noise_type = "空间噪声主导"
        icon = "🔵"
        recommendation = {
            'type': 'spatial',
            'use_denoising': temporal_ratio > 2,
            'denoise_type': 'conv',
            'use_advanced_graph': True,
            'graph_heads': 4
        }
    else:
        noise_type = "时空耦合噪声"
        icon = "🟣"
        recommendation = {
            'type': 'spatiotemporal',
            'use_denoising': True,
            'denoise_type': 'attention',
            'use_advanced_graph': True,
            'graph_heads': 4
        }
    
    print(f"\n{icon} 【{noise_type}】")
    print(f"   时间异常: {temporal_ratio:.2f}%")
    print(f"   空间异常: {spatial_ratio:.2f}%")
    print(f"   比值: {temporal_ratio/spatial_ratio:.2f}" if spatial_ratio > 0 else "   比值: inf")
    
    print(f"\n💡 配置建议:")
    print(f"   use_denoising: {recommendation['use_denoising']}")
    if recommendation['use_denoising']:
        print(f"   denoise_type: '{recommendation['denoise_type']}'")
    print(f"   use_advanced_graph: {recommendation['use_advanced_graph']}")
    if recommendation.get('graph_heads'):
        print(f"   graph_heads: {recommendation['graph_heads']}")
    
    print(f"\n🌐 空间结构评估:")
    if avg_moran > 0.7:
        print(f"   ✅ 空间自相关强 ({avg_moran:.3f}) - 空间结构良好")
    elif avg_moran > 0.4:
        print(f"   🟡 空间自相关中等 ({avg_moran:.3f}) - 空间结构可接受")
    else:
        print(f"   🔴 空间自相关弱 ({avg_moran:.3f}) - 空间结构混乱")
        print(f"      建议: 使用自适应图学习重新学习邻接矩阵")
    
    return {
        'dataset': dataset_name,
        'temporal_outlier_ratio': temporal_ratio,
        'spatial_outlier_ratio': spatial_ratio,
        'spatiotemporal_outlier_ratio': st_ratio,
        'avg_spatial_autocorrelation': avg_moran,
        'noise_type': noise_type,
        'recommendation': recommendation
    }


def main():
    """主函数"""
    print("\n" + "🌐 " + "="*68 + " 🌐")
    print("🌐  时空联合噪声分析工具（文本版本）")
    print("🌐 " + "="*68 + " 🌐")
    
    datasets = []
    datasets_dir = 'datasets'
    
    if os.path.exists(datasets_dir):
        for item in os.listdir(datasets_dir):
            item_path = os.path.join(datasets_dir, item)
            if os.path.isdir(item_path):
                train_file = os.path.join(item_path, 'train_data.npy')
                if os.path.exists(train_file):
                    datasets.append(item)
    
    datasets = sorted(datasets)
    
    if not datasets:
        print("\n❌ 未找到任何数据集！")
        return
    
    print(f"\n✅ 找到 {len(datasets)} 个数据集: {', '.join(datasets)}")
    
    # 分析所有数据集
    all_metrics = []
    
    for dataset_name in datasets:
        metrics = analyze_dataset(dataset_name)
        if metrics:
            all_metrics.append(metrics)
    
    # 汇总对比
    if len(all_metrics) > 1:
        print(f"\n{'='*70}")
        print(f"📊 汇总对比")
        print(f"{'='*70}")
        
        print(f"\n{'数据集':<15} {'噪声类型':<12} {'时间%':<8} {'空间%':<8} {'Moran':<8} {'建议'}")
        print(f"{'-'*70}")
        
        for m in all_metrics:
            dataset = m['dataset']
            noise_type = m['noise_type'][:4]
            temp = m['temporal_outlier_ratio']
            spat = m['spatial_outlier_ratio']
            moran = m['avg_spatial_autocorrelation']
            rec = m['recommendation']
            
            if rec['use_denoising'] and rec['use_advanced_graph']:
                suggest = "去噪+图学习"
            elif rec['use_denoising']:
                suggest = f"去噪({rec['denoise_type']})"
            elif rec['use_advanced_graph']:
                suggest = "图学习"
            else:
                suggest = "无需特殊处理"
            
            print(f"{dataset:<15} {noise_type:<12} {temp:>6.2f}% {spat:>6.2f}% {moran:>7.3f} {suggest}")
    
    print(f"\n{'='*70}")
    print(f"✅ 分析完成！")
    print(f"{'='*70}")
    
    print(f"\n💡 总结:")
    print(f"   • 时间噪声主导的数据集适合使用去噪模块")
    print(f"   • 空间噪声主导的数据集适合使用动态图学习")
    print(f"   • 时空耦合噪声需要两者结合")
    print(f"   • 低空间自相关(Moran's I<0.4)建议使用自适应图学习\n")


if __name__ == '__main__':
    main()
