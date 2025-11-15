"""
可视化原始数据 - 快速检查数据质量
"""
import numpy as np
import matplotlib.pyplot as plt
import os
plt.rcParams['font.sans-serif'] = ['SimHei']


def load_dataset(dataset_name='PEMS03', mode='train'):
    """加载数据集"""
    data_path = f'datasets/{dataset_name}/{mode}_data.npy'
    
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        print(f"请检查数据集路径")
        return None
    
    data = np.load(data_path)
    print(f"✅ 加载数据: {data_path}")
    print(f"   数据形状: {data.shape}")
    print(f"   数据类型: {data.dtype}")
    print(f"   数值范围: [{data.min():.2f}, {data.max():.2f}]")
    print(f"   均值: {data.mean():.2f}")
    print(f"   标准差: {data.std():.2f}")
    
    return data


def plot_time_series(data, num_samples=5, sample_nodes=None, save_path='figure/raw_data_time_series.png'):
    """绘制时间序列"""
    T, N = data.shape
    
    if sample_nodes is None:
        # 随机选择节点
        sample_nodes = np.random.choice(N, num_samples, replace=False)
    else:
        num_samples = len(sample_nodes)
    
    # 只显示前500个时间步以便查看细节
    time_window = min(500, T)
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(14, 2.5 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for idx, node_id in enumerate(sample_nodes):
        ax = axes[idx]
        time_series = data[:time_window, node_id]
        
        ax.plot(time_series, linewidth=1.0, alpha=0.8, color='steelblue')
        ax.set_title(f'节点 {node_id} 的时间序列', fontsize=11, fontweight='bold')
        ax.set_xlabel('时间步', fontsize=10)
        ax.set_ylabel('数值', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 添加统计信息
        mean_val = time_series.mean()
        std_val = time_series.std()
        ax.axhline(mean_val, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'均值: {mean_val:.2f}')
        ax.fill_between(range(time_window), mean_val - std_val, mean_val + std_val, 
                        color='red', alpha=0.1, label=f'±1σ: {std_val:.2f}')
        ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 时间序列图已保存: {save_path}")
    plt.close()


def plot_distribution(data, save_path='figure/raw_data_distribution.png'):
    """绘制数据分布"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 全局分布
    ax1 = axes[0]
    ax1.hist(data.flatten(), bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_title('全局数据分布', fontsize=12, fontweight='bold')
    ax1.set_xlabel('数值', fontsize=10)
    ax1.set_ylabel('频数', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_val = data.mean()
    std_val = data.std()
    ax1.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'均值: {mean_val:.2f}')
    ax1.axvline(mean_val + std_val, color='orange', linestyle='--', linewidth=1, label=f'+1σ: {mean_val+std_val:.2f}')
    ax1.axvline(mean_val - std_val, color='orange', linestyle='--', linewidth=1, label=f'-1σ: {mean_val-std_val:.2f}')
    ax1.legend(fontsize=9)
    
    # 每个节点的均值和标准差
    ax2 = axes[1]
    node_means = data.mean(axis=0)
    node_stds = data.std(axis=0)
    
    ax2.scatter(node_means, node_stds, alpha=0.5, s=20, color='steelblue')
    ax2.set_title('各节点统计特征', fontsize=12, fontweight='bold')
    ax2.set_xlabel('节点均值', fontsize=10)
    ax2.set_ylabel('节点标准差', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 数据分布图已保存: {save_path}")
    plt.close()


def plot_correlation_heatmap(data, max_nodes=50, save_path='figure/raw_data_correlation.png'):
    """绘制节点相关性热图"""
    T, N = data.shape
    
    # 只显示部分节点以便查看
    sample_nodes = min(max_nodes, N)
    node_indices = np.linspace(0, N-1, sample_nodes, dtype=int)
    
    # 计算相关性矩阵
    data_sample = data[:, node_indices]
    corr_matrix = np.corrcoef(data_sample.T)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    ax.set_title(f'节点相关性热图 (前{sample_nodes}个节点)', fontsize=12, fontweight='bold')
    ax.set_xlabel('节点索引', fontsize=10)
    ax.set_ylabel('节点索引', fontsize=10)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('相关系数', fontsize=10)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 相关性热图已保存: {save_path}")
    plt.close()


def analyze_basic_stats(data):
    """分析基本统计信息"""
    print("\n" + "="*60)
    print("📈 基本统计分析")
    print("="*60)
    
    T, N = data.shape
    
    print(f"\n数据维度:")
    print(f"  时间步数: {T}")
    print(f"  节点数量: {N}")
    print(f"  总样本数: {T * N:,}")
    
    print(f"\n全局统计:")
    print(f"  最小值: {data.min():.4f}")
    print(f"  最大值: {data.max():.4f}")
    print(f"  均值: {data.mean():.4f}")
    print(f"  中位数: {np.median(data):.4f}")
    print(f"  标准差: {data.std():.4f}")
    print(f"  变异系数: {data.std()/data.mean():.4f}")
    
    # 检测异常值 (使用3σ准则)
    mean_val = data.mean()
    std_val = data.std()
    outliers = np.abs(data - mean_val) > 3 * std_val
    outlier_ratio = outliers.sum() / data.size * 100
    
    print(f"\n异常值检测 (3σ准则):")
    print(f"  异常值数量: {outliers.sum():,}")
    print(f"  异常值比例: {outlier_ratio:.2f}%")
    
    if outlier_ratio > 5:
        print(f"  ⚠️  警告: 异常值比例较高，建议使用去噪")
    elif outlier_ratio > 1:
        print(f"  ℹ️  提示: 有一定异常值，可以考虑去噪")
    else:
        print(f"  ✅ 异常值比例较低，数据质量良好")
    
    # 检查数据变化率
    diff = np.diff(data, axis=0)
    change_rate = np.abs(diff).mean()
    
    print(f"\n时间序列特征:")
    print(f"  平均变化率: {change_rate:.4f}")
    print(f"  最大变化: {np.abs(diff).max():.4f}")
    
    return {
        'outlier_ratio': outlier_ratio,
        'change_rate': change_rate,
        'std': std_val
    }


def generate_recommendation(stats):
    """根据统计信息生成建议"""
    print("\n" + "="*60)
    print("💡 去噪建议")
    print("="*60)
    
    outlier_ratio = stats['outlier_ratio']
    change_rate = stats['change_rate']
    std = stats['std']
    
    # 综合评分
    score = 0
    
    if outlier_ratio > 5:
        score += 3
        print(f"\n❌ 异常值比例高 ({outlier_ratio:.2f}%) → 强烈建议去噪")
    elif outlier_ratio > 1:
        score += 2
        print(f"\n⚠️  异常值比例中等 ({outlier_ratio:.2f}%) → 建议去噪")
    else:
        score += 0
        print(f"\n✅ 异常值比例低 ({outlier_ratio:.2f}%) → 数据质量好")
    
    if std > 50:
        score += 2
        print(f"⚠️  标准差较大 ({std:.2f}) → 建议去噪")
    elif std > 20:
        score += 1
        print(f"ℹ️  标准差中等 ({std:.2f}) → 可以考虑去噪")
    else:
        print(f"✅ 标准差较小 ({std:.2f}) → 数据稳定")
    
    print("\n" + "-"*60)
    print("📝 推荐配置:")
    print("-"*60)
    
    if score >= 4:
        print("\n🔴 强烈建议使用注意力去噪:")
        print("```yaml")
        print("use_denoising: True")
        print("denoise_type: 'attention'")
        print("```")
    elif score >= 2:
        print("\n🟡 建议使用卷积去噪:")
        print("```yaml")
        print("use_denoising: True")
        print("denoise_type: 'conv'")
        print("```")
    else:
        print("\n🟢 数据质量良好，可以不使用去噪:")
        print("```yaml")
        print("use_denoising: False")
        print("```")
        print("\n但也可以尝试轻量级去噪看是否有提升:")
        print("```yaml")
        print("use_denoising: True")
        print("denoise_type: 'conv'")
        print("```")


def main():
    """主函数"""
    print("\n" + "🔍 " + "="*58 + " 🔍")
    print("🔍  原始数据可视化与分析工具")
    print("🔍 " + "="*58 + " 🔍\n")
    
    # 加载数据
    dataset_name = 'PEMS03'
    mode = 'val'
    
    data = load_dataset(dataset_name, mode)
    
    if data is None:
        print("\n❌ 无法加载数据，程序退出")
        return
    
    # 基本统计分析
    stats = analyze_basic_stats(data)
    
    # 可视化
    print("\n" + "="*60)
    print("📊 生成可视化图表...")
    print("="*60)
    
    # 时间序列图 - 选择几个代表性节点
    sample_nodes = [0, 50, 100, 150, 200]  # 可以自定义
    plot_time_series(data, sample_nodes=sample_nodes)
    
    # 数据分布图
    plot_distribution(data)
    
    # 相关性热图
    plot_correlation_heatmap(data)
    
    # 生成建议
    generate_recommendation(stats)
    
    print("\n" + "="*60)
    print("✅ 分析完成！")
    print("="*60)
    print("\n📂 所有图表已保存到 'figure/' 目录")
    print("\n💡 下一步:")
    print("   1. 查看生成的图表")
    print("   2. 根据建议配置去噪参数")
    print("   3. 运行 analyze_noise.py 进行深入分析")
    print("   4. 开始训练模型\n")


if __name__ == '__main__':
    main()
