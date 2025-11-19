"""
批量噪声分析 - 对所有数据集进行噪声分析
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import os
import glob
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


def estimate_snr(data, window_size=50):
    """
    估计信噪比 (Signal-to-Noise Ratio)
    使用滑动窗口估计局部信号和噪声
    """
    T, N = data.shape
    snr_values = np.zeros(N)
    
    for node in range(N):
        time_series = data[:, node]
        
        # 使用移动平均作为信号
        signal_estimate = np.convolve(time_series, np.ones(window_size)/window_size, mode='same')
        
        # 噪声 = 原始数据 - 信号
        noise = time_series - signal_estimate
        
        # 计算信号功率和噪声功率
        signal_power = np.mean(signal_estimate ** 2)
        noise_power = np.mean(noise ** 2)
        
        # SNR (dB)
        if noise_power > 0:
            snr_values[node] = 10 * np.log10(signal_power / noise_power)
        else:
            snr_values[node] = 100  # 非常干净
    
    return snr_values


def analyze_frequency_spectrum(data, sample_rate=1.0, num_nodes=10):
    """分析频谱 - 检测高频噪声"""
    T, N = data.shape
    
    # 随机选择几个节点
    sample_nodes = np.random.choice(N, min(num_nodes, N), replace=False)
    
    all_freqs = []
    all_power = []
    
    for node in sample_nodes:
        time_series = data[:, node]
        
        # FFT
        yf = fft(time_series)
        xf = fftfreq(T, 1/sample_rate)[:T//2]
        power = 2.0/T * np.abs(yf[:T//2])
        
        all_freqs.append(xf)
        all_power.append(power)
    
    # 平均频谱
    avg_power = np.mean(all_power, axis=0)
    
    return xf, avg_power


def detect_outliers(data, method='iqr'):
    """
    检测异常值
    method: 'iqr' (四分位距) 或 'zscore' (Z分数)
    """
    T, N = data.shape
    
    if method == 'iqr':
        # IQR方法
        q1 = np.percentile(data, 25, axis=0)
        q3 = np.percentile(data, 75, axis=0)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = (data < lower_bound) | (data > upper_bound)
        
    elif method == 'zscore':
        # Z-score方法
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        z_scores = np.abs((data - mean) / (std + 1e-8))
        
        outliers = z_scores > 3
    
    outlier_ratio = (outliers.sum() / outliers.size) * 100
    
    return outliers, outlier_ratio


def analyze_autocorrelation(data, max_lag=50):
    """分析自相关函数"""
    T, N = data.shape
    
    # 计算每个节点的自相关
    all_acf = []
    
    for node in range(N):
        time_series = data[:, node]
        
        # 标准化
        ts_norm = (time_series - np.mean(time_series)) / (np.std(time_series) + 1e-8)
        
        # 计算自相关
        acf = np.correlate(ts_norm, ts_norm, mode='full')[len(ts_norm)-1:]
        acf = acf[:max_lag+1] / acf[0]
        
        all_acf.append(acf)
    
    # 平均自相关
    avg_acf = np.mean(all_acf, axis=0)
    
    return avg_acf


def plot_comprehensive_analysis(data, dataset_name, save_dir='figure'):
    """
    生成综合噪声分析报告（4合1图表）
    
    Args:
        data: 数据数组
        dataset_name: 数据集名称
        save_dir: 保存目录
    """
    fig = plt.figure(figsize=(16, 10))
    
    # 1. SNR分布
    ax1 = plt.subplot(2, 2, 1)
    snr_values = estimate_snr(data)
    
    # 创建热图显示
    T, N = data.shape
    n_rows = int(np.sqrt(N))
    n_cols = int(np.ceil(N / n_rows))
    snr_grid = np.full((n_rows, n_cols), np.nan)
    snr_grid.flat[:N] = snr_values
    
    im1 = ax1.imshow(snr_grid, cmap='RdYlGn', aspect='auto', vmin=0, vmax=30)
    ax1.set_title(f'{dataset_name} - 各节点信噪比(SNR)分布', fontsize=12, fontweight='bold')
    ax1.set_xlabel('列索引', fontsize=10)
    ax1.set_ylabel('行索引', fontsize=10)
    plt.colorbar(im1, ax=ax1, label='SNR (dB)')
    
    # 添加统计信息
    avg_snr = np.mean(snr_values)
    ax1.text(0.02, 0.98, f'平均SNR: {avg_snr:.2f} dB', 
             transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. 频谱分析
    ax2 = plt.subplot(2, 2, 2)
    freqs, power = analyze_frequency_spectrum(data)
    
    ax2.semilogy(freqs, power, linewidth=1.5, color='steelblue')
    ax2.set_title(f'{dataset_name} - 平均功率谱密度', fontsize=12, fontweight='bold')
    ax2.set_xlabel('频率', fontsize=10)
    ax2.set_ylabel('功率 (对数尺度)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 标记高频区域
    high_freq_threshold = freqs.max() * 0.3
    high_freq_idx = freqs > high_freq_threshold
    high_freq_power = power[high_freq_idx].sum()
    total_power = power.sum()
    high_freq_ratio = high_freq_power / total_power * 100
    
    ax2.axvline(high_freq_threshold, color='red', linestyle='--', linewidth=1.5, 
                label=f'高频区 (>{high_freq_threshold:.2f})')
    ax2.text(0.02, 0.98, f'高频能量占比: {high_freq_ratio:.2f}%', 
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.legend(fontsize=9)
    
    # 3. 异常值检测
    ax3 = plt.subplot(2, 2, 3)
    outliers, outlier_ratio = detect_outliers(data, method='zscore')
    
    # 显示异常值分布
    outlier_counts = outliers.sum(axis=0)
    ax3.bar(range(N), outlier_counts, color='coral', alpha=0.7, edgecolor='black')
    ax3.set_title(f'{dataset_name} - 各节点异常值数量', fontsize=12, fontweight='bold')
    ax3.set_xlabel('节点索引', fontsize=10)
    ax3.set_ylabel('异常值数量', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    ax3.text(0.02, 0.98, f'总异常值比例: {outlier_ratio:.2f}%', 
             transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. 自相关分析
    ax4 = plt.subplot(2, 2, 4)
    acf = analyze_autocorrelation(data)
    lags = range(len(acf))
    
    ax4.plot(lags, acf, linewidth=2, color='steelblue', marker='o', markersize=4)
    ax4.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax4.axhline(0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='中等相关')
    ax4.set_title(f'{dataset_name} - 平均自相关函数', fontsize=12, fontweight='bold')
    ax4.set_xlabel('滞后量', fontsize=10)
    ax4.set_ylabel('相关系数', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9)
    
    # 找到第一个低相关的滞后
    low_corr_lag = np.argmax(np.abs(acf[1:]) < 0.3) + 1 if any(np.abs(acf[1:]) < 0.3) else len(acf)
    ax4.text(0.02, 0.98, f'快速衰减滞后: {low_corr_lag}', 
             transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存文件名包含数据集名称
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'noise_analysis_{dataset_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n  📊 分析报告已保存: {save_path}")
    plt.close()
    
    return {
        'dataset': dataset_name,
        'avg_snr': avg_snr,
        'high_freq_ratio': high_freq_ratio,
        'outlier_ratio': outlier_ratio,
        'low_corr_lag': low_corr_lag
    }


def generate_recommendation(metrics):
    """根据指标生成建议"""
    dataset = metrics['dataset']
    avg_snr = metrics['avg_snr']
    high_freq_ratio = metrics['high_freq_ratio']
    outlier_ratio = metrics['outlier_ratio']
    
    print(f"\n  📋 【{dataset}】噪声分析报告:")
    print(f"     • 平均信噪比: {avg_snr:.2f} dB")
    print(f"     • 高频能量占比: {high_freq_ratio:.2f}%")
    print(f"     • 异常值比例: {outlier_ratio:.2f}%")
    
    # 评分系统
    score = 0
    
    if avg_snr < 15:
        score += 2
    elif avg_snr < 20:
        score += 1
    
    if high_freq_ratio > 15:
        score += 2
    elif high_freq_ratio > 10:
        score += 1
    
    if outlier_ratio > 5:
        score += 2
    elif outlier_ratio > 2:
        score += 1
    
    print(f"     • 噪声评分: {score}/6", end="")
    
    # 推荐方案
    if score >= 4:
        print(" 🔴 严重")
        print(f"     💡 推荐: 使用注意力去噪 (denoise_type='attention')")
    elif score >= 2:
        print(" 🟡 中等")
        print(f"     💡 推荐: 使用卷积去噪 (denoise_type='conv')")
    else:
        print(" 🟢 良好")
        print(f"     💡 推荐: 可以不使用去噪，或做对比实验")
    
    return score


def find_all_datasets():
    """查找所有可用的数据集"""
    datasets_dir = 'datasets'
    
    if not os.path.exists(datasets_dir):
        print(f"❌ 数据集目录不存在: {datasets_dir}")
        return []
    
    # 查找所有包含train_data.npy的子目录
    dataset_names = []
    for item in os.listdir(datasets_dir):
        item_path = os.path.join(datasets_dir, item)
        if os.path.isdir(item_path):
            train_file = os.path.join(item_path, 'train_data.npy')
            if os.path.exists(train_file):
                dataset_names.append(item)
    
    return sorted(dataset_names)


def create_summary_report(all_metrics, save_dir='figure'):
    """创建汇总对比报告"""
    if not all_metrics:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    datasets = [m['dataset'] for m in all_metrics]
    snrs = [m['avg_snr'] for m in all_metrics]
    high_freqs = [m['high_freq_ratio'] for m in all_metrics]
    outliers = [m['outlier_ratio'] for m in all_metrics]
    
    # 1. SNR对比
    ax1 = axes[0, 0]
    bars1 = ax1.bar(datasets, snrs, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_ylabel('平均SNR (dB)', fontsize=11)
    ax1.set_title('各数据集平均信噪比对比', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 2. 高频能量对比
    ax2 = axes[0, 1]
    bars2 = ax2.bar(datasets, high_freqs, color='coral', alpha=0.7, edgecolor='black')
    ax2.set_ylabel('高频能量占比 (%)', fontsize=11)
    ax2.set_title('各数据集高频能量占比对比', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 3. 异常值比例对比
    ax3 = axes[1, 0]
    bars3 = ax3.bar(datasets, outliers, color='lightgreen', alpha=0.7, edgecolor='black')
    ax3.set_ylabel('异常值比例 (%)', fontsize=11)
    ax3.set_title('各数据集异常值比例对比', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=9)
    
    # 4. 综合评分雷达图
    ax4 = axes[1, 1]
    
    # 归一化指标用于雷达图
    snrs_norm = [(30 - s) / 30 * 100 for s in snrs]  # 反转，越低越好
    
    x = np.arange(len(datasets))
    width = 0.25
    
    ax4.bar(x - width, snrs_norm, width, label='SNR指标', alpha=0.7)
    ax4.bar(x, high_freqs, width, label='高频噪声', alpha=0.7)
    ax4.bar(x + width, outliers, width, label='异常值', alpha=0.7)
    
    ax4.set_ylabel('指标值', fontsize=11)
    ax4.set_title('噪声综合指标对比', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(datasets, rotation=45)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存汇总报告
    summary_path = os.path.join(save_dir, 'noise_analysis_summary.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 汇总对比报告已保存: {summary_path}")
    plt.close()


def main():
    """主函数"""
    print("\n" + "🔬 " + "="*58 + " 🔬")
    print("🔬  批量噪声分析工具 - 分析所有数据集")
    print("🔬 " + "="*58 + " 🔬\n")
    
    # 查找所有数据集
    print("🔍 查找可用数据集...")
    dataset_names = find_all_datasets()
    
    if not dataset_names:
        print("\n❌ 未找到任何数据集！")
        print("💡 请确保 datasets/ 目录下有包含 train_data.npy 的子目录")
        return
    
    print(f"\n✅ 找到 {len(dataset_names)} 个数据集:")
    for i, name in enumerate(dataset_names, 1):
        print(f"   {i}. {name}")
    
    # 分析每个数据集
    all_metrics = []
    
    print("\n" + "="*60)
    print("📊 开始批量分析...")
    print("="*60)
    
    for i, dataset_name in enumerate(dataset_names, 1):
        print(f"\n[{i}/{len(dataset_names)}] 分析数据集: {dataset_name}")
        print("-" * 60)
        
        # 加载数据
        data = load_dataset(dataset_name, mode='train')
        
        if data is None:
            print(f"  ⚠️  跳过 {dataset_name}")
            continue
        
        # 生成分析报告
        metrics = plot_comprehensive_analysis(data, dataset_name)
        
        # 生成建议
        score = generate_recommendation(metrics)
        
        all_metrics.append(metrics)
    
    # 创建汇总报告
    if all_metrics:
        print("\n" + "="*60)
        print("📊 生成汇总对比报告...")
        print("="*60)
        create_summary_report(all_metrics)
    
    # 打印总结
    print("\n" + "="*60)
    print("✅ 批量分析完成！")
    print("="*60)
    
    print(f"\n📂 生成的文件:")
    print(f"   • 个别数据集报告: figure/noise_analysis_<数据集名>.png")
    print(f"   • 汇总对比报告: figure/noise_analysis_summary.png")
    
    print("\n💡 建议配置总结:")
    print("-" * 60)
    for metrics in all_metrics:
        dataset = metrics['dataset']
        avg_snr = metrics['avg_snr']
        
        if avg_snr < 15:
            denoise = "attention"
            icon = "🔴"
        elif avg_snr < 20:
            denoise = "conv"
            icon = "🟡"
        else:
            denoise = "None (可选)"
            icon = "🟢"
        
        print(f"  {icon} {dataset:15s} → denoise_type: '{denoise}'")
    
    print("\n💡 下一步:")
    print("   1. 查看各数据集的详细分析报告")
    print("   2. 查看汇总对比报告了解整体情况")
    print("   3. 根据建议配置各数据集的去噪参数")
    print("   4. 运行对比实验验证去噪效果\n")


if __name__ == '__main__':
    main()
