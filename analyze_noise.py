"""
深度噪声分析 - 评估数据噪声水平
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import os
plt.rcParams['font.sans-serif'] = ['SimHei']


def load_dataset(dataset_name='PEMS03', mode='train'):
    """加载数据集"""
    data_path = f'datasets/{dataset_name}/{mode}_data.npy'
    
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        return None
    
    data = np.load(data_path)
    print(f"✅ 加载数据: {data_path}")
    print(f"   数据形状: {data.shape}")
    
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
    """
    分析频谱 - 检测高频噪声
    """
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
        
    else:  # zscore
        # Z-score方法
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        z_scores = np.abs((data - mean) / std)
        outliers = z_scores > 3
    
    outlier_ratio = outliers.sum() / data.size * 100
    
    return outliers, outlier_ratio


def analyze_autocorrelation(data, max_lag=50, num_samples=5):
    """
    分析自相关性 - 检测时序模式和噪声
    """
    T, N = data.shape
    sample_nodes = np.random.choice(N, min(num_samples, N), replace=False)
    
    all_acf = []
    
    for node in sample_nodes:
        time_series = data[:, node]
        time_series = (time_series - time_series.mean()) / time_series.std()
        
        acf = []
        for lag in range(max_lag):
            if lag == 0:
                acf.append(1.0)
            else:
                correlation = np.corrcoef(time_series[:-lag], time_series[lag:])[0, 1]
                acf.append(correlation)
        
        all_acf.append(acf)
    
    avg_acf = np.mean(all_acf, axis=0)
    
    return avg_acf


def plot_comprehensive_analysis(data, save_path='figure/noise_analysis_report.png'):
    """
    生成综合噪声分析报告（4合1图表）
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
    ax1.set_title('各节点信噪比(SNR)分布', fontsize=12, fontweight='bold')
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
    ax2.set_title('平均功率谱密度', fontsize=12, fontweight='bold')
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
    outliers, outlier_ratio = detect_outliers(data, method='iqr')
    
    # 显示异常值分布
    outlier_counts = outliers.sum(axis=0)
    ax3.bar(range(N), outlier_counts, color='coral', alpha=0.7, edgecolor='black')
    ax3.set_title('各节点异常值数量', fontsize=12, fontweight='bold')
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
    ax4.set_title('平均自相关函数', fontsize=12, fontweight='bold')
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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 综合分析报告已保存: {save_path}")
    plt.close()
    
    return {
        'avg_snr': avg_snr,
        'high_freq_ratio': high_freq_ratio,
        'outlier_ratio': outlier_ratio,
        'low_corr_lag': low_corr_lag
    }


def generate_detailed_recommendation(metrics):
    """根据多个指标生成详细建议"""
    print("\n" + "="*60)
    print("📋 噪声分析报告")
    print("="*60)
    
    avg_snr = metrics['avg_snr']
    high_freq_ratio = metrics['high_freq_ratio']
    outlier_ratio = metrics['outlier_ratio']
    
    print(f"\n📊 关键指标:")
    print(f"  • 平均信噪比: {avg_snr:.2f} dB")
    print(f"  • 高频能量占比: {high_freq_ratio:.2f}%")
    print(f"  • 异常值比例: {outlier_ratio:.2f}%")
    
    # 评分系统
    score = 0
    reasons = []
    
    # SNR评估
    if avg_snr < 10:
        score += 3
        reasons.append(f"❌ SNR过低 ({avg_snr:.2f} dB < 10 dB)")
    elif avg_snr < 20:
        score += 2
        reasons.append(f"⚠️  SNR较低 ({avg_snr:.2f} dB < 20 dB)")
    else:
        reasons.append(f"✅ SNR良好 ({avg_snr:.2f} dB >= 20 dB)")
    
    # 高频噪声评估
    if high_freq_ratio > 30:
        score += 3
        reasons.append(f"❌ 高频噪声严重 ({high_freq_ratio:.2f}% > 30%)")
    elif high_freq_ratio > 10:
        score += 2
        reasons.append(f"⚠️  高频噪声明显 ({high_freq_ratio:.2f}% > 10%)")
    else:
        reasons.append(f"✅ 高频噪声较少 ({high_freq_ratio:.2f}% <= 10%)")
    
    # 异常值评估
    if outlier_ratio > 5:
        score += 2
        reasons.append(f"❌ 异常值过多 ({outlier_ratio:.2f}% > 5%)")
    elif outlier_ratio > 1:
        score += 1
        reasons.append(f"⚠️  有一定异常值 ({outlier_ratio:.2f}% > 1%)")
    else:
        reasons.append(f"✅ 异常值很少 ({outlier_ratio:.2f}% <= 1%)")
    
    print(f"\n🔍 诊断结果:")
    for reason in reasons:
        print(f"  {reason}")
    
    print(f"\n💯 噪声评分: {score}/8")
    
    print("\n" + "-"*60)
    print("💡 推荐方案:")
    print("-"*60)
    
    if score >= 6:
        print("\n🔴 数据噪声严重，强烈推荐使用注意力去噪:")
        print("\n```yaml")
        print("use_denoising: True")
        print("denoise_type: 'attention'")
        print("```")
        print("\n理由: 数据包含严重噪声，需要强大的自适应去噪能力")
        
    elif score >= 3:
        print("\n🟡 数据有一定噪声，推荐使用卷积去噪:")
        print("\n```yaml")
        print("use_denoising: True")
        print("denoise_type: 'conv'")
        print("```")
        print("\n理由: 轻量级去噪即可处理，兼顾效率和效果")
        
    else:
        print("\n🟢 数据质量良好，可以不使用去噪:")
        print("\n```yaml")
        print("use_denoising: False")
        print("```")
        print("\n但建议做对比实验验证:")
        print("\n1. Baseline (无去噪)")
        print("2. Conv去噪")
        print("3. 对比验证集性能")


def main():
    """主函数"""
    print("\n" + "🔬 " + "="*58 + " 🔬")
    print("🔬  深度噪声分析工具")
    print("🔬 " + "="*58 + " 🔬\n")
    
    # 加载数据
    dataset_name = 'METR-LA'
    mode = 'train'
    
    data = load_dataset(dataset_name, mode)
    
    if data is None:
        print("\n❌ 无法加载数据，程序退出")
        return
    
    print("\n" + "="*60)
    print("📊 执行噪声分析...")
    print("="*60)
    
    # 生成综合分析报告
    metrics = plot_comprehensive_analysis(data)
    
    # 生成详细建议
    generate_detailed_recommendation(metrics)
    
    print("\n" + "="*60)
    print("✅ 分析完成！")
    print("="*60)
    print("\n📂 分析报告已保存: figure/noise_analysis_report.png")
    print("\n💡 下一步:")
    print("   1. 查看噪声分析报告图表")
    print("   2. 根据推荐配置去噪参数")
    print("   3. 运行对比实验验证效果")
    print("   4. 选择最佳配置进行完整训练\n")


if __name__ == '__main__':
    main()
