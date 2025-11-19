"""
异常值检测示例和可视化
演示如何理解异常值分析的结果
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def demo_iqr_method():
    """演示IQR方法的工作原理"""
    
    # 生成示例数据（包含一些异常值）
    np.random.seed(42)
    normal_data = np.random.normal(50, 10, 100)  # 正常数据
    outliers_low = np.random.uniform(0, 20, 5)   # 低异常值
    outliers_high = np.random.uniform(90, 120, 5) # 高异常值
    
    data = np.concatenate([normal_data, outliers_low, outliers_high])
    np.random.shuffle(data)
    
    # 计算IQR
    q1 = np.percentile(data, 25)
    q2 = np.percentile(data, 50)  # 中位数
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # 识别异常值
    outliers_mask = (data < lower_bound) | (data > upper_bound)
    outlier_count = outliers_mask.sum()
    outlier_ratio = outlier_count / len(data) * 100
    
    # 创建可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 图1: 箱线图展示IQR
    ax1 = axes[0]
    bp = ax1.boxplot([data], vert=True, patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_alpha(0.7)
    
    # 标注关键点
    ax1.axhline(q1, color='green', linestyle='--', linewidth=2, label=f'Q1 = {q1:.1f}')
    ax1.axhline(q2, color='orange', linestyle='--', linewidth=2, label=f'Q2(中位数) = {q2:.1f}')
    ax1.axhline(q3, color='green', linestyle='--', linewidth=2, label=f'Q3 = {q3:.1f}')
    ax1.axhline(lower_bound, color='red', linestyle=':', linewidth=2, label=f'下界 = {lower_bound:.1f}')
    ax1.axhline(upper_bound, color='red', linestyle=':', linewidth=2, label=f'上界 = {upper_bound:.1f}')
    
    # 添加IQR区域
    ax1.add_patch(Rectangle((0.7, q1), 0.6, iqr, 
                            facecolor='yellow', alpha=0.3, label=f'IQR = {iqr:.1f}'))
    
    ax1.set_ylabel('数据值', fontsize=12)
    ax1.set_title('箱线图：IQR方法原理', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([])
    
    # 图2: 散点图显示异常值
    ax2 = axes[1]
    normal_indices = np.where(~outliers_mask)[0]
    outlier_indices = np.where(outliers_mask)[0]
    
    ax2.scatter(normal_indices, data[~outliers_mask], 
               c='steelblue', alpha=0.6, s=50, label='正常值')
    ax2.scatter(outlier_indices, data[outliers_mask], 
               c='red', alpha=0.8, s=100, marker='X', label='异常值')
    
    ax2.axhline(lower_bound, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax2.axhline(upper_bound, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax2.fill_between([0, len(data)], lower_bound, upper_bound, 
                     color='green', alpha=0.1, label='正常范围')
    
    ax2.set_xlabel('数据点索引', fontsize=12)
    ax2.set_ylabel('数据值', fontsize=12)
    ax2.set_title(f'异常值分布 (检出{outlier_count}个, {outlier_ratio:.1f}%)', 
                 fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 图3: 直方图显示数据分布
    ax3 = axes[2]
    ax3.hist(data[~outliers_mask], bins=20, color='steelblue', 
            alpha=0.7, edgecolor='black', label='正常值分布')
    ax3.hist(data[outliers_mask], bins=10, color='red', 
            alpha=0.7, edgecolor='black', label='异常值分布')
    
    ax3.axvline(q1, color='green', linestyle='--', linewidth=2)
    ax3.axvline(q3, color='green', linestyle='--', linewidth=2)
    ax3.axvline(lower_bound, color='red', linestyle=':', linewidth=2)
    ax3.axvline(upper_bound, color='red', linestyle=':', linewidth=2)
    
    ax3.set_xlabel('数据值', fontsize=12)
    ax3.set_ylabel('频数', fontsize=12)
    ax3.set_title('数据分布直方图', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('figure/outlier_demo_iqr.png', dpi=300, bbox_inches='tight')
    print(f"✅ IQR方法演示图已保存: figure/outlier_demo_iqr.png")
    plt.close()
    
    # 打印统计信息
    print("\n" + "="*60)
    print("IQR 异常值检测结果")
    print("="*60)
    print(f"数据点总数: {len(data)}")
    print(f"Q1 (25%分位数): {q1:.2f}")
    print(f"Q2 (中位数): {q2:.2f}")
    print(f"Q3 (75%分位数): {q3:.2f}")
    print(f"IQR (四分位距): {iqr:.2f}")
    print(f"下界: {lower_bound:.2f}")
    print(f"上界: {upper_bound:.2f}")
    print(f"\n异常值数量: {outlier_count}")
    print(f"异常值比例: {outlier_ratio:.2f}%")
    print(f"低端异常值: {(data < lower_bound).sum()}个")
    print(f"高端异常值: {(data > upper_bound).sum()}个")


def demo_multi_node_outliers():
    """演示多节点异常值分析（类似实际数据集）"""
    
    np.random.seed(42)
    
    # 模拟10个节点，每个节点200个时间步
    T, N = 200, 10
    data = np.random.normal(50, 10, (T, N))
    
    # 为不同节点添加不同程度的异常值
    # 节点0-2: 低异常值 (2-3%)
    for node in range(3):
        n_outliers = int(T * 0.025)
        outlier_indices = np.random.choice(T, n_outliers, replace=False)
        data[outlier_indices, node] += np.random.choice([-40, 50], n_outliers)
    
    # 节点3-5: 中等异常值 (5-6%)
    for node in range(3, 6):
        n_outliers = int(T * 0.055)
        outlier_indices = np.random.choice(T, n_outliers, replace=False)
        data[outlier_indices, node] += np.random.choice([-40, 50], n_outliers)
    
    # 节点6-7: 高异常值 (10%)
    for node in range(6, 8):
        n_outliers = int(T * 0.1)
        outlier_indices = np.random.choice(T, n_outliers, replace=False)
        data[outlier_indices, node] += np.random.choice([-40, 50], n_outliers)
    
    # 节点8-9: 极少异常值 (<1%)
    # 保持原样，自然产生的极少异常值
    
    # 检测异常值
    q1 = np.percentile(data, 25, axis=0)
    q3 = np.percentile(data, 75, axis=0)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = (data < lower_bound) | (data > upper_bound)
    outlier_counts = outliers.sum(axis=0)
    total_outlier_ratio = outliers.sum() / outliers.size * 100
    
    # 创建可视化（模拟实际分析图）
    fig = plt.figure(figsize=(16, 10))
    
    # 子图1: 异常值数量柱状图（主要图）
    ax1 = plt.subplot(2, 2, 1)
    colors = ['green' if c < 10 else 'orange' if c < 15 else 'red' 
             for c in outlier_counts]
    bars = ax1.bar(range(N), outlier_counts, color=colors, 
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for i, (bar, count) in enumerate(zip(bars, outlier_counts)):
        height = bar.get_height()
        percentage = count / T * 100
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_xlabel('节点索引', fontsize=11)
    ax1.set_ylabel('异常值数量', fontsize=11)
    ax1.set_title('各节点异常值数量分布', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加整体统计信息
    ax1.text(0.02, 0.98, f'总异常值比例: {total_outlier_ratio:.2f}%', 
             transform=ax1.transAxes, fontsize=11, fontweight='bold',
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 子图2: 热图显示异常值位置
    ax2 = plt.subplot(2, 2, 2)
    im = ax2.imshow(outliers.T, aspect='auto', cmap='RdYlGn_r', 
                   interpolation='nearest')
    ax2.set_xlabel('时间步', fontsize=11)
    ax2.set_ylabel('节点索引', fontsize=11)
    ax2.set_title('异常值时空分布热图', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax2, label='异常值 (红=True)')
    
    # 子图3: 异常值比例对比
    ax3 = plt.subplot(2, 2, 3)
    percentages = outlier_counts / T * 100
    ax3.barh(range(N), percentages, color=colors, alpha=0.7, edgecolor='black')
    ax3.axvline(2, color='green', linestyle='--', linewidth=2, 
               alpha=0.5, label='优秀阈值 (2%)')
    ax3.axvline(5, color='orange', linestyle='--', linewidth=2, 
               alpha=0.5, label='正常阈值 (5%)')
    ax3.set_xlabel('异常值比例 (%)', fontsize=11)
    ax3.set_ylabel('节点索引', fontsize=11)
    ax3.set_title('各节点异常值比例', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 子图4: 示例节点时间序列
    ax4 = plt.subplot(2, 2, 4)
    
    # 选择一个高异常值节点和一个低异常值节点
    high_outlier_node = np.argmax(outlier_counts)
    low_outlier_node = np.argmin(outlier_counts)
    
    # 绘制高异常值节点
    time_steps = range(T)
    ts_high = data[:, high_outlier_node]
    outlier_mask_high = outliers[:, high_outlier_node]
    
    ax4.plot(time_steps, ts_high, 'b-', alpha=0.5, linewidth=1, 
            label=f'节点{high_outlier_node} (高异常)')
    ax4.scatter(np.where(outlier_mask_high)[0], 
               ts_high[outlier_mask_high],
               color='red', s=50, marker='X', zorder=5, label='异常值')
    
    # 绘制正常边界
    ax4.axhline(upper_bound[high_outlier_node], color='red', 
               linestyle=':', alpha=0.5)
    ax4.axhline(lower_bound[high_outlier_node], color='red', 
               linestyle=':', alpha=0.5)
    
    ax4.set_xlabel('时间步', fontsize=11)
    ax4.set_ylabel('流量值', fontsize=11)
    ax4.set_title('时间序列示例（含异常值标记）', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figure/outlier_demo_multi_node.png', dpi=300, bbox_inches='tight')
    print(f"✅ 多节点异常值分析图已保存: figure/outlier_demo_multi_node.png")
    plt.close()
    
    # 打印分析报告
    print("\n" + "="*60)
    print("多节点异常值分析报告")
    print("="*60)
    print(f"数据形状: ({T} 时间步, {N} 节点)")
    print(f"总异常值比例: {total_outlier_ratio:.2f}%")
    print(f"\n各节点异常值详情:")
    print("-" * 60)
    
    for i in range(N):
        count = outlier_counts[i]
        pct = count / T * 100
        status = "🟢 优秀" if pct < 2 else "🟡 正常" if pct < 5 else "🔴 偏高"
        print(f"  节点 {i}: {count:3d}个 ({pct:5.2f}%) {status}")
    
    print("\n" + "="*60)
    print("建议:")
    print("="*60)
    if total_outlier_ratio < 2:
        print("✅ 数据质量优秀，可以不使用去噪")
    elif total_outlier_ratio < 5:
        print("🟡 数据质量良好，建议使用轻量级去噪（conv）")
    else:
        print("🔴 异常值较多，建议使用强力去噪（attention）")
    
    print(f"\n高异常值节点（需重点关注）:")
    top_k = 3
    top_nodes = np.argsort(outlier_counts)[-top_k:][::-1]
    for i, node_id in enumerate(top_nodes):
        count = outlier_counts[node_id]
        pct = count / T * 100
        print(f"  {i+1}. 节点{node_id}: {count}个 ({pct:.2f}%)")


def create_interpretation_guide():
    """创建图表解读指南"""
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 情况1: 健康数据
    ax1 = axes[0]
    healthy_counts = [5, 4, 6, 5, 4, 5, 6, 4, 5, 6]
    bars1 = ax1.bar(range(10), healthy_counts, color='green', 
                    alpha=0.7, edgecolor='black')
    ax1.set_ylabel('异常值数量', fontsize=11)
    ax1.set_title('✅ 情况1: 健康数据 - 柱子低且均匀', 
                 fontsize=12, fontweight='bold', color='green')
    ax1.set_ylim(0, 30)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.text(0.5, 0.85, 
            '解读：\n• 所有节点异常值都很少\n• 分布均匀，无突出节点\n• 数据质量好\n• 建议：可以不去噪',
            transform=ax1.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # 情况2: 部分问题
    ax2 = axes[1]
    partial_counts = [5, 4, 24, 6, 5, 4, 22, 5, 6, 4]
    colors2 = ['green' if c < 10 else 'red' for c in partial_counts]
    bars2 = ax2.bar(range(10), partial_counts, color=colors2, 
                    alpha=0.7, edgecolor='black')
    ax2.set_ylabel('异常值数量', fontsize=11)
    ax2.set_title('⚠️ 情况2: 部分问题 - 个别节点突出', 
                 fontsize=12, fontweight='bold', color='orange')
    ax2.set_ylim(0, 30)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 标记问题节点
    problem_nodes = [i for i, c in enumerate(partial_counts) if c > 10]
    for node in problem_nodes:
        ax2.annotate('⚠️ 问题节点', 
                    xy=(node, partial_counts[node]), 
                    xytext=(node, partial_counts[node] + 3),
                    ha='center', fontsize=9, color='red',
                    arrowprops=dict(arrowstyle='->', color='red'))
    
    ax2.text(0.5, 0.85,
            f'解读：\n• 节点{problem_nodes}异常值明显偏高\n• 可能是特殊位置或传感器问题\n• 其他节点正常\n• 建议：使用轻量级去噪（conv）',
            transform=ax2.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # 情况3: 严重问题
    ax3 = axes[2]
    severe_counts = [20, 18, 25, 22, 19, 24, 21, 23, 20, 22]
    bars3 = ax3.bar(range(10), severe_counts, color='red', 
                    alpha=0.7, edgecolor='black')
    ax3.set_xlabel('节点索引', fontsize=11)
    ax3.set_ylabel('异常值数量', fontsize=11)
    ax3.set_title('🔴 情况3: 严重问题 - 普遍偏高', 
                 fontsize=12, fontweight='bold', color='red')
    ax3.set_ylim(0, 30)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.text(0.5, 0.85,
            '解读：\n• 所有节点异常值都很高\n• 系统性数据质量问题\n• 可能是传感器网络问题\n• 建议：使用强力去噪（attention）',
            transform=ax3.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('figure/outlier_interpretation_guide.png', dpi=300, bbox_inches='tight')
    print(f"✅ 图表解读指南已保存: figure/outlier_interpretation_guide.png")
    plt.close()


def main():
    """运行所有演示"""
    print("\n" + "📊 " + "="*58 + " 📊")
    print("📊  异常值分析可视化演示")
    print("📊 " + "="*58 + " 📊\n")
    
    import os
    os.makedirs('figure', exist_ok=True)
    
    print("\n1️⃣ 生成 IQR 方法原理演示...")
    print("-" * 60)
    demo_iqr_method()
    
    print("\n2️⃣ 生成多节点异常值分析演示...")
    print("-" * 60)
    demo_multi_node_outliers()
    
    print("\n3️⃣ 生成图表解读指南...")
    print("-" * 60)
    create_interpretation_guide()
    
    print("\n" + "="*60)
    print("✅ 所有演示图表生成完成！")
    print("="*60)
    print("\n📂 生成的文件:")
    print("   • figure/outlier_demo_iqr.png - IQR方法原理")
    print("   • figure/outlier_demo_multi_node.png - 多节点分析示例")
    print("   • figure/outlier_interpretation_guide.png - 图表解读指南")
    print("\n💡 建议:")
    print("   1. 查看 outlier_demo_iqr.png 理解IQR方法")
    print("   2. 查看 outlier_demo_multi_node.png 了解实际分析图")
    print("   3. 查看 outlier_interpretation_guide.png 学习如何解读")
    print("   4. 阅读 OUTLIER_ANALYSIS_GUIDE.md 获取详细说明\n")


if __name__ == '__main__':
    main()
