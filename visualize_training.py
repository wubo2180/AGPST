"""
可视化训练结果
绘制MAE、RMSE、MAPE随epoch变化的曲线
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

def plot_training_results(csv_file, output_file='training_curves.png'):
    """
    绘制训练曲线
    
    Args:
        csv_file: CSV文件路径
        output_file: 输出图片路径
    """
    # 读取数据
    df = pd.read_csv(csv_file)
    
    print(f"📊 读取 {len(df)} 条记录")
    
    # 创建图表
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Training Results - AGPST Model', fontsize=16, fontweight='bold')
    
    # 绘制MAE
    axes[0].plot(df['Epoch'], df['MAE'], 'b-', linewidth=2, label='MAE')
    axes[0].axhline(y=df['MAE'].min(), color='r', linestyle='--', alpha=0.5, label=f'Min: {df["MAE"].min():.4f}')
    axes[0].set_ylabel('MAE', fontsize=12, fontweight='bold')
    axes[0].set_title('Mean Absolute Error (MAE)', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # 绘制RMSE
    axes[1].plot(df['Epoch'], df['RMSE'], 'g-', linewidth=2, label='RMSE')
    axes[1].axhline(y=df['RMSE'].min(), color='r', linestyle='--', alpha=0.5, label=f'Min: {df["RMSE"].min():.4f}')
    axes[1].set_ylabel('RMSE', fontsize=12, fontweight='bold')
    axes[1].set_title('Root Mean Square Error (RMSE)', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # 绘制MAPE
    axes[2].plot(df['Epoch'], df['MAPE'], 'orange', linewidth=2, label='MAPE')
    axes[2].axhline(y=df['MAPE'].min(), color='r', linestyle='--', alpha=0.5, label=f'Min: {df["MAPE"].min():.4f}')
    axes[2].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('MAPE', fontsize=12, fontweight='bold')
    axes[2].set_title('Mean Absolute Percentage Error (MAPE)', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存: {output_file}")
    
    # 打印关键信息
    print("\n" + "="*70)
    print("📈 训练结果摘要")
    print("="*70)
    
    best_mae_idx = df['MAE'].idxmin()
    best_rmse_idx = df['RMSE'].idxmin()
    best_mape_idx = df['MAPE'].idxmin()
    
    print(f"\n🏆 最佳 MAE (Epoch {df.loc[best_mae_idx, 'Epoch']}):")
    print(f"   MAE:  {df.loc[best_mae_idx, 'MAE']:.4f}")
    print(f"   RMSE: {df.loc[best_mae_idx, 'RMSE']:.4f}")
    print(f"   MAPE: {df.loc[best_mae_idx, 'MAPE']:.4f}")
    
    print(f"\n🏆 最佳 RMSE (Epoch {df.loc[best_rmse_idx, 'Epoch']}):")
    print(f"   MAE:  {df.loc[best_rmse_idx, 'MAE']:.4f}")
    print(f"   RMSE: {df.loc[best_rmse_idx, 'RMSE']:.4f}")
    print(f"   MAPE: {df.loc[best_rmse_idx, 'MAPE']:.4f}")
    
    print(f"\n🏆 最佳 MAPE (Epoch {df.loc[best_mape_idx, 'Epoch']}):")
    print(f"   MAE:  {df.loc[best_mape_idx, 'MAE']:.4f}")
    print(f"   RMSE: {df.loc[best_mape_idx, 'RMSE']:.4f}")
    print(f"   MAPE: {df.loc[best_mape_idx, 'MAPE']:.4f}")
    
    # 最后10个epoch的平均性能
    last_10 = df.tail(10)
    print(f"\n📊 最后10个Epoch的平均性能:")
    print(f"   MAE:  {last_10['MAE'].mean():.4f} ± {last_10['MAE'].std():.4f}")
    print(f"   RMSE: {last_10['RMSE'].mean():.4f} ± {last_10['RMSE'].std():.4f}")
    print(f"   MAPE: {last_10['MAPE'].mean():.4f} ± {last_10['MAPE'].std():.4f}")
    
    # 性能改进
    initial_mae = df.loc[0, 'MAE']
    final_mae = df.loc[len(df)-1, 'MAE']
    improvement = (initial_mae - final_mae) / initial_mae * 100
    
    print(f"\n📈 整体改进:")
    print(f"   初始 MAE: {initial_mae:.4f}")
    print(f"   最终 MAE: {final_mae:.4f}")
    print(f"   改进率: {improvement:.2f}%")
    
    print("="*70)

if __name__ == "__main__":
    try:
        plot_training_results('training_results.csv', 'training_curves.png')
        print("\n✅ 可视化完成!")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n提示: 需要安装 pandas 和 matplotlib")
        print("   pip install pandas matplotlib")
