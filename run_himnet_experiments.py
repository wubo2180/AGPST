"""
HimNet 启发改进 - 快速验证脚本

对比实验:
1. Phase 1 Baseline
2. Phase 1 Optimized
3. HimNet Version

目标: 验证 HimNet 设计理念的有效性
"""

import subprocess
import time
from pathlib import Path

# 实验配置
EXPERIMENTS = [
    {
        'name': 'Phase1_Baseline',
        'config': 'parameters/PEMS03.yaml',
        'description': 'Original alternating architecture (baseline)',
        'expected_mae': 5.4
    },
    {
        'name': 'Phase1_Optimized',
        'config': 'parameters/PEMS03_alternating_optimized.yaml',
        'description': 'Asymmetric depth + cross-attention fusion',
        'expected_mae': 4.8
    },
    {
        'name': 'HimNet_Inspired',
        'config': 'parameters/PEMS03_alternating_himnet.yaml',
        'description': 'Node heterogeneity + GCN hybrid + Huber loss',
        'expected_mae': 4.5
    }
]

EPOCHS = 10  # 快速验证
DEVICE = 'cuda'


def run_experiment(exp_config):
    """运行单个实验"""
    print(f"\n{'='*80}")
    print(f"🚀 Running: {exp_config['name']}")
    print(f"📝 Description: {exp_config['description']}")
    print(f"🎯 Expected MAE: {exp_config['expected_mae']}")
    print(f"⚙️  Config: {exp_config['config']}")
    print(f"{'='*80}\n")
    
    # 构建命令
    cmd = [
        'python', 'main.py',
        '--cfg', exp_config['config'],
        '--epochs', str(EPOCHS)
    ]
    
    # 运行实验
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1小时超时
        )
        
        elapsed_time = time.time() - start_time
        
        # 解析结果
        if result.returncode == 0:
            print(f"✅ {exp_config['name']} completed successfully!")
            print(f"⏱️  Time: {elapsed_time/60:.2f} minutes")
            
            # 尝试从输出中提取 MAE
            output = result.stdout
            if 'MAE' in output:
                # 简单的 MAE 提取 (需要根据实际输出调整)
                lines = output.split('\n')
                for line in lines:
                    if 'MAE' in line and '@10' in line:
                        print(f"📊 Result: {line.strip()}")
        else:
            print(f"❌ {exp_config['name']} failed!")
            print(f"Error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {exp_config['name']} timed out after 1 hour!")
    except Exception as e:
        print(f"💥 {exp_config['name']} crashed: {str(e)}")


def main():
    """主函数"""
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║     HimNet 启发改进 - 快速验证实验                                ║
    ║                                                                 ║
    ║  对比 3 个版本:                                                  ║
    ║  1. Phase 1 Baseline (原始交替架构)                               ║
    ║  2. Phase 1 Optimized (非对称深度 + 交叉注意力)                     ║
    ║  3. HimNet Inspired (节点异质性 + GCN混合 + Huber损失)             ║
    ║                                                                 ║
    ║  数据集: PEMS03 (358 nodes)                                      ║
    ║  训练轮数: 10 epochs (快速验证)                                    ║
    ║  预期总时间: 1-2 小时                                             ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    # 检查配置文件是否存在
    print("\n📋 Checking configuration files...")
    all_exist = True
    for exp in EXPERIMENTS:
        config_path = Path(exp['config'])
        if config_path.exists():
            print(f"  ✅ {exp['config']}")
        else:
            print(f"  ❌ {exp['config']} NOT FOUND!")
            all_exist = False
    
    if not all_exist:
        print("\n❌ Some configuration files are missing. Please check!")
        return
    
    # 运行实验
    results = []
    total_start = time.time()
    
    for exp in EXPERIMENTS:
        result = run_experiment(exp)
        results.append(result)
        
        # 实验间短暂休息
        print("\n⏸️  Waiting 10 seconds before next experiment...\n")
        time.sleep(10)
    
    total_time = time.time() - total_start
    
    # 总结
    print(f"\n{'='*80}")
    print("📊 EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"⏱️  Total Time: {total_time/60:.2f} minutes")
    print(f"\n🎯 Expected Improvements:")
    print(f"  Phase 1 Baseline → Optimized: ~11% (5.4 → 4.8)")
    print(f"  Phase 1 Baseline → HimNet: ~17% (5.4 → 4.5)")
    print(f"  Phase 1 Optimized → HimNet: ~6% (4.8 → 4.5)")
    print(f"\n📈 Check detailed results in:")
    print(f"  - checkpoints/PEMS03_AlternatingST/")
    print(f"  - checkpoints/PEMS03_AlternatingST_Optimized/")
    print(f"  - checkpoints/PEMS03_AlternatingST_HimNet/")
    print(f"{'='*80}\n")
    
    print("""
    📝 Next Steps:
    1. If HimNet version performs best (MAE < 4.5):
       → Run full 150-epoch training
       → Test on PEMS04/07/08
       → Consider adding Kalman filter post-processing
    
    2. If Optimized version is sufficient (MAE < 4.8):
       → Simpler architecture, less parameters
       → Easier to explain in paper
    
    3. If both fail to improve over baseline:
       → Stick with Phase 1 baseline
       → Write paper emphasizing simplicity
       → Use failed experiments as ablation studies
    """)


if __name__ == '__main__':
    main()
