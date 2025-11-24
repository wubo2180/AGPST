"""
消融实验结果分析和可视化脚本

用法:
    python analyze_ablation.py --dataset PEMS03 --plot
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


class AblationAnalyzer:
    """消融实验结果分析器"""
    
    def __init__(self, dataset='PEMS03', results_dir='checkpoints'):
        self.dataset = dataset
        self.results_dir = Path(results_dir) / dataset / 'ablation'
        self.experiments = [
            'full_model',
            'wo_temporal',
            'wo_spatial',
            'wo_stage2',
            'embedding_only',
            'wo_denoising'
        ]
        self.experiment_names = {
            'full_model': 'Full Model',
            'wo_temporal': 'w/o Temporal',
            'wo_spatial': 'w/o Spatial',
            'wo_stage2': 'w/o Stage 2',
            'embedding_only': 'Embedding Only',
            'wo_denoising': 'w/o Denoising'
        }
        
    def load_results(self):
        """从日志文件加载实验结果"""
        results = {}
        
        for exp in self.experiments:
            exp_dir = self.results_dir / exp
            
            # 尝试加载 JSON 结果
            json_file = exp_dir / 'results.json'
            if json_file.exists():
                with open(json_file, 'r') as f:
                    results[exp] = json.load(f)
            else:
                # 尝试从日志文件解析
                log_file = exp_dir / 'train.log'
                if log_file.exists():
                    results[exp] = self._parse_log(log_file)
                else:
                    print(f"⚠️  未找到 {exp} 的结果文件")
                    results[exp] = None
        
        return results
    
    def _parse_log(self, log_file):
        """从训练日志解析最优结果"""
        # 简化版: 假设日志格式
        # TODO: 根据实际日志格式调整
        best_mae = float('inf')
        best_rmse = float('inf')
        best_mape = float('inf')
        
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    if 'MAE:' in line:
                        # 解析 MAE
                        parts = line.split('MAE:')
                        if len(parts) > 1:
                            mae = float(parts[1].split()[0])
                            if mae < best_mae:
                                best_mae = mae
                    # 类似地解析 RMSE 和 MAPE
            
            return {
                'MAE': best_mae if best_mae != float('inf') else None,
                'RMSE': best_rmse if best_rmse != float('inf') else None,
                'MAPE': best_mape if best_mape != float('inf') else None
            }
        except Exception as e:
            print(f"❌ 解析日志失败: {e}")
            return None
    
    def create_results_table(self, results):
        """创建结果对比表"""
        data = []
        baseline_mae = results['full_model']['MAE'] if results['full_model'] else None
        
        for exp in self.experiments:
            if results[exp] is None:
                continue
            
            exp_name = self.experiment_names[exp]
            mae = results[exp].get('MAE')
            rmse = results[exp].get('RMSE')
            mape = results[exp].get('MAPE')
            
            if baseline_mae and mae:
                delta = ((mae - baseline_mae) / baseline_mae) * 100
                delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
            else:
                delta_str = "-"
            
            data.append({
                'Experiment': exp_name,
                'MAE': f"{mae:.2f}" if mae else "-",
                'RMSE': f"{rmse:.2f}" if rmse else "-",
                'MAPE': f"{mape:.2f}%" if mape else "-",
                'Δ MAE': delta_str
            })
        
        df = pd.DataFrame(data)
        return df
    
    def plot_bar_chart(self, results, metric='MAE', save_path='figure/ablation_bar.pdf'):
        """绘制柱状图对比"""
        plt.figure(figsize=(10, 6))
        
        # 准备数据
        experiments = []
        values = []
        colors = []
        
        for exp in self.experiments:
            if results[exp] and metric in results[exp] and results[exp][metric]:
                experiments.append(self.experiment_names[exp])
                values.append(results[exp][metric])
                
                # 根据实验类型设置颜色
                if exp == 'full_model':
                    colors.append('green')
                elif exp == 'embedding_only':
                    colors.append('darkred')
                elif 'wo_' in exp:
                    colors.append('red')
                else:
                    colors.append('orange')
        
        # 绘制柱状图
        bars = plt.bar(experiments, values, color=colors, alpha=0.7, edgecolor='black')
        
        # 添加基线
        if 'full_model' in results and results['full_model']:
            baseline = results['full_model'][metric]
            plt.axhline(y=baseline, color='blue', linestyle='--', 
                       linewidth=2, label=f'Baseline ({baseline:.2f})')
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 设置标签和标题
        plt.ylabel(metric, fontsize=14, fontweight='bold')
        plt.title(f'Ablation Study: {metric} Comparison on {self.dataset}', 
                 fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.legend(fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # 保存
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 柱状图已保存: {save_path}")
        plt.close()
    
    def plot_radar_chart(self, results, save_path='figure/ablation_radar.pdf'):
        """绘制雷达图 (多指标对比)"""
        # 准备数据
        categories = ['MAE', 'RMSE', 'MAPE']
        
        # 归一化: 所有指标转为 [0, 1],越小越好
        baseline = results['full_model']
        if not baseline:
            print("⚠️  无法绘制雷达图: 缺少 baseline 数据")
            return
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        # 为每个实验绘制
        for exp in ['full_model', 'wo_temporal', 'wo_spatial', 'wo_stage2']:
            if results[exp] is None:
                continue
            
            values = []
            for cat in categories:
                val = results[exp].get(cat)
                if val is None:
                    values.append(0)
                else:
                    # 归一化到 [0, 1]
                    baseline_val = baseline.get(cat, 1)
                    normalized = 1 - (val / baseline_val)  # 越小越好,转为越大越好
                    values.append(max(0, min(1, normalized)))
            
            values += values[:1]  # 闭合
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=self.experiment_names[exp])
            ax.fill(angles, values, alpha=0.15)
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_title('Ablation Study: Multi-metric Comparison', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
        ax.grid(True)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 雷达图已保存: {save_path}")
        plt.close()
    
    def generate_latex_table(self, results, save_path='results/ablation_table.tex'):
        """生成 LaTeX 表格代码"""
        baseline_mae = results['full_model']['MAE'] if results['full_model'] else None
        
        latex = r"""\begin{table}[t]
\centering
\caption{Ablation study results on """ + self.dataset + r""" dataset.}
\label{tab:ablation}
\begin{tabular}{lcccr}
\toprule
Configuration & MAE $\downarrow$ & RMSE $\downarrow$ & MAPE (\%) $\downarrow$ & $\Delta$ MAE \\
\midrule
"""
        
        for exp in self.experiments:
            if results[exp] is None:
                continue
            
            exp_name = self.experiment_names[exp]
            mae = results[exp].get('MAE')
            rmse = results[exp].get('RMSE')
            mape = results[exp].get('MAPE')
            
            if baseline_mae and mae:
                delta = ((mae - baseline_mae) / baseline_mae) * 100
                delta_str = f"+{delta:.1f}\%" if delta > 0 else f"{delta:.1f}\%"
            else:
                delta_str = "-"
            
            # 加粗最优值
            mae_str = f"\\textbf{{{mae:.2f}}}" if exp == 'full_model' else f"{mae:.2f}"
            
            latex += f"{exp_name} & {mae_str} & {rmse:.2f} & {mape:.1f} & {delta_str} \\\\\n"
        
        latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
        
        # 保存
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(latex)
        
        print(f"✅ LaTeX 表格已保存: {save_path}")
        return latex
    
    def print_summary(self, results):
        """打印结果摘要"""
        print("\n" + "="*60)
        print(f"消融实验结果摘要 - {self.dataset}")
        print("="*60 + "\n")
        
        df = self.create_results_table(results)
        print(df.to_string(index=False))
        
        # 统计分析
        print("\n" + "-"*60)
        print("关键发现:")
        print("-"*60)
        
        if results['full_model'] and results['wo_temporal']:
            baseline = results['full_model']['MAE']
            wo_temp = results['wo_temporal']['MAE']
            impact = ((wo_temp - baseline) / baseline) * 100
            print(f"1. 时间编码器贡献: {impact:.1f}% MAE 改进")
        
        if results['full_model'] and results['wo_spatial']:
            baseline = results['full_model']['MAE']
            wo_spat = results['wo_spatial']['MAE']
            impact = ((wo_spat - baseline) / baseline) * 100
            print(f"2. 空间编码器贡献: {impact:.1f}% MAE 改进")
        
        if results['full_model'] and results['wo_stage2']:
            baseline = results['full_model']['MAE']
            wo_stage2 = results['wo_stage2']['MAE']
            impact = ((wo_stage2 - baseline) / baseline) * 100
            print(f"3. 第二阶段贡献: {impact:.1f}% MAE 改进")
        
        if results['full_model'] and results['wo_denoising']:
            baseline = results['full_model']['MAE']
            wo_denoise = results['wo_denoising']['MAE']
            impact = ((wo_denoise - baseline) / baseline) * 100
            print(f"4. 去噪模块贡献: {impact:.1f}% MAE 改进")
        
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='分析消融实验结果')
    parser.add_argument('--dataset', type=str, default='PEMS03',
                       help='数据集名称')
    parser.add_argument('--results_dir', type=str, default='checkpoints',
                       help='结果目录')
    parser.add_argument('--plot', action='store_true',
                       help='生成可视化图表')
    parser.add_argument('--latex', action='store_true',
                       help='生成 LaTeX 表格')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = AblationAnalyzer(dataset=args.dataset, results_dir=args.results_dir)
    
    print(f"\n📊 加载消融实验结果...")
    results = analyzer.load_results()
    
    # 打印摘要
    analyzer.print_summary(results)
    
    # 生成图表
    if args.plot:
        print("\n📈 生成可视化图表...")
        analyzer.plot_bar_chart(results, metric='MAE', 
                               save_path=f'figure/ablation_{args.dataset}_MAE.pdf')
        analyzer.plot_bar_chart(results, metric='RMSE',
                               save_path=f'figure/ablation_{args.dataset}_RMSE.pdf')
        analyzer.plot_radar_chart(results,
                                 save_path=f'figure/ablation_{args.dataset}_radar.pdf')
    
    # 生成 LaTeX 表格
    if args.latex:
        print("\n📝 生成 LaTeX 表格...")
        latex_code = analyzer.generate_latex_table(results,
                                                   save_path=f'results/ablation_{args.dataset}.tex')
        print("\nLaTeX 代码:")
        print(latex_code)
    
    print("\n✅ 分析完成!")


if __name__ == '__main__':
    main()
