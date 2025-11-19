"""
解析训练日志文件并转换为CSV格式
"""
import re
import csv

def parse_mae_file(input_file, output_file):
    """
    解析MAE日志文件并保存为CSV
    
    Args:
        input_file: 输入日志文件路径
        output_file: 输出CSV文件路径
    """
    results = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    epoch = 0
    for line in lines:
        line = line.strip()
        
        # 匹配 Epoch 标记
        epoch_match = re.match(r'={12} Epoch (\d+)/\d+ ={12}', line)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            continue
        
        # 匹配指标行
        # 格式: "Overall - Test MAE: 23.1919, Test RMSE: 37.0716, Test MAPE: 0.4243"
        # 或: "Test MAE: 14.6546, Test RMSE: 25.4121, Test MAPE: 0.1472"
        metric_match = re.search(
            r'(?:Overall - )?Test MAE: ([\d.]+), Test RMSE: ([\d.]+), Test MAPE: ([\d.]+)',
            line
        )
        
        if metric_match:
            mae = float(metric_match.group(1))
            rmse = float(metric_match.group(2))
            mape = float(metric_match.group(3))
            
            results.append({
                'Epoch': epoch,
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape
            })
    
    # 写入CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        if results:
            fieldnames = ['Epoch', 'MAE', 'RMSE', 'MAPE']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(results)
    
    print(f"✅ 成功解析 {len(results)} 条记录")
    print(f"📁 已保存到: {output_file}")
    
    # 打印统计信息
    if results:
        mae_values = [r['MAE'] for r in results]
        rmse_values = [r['RMSE'] for r in results]
        mape_values = [r['MAPE'] for r in results]
        
        print(f"\n📊 统计信息:")
        print(f"   MAE  - Min: {min(mae_values):.4f}, Max: {max(mae_values):.4f}, Avg: {sum(mae_values)/len(mae_values):.4f}")
        print(f"   RMSE - Min: {min(rmse_values):.4f}, Max: {max(rmse_values):.4f}, Avg: {sum(rmse_values)/len(rmse_values):.4f}")
        print(f"   MAPE - Min: {min(mape_values):.4f}, Max: {max(mape_values):.4f}, Avg: {sum(mape_values)/len(mape_values):.4f}")
        
        # 找到最佳epoch
        best_mae_idx = mae_values.index(min(mae_values))
        best_epoch = results[best_mae_idx]['Epoch']
        print(f"\n🏆 最佳性能 (Epoch {best_epoch}):")
        print(f"   MAE: {results[best_mae_idx]['MAE']:.4f}")
        print(f"   RMSE: {results[best_mae_idx]['RMSE']:.4f}")
        print(f"   MAPE: {results[best_mae_idx]['MAPE']:.4f}")

if __name__ == "__main__":
    input_file = "mae"
    output_file = "training_results.csv"
    
    print("=" * 70)
    print("解析训练日志 -> CSV")
    print("=" * 70)
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print("-" * 70)
    
    parse_mae_file(input_file, output_file)
    
    print("=" * 70)
    print("✅ 完成!")
    print("=" * 70)
