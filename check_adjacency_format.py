"""
检查邻接矩阵文件格式
"""
import pickle
import numpy as np
import os

datasets = ['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08']

print("\n" + "="*60)
print("检查邻接矩阵文件格式")
print("="*60)

for dataset in datasets:
    adj_path = f'datasets/{dataset}/adj_mx.pkl'
    
    if not os.path.exists(adj_path):
        print(f"\n❌ {dataset}: 文件不存在")
        continue
    
    print(f"\n📂 {dataset}:")
    print(f"   路径: {adj_path}")
    
    try:
        with open(adj_path, 'rb') as f:
            pkl_data = pickle.load(f, encoding='latin1')
        
        print(f"   ✅ 加载成功")
        print(f"   类型: {type(pkl_data)}")
        
        if isinstance(pkl_data, tuple):
            print(f"   元组长度: {len(pkl_data)}")
            for i, item in enumerate(pkl_data):
                print(f"   元素[{i}]类型: {type(item)}", end="")
                if isinstance(item, np.ndarray):
                    print(f" - 形状: {item.shape}, dtype: {item.dtype}")
                elif isinstance(item, list):
                    print(f" - 长度: {len(item)}")
                elif isinstance(item, dict):
                    print(f" - 键数量: {len(item)}")
                else:
                    print()
        
        elif isinstance(pkl_data, np.ndarray):
            print(f"   形状: {pkl_data.shape}")
            print(f"   dtype: {pkl_data.dtype}")
        
        # 尝试提取邻接矩阵
        if isinstance(pkl_data, tuple):
            # 找到最大的2D数组，很可能是邻接矩阵
            for i, item in enumerate(pkl_data):
                if isinstance(item, np.ndarray) and len(item.shape) == 2:
                    print(f"   💡 可能的邻接矩阵: 元素[{i}], 形状 {item.shape}")
        
    except Exception as e:
        print(f"   ❌ 加载失败: {e}")

print("\n" + "="*60)
