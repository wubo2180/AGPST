"""
快速测试不同空间编码器的脚本

用法:
    python test_spatial_encoders.py --encoder hybrid --epochs 10
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import time

# 导入模型
import sys
sys.path.append('.')
from basicts.mask.alternating_st import AlternatingSTModel


def create_dummy_data(batch_size=8, num_nodes=358, seq_len=12):
    """创建虚拟数据用于测试"""
    # 输入: (B, T, N, 1)
    history_data = torch.randn(batch_size, seq_len, num_nodes, 1)
    
    # 邻接矩阵: 简单的环形图 (每个节点连接前后节点)
    adj_mx = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        # 连接前一个节点
        adj_mx[i, (i - 1) % num_nodes] = 1
        # 连接后一个节点
        adj_mx[i, (i + 1) % num_nodes] = 1
        # 自环
        adj_mx[i, i] = 1
    
    # 归一化: D^(-1/2) A D^(-1/2)
    rowsum = np.array(adj_mx.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    adj_normalized = adj_mx.dot(d_mat_inv_sqrt).T.dot(d_mat_inv_sqrt)
    adj_tensor = torch.FloatTensor(adj_normalized)
    
    return history_data, adj_tensor


def test_encoder(encoder_type='hybrid', num_epochs=10, device='cuda'):
    """
    测试指定编码器的性能
    
    Args:
        encoder_type: 'transformer', 'gcn', 'chebnet', 'gat', 'hybrid'
        num_epochs: 训练轮数
        device: 'cuda' or 'cpu'
    """
    print(f"\n{'='*60}")
    print(f"测试空间编码器: {encoder_type.upper()}")
    print(f"{'='*60}\n")
    
    # 创建模型
    model = AlternatingSTModel(
        num_nodes=358,
        in_steps=12,
        out_steps=12,
        input_dim=1,
        embed_dim=64,  # 减小以加速测试
        num_heads=4,
        temporal_depth_1=2,
        spatial_depth_1=1 if encoder_type == 'hybrid' else 2,
        temporal_depth_2=2,
        spatial_depth_2=1 if encoder_type == 'hybrid' else 2,
        fusion_type='gated',
        dropout=0.05,
        use_denoising=True,
        denoise_type='conv',
        spatial_encoder_type=encoder_type,
        gnn_K=3  # ChebNet 的 K 值
    ).to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 模型参数:")
    print(f"   总参数: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    print(f"   参数大小: {total_params * 4 / 1024 / 1024:.2f} MB\n")
    
    # 创建虚拟数据
    history_data, adj_mx = create_dummy_data()
    history_data = history_data.to(device)
    adj_mx = adj_mx.to(device) if encoder_type != 'transformer' else None
    
    # 虚拟目标 (用于计算损失)
    target = torch.randn(8, 12, 358, 1).to(device)
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.L1Loss()
    
    # 训练循环
    print("🚀 开始训练...\n")
    times = []
    losses = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # 前向传播
        model.train()
        optimizer.zero_grad()
        
        if adj_mx is not None:
            prediction = model(history_data, adj_mx=adj_mx)
        else:
            prediction = model(history_data)
        
        # 计算损失
        loss = criterion(prediction, target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        epoch_time = time.time() - start_time
        times.append(epoch_time)
        losses.append(loss.item())
        
        print(f"Epoch [{epoch+1:2d}/{num_epochs}] | "
              f"Loss: {loss.item():.4f} | "
              f"Time: {epoch_time:.3f}s")
    
    # 统计结果
    avg_time = np.mean(times[1:])  # 跳过第一个 epoch (预热)
    final_loss = losses[-1]
    
    print(f"\n{'='*60}")
    print(f"📈 测试结果:")
    print(f"   编码器: {encoder_type.upper()}")
    print(f"   平均训练时间: {avg_time:.3f} s/epoch")
    print(f"   最终损失: {final_loss:.4f}")
    print(f"   总参数量: {total_params:,}")
    print(f"{'='*60}\n")
    
    return {
        'encoder': encoder_type,
        'avg_time': avg_time,
        'final_loss': final_loss,
        'params': total_params
    }


def main():
    parser = argparse.ArgumentParser(description='测试不同的空间编码器')
    parser.add_argument('--encoder', type=str, default='all',
                       choices=['all', 'transformer', 'gcn', 'chebnet', 'gat', 'hybrid'],
                       help='空间编码器类型')
    parser.add_argument('--epochs', type=int, default=10,
                       help='训练轮数')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='设备类型')
    
    args = parser.parse_args()
    
    # 检查 CUDA 可用性
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA 不可用,切换到 CPU")
        args.device = 'cpu'
    
    # 测试编码器
    if args.encoder == 'all':
        # 测试所有编码器
        encoders = ['transformer', 'gcn', 'chebnet', 'gat', 'hybrid']
        results = []
        
        for encoder in encoders:
            try:
                result = test_encoder(encoder, args.epochs, args.device)
                results.append(result)
            except Exception as e:
                print(f"❌ {encoder.upper()} 测试失败: {e}\n")
        
        # 打印对比结果
        if results:
            print("\n" + "="*80)
            print("🏆 综合对比结果")
            print("="*80)
            print(f"{'编码器':<15} {'平均时间 (s)':<15} {'最终损失':<15} {'参数量':<15}")
            print("-"*80)
            
            for r in results:
                print(f"{r['encoder']:<15} {r['avg_time']:<15.3f} "
                      f"{r['final_loss']:<15.4f} {r['params']:<15,}")
            
            print("="*80)
            
            # 找出最优
            fastest = min(results, key=lambda x: x['avg_time'])
            best_loss = min(results, key=lambda x: x['final_loss'])
            smallest = min(results, key=lambda x: x['params'])
            
            print(f"\n⚡ 最快: {fastest['encoder'].upper()} ({fastest['avg_time']:.3f} s/epoch)")
            print(f"🎯 损失最低: {best_loss['encoder'].upper()} ({best_loss['final_loss']:.4f})")
            print(f"💾 参数最少: {smallest['encoder'].upper()} ({smallest['params']:,})")
            print("="*80 + "\n")
    else:
        # 测试单个编码器
        test_encoder(args.encoder, args.epochs, args.device)


if __name__ == '__main__':
    main()
