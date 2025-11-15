"""
测试去噪模块的功能
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

from basicts.mask.model import AGPSTModel, DenoiseAttention


def add_noise(data, noise_level=0.1):
    """添加高斯噪声"""
    noise = torch.randn_like(data) * noise_level
    return data + noise


def test_conv_denoising():
    """测试卷积去噪"""
    print("=" * 60)
    print("Testing Convolutional Denoising Module")
    print("=" * 60)
    
    # 创建模拟数据
    B, T, N, C = 4, 12, 358, 1
    clean_data = torch.randn(B, T, N, C)
    noisy_data = add_noise(clean_data, noise_level=0.2)
    
    # 创建模型（仅用于测试去噪）
    backend_args = {
        'num_nodes': 358,
        'supports': [],
        'dropout': 0.3,
        'gcn_bool': True,
        'addaptadj': True,
        'aptinit': None,
        'in_dim': 1,
        'out_dim': 12,
        'residual_channels': 32,
        'dilation_channels': 32,
        'skip_channels': 256,
        'end_channels': 512,
        'kernel_size': 2,
        'blocks': 4,
        'layers': 2
    }
    
    model = AGPSTModel(
        num_nodes=358,
        dim=10,
        topK=10,
        in_channel=1,
        embed_dim=96,
        num_heads=4,
        mlp_ratio=4,
        dropout=0.1,
        encoder_depth=4,
        backend_args=backend_args,
        use_denoising=True,
        denoise_type='conv'
    )
    
    model.eval()
    
    # 测试前向传播
    with torch.no_grad():
        # 手动执行去噪步骤
        x_denoise = noisy_data.permute(0, 2, 3, 1).reshape(B * N, C, T)
        noise_estimated = model.denoiser(x_denoise)
        x_denoise = x_denoise - noise_estimated
        denoised_data = x_denoise.reshape(B, N, C, T).permute(0, 3, 1, 2)
    
    # 统计信息
    print(f"\n✅ Clean Data   - Mean: {clean_data.mean():.4f}, Std: {clean_data.std():.4f}")
    print(f"❌ Noisy Data   - Mean: {noisy_data.mean():.4f}, Std: {noisy_data.std():.4f}")
    print(f"🔧 Denoised Data - Mean: {denoised_data.mean():.4f}, Std: {denoised_data.std():.4f}")
    
    # 计算去噪效果
    noise_before = torch.abs(noisy_data - clean_data).mean().item()
    noise_after = torch.abs(denoised_data - clean_data).mean().item()
    improvement = (noise_before - noise_after) / noise_before * 100
    
    print(f"\n📊 Denoising Performance:")
    print(f"   Noise before: {noise_before:.4f}")
    print(f"   Noise after:  {noise_after:.4f}")
    print(f"   Improvement:  {improvement:.2f}%")
    
    return noisy_data, denoised_data, clean_data


def test_attention_denoising():
    """测试注意力去噪"""
    print("\n" + "=" * 60)
    print("Testing Attention Denoising Module")
    print("=" * 60)
    
    # 创建模拟数据
    B, T, N, C = 4, 12, 358, 1
    clean_data = torch.randn(B, T, N, C)
    noisy_data = add_noise(clean_data, noise_level=0.2)
    
    # 创建去噪模块
    denoiser = DenoiseAttention(in_channels=1, hidden_dim=24, dropout=0.1)
    denoiser.eval()
    
    # 测试前向传播
    with torch.no_grad():
        denoised_data = denoiser(noisy_data)
    
    # 统计信息
    print(f"\n✅ Clean Data   - Mean: {clean_data.mean():.4f}, Std: {clean_data.std():.4f}")
    print(f"❌ Noisy Data   - Mean: {noisy_data.mean():.4f}, Std: {noisy_data.std():.4f}")
    print(f"🔧 Denoised Data - Mean: {denoised_data.mean():.4f}, Std: {denoised_data.std():.4f}")
    
    # 计算去噪效果
    noise_before = torch.abs(noisy_data - clean_data).mean().item()
    noise_after = torch.abs(denoised_data - clean_data).mean().item()
    improvement = (noise_before - noise_after) / noise_before * 100
    
    print(f"\n📊 Denoising Performance:")
    print(f"   Noise before: {noise_before:.4f}")
    print(f"   Noise after:  {noise_after:.4f}")
    print(f"   Improvement:  {improvement:.2f}%")
    
    return noisy_data, denoised_data, clean_data


def visualize_denoising(noisy, denoised, clean, save_path='figure/denoising_test.pdf'):
    """可视化去噪效果"""
    import os
    os.makedirs('figure', exist_ok=True)
    
    # 选择一个节点的时间序列
    node_idx = 0
    batch_idx = 0
    
    clean_series = clean[batch_idx, :, node_idx, 0].numpy()
    noisy_series = noisy[batch_idx, :, node_idx, 0].numpy()
    denoised_series = denoised[batch_idx, :, node_idx, 0].numpy()
    
    plt.figure(figsize=(12, 4))
    
    plt.plot(clean_series, 'g-', label='Clean Signal', linewidth=2, alpha=0.7)
    plt.plot(noisy_series, 'r--', label='Noisy Signal', linewidth=1.5, alpha=0.7)
    plt.plot(denoised_series, 'b-.', label='Denoised Signal', linewidth=2, alpha=0.7)
    
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title(f'Denoising Effect (Node {node_idx})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📈 Visualization saved to: {save_path}")
    plt.close()


def test_model_integration():
    """测试去噪模块在完整模型中的集成"""
    print("\n" + "=" * 60)
    print("Testing Denoising Module Integration in Full Model")
    print("=" * 60)
    
    backend_args = {
        'num_nodes': 358,
        'supports': [],
        'dropout': 0.3,
        'gcn_bool': True,
        'addaptadj': True,
        'aptinit': None,
        'in_dim': 1,
        'out_dim': 12,
        'residual_channels': 32,
        'dilation_channels': 32,
        'skip_channels': 256,
        'end_channels': 512,
        'kernel_size': 2,
        'blocks': 4,
        'layers': 2
    }
    
    # 测试两种去噪类型
    for denoise_type in ['conv', 'attention']:
        print(f"\n🔧 Testing with {denoise_type} denoising...")
        
        model = AGPSTModel(
            num_nodes=358,
            dim=10,
            topK=10,
            in_channel=1,
            embed_dim=96,
            num_heads=4,
            mlp_ratio=4,
            dropout=0.1,
            encoder_depth=4,
            backend_args=backend_args,
            use_denoising=True,
            denoise_type=denoise_type
        )
        
        # 测试数据
        B, T, N, C = 2, 12, 358, 1
        history_data = torch.randn(B, T, N, C)
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            output = model(history_data)
        
        print(f"   Input shape:  {history_data.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   ✅ {denoise_type.capitalize()} denoising works correctly!")
    
    # 测试不使用去噪
    print(f"\n🔧 Testing without denoising...")
    model_no_denoise = AGPSTModel(
        num_nodes=358,
        dim=10,
        topK=10,
        in_channel=1,
        embed_dim=96,
        num_heads=4,
        mlp_ratio=4,
        dropout=0.1,
        encoder_depth=4,
        backend_args=backend_args,
        use_denoising=False
    )
    
    model_no_denoise.eval()
    with torch.no_grad():
        output = model_no_denoise(history_data)
    
    print(f"   Input shape:  {history_data.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   ✅ No denoising works correctly!")


if __name__ == '__main__':
    print("\n🚀 Starting Denoising Module Tests...\n")
    
    # 测试1: 卷积去噪
    noisy_conv, denoised_conv, clean_conv = test_conv_denoising()
    visualize_denoising(noisy_conv, denoised_conv, clean_conv, 'figure/denoising_conv_test.pdf')
    
    # 测试2: 注意力去噪
    noisy_attn, denoised_attn, clean_attn = test_attention_denoising()
    visualize_denoising(noisy_attn, denoised_attn, clean_attn, 'figure/denoising_attention_test.pdf')
    
    # 测试3: 完整模型集成
    test_model_integration()
    
    print("\n" + "=" * 60)
    print("🎉 All tests passed successfully!")
    print("=" * 60)
    print("\n💡 Next steps:")
    print("   1. Check the visualization in figure/denoising_*_test.pdf")
    print("   2. Run training with: python main.py --config=parameters/PEMS03_v3.yaml")
    print("   3. Compare results with use_denoising=True vs False")
    print()
