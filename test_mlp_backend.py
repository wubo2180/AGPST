"""
测试 MLP 后端迁移后的模型
验证模型可以正常初始化和前向传播
"""

import torch
import sys
sys.path.append('.')

from basicts.mask.model import AGPSTModel

def test_model_initialization():
    """测试模型初始化"""
    print("=" * 60)
    print("测试 1: 模型初始化")
    print("=" * 60)
    
    model = AGPSTModel(
        num_nodes=358,
        dim=40,
        topK=10,
        in_channel=1,
        embed_dim=96,
        num_heads=4,
        mlp_ratio=4,
        dropout=0.1,
        encoder_depth=4,
        use_denoising=True,
        denoise_type='conv',
        use_advanced_graph=True,
        graph_heads=4,
        pred_len=12  # 新参数
    )
    
    print("✅ 模型初始化成功！")
    print(f"   - 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - 可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return model

def test_forward_pass(model):
    """测试前向传播"""
    print("\n" + "=" * 60)
    print("测试 2: 前向传播")
    print("=" * 60)
    
    # 创建测试输入
    batch_size = 4
    seq_len = 12
    num_nodes = 358
    in_channels = 1
    
    history_data = torch.randn(batch_size, seq_len, num_nodes, in_channels)
    print(f"输入形状: {history_data.shape}")
    
    # 前向传播
    with torch.no_grad():
        prediction = model(history_data)
    
    print(f"输出形状: {prediction.shape}")
    
    # 验证输出形状
    expected_shape = (batch_size, 12, num_nodes, 1)
    assert prediction.shape == expected_shape, f"期望形状 {expected_shape}, 得到 {prediction.shape}"
    
    print(f"✅ 前向传播成功！")
    print(f"   - 输入: (B={batch_size}, T={seq_len}, N={num_nodes}, C={in_channels})")
    print(f"   - 输出: (B={batch_size}, pred_len=12, N={num_nodes}, C=1)")
    print(f"   - 预测值范围: [{prediction.min():.4f}, {prediction.max():.4f}]")
    
    return prediction

def test_different_denoise_modes(model_params):
    """测试不同去噪模式"""
    print("\n" + "=" * 60)
    print("测试 3: 不同去噪模式")
    print("=" * 60)
    
    modes = ['conv', 'attention']
    
    for mode in modes:
        print(f"\n测试去噪模式: {mode}")
        params = model_params.copy()
        params['denoise_type'] = mode
        
        model = AGPSTModel(**params)
        history_data = torch.randn(2, 12, 358, 1)
        
        with torch.no_grad():
            prediction = model(history_data)
        
        print(f"  ✅ {mode} 模式正常工作")
        print(f"     输出形状: {prediction.shape}")

def test_without_denoising(model_params):
    """测试无去噪模式"""
    print("\n" + "=" * 60)
    print("测试 4: 无去噪模式")
    print("=" * 60)
    
    params = model_params.copy()
    params['use_denoising'] = False
    
    model = AGPSTModel(**params)
    history_data = torch.randn(2, 12, 358, 1)
    
    with torch.no_grad():
        prediction = model(history_data)
    
    print(f"✅ 无去噪模式正常工作")
    print(f"   输出形状: {prediction.shape}")

def test_gradient_flow():
    """测试梯度流"""
    print("\n" + "=" * 60)
    print("测试 5: 梯度流")
    print("=" * 60)
    
    model = AGPSTModel(
        num_nodes=358,
        dim=40,
        topK=10,
        in_channel=1,
        embed_dim=96,
        num_heads=4,
        mlp_ratio=4,
        dropout=0.1,
        encoder_depth=4,
        use_denoising=True,
        denoise_type='conv',
        use_advanced_graph=True,
        graph_heads=4,
        pred_len=12
    )
    
    # 创建测试数据
    history_data = torch.randn(2, 12, 358, 1)
    target = torch.randn(2, 12, 358, 1)
    
    # 前向传播
    prediction = model(history_data)
    
    # 计算损失
    loss = torch.nn.functional.mse_loss(prediction, target)
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    has_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total_params = sum(1 for p in model.parameters() if p.requires_grad)
    
    print(f"✅ 梯度流测试通过")
    print(f"   - 损失值: {loss.item():.6f}")
    print(f"   - 有梯度的参数: {has_grad}/{total_params}")
    print(f"   - 平均梯度范数: {sum(p.grad.norm() for p in model.parameters() if p.grad is not None) / has_grad:.6f}")

def test_different_pred_lengths():
    """测试不同预测长度"""
    print("\n" + "=" * 60)
    print("测试 6: 不同预测长度")
    print("=" * 60)
    
    pred_lengths = [3, 6, 12, 24]
    
    for pred_len in pred_lengths:
        print(f"\n测试预测长度: {pred_len}")
        
        model = AGPSTModel(
            num_nodes=358,
            dim=40,
            topK=10,
            in_channel=1,
            embed_dim=96,
            num_heads=4,
            mlp_ratio=4,
            dropout=0.1,
            encoder_depth=4,
            pred_len=pred_len  # 不同的预测长度
        )
        
        history_data = torch.randn(2, 12, 358, 1)
        
        with torch.no_grad():
            prediction = model(history_data)
        
        expected_shape = (2, pred_len, 358, 1)
        assert prediction.shape == expected_shape, f"期望 {expected_shape}, 得到 {prediction.shape}"
        
        print(f"  ✅ pred_len={pred_len} 正常工作")
        print(f"     输出形状: {prediction.shape}")

def main():
    print("\n" + "🚀" * 30)
    print("MLP 后端迁移验证测试")
    print("🚀" * 30 + "\n")
    
    # 基础参数
    model_params = {
        'num_nodes': 358,
        'dim': 40,
        'topK': 10,
        'in_channel': 1,
        'embed_dim': 96,
        'num_heads': 4,
        'mlp_ratio': 4,
        'dropout': 0.1,
        'encoder_depth': 4,
        'use_denoising': True,
        'denoise_type': 'conv',
        'use_advanced_graph': True,
        'graph_heads': 4,
        'pred_len': 12
    }
    
    try:
        # 测试 1: 初始化
        model = test_model_initialization()
        
        # 测试 2: 前向传播
        test_forward_pass(model)
        
        # 测试 3: 不同去噪模式
        test_different_denoise_modes(model_params)
        
        # 测试 4: 无去噪模式
        test_without_denoising(model_params)
        
        # 测试 5: 梯度流
        test_gradient_flow()
        
        # 测试 6: 不同预测长度
        test_different_pred_lengths()
        
        # 总结
        print("\n" + "=" * 60)
        print("🎉 所有测试通过！")
        print("=" * 60)
        print("\n✅ MLP 后端迁移成功！")
        print("✅ 模型可以正常初始化和前向传播")
        print("✅ 支持不同的去噪模式和预测长度")
        print("✅ 梯度流正常，可以进行训练")
        print("\n下一步: 运行完整的训练实验")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ 测试失败！")
        print("=" * 60)
        print(f"\n错误信息: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
