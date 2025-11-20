"""
测试交替时空架构 (Alternating Spatio-Temporal Architecture)

快速验证新架构是否能正常运行
"""

import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basicts.mask.alternating_st import (
    AlternatingSTModel,
    TemporalEncoder,
    SpatialEncoder,
    FusionLayer,
    STDecoder
)


def test_temporal_encoder():
    """测试时间编码器"""
    print("\n" + "="*60)
    print("测试 TemporalEncoder")
    print("="*60)
    
    B, N, T, D = 4, 358, 12, 96
    
    encoder = TemporalEncoder(
        d_model=D,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    )
    
    x = torch.randn(B, N, T, D)
    out = encoder(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    print(f"✅ TemporalEncoder 测试通过!")
    
    assert out.shape == x.shape, "输出形状不匹配!"
    return True


def test_spatial_encoder():
    """测试空间编码器"""
    print("\n" + "="*60)
    print("测试 SpatialEncoder")
    print("="*60)
    
    B, N, T, D = 4, 358, 12, 96
    
    encoder = SpatialEncoder(
        num_nodes=N,
        d_model=D,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    )
    
    x = torch.randn(B, N, T, D)
    out = encoder(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    print(f"✅ SpatialEncoder 测试通过!")
    
    assert out.shape == x.shape, "输出形状不匹配!"
    return True


def test_fusion_layer():
    """测试融合层"""
    print("\n" + "="*60)
    print("测试 FusionLayer")
    print("="*60)
    
    B, N, T, D = 4, 358, 12, 96
    
    for fusion_type in ['concat', 'gated', 'cross_attn']:
        print(f"\n测试融合类型: {fusion_type}")
        
        fusion = FusionLayer(
            d_model=D,
            fusion_type=fusion_type,
            dropout=0.1
        )
        
        temporal_feat = torch.randn(B, N, T, D)
        spatial_feat = torch.randn(B, N, T, D)
        
        fused = fusion(temporal_feat, spatial_feat)
        
        print(f"  时间特征形状: {temporal_feat.shape}")
        print(f"  空间特征形状: {spatial_feat.shape}")
        print(f"  融合后形状: {fused.shape}")
        
        assert fused.shape == (B, N, T, D), f"{fusion_type} 输出形状不匹配!"
        print(f"  ✅ {fusion_type} 融合测试通过!")
    
    return True


def test_st_decoder():
    """测试时空解码器"""
    print("\n" + "="*60)
    print("测试 STDecoder")
    print("="*60)
    
    B, N, T, D = 4, 358, 12, 96
    
    decoder = STDecoder(
        d_model=D,
        num_heads=4,
        dropout=0.1
    )
    
    fused_features = torch.randn(B, N, T, D)
    temporal_comp, spatial_comp = decoder(fused_features)
    
    print(f"输入形状: {fused_features.shape}")
    print(f"时间分量形状: {temporal_comp.shape}")
    print(f"空间分量形状: {spatial_comp.shape}")
    print(f"✅ STDecoder 测试通过!")
    
    assert temporal_comp.shape == (B, N, T, D), "时间分量形状不匹配!"
    assert spatial_comp.shape == (B, N, T, D), "空间分量形状不匹配!"
    return True


def test_full_model():
    """测试完整模型"""
    print("\n" + "="*60)
    print("测试 AlternatingSTModel (完整架构)")
    print("="*60)
    
    # PEMS03 配置
    B = 4
    N = 358
    T_in = 12
    T_out = 12
    
    model = AlternatingSTModel(
        num_nodes=N,
        in_steps=T_in,
        out_steps=T_out,
        input_dim=1,
        embed_dim=96,
        num_heads=4,
        temporal_depth_1=2,
        spatial_depth_1=2,
        temporal_depth_2=2,
        spatial_depth_2=2,
        fusion_type='gated',
        dropout=0.05,
        use_denoising=True
    )
    
    # 输入格式: (B, T, N, 1) - 标准格式
    x = torch.randn(B, T_in, N, 1)
    
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    with torch.no_grad():
        out = model(x)
    
    print(f"输出形状: {out.shape}")
    print(f"预期输出形状: ({B}, {T_out}, {N}, 1)")
    
    assert out.shape == (B, T_out, N, 1), f"输出形状不匹配! 预期 ({B}, {T_out}, {N}, 1), 实际 {out.shape}"
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型参数统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  参数大小: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    print(f"\n✅ AlternatingSTModel 完整测试通过!")
    print(f"🎉 恭喜! 新架构可以正常运行!")
    
    return True


def test_forward_backward():
    """测试前向+反向传播"""
    print("\n" + "="*60)
    print("测试前向+反向传播 (确保梯度流通畅)")
    print("="*60)
    
    B, N, T_in, T_out = 2, 358, 12, 12
    
    model = AlternatingSTModel(
        num_nodes=N,
        in_steps=T_in,
        out_steps=T_out,
        input_dim=1,
        embed_dim=64,  # 减小维度以加速测试
        num_heads=4,
        temporal_depth_1=1,
        spatial_depth_1=1,
        temporal_depth_2=1,
        spatial_depth_2=1,
        fusion_type='gated',
        dropout=0.1,
        use_denoising=True
    )
    
    x = torch.randn(B, T_in, N, 1)
    target = torch.randn(B, T_out, N, 1)
    
    # 前向传播
    output = model(x)
    
    # 计算损失
    loss = torch.nn.functional.mse_loss(output, target)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"损失值: {loss.item():.4f}")
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            grad_norm = param.grad.norm().item()
            print(f"  {name}: grad_norm = {grad_norm:.6f}")
            if grad_norm == 0:
                print(f"    ⚠️ 警告: {name} 梯度为0!")
    
    assert has_grad, "没有参数有梯度!"
    print(f"\n✅ 前向+反向传播测试通过!")
    print(f"✅ 梯度流正常!")
    
    return True


def main():
    """运行所有测试"""
    print("\n" + "🚀"*30)
    print("交替时空架构 (Alternating ST) 单元测试")
    print("🚀"*30)
    
    tests = [
        ("TemporalEncoder", test_temporal_encoder),
        ("SpatialEncoder", test_spatial_encoder),
        ("FusionLayer", test_fusion_layer),
        ("STDecoder", test_st_decoder),
        ("完整模型", test_full_model),
        ("前向+反向传播", test_forward_backward),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n❌ {name} 测试失败!")
            print(f"错误: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    print(f"✅ 通过: {passed}/{len(tests)}")
    print(f"❌ 失败: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉🎉🎉 所有测试通过! 架构准备就绪! 🎉🎉🎉")
        print("\n下一步:")
        print("  1. 运行训练: python main.py --cfg parameters/PEMS03_alternating.yaml")
        print("  2. 监控性能: 目标 MAE < 15")
        print("  3. 对比 baseline: MAE 14.57")
    else:
        print("\n⚠️ 有测试失败,请修复后再训练!")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
