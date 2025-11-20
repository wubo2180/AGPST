"""
性能诊断脚本
检查模型输出、梯度、学习率等关键指标
"""

import torch
import sys
sys.path.append('.')

from basicts.mask.model import AGPSTModel

def diagnose_model():
    print("=" * 60)
    print("性能诊断 - Encoder-Decoder 架构")
    print("=" * 60)
    
    # 模型配置
    config = {
        'num_nodes': 358,
        'dim': 10,
        'topK': 10,
        'in_channel': 1,
        'embed_dim': 96,
        'num_heads': 4,
        'mlp_ratio': 4,
        'dropout': 0.1,
        'encoder_depth': 4,
        'decoder_depth': 1,  # ⭐ 修复后: 1层
        'use_denoising': False,
        'denoise_type': 'conv',
        'use_advanced_graph': True,
        'graph_heads': 4,
        'pred_len': 12
    }
    
    print("\n当前配置:")
    print(f"  Encoder深度: {config['encoder_depth']}")
    print(f"  Decoder深度: {config['decoder_depth']} ⭐ (修复后)")
    print(f"  学习率建议: 0.0003 ⭐ (修复后)")
    print(f"  批次大小建议: 64 ⭐ (修复后)")
    print()
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AGPSTModel(**config).to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    projection_params = sum(p.numel() for p in model.output_projection.parameters())
    
    print("参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  Encoder: {encoder_params:,} ({encoder_params/total_params*100:.1f}%)")
    print(f"  Decoder: {decoder_params:,} ({decoder_params/total_params*100:.1f}%)")
    print(f"  Projection: {projection_params:,} ({projection_params/total_params*100:.1f}%)")
    print()
    
    # 测试前向传播
    batch_size = 32
    history = torch.randn(batch_size, 12, 358, 1).to(device)
    target = torch.randn(batch_size, 12, 358, 1).to(device)
    
    print("=" * 60)
    print("前向传播测试")
    print("=" * 60)
    
    model.train()
    prediction = model(history)
    
    print(f"输入形状: {history.shape}")
    print(f"输出形状: {prediction.shape}")
    print()
    
    # 检查输出范围
    print("输出统计:")
    print(f"  预测值范围: [{prediction.min().item():.4f}, {prediction.max().item():.4f}]")
    print(f"  预测值均值: {prediction.mean().item():.4f}")
    print(f"  预测值标准差: {prediction.std().item():.4f}")
    print(f"  目标值范围: [{target.min().item():.4f}, {target.max().item():.4f}]")
    print()
    
    # 检查是否有异常值
    if torch.isnan(prediction).any():
        print("❌ 警告: 输出包含 NaN!")
    elif torch.isinf(prediction).any():
        print("❌ 警告: 输出包含 Inf!")
    else:
        print("✅ 输出值正常")
    print()
    
    # 测试梯度
    print("=" * 60)
    print("梯度测试")
    print("=" * 60)
    
    loss = torch.nn.functional.mse_loss(prediction, target)
    print(f"损失值: {loss.item():.6f}")
    
    loss.backward()
    
    # 检查关键组件的梯度
    print("\n关键组件梯度范数:")
    
    components = {
        'future_queries': model.future_queries,
        'encoder_pos_embed': model.encoder_pos_embed,
        'decoder_pos_embed': model.decoder_pos_embed,
    }
    
    for name, param in components.items():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"  {name:20s}: {grad_norm:.6f}")
        else:
            print(f"  {name:20s}: No gradient")
    
    # 检查所有参数的梯度范围
    print("\n梯度范围分析:")
    grad_norms = []
    large_grads = []
    tiny_grads = []
    no_grads = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                
                if grad_norm > 10.0:
                    large_grads.append((name, grad_norm))
                elif grad_norm < 1e-6:
                    tiny_grads.append((name, grad_norm))
            else:
                no_grads.append(name)
    
    if grad_norms:
        print(f"  平均梯度范数: {sum(grad_norms)/len(grad_norms):.6f}")
        print(f"  最大梯度范数: {max(grad_norms):.6f}")
        print(f"  最小梯度范数: {min(grad_norms):.6f}")
    
    if large_grads:
        print(f"\n⚠️  过大的梯度 (> 10.0): {len(large_grads)} 个")
        for name, norm in large_grads[:3]:
            print(f"    - {name}: {norm:.4f}")
    
    if tiny_grads:
        print(f"\n⚠️  过小的梯度 (< 1e-6): {len(tiny_grads)} 个")
        for name, norm in tiny_grads[:3]:
            print(f"    - {name}: {norm:.8f}")
    
    if no_grads:
        print(f"\n⚠️  未计算梯度: {len(no_grads)} 个参数")
    
    if not large_grads and not tiny_grads:
        print("\n✅ 梯度范围正常")
    
    print()
    
    # 输出投影层分析
    print("=" * 60)
    print("输出投影层分析")
    print("=" * 60)
    
    print("\n当前结构:")
    for i, layer in enumerate(model.output_projection):
        print(f"  Layer {i}: {layer}")
    
    print(f"\n投影层参数量: {projection_params:,}")
    print(f"占总参数比例: {projection_params/total_params*100:.2f}%")
    
    # 检查投影层的输出
    test_input = torch.randn(32, 12, 96).to(device)
    test_output = model.output_projection(test_input)
    print(f"\n投影层测试:")
    print(f"  输入: {test_input.shape}")
    print(f"  输出: {test_output.shape}")
    print(f"  输出范围: [{test_output.min().item():.4f}, {test_output.max().item():.4f}]")
    
    print()
    
    # 建议
    print("=" * 60)
    print("优化建议")
    print("=" * 60)
    
    print("\n✅ 已修复的问题:")
    print("  1. 输出投影层: 增强为 96→96→48→1 (带 LayerNorm + GELU)")
    print("  2. 未来查询初始化: 改用 Xavier 初始化")
    print("  3. 解码器深度: 从 2 层减少到 1 层")
    print()
    
    print("📝 配置文件建议 (parameters/PEMS03.yaml):")
    print("  decoder_depth: 1     # ⭐ 已修改")
    print("  lr: 0.0003           # ⭐ 已修改 (从 0.001)")
    print("  batch_size: 64       # ⭐ 已修改 (从 32)")
    print()
    
    print("🎯 预期性能提升:")
    print("  当前 MAE: ~22.03")
    print("  修复后预期: ~16-18 (第一阶段)")
    print("  最终目标: ~14.5-15 (全部优化)")
    print()
    
    print("🚀 下一步:")
    print("  1. 使用修复后的配置重新训练")
    print("  2. 监控训练损失曲线是否稳定下降")
    print("  3. 如果性能仍不佳，继续应用进阶优化")
    print()
    
    print("=" * 60)
    print("诊断完成!")
    print("=" * 60)


if __name__ == '__main__':
    diagnose_model()
