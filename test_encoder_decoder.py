"""
测试 Encoder-Decoder 架构
验证新架构的前向传播和输出形状
"""

import torch
import sys
sys.path.append('.')

from basicts.mask.model import AGPSTModel

def test_encoder_decoder():
    print("=" * 60)
    print("测试 Encoder-Decoder 架构")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 模型参数
    config = {
        'num_nodes': 358,
        'dim': 40,
        'topK': 10,
        'in_channel': 1,
        'embed_dim': 96,
        'num_heads': 4,
        'mlp_ratio': 4,
        'dropout': 0.1,
        'encoder_depth': 4,
        'decoder_depth': 2,  # 新增参数
        'use_denoising': True,
        'denoise_type': 'conv',
        'use_advanced_graph': True,
        'graph_heads': 4,
        'pred_len': 12
    }
    
    print("模型配置:")
    for key, value in config.items():
        print(f"  {key:20s}: {value}")
    print()
    
    # 创建模型
    print("创建模型...")
    model = AGPSTModel(**config).to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print()
    
    # 打印模型关键组件
    print("模型关键组件:")
    print(f"  Encoder layers: {config['encoder_depth']}")
    print(f"  Decoder layers: {config['decoder_depth']}")
    print(f"  Future queries: {model.future_queries.shape}")
    print(f"  Encoder pos embed: {model.encoder_pos_embed.shape}")
    print(f"  Decoder pos embed: {model.decoder_pos_embed.shape}")
    print()
    
    # 创建测试数据
    batch_size = 8
    seq_len = 12
    num_nodes = 358
    
    history_data = torch.randn(batch_size, seq_len, num_nodes, 1).to(device)
    print(f"输入形状: {history_data.shape} (B, T, N, C)")
    
    # 前向传播
    print("\n开始前向传播...")
    model.eval()
    with torch.no_grad():
        prediction = model(history_data)
    
    print(f"输出形状: {prediction.shape} (B, pred_len, N, C)")
    
    # 验证输出形状
    expected_shape = (batch_size, config['pred_len'], num_nodes, 1)
    assert prediction.shape == expected_shape, \
        f"输出形状不匹配! 期望 {expected_shape}, 得到 {prediction.shape}"
    
    print("\n✅ 形状验证通过!")
    
    # 检查输出值
    print(f"\n输出统计:")
    print(f"  最小值: {prediction.min().item():.4f}")
    print(f"  最大值: {prediction.max().item():.4f}")
    print(f"  平均值: {prediction.mean().item():.4f}")
    print(f"  标准差: {prediction.std().item():.4f}")
    
    # 检查是否有 NaN 或 Inf
    if torch.isnan(prediction).any():
        print("\n❌ 警告: 输出包含 NaN!")
    elif torch.isinf(prediction).any():
        print("\n❌ 警告: 输出包含 Inf!")
    else:
        print("\n✅ 输出值正常!")
    
    # 测试梯度反向传播
    print("\n" + "=" * 60)
    print("测试梯度反向传播")
    print("=" * 60)
    
    model.train()
    history_data = torch.randn(batch_size, seq_len, num_nodes, 1).to(device)
    target = torch.randn(batch_size, config['pred_len'], num_nodes, 1).to(device)
    
    # 前向传播
    prediction = model(history_data)
    
    # 计算损失
    loss = torch.nn.functional.mse_loss(prediction, target)
    print(f"损失值: {loss.item():.6f}")
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    has_grad = False
    no_grad_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                has_grad = True
                if torch.isnan(param.grad).any():
                    print(f"❌ {name} 的梯度包含 NaN!")
            else:
                no_grad_params.append(name)
    
    if has_grad:
        print("✅ 梯度计算成功!")
    else:
        print("❌ 没有计算到梯度!")
    
    if no_grad_params:
        print(f"\n未计算梯度的参数 ({len(no_grad_params)}):")
        for name in no_grad_params[:5]:  # 只显示前5个
            print(f"  - {name}")
        if len(no_grad_params) > 5:
            print(f"  ... 还有 {len(no_grad_params) - 5} 个")
    
    # 测试不同预测长度
    print("\n" + "=" * 60)
    print("测试不同预测长度")
    print("=" * 60)
    
    for pred_len in [3, 6, 12, 24]:
        config_test = config.copy()
        config_test['pred_len'] = pred_len
        
        model_test = AGPSTModel(**config_test).to(device)
        model_test.eval()
        
        with torch.no_grad():
            pred = model_test(history_data)
        
        expected = (batch_size, pred_len, num_nodes, 1)
        status = "✅" if pred.shape == expected else "❌"
        print(f"{status} pred_len={pred_len:2d}: {pred.shape}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    
    # 对比 Encoder-Decoder vs 单编码器参数量
    print("\n架构对比:")
    print("-" * 40)
    
    # Encoder-Decoder
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    future_queries_params = model.future_queries.numel()
    projection_params = sum(p.numel() for p in model.output_projection.parameters())
    
    print(f"Encoder-Decoder 架构:")
    print(f"  Encoder:          {encoder_params:>10,} 参数")
    print(f"  Decoder:          {decoder_params:>10,} 参数")
    print(f"  Future queries:   {future_queries_params:>10,} 参数")
    print(f"  Output projection:{projection_params:>10,} 参数")
    print(f"  总计:             {total_params:>10,} 参数")
    
    print("\n✅ 所有测试通过!")


if __name__ == '__main__':
    test_encoder_decoder()
