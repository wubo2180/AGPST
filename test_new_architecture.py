"""
快速测试新的模块架构
"""
import sys
import torch

def test_imports():
    """测试导入"""
    print("=" * 60)
    print("测试1: 导入模块")
    print("=" * 60)
    
    try:
        from basicts.mask import (
            AGPSTModel,
            ForecastingWithAdaptiveGraph,
            DynamicGraphConv,
            AdaptiveGraphLearner,
            PatchEmbedding,
            TransformerLayers,
            PositionalEncoding
        )
        print("✅ 所有模块导入成功!")
        print(f"   - AGPSTModel: {AGPSTModel}")
        print(f"   - ForecastingWithAdaptiveGraph (alias): {ForecastingWithAdaptiveGraph}")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False


def test_model_creation():
    """测试模型创建"""
    print("\n" + "=" * 60)
    print("测试2: 创建模型")
    print("=" * 60)
    
    try:
        from basicts.mask import AGPSTModel
        
        model = AGPSTModel(
            num_nodes=358,
            dim=10,
            topK=10,
            patch_size=12,
            in_channel=1,
            embed_dim=96,
            num_heads=4,
            graph_heads=4,
            mlp_ratio=4,
            dropout=0.1,
            encoder_depth=4,
            backend_args={
                'num_nodes': 358,
                'supports': None,
                'dropout': 0.3,
                'gcn_bool': True,
                'addaptadj': True,
                'aptinit': None,
                'in_dim': 96,
                'out_dim': 12,
                'residual_channels': 32,
                'dilation_channels': 32,
                'skip_channels': 256,
                'end_channels': 512,
                'kernel_size': 2,
                'blocks': 4,
                'layers': 2
            }
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print("✅ 模型创建成功!")
        print(f"   - 总参数量: {total_params:,}")
        print(f"   - 可训练参数: {trainable_params:,}")
        print(f"   - 参数量 (MB): {total_params * 4 / 1024 / 1024:.2f}")
        
        return model
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_forward_pass(model):
    """测试前向传播"""
    print("\n" + "=" * 60)
    print("测试3: 前向传播")
    print("=" * 60)
    
    if model is None:
        print("❌ 跳过（模型未创建）")
        return False
    
    try:
        # 创建测试数据
        B, N, C = 2, 358, 1  # 小batch用于测试
        short_len, long_len = 12, 864
        
        history_data = torch.randn(B, short_len, N, C)
        long_history_data = torch.randn(B, long_len, N, C)
        
        print(f"   输入数据:")
        print(f"   - history_data: {history_data.shape}")
        print(f"   - long_history_data: {long_history_data.shape}")
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            prediction = model(history_data, long_history_data)
        
        print(f"   输出数据:")
        print(f"   - prediction: {prediction.shape}")
        print(f"   - 期望shape: (B={B}, T=12, N={N}, C={C})")
        
        # 验证shape
        expected_shape = (B, short_len, N, C)
        if prediction.shape == expected_shape:
            print("✅ 前向传播成功，输出shape正确!")
            
            # 检查是否有NaN
            if torch.isnan(prediction).any():
                print("⚠️  警告: 输出包含NaN值")
                return False
            else:
                print("✅ 输出数据正常（无NaN）")
                return True
        else:
            print(f"❌ 输出shape不匹配: {prediction.shape} != {expected_shape}")
            return False
            
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_components():
    """测试各个组件"""
    print("\n" + "=" * 60)
    print("测试4: 独立组件")
    print("=" * 60)
    
    try:
        from basicts.mask import PatchEmbedding, DynamicGraphConv, TransformerLayers
        
        B, N, L, C = 2, 358, 864, 1
        
        # 测试PatchEmbedding
        print("   测试 PatchEmbedding...")
        patch_embed = PatchEmbedding(patch_size=12, in_channel=1, embed_dim=96)
        long_history = torch.randn(B, N, L, C)
        patches = patch_embed(long_history)
        print(f"   ✅ PatchEmbedding: {long_history.shape} → {patches.shape}")
        
        # 测试DynamicGraphConv
        print("   测试 DynamicGraphConv...")
        graph_conv = DynamicGraphConv(embed_dim=96, num_nodes=N, node_dim=10)
        graph_features, adj, loss = graph_conv(patches)
        print(f"   ✅ DynamicGraphConv: {patches.shape} → {graph_features.shape}")
        print(f"      - 邻接矩阵: {adj.shape}")
        print(f"      - 对比损失: {loss.item() if loss is not None else 'None'}")
        
        # 测试TransformerLayers
        print("   测试 TransformerLayers...")
        transformer = TransformerLayers(hidden_dim=96, nlayers=4, mlp_ratio=4)
        temporal_features = transformer(graph_features)
        print(f"   ✅ TransformerLayers: {graph_features.shape} → {temporal_features.shape}")
        
        print("✅ 所有组件测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 组件测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "🔍 AGPST 模块架构测试" + "\n")
    
    results = {
        "导入测试": False,
        "模型创建": False,
        "前向传播": False,
        "组件测试": False
    }
    
    # 测试1: 导入
    results["导入测试"] = test_imports()
    
    # 测试2: 模型创建
    if results["导入测试"]:
        model = test_model_creation()
        results["模型创建"] = model is not None
        
        # 测试3: 前向传播
        if results["模型创建"]:
            results["前向传播"] = test_forward_pass(model)
    
    # 测试4: 组件测试
    if results["导入测试"]:
        results["组件测试"] = test_components()
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过! 新架构工作正常!")
    else:
        print("⚠️  部分测试失败，请检查上述错误信息")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
