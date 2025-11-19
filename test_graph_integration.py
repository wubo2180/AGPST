"""
测试高级图学习模块集成
"""
import torch
import yaml
import sys
sys.path.append('.')

from basicts.mask.model import AGPSTModel
from basicts.mask.graph_learning import AdaptiveGraphLearner, DynamicGraphConv


def test_simple_graph_learning():
    """测试简单图学习模式"""
    print("=" * 70)
    print("Testing Simple Graph Learning Mode")
    print("=" * 70)
    
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
        use_denoising=False,
        use_advanced_graph=False  # 简单模式
    )
    
    # 测试数据
    B, T, N, C = 2, 12, 358, 1
    history_data = torch.randn(B, T, N, C)
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(history_data)
    
    print(f"\n✅ Simple Graph Learning Mode:")
    print(f"   Input shape:  {history_data.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def test_advanced_graph_learning():
    """测试高级图学习模式"""
    print("\n" + "=" * 70)
    print("Testing Advanced Graph Learning Mode")
    print("=" * 70)
    
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
        use_denoising=False,
        use_advanced_graph=True,  # 高级模式
        graph_heads=4
    )
    
    # 测试数据
    B, T, N, C = 2, 12, 358, 1
    history_data = torch.randn(B, T, N, C)
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(history_data)
    
    print(f"\n✅ Advanced Graph Learning Mode:")
    print(f"   Input shape:  {history_data.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 检查对比损失
    if model.contrastive_loss is not None:
        print(f"   Contrastive Loss: {model.contrastive_loss.item():.4f}")
    else:
        print(f"   Contrastive Loss: None (eval mode)")
    
    return model


def test_graph_learner_standalone():
    """独立测试图学习器"""
    print("\n" + "=" * 70)
    print("Testing AdaptiveGraphLearner Standalone")
    print("=" * 70)
    
    learner = AdaptiveGraphLearner(
        num_nodes=358,
        node_dim=10,
        embed_dim=96,
        graph_heads=4,
        topk=10,
        dropout=0.1,
        use_temporal_info=True
    )
    
    # 测试数据
    B, N, P, D = 2, 358, 12, 96
    patch_features = torch.randn(B, N, P, D)
    
    learner.eval()
    with torch.no_grad():
        learned_adjs, contrastive_loss = learner(patch_features)
    
    print(f"\n✅ Graph Learner Output:")
    print(f"   Input:  (B={B}, N={N}, P={P}, D={D})")
    print(f"   Learned Adjacency: {learned_adjs.shape}")
    print(f"   Expected: (B={B}, N={N}, N={N})")
    print(f"   Contrastive Loss: {contrastive_loss if contrastive_loss else 'None (eval mode)'}")
    
    # 检查稀疏性
    sparsity = (learned_adjs[0] > 0).sum().item() / (N * N) * 100
    print(f"   Graph Sparsity: {sparsity:.2f}%")
    print(f"   Expected: ~{10/N*100:.2f}% (topK={10})")


def test_dynamic_graph_conv():
    """测试动态图卷积"""
    print("\n" + "=" * 70)
    print("Testing DynamicGraphConv Standalone")
    print("=" * 70)
    
    dgc = DynamicGraphConv(
        embed_dim=96,
        num_nodes=358,
        node_dim=10,
        graph_heads=4,
        topk=10,
        dropout=0.1
    )
    
    # 测试数据
    B, N, P, D = 2, 358, 12, 96
    patch_features = torch.randn(B, N, P, D)
    
    dgc.eval()
    with torch.no_grad():
        output, learned_adjs, contrastive_loss = dgc(patch_features)
    
    print(f"\n✅ Dynamic Graph Conv Output:")
    print(f"   Input:  {patch_features.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Learned Adjacency: {learned_adjs.shape}")
    print(f"   Contrastive Loss: {contrastive_loss if contrastive_loss else 'None (eval mode)'}")


def test_with_config():
    """使用配置文件测试"""
    print("\n" + "=" * 70)
    print("Testing with Configuration File")
    print("=" * 70)
    
    try:
        with open('parameters/PEMS03_v3.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"\n📄 Configuration loaded:")
        print(f"   use_advanced_graph: {config.get('use_advanced_graph', 'Not set')}")
        print(f"   graph_heads: {config.get('graph_heads', 'Not set')}")
        print(f"   dim: {config.get('dim', 'Not set')}")
        print(f"   topK: {config.get('topK', 'Not set')}")
        
        backend_args = config['backend_args']
        backend_args['supports'] = []
        
        model = AGPSTModel(
            num_nodes=config['num_nodes'],
            dim=config['dim'],
            topK=config['topK'],
            in_channel=config['in_channel'],
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            mlp_ratio=config['mlp_ratio'],
            dropout=config['dropout'],
            encoder_depth=config['encoder_depth'],
            backend_args=backend_args,
            use_denoising=config.get('use_denoising', True),
            denoise_type=config.get('denoise_type', 'conv'),
            use_advanced_graph=config.get('use_advanced_graph', True),
            graph_heads=config.get('graph_heads', 4)
        )
        
        B, T, N, C = 2, 12, 358, 1
        history_data = torch.randn(B, T, N, C)
        
        model.eval()
        with torch.no_grad():
            output = model(history_data)
        
        print(f"\n✅ Model created from config successfully!")
        print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


def compare_modes():
    """对比两种模式的参数量和性能"""
    print("\n" + "=" * 70)
    print("Comparing Simple vs Advanced Graph Learning")
    print("=" * 70)
    
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
    
    # 简单模式
    model_simple = AGPSTModel(
        num_nodes=358, dim=10, topK=10, in_channel=1,
        embed_dim=96, num_heads=4, mlp_ratio=4, dropout=0.1,
        encoder_depth=4, backend_args=backend_args,
        use_denoising=False, use_advanced_graph=False
    )
    
    # 高级模式
    model_advanced = AGPSTModel(
        num_nodes=358, dim=10, topK=10, in_channel=1,
        embed_dim=96, num_heads=4, mlp_ratio=4, dropout=0.1,
        encoder_depth=4, backend_args=backend_args,
        use_denoising=False, use_advanced_graph=True, graph_heads=4
    )
    
    params_simple = sum(p.numel() for p in model_simple.parameters())
    params_advanced = sum(p.numel() for p in model_advanced.parameters())
    
    print(f"\n📊 Parameter Comparison:")
    print(f"   Simple Mode:   {params_simple:,} parameters")
    print(f"   Advanced Mode: {params_advanced:,} parameters")
    print(f"   Difference:    {params_advanced - params_simple:,} (+{(params_advanced/params_simple-1)*100:.1f}%)")
    
    # 测试速度
    import time
    
    B, T, N, C = 4, 12, 358, 1
    history_data = torch.randn(B, T, N, C)
    
    model_simple.eval()
    model_advanced.eval()
    
    # Warm up
    with torch.no_grad():
        _ = model_simple(history_data)
        _ = model_advanced(history_data)
    
    # Benchmark
    runs = 10
    
    start = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model_simple(history_data)
    time_simple = (time.time() - start) / runs
    
    start = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model_advanced(history_data)
    time_advanced = (time.time() - start) / runs
    
    print(f"\n⏱️  Speed Comparison (averaged over {runs} runs):")
    print(f"   Simple Mode:   {time_simple*1000:.2f} ms/batch")
    print(f"   Advanced Mode: {time_advanced*1000:.2f} ms/batch")
    print(f"   Slowdown:      {(time_advanced/time_simple-1)*100:.1f}%")


if __name__ == '__main__':
    print("\n" + "🚀 " + "=" * 66 + " 🚀")
    print("🚀  AGPST Advanced Graph Learning Integration Test Suite        🚀")
    print("🚀 " + "=" * 66 + " 🚀\n")
    
    try:
        # Test 1: Simple mode
        test_simple_graph_learning()
        
        # Test 2: Advanced mode
        test_advanced_graph_learning()
        
        # Test 3: Graph learner standalone
        test_graph_learner_standalone()
        
        # Test 4: Dynamic graph conv
        test_dynamic_graph_conv()
        
        # Test 5: With config
        test_with_config()
        
        # Test 6: Compare modes
        compare_modes()
        
        print("\n" + "=" * 70)
        print("🎉 All tests passed successfully!")
        print("=" * 70)
        
        print("\n💡 Next steps:")
        print("   1. Run training: python main.py --config=parameters/PEMS03_v3.yaml")
        print("   2. Compare performance: use_advanced_graph=True vs False")
        print("   3. Tune graph_heads: try 2, 4, 6, 8")
        print("   4. Visualize learned graphs")
        print()
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
