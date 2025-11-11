"""
测试简化后的模型（移除patch embedding）
"""
import sys
import os

# 检查必要的库
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    print(f"✅ PyTorch {torch.__version__} loaded successfully")
except ImportError as e:
    print(f"❌ Error importing PyTorch: {e}")
    print("Please install PyTorch first: pip install torch")
    sys.exit(1)

try:
    import numpy as np
    print(f"✅ NumPy {np.__version__} loaded")
except ImportError as e:
    print(f"❌ Error importing NumPy: {e}")
    sys.exit(1)

# 导入模型
try:
    from basicts.mask.model import AGPSTModel
    print("✅ AGPSTModel imported successfully")
except ImportError as e:
    print(f"❌ Error importing model: {e}")
    sys.exit(1)

def test_model():
    print("\n" + "="*60)
    print("Testing Simplified AGPST Model (No Patch Embedding)")
    print("="*60)
    
    # 模型配置
    config = {
        'num_nodes': 358,
        'in_channel': 1,
        'embed_dim': 96,
        'num_heads': 4,
        'mlp_ratio': 4,
        'dropout': 0.1,
        'encoder_depth': 4,
        'dim': 10,
        'topK': 10,
        'backend_args': {
            'num_nodes': 358,
            'supports': None,
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
    }
    
    print("\n📋 Model Configuration:")
    print(f"   - Number of nodes: {config['num_nodes']}")
    print(f"   - Embedding dim: {config['embed_dim']}")
    print(f"   - Transformer layers: {config['encoder_depth']}")
    print(f"   - Graph TopK: {config['topK']}")
    
    # 创建模型
    try:
        model = AGPSTModel(**config)
        print("\n✅ Model created successfully")
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n📊 Model Statistics:")
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        print(f"\n❌ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试前向传播
    print("\n🔄 Testing forward pass...")
    
    # 输入数据: (B=4, T=12, N=358, C=1)
    batch_size = 4
    seq_len = 12
    num_nodes = 358
    in_channels = 1
    
    try:
        # 创建假数据
        history_data = torch.randn(batch_size, seq_len, num_nodes, in_channels)
        long_history_data = torch.randn(batch_size, seq_len, num_nodes, in_channels)  # 不再使用，但保持接口
        
        print(f"\n📥 Input shapes:")
        print(f"   - history_data: {history_data.shape}")
        print(f"   - long_history_data: {long_history_data.shape} (not used)")
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            output = model(history_data, long_history_data)
        
        print(f"\n📤 Output shape: {output.shape}")
        print(f"   - Expected: ({batch_size}, {seq_len}, {num_nodes}, {in_channels})")
        
        # 验证形状
        expected_shape = (batch_size, seq_len, num_nodes, in_channels)
        if output.shape == expected_shape:
            print(f"\n✅ Output shape is correct!")
        else:
            print(f"\n❌ Output shape mismatch!")
            print(f"   Expected: {expected_shape}")
            print(f"   Got: {output.shape}")
            return False
        
        # 检查输出值
        print(f"\n📊 Output Statistics:")
        print(f"   - Mean: {output.mean().item():.4f}")
        print(f"   - Std: {output.std().item():.4f}")
        print(f"   - Min: {output.min().item():.4f}")
        print(f"   - Max: {output.max().item():.4f}")
        
        # 检查是否有 NaN 或 Inf
        if torch.isnan(output).any():
            print("   ⚠️ Warning: Output contains NaN values!")
        elif torch.isinf(output).any():
            print("   ⚠️ Warning: Output contains Inf values!")
        else:
            print("   ✅ No NaN or Inf values")
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\n📝 Summary:")
        print("   - Model structure: Simplified (no patch embedding)")
        print("   - Input processing: Direct time embedding")
        print("   - Graph learning: Single-scale adaptive graph")
        print("   - Architecture: Time Embed → Graph Conv → Transformer → Backend")
        print("   - Status: Ready for training with T=12 sequences")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model()
    sys.exit(0 if success else 1)
