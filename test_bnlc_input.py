"""
测试修改后的PatchEmbedding处理(B, N, L, C)输入格式
"""

import torch
from basicts.mask.patch import PatchEmbedding

def test_bnlc_input_format():
    """测试(B, N, L, C)输入格式"""
    
    print("=== 测试PatchEmbedding处理(B, N, L, C)输入 ===")
    
    # 创建PatchEmbedding
    patch_embedding = PatchEmbedding(
        patch_size=12,
        in_channel=1,
        embed_dim=96,
        norm_layer=None
    )
    
    print(f"PatchEmbedding配置:")
    print(f"  patch_size: 12")
    print(f"  in_channel: 1")
    print(f"  embed_dim: 96")
    
    # 创建测试数据 (B, N, L, C)
    B, N, L, C = 4, 358, 864, 1
    test_data = torch.randn(B, N, L, C)
    
    print(f"\n输入数据:")
    print(f"  形状: {test_data.shape}")
    print(f"  格式: (B, N, L, C)")
    print(f"  B={B}, N={N}, L={L}, C={C}")
    
    # 前向传播
    with torch.no_grad():
        try:
            output = patch_embedding(test_data)
            
            print(f"\n输出数据:")
            print(f"  形状: {output.shape}")
            print(f"  格式: (B, N, P, d)")
            
            # 验证输出维度
            expected_P = L // 12  # 864 // 12 = 72
            expected_shape = (B, N, expected_P, 96)
            
            print(f"\n维度验证:")
            print(f"  期望形状: {expected_shape}")
            print(f"  实际形状: {output.shape}")
            print(f"  维度正确: {'OK' if output.shape == expected_shape else 'FAIL'}")
            
            # 详细分析
            print(f"\n详细分析:")
            print(f"  B (batch_size): {output.shape[0]} = {B}")
            print(f"  N (num_nodes): {output.shape[1]} = {N}")
            print(f"  P (num_patches): {output.shape[2]} = {L}/{12} = {expected_P}")
            print(f"  d (embed_dim): {output.shape[3]} = 96")
            
            return output.shape == expected_shape
            
        except Exception as e:
            print(f"\n前向传播失败: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_different_input_sizes():
    """测试不同输入尺寸"""
    
    print(f"\n=== 测试不同输入尺寸 ===")
    
    patch_embedding = PatchEmbedding(12, 1, 96, None)
    
    test_cases = [
        {"shape": (2, 100, 240, 1), "desc": "小规模数据"},
        {"shape": (4, 358, 864, 1), "desc": "PEMS03数据"},
        {"shape": (8, 207, 1440, 1), "desc": "大规模数据"},
    ]
    
    all_passed = True
    
    for case in test_cases:
        B, N, L, C = case["shape"]
        desc = case["desc"]
        
        print(f"\n测试 {desc}: {case['shape']}")
        
        try:
            test_data = torch.randn(B, N, L, C)
            
            with torch.no_grad():
                output = patch_embedding(test_data)
                
            expected_P = L // 12
            expected_shape = (B, N, expected_P, 96)
            
            success = output.shape == expected_shape
            print(f"  输出形状: {output.shape}")
            print(f"  期望形状: {expected_shape}")
            print(f"  结果: {'OK' if success else 'FAIL'}")
            
            if not success:
                all_passed = False
                
        except Exception as e:
            print(f"  测试失败: {e}")
            all_passed = False
    
    return all_passed

def test_compatibility_with_dynamic_graph():
    """测试与动态图学习的兼容性"""
    
    print(f"\n=== 测试与PostPatchDynamicGraphConv兼容性 ===")
    
    try:
        from basicts.mask.post_patch_adaptive_graph import PostPatchDynamicGraphConv
        
        # 创建组件
        patch_embedding = PatchEmbedding(12, 1, 96, None)
        dynamic_graph = PostPatchDynamicGraphConv(
            embed_dim=96,
            num_nodes=358,
            node_dim=10,
            graph_heads=4,  # 使用新的参数名
            topk=6,
            dropout=0.1
        )
        
        print("组件创建成功")
        
        # 测试完整数据流 (B, N, L, C)
        test_data = torch.randn(4, 358, 864, 1)
        print(f"原始输入: {test_data.shape} (B, N, L, C)")
        
        with torch.no_grad():
            # Step 1: Patch Embedding
            patches = patch_embedding(test_data)
            print(f"PatchEmbedding输出: {patches.shape} (B, N, P, d)")
            
            # Step 2: 动态图学习 (直接兼容)
            enhanced_patches, learned_adj = dynamic_graph(patches)
            print(f"动态图学习输出: {enhanced_patches.shape} (B, N, P, D)")
            print(f"学习的邻接矩阵: {learned_adj.shape} (B, N, N)")
            
            print("OK: 完整流水线测试成功!")
            return True
            
    except Exception as e:
        print(f"兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 测试PatchEmbedding处理(B, N, L, C)输入格式")
    print("=" * 60)
    
    # 测试基本功能
    basic_test = test_bnlc_input_format()
    
    # 测试不同尺寸
    size_test = test_different_input_sizes()
    
    # 测试兼容性
    compat_test = test_compatibility_with_dynamic_graph()
    
    print("\n" + "=" * 60)
    if basic_test and size_test and compat_test:
        print("🎉 所有测试通过!")
        print("✅ PatchEmbedding正确处理 (B, N, L, C) 输入")
        print("✅ 输出格式 (B, N, P, d) 正确")
        print("✅ 与PostPatchDynamicGraphConv完全兼容")
        
        print(f"\n📝 数据流总结:")
        print(f"输入: (B, N, L, C) = (4, 358, 864, 1)")
        print(f"↓ 转换: (B, N, L, C) → (B, N, C, L)")
        print(f"↓ PatchEmbedding")
        print(f"输出: (B, N, P, d) = (4, 358, 72, 96)")
        print(f"↓ PostPatchDynamicGraphConv")
        print(f"最终: (B, N, P, D) + adj(B, N, N)")
    else:
        print("❌ 部分测试失败，请检查代码")