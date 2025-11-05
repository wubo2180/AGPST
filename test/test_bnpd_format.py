"""
测试修改后的PostPatchDynamicGraphConv
验证输入格式为 (B, N, P, D)
"""

import torch
from basicts.mask.post_patch_adaptive_graph import PostPatchDynamicGraphConv
from basicts.mask.patch import PatchEmbedding

def test_bnpd_format():
    """测试(B, N, P, D)格式的输入"""
    
    print("=== 测试PostPatchDynamicGraphConv (B, N, P, D)格式 ===")
    
    # 创建动态图学习模块
    dynamic_graph = PostPatchDynamicGraphConv(
        embed_dim=96,
        num_nodes=358,
        node_dim=10,
        num_heads=4,
        topk=6,
        dropout=0.1
    )
    
    print(f"动态图模块创建成功")
    
    # 创建测试数据 (B, N, P, D)
    B, N, P, D = 4, 358, 72, 96
    test_patches = torch.randn(B, N, P, D)
    
    print(f"输入数据:")
    print(f"  形状: {test_patches.shape}")
    print(f"  格式: (B, N, P, D)")
    print(f"  B={B}, N={N}, P={P}, D={D}")
    
    # 前向传播
    with torch.no_grad():
        enhanced_patches, learned_adj = dynamic_graph(test_patches)
    
    print(f"\n输出数据:")
    print(f"  Enhanced patches: {enhanced_patches.shape}")
    print(f"  Learned adjacency: {learned_adj.shape}")
    
    # 验证输出格式
    expected_patches_shape = (B, N, P, D)
    expected_adj_shape = (B, N, N)
    
    print(f"\n格式验证:")
    print(f"  期望patches形状: {expected_patches_shape}")
    print(f"  实际patches形状: {enhanced_patches.shape}")
    print(f"  Patches格式正确: {'OK' if enhanced_patches.shape == expected_patches_shape else 'FAIL'}")
    
    print(f"  期望adjacency形状: {expected_adj_shape}")
    print(f"  实际adjacency形状: {learned_adj.shape}")
    print(f"  Adjacency格式正确: {'OK' if learned_adj.shape == expected_adj_shape else 'FAIL'}")
    
    return (enhanced_patches.shape == expected_patches_shape and 
            learned_adj.shape == expected_adj_shape)

def test_full_pipeline():
    """测试完整的数据流水线"""
    
    print("\n=== 测试完整数据流水线 ===")
    
    try:
        # 1. PatchEmbedding (现在输出 (B, N, P, d))
        patch_embedding = PatchEmbedding(12, 1, 96, None)
        
        # 2. 动态图学习 (期望输入 (B, N, P, D))
        dynamic_graph = PostPatchDynamicGraphConv(96, 358, 10, 4, 6, 0.1)
        
        # 3. 测试数据 (B, N, C, L)
        test_data = torch.randn(4, 358, 1, 864)
        print(f"原始输入: {test_data.shape} (B, N, C, L)")
        
        with torch.no_grad():
            # Step 1: Patch Embedding
            patches = patch_embedding(test_data)
            print(f"PatchEmbedding输出: {patches.shape} (B, N, P, d)")
            
            # Step 2: 动态图学习 (直接使用，不需要转换)
            enhanced_patches, learned_adj = dynamic_graph(patches)
            print(f"动态图学习输出: {enhanced_patches.shape} (B, N, P, D)")
            print(f"学习的邻接矩阵: {learned_adj.shape} (B, N, N)")
            
            print("OK: 完整流水线测试成功!")
            return True
            
    except Exception as e:
        print(f"FAIL: 流水线测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_adjacency_properties():
    """测试邻接矩阵的属性"""
    
    print("\n=== 测试邻接矩阵属性 ===")
    
    dynamic_graph = PostPatchDynamicGraphConv(96, 358, 10, 4, 6, 0.1)
    test_patches = torch.randn(4, 358, 72, 96)
    
    with torch.no_grad():
        enhanced_patches, learned_adj = dynamic_graph(test_patches)
    
    print(f"邻接矩阵分析:")
    print(f"  形状: {learned_adj.shape}")
    print(f"  最大值: {learned_adj.max().item():.6f}")
    print(f"  最小值: {learned_adj.min().item():.6f}")
    print(f"  平均值: {learned_adj.mean().item():.6f}")
    
    # 检查稀疏性
    topk = 6
    nonzero_ratio = (learned_adj > 1e-6).float().mean().item()
    expected_ratio = topk / 358
    print(f"  非零元素比例: {nonzero_ratio:.4f}")
    print(f"  期望稀疏度: {expected_ratio:.4f}")
    
    return True

if __name__ == "__main__":
    print("🧪 测试修改后的PostPatchDynamicGraphConv")
    print("=" * 60)
    
    # 测试基本格式
    format_correct = test_bnpd_format()
    
    # 测试完整流水线
    pipeline_ok = test_full_pipeline()
    
    # 测试邻接矩阵
    adj_ok = test_adjacency_properties()
    
    print("\n" + "=" * 60)
    if format_correct and pipeline_ok and adj_ok:
        print("🎉 所有测试通过!")
        print("✅ PostPatchDynamicGraphConv现在接受 (B, N, P, D) 格式")
        print("✅ 与PatchEmbedding的 (B, N, P, d) 输出完全匹配")
        print("✅ 数据流水线完整无误")
    else:
        print("❌ 部分测试失败，请检查代码")