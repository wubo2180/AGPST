"""
测试修改后的PatchEmbedding输出格式
验证输出是否为 (B, N, P, d) 格式
"""

import torch
from basicts.mask.patch import PatchEmbedding

def test_patch_embedding_format():
    """测试PatchEmbedding的输出格式"""
    
    print("=== 测试PatchEmbedding输出格式 ===")
    
    # 创建PatchEmbedding实例
    patch_size = 12
    in_channel = 1
    embed_dim = 96
    
    patch_embedding = PatchEmbedding(
        patch_size=patch_size,
        in_channel=in_channel,
        embed_dim=embed_dim,
        norm_layer=None
    )
    
    print(f"PatchEmbedding配置:")
    print(f"  patch_size: {patch_size}")
    print(f"  in_channel: {in_channel}")  
    print(f"  embed_dim: {embed_dim}")
    
    # 创建测试数据 (B, N, C, L)
    B, N, C, L = 4, 358, 1, 864
    test_data = torch.randn(B, N, C, L)
    
    print(f"\n输入数据:")
    print(f"  形状: {test_data.shape}")
    print(f"  格式: (B, N, C, L)")
    
    # 前向传播
    with torch.no_grad():
        output = patch_embedding(test_data)
    
    print(f"\n输出数据:")
    print(f"  形状: {output.shape}")
    print(f"  格式: (B, N, P, d)")
    
    # 验证输出格式
    expected_P = L // patch_size  # 864 // 12 = 72
    expected_shape = (B, N, expected_P, embed_dim)
    
    print(f"\n格式验证:")
    print(f"  期望形状: {expected_shape}")
    print(f"  实际形状: {output.shape}")
    print(f"  格式正确: {'✅' if output.shape == expected_shape else '❌'}")
    
    # 详细维度分析
    print(f"\n维度分析:")
    print(f"  B (batch_size): {output.shape[0]} = {B}")
    print(f"  N (num_nodes): {output.shape[1]} = {N}")  
    print(f"  P (num_patches): {output.shape[2]} = {L}/{patch_size} = {expected_P}")
    print(f"  d (embed_dim): {output.shape[3]} = {embed_dim}")
    
    return output.shape == expected_shape

def test_compatibility_with_dynamic_graph():
    """测试与PostPatchDynamicGraphConv的兼容性"""
    
    print("\n=== 测试与动态图学习的兼容性 ===")
    
    try:
        from basicts.mask.post_patch_adaptive_graph import PostPatchDynamicGraphConv
        
        # 创建组件
        patch_embedding = PatchEmbedding(12, 1, 96, None)
        dynamic_graph = PostPatchDynamicGraphConv(
            embed_dim=96,
            num_nodes=358,
            node_dim=10,
            num_heads=4,
            topk=6,
            dropout=0.1
        )
        
        # 测试数据流
        test_data = torch.randn(4, 358, 1, 864)  # (B, N, C, L)
        
        with torch.no_grad():
            # Step 1: Patch Embedding
            patches = patch_embedding(test_data)  # 期望: (B, N, P, d)
            print(f"PatchEmbedding输出: {patches.shape}")
            
            # Step 2: 动态图学习 (期望输入格式为 (B, P, N, d))
            patches_for_graph = patches.permute(0, 2, 1, 3)  # (B, N, P, d) -> (B, P, N, d)
            print(f"转换为图学习格式: {patches_for_graph.shape}")
            
            # Step 3: 动态图学习
            enhanced_patches, learned_adj = dynamic_graph(patches_for_graph)
            print(f"动态图学习输出: {enhanced_patches.shape}")
            print(f"学习的邻接矩阵: {learned_adj.shape}")
            
            print("✅ 与PostPatchDynamicGraphConv兼容!")
            
    except Exception as e:
        print(f"❌ 兼容性测试失败: {e}")
        return False
        
    return True

if __name__ == "__main__":
    print("🧪 测试修改后的PatchEmbedding格式")
    print("=" * 50)
    
    # 测试输出格式
    format_correct = test_patch_embedding_format()
    
    # 测试兼容性
    compatibility_ok = test_compatibility_with_dynamic_graph()
    
    print("\n" + "=" * 50)
    if format_correct and compatibility_ok:
        print("🎉 所有测试通过!")
        print("✅ PatchEmbedding现在输出 (B, N, P, d) 格式")
        print("✅ 与PostPatchDynamicGraphConv完全兼容")
    else:
        print("❌ 测试失败，请检查代码")