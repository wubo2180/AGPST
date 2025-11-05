"""
多尺度Patch Embedding测试脚本
用于验证多尺度patch embedding的功能是否正常
"""
import torch
import sys
sys.path.append('.')

from basicts.mask.patch import PatchEmbedding

def test_single_scale():
    """测试单一尺度（原有功能）"""
    print("=" * 50)
    print("测试单一尺度Patch Embedding")
    print("=" * 50)
    
    # 参数设置
    batch_size = 2
    seq_len = 144  # 12小时 * 12个时间步
    num_nodes = 10
    topK = 5
    in_channel = 1
    patch_size = 12
    embed_dim = 64
    
    # 创建模型
    patch_embed = PatchEmbedding(
        patch_size=patch_size,
        in_channel=in_channel,
        embed_dim=embed_dim,
        num_nodes=num_nodes,
        topK=topK,
        norm_layer=None,
        patch_sizes=None  # 不指定，使用单一尺度
    )
    
    # 创建测试数据 [B, L, N, K, C]
    test_data = torch.randn(batch_size, seq_len, num_nodes, topK, in_channel)
    
    print(f"输入形状: {test_data.shape}")
    
    # 前向传播
    output = patch_embed(test_data)
    
    print(f"输出形状: {output.shape}")
    expected_patches = seq_len // patch_size
    print(f"期望patch数: {expected_patches}")
    print(f"实际patch数: {output.shape[2]}")
    print(f"嵌入维度: {output.shape[1]}")
    
    assert output.shape == (batch_size, embed_dim, expected_patches, num_nodes, topK), \
        f"输出形状不匹配! 期望 {(batch_size, embed_dim, expected_patches, num_nodes, topK)}, 得到 {output.shape}"
    
    print("✓ 单一尺度测试通过!\n")
    return True


def test_multi_scale():
    """测试多尺度patch embedding"""
    print("=" * 50)
    print("测试多尺度Patch Embedding")
    print("=" * 50)
    
    # 参数设置
    batch_size = 2
    seq_len = 144  # 12小时 * 12个时间步
    num_nodes = 10
    topK = 5
    in_channel = 1
    patch_size = 12
    patch_sizes = [6, 12, 24]  # 多尺度
    embed_dim = 96
    
    # 创建模型
    patch_embed = PatchEmbedding(
        patch_size=patch_size,
        in_channel=in_channel,
        embed_dim=embed_dim,
        num_nodes=num_nodes,
        topK=topK,
        norm_layer=None,
        patch_sizes=patch_sizes  # 指定多尺度
    )
    
    # 创建测试数据 [B, L, N, K, C]
    test_data = torch.randn(batch_size, seq_len, num_nodes, topK, in_channel)
    
    print(f"输入形状: {test_data.shape}")
    print(f"多尺度配置: {patch_sizes}")
    
    # 前向传播
    output = patch_embed(test_data)
    
    print(f"输出形状: {output.shape}")
    
    # 计算最小patch数
    min_patches = min([seq_len // ps for ps in patch_sizes])
    print(f"各尺度patch数: {[seq_len // ps for ps in patch_sizes]}")
    print(f"对齐后patch数: {min_patches}")
    print(f"实际patch数: {output.shape[2]}")
    print(f"嵌入维度: {output.shape[1]}")
    
    assert output.shape == (batch_size, embed_dim, min_patches, num_nodes, topK), \
        f"输出形状不匹配! 期望 {(batch_size, embed_dim, min_patches, num_nodes, topK)}, 得到 {output.shape}"
    
    print("✓ 多尺度测试通过!\n")
    return True


def test_parameter_count():
    """比较单尺度和多尺度的参数量"""
    print("=" * 50)
    print("参数量对比")
    print("=" * 50)
    
    num_nodes = 10
    topK = 5
    in_channel = 1
    embed_dim = 96
    patch_size = 12
    
    # 单尺度
    single_scale = PatchEmbedding(
        patch_size=patch_size,
        in_channel=in_channel,
        embed_dim=embed_dim,
        num_nodes=num_nodes,
        topK=topK,
        norm_layer=None,
        patch_sizes=None
    )
    
    # 多尺度
    multi_scale = PatchEmbedding(
        patch_size=patch_size,
        in_channel=in_channel,
        embed_dim=embed_dim,
        num_nodes=num_nodes,
        topK=topK,
        norm_layer=None,
        patch_sizes=[6, 12, 24]
    )
    
    single_params = sum(p.numel() for p in single_scale.parameters())
    multi_params = sum(p.numel() for p in multi_scale.parameters())
    
    print(f"单尺度参数量: {single_params:,}")
    print(f"多尺度参数量: {multi_params:,}")
    print(f"参数增加: {multi_params - single_params:,} ({(multi_params/single_params - 1)*100:.2f}%)")
    print()
    
    return True


if __name__ == "__main__":
    print("\n" + "="*50)
    print("开始测试多尺度Patch Embedding")
    print("="*50 + "\n")
    
    try:
        # 测试单一尺度
        test_single_scale()
        
        # 测试多尺度
        test_multi_scale()
        
        # 参数量对比
        test_parameter_count()
        
        print("="*50)
        print("✓ 所有测试通过!")
        print("="*50)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
