"""
测试修改后的PatchEmbedding是否正确处理(B, L, N, C)格式的数据
"""
import torch
import torch.nn as nn
import sys
import os

# 添加路径以便导入模块
sys.path.append('.')

from basicts.mask.patch import PatchEmbedding

def test_patch_embedding():
    """测试PatchEmbedding对(B, L, N, C)数据的处理"""
    
    # 模拟您的实际数据
    B, L, N, C = 4, 864, 358, 1
    patch_size = 12
    embed_dim = 96
    
    print("🧪 测试PatchEmbedding修改版本")
    print(f"输入数据维度: (B={B}, L={L}, N={N}, C={C})")
    print(f"Patch配置: patch_size={patch_size}, embed_dim={embed_dim}")
    
    # 创建测试数据
    test_data = torch.randn(B, L, N, C)
    print(f"✅ 测试数据形状: {test_data.shape}")
    
    # 测试单尺度patch embedding
    print("\n📊 测试单尺度Patch Embedding:")
    single_scale_patch = PatchEmbedding(
        patch_size=patch_size,
        in_channel=C,
        embed_dim=embed_dim,
        num_nodes=N,
        topK=6,  # 这个参数现在不影响结果
        norm_layer=None,
        patch_sizes=None  # 单尺度
    )
    
    try:
        output_single = single_scale_patch(test_data)
        expected_patches = L // patch_size  # 864 // 12 = 72
        print(f"✅ 单尺度输出形状: {output_single.shape}")
        print(f"✅ 预期形状: (B={B}, embed_dim={embed_dim}, P={expected_patches}, N={N})")
        assert output_single.shape == (B, embed_dim, expected_patches, N)
        print("✅ 单尺度测试通过！")
    except Exception as e:
        print(f"❌ 单尺度测试失败: {e}")
        return False
    
    # 测试多尺度patch embedding
    print("\n📊 测试多尺度Patch Embedding:")
    patch_sizes = [6, 12, 24]  # 多尺度
    multi_scale_patch = PatchEmbedding(
        patch_size=patch_size,  # 主patch size
        in_channel=C,
        embed_dim=embed_dim,
        num_nodes=N,
        topK=6,
        norm_layer=None,
        patch_sizes=patch_sizes  # 多尺度
    )
    
    try:
        output_multi = multi_scale_patch(test_data)
        min_patches = min([L // p for p in patch_sizes])  # 最小patch数
        print(f"✅ 多尺度输出形状: {output_multi.shape}")
        print(f"✅ 预期形状: (B={B}, embed_dim={embed_dim}, P={min_patches}, N={N})")
        assert output_multi.shape == (B, embed_dim, min_patches, N)
        print("✅ 多尺度测试通过！")
    except Exception as e:
        print(f"❌ 多尺度测试失败: {e}")
        return False
    
    # 测试不同patch size的影响
    print("\n📊 测试不同Patch Size:")
    for p_size in [6, 12, 24]:
        patch_layer = PatchEmbedding(
            patch_size=p_size,
            in_channel=C,
            embed_dim=embed_dim,
            num_nodes=N,
            topK=6,
            norm_layer=None
        )
        
        try:
            output = patch_layer(test_data)
            expected_p = L // p_size
            print(f"  Patch size {p_size}: 输出 {output.shape}, 预期patches={expected_p}")
            assert output.shape == (B, embed_dim, expected_p, N)
        except Exception as e:
            print(f"  ❌ Patch size {p_size} 测试失败: {e}")
            return False
    
    print("\n🎉 所有测试通过！PatchEmbedding已成功适配(B, L, N, C)格式")
    return True

def demonstrate_usage():
    """演示如何在实际模型中使用"""
    print("\n🔧 演示实际使用方法:")
    
    # PEMS03数据集配置
    config = {
        'B': 4,           # batch size
        'L': 864,         # 时间序列长度
        'N': 358,         # 节点数 
        'C': 1,           # 特征维度
        'patch_size': 12, # patch大小
        'embed_dim': 96,  # 嵌入维度
        'patch_sizes': [6, 12, 24]  # 多尺度
    }
    
    # 创建patch embedding层
    patch_embedding = PatchEmbedding(
        patch_size=config['patch_size'],
        in_channel=config['C'],
        embed_dim=config['embed_dim'],
        num_nodes=config['N'],
        topK=6,  # 在新版本中这个参数不影响输出维度
        norm_layer=nn.LayerNorm(config['embed_dim']),  # 可以添加normalization
        patch_sizes=config['patch_sizes']
    )
    
    # 模拟数据
    traffic_data = torch.randn(config['B'], config['L'], config['N'], config['C'])
    print(f"原始交通数据: {traffic_data.shape}")
    
    # Patch embedding
    patches = patch_embedding(traffic_data)
    print(f"Patch嵌入后: {patches.shape}")
    
    # 为AdaptiveGraphLearner准备数据
    # 需要转换为 (B, P, N, D) 格式
    B, D, P, N = patches.shape
    patches_for_graph = patches.permute(0, 2, 3, 1)  # (B, P, N, D)
    print(f"图学习输入格式: {patches_for_graph.shape}")
    
    print("\n📋 数据流程总结:")
    print(f"1. 原始数据: (B={config['B']}, L={config['L']}, N={config['N']}, C={config['C']})")
    print(f"2. Patch嵌入: (B={B}, D={D}, P={P}, N={N})")
    print(f"3. 图学习输入: (B={B}, P={P}, N={N}, D={D})")
    print(f"4. 压缩比: 时间维度从{config['L']}压缩到{P} (压缩{config['L']//P}倍)")

if __name__ == "__main__":
    print("🚀 测试修改后的PatchEmbedding模块\n")
    
    # 运行测试
    if test_patch_embedding():
        demonstrate_usage()
    else:
        print("❌ 测试失败，请检查代码")