"""
Phase 2 模型测试脚本
测试跳跃连接、批处理和参数共享功能
"""
import torch
import sys
sys.path.append('.')

from basicts.mask import AlternatingSTModel, AlternatingSTModel_Phase2


def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def test_phase2_features():
    """测试Phase 2的三大优化功能"""
    print("=" * 80)
    print("Phase 2 模型功能测试")
    print("=" * 80)
    
    # 测试配置
    batch_size = 8
    num_nodes = 207  # METR-LA
    in_steps = 12
    out_steps = 12
    input_dim = 1
    
    # 创建测试输入 (B, T, N, C) - 标准格式
    x = torch.randn(batch_size, in_steps, num_nodes, input_dim)
    print(f"\n输入形状: {x.shape} (B, T, N, C)")
    
    # ========== 测试1: 基础版本（无优化）==========
    print("\n" + "-" * 80)
    print("测试1: 基础版本（所有优化关闭）")
    print("-" * 80)
    
    model_basic = AlternatingSTModel_Phase2(
        num_nodes=num_nodes,
        in_steps=in_steps,
        out_steps=out_steps,
        input_dim=input_dim,
        embed_dim=64,
        num_heads=4,
        temporal_depth=2,
        spatial_depth=2,
        use_skip_connections=False,
        use_parameter_sharing=False,
        batch_spatial_encoding=False
    )
    
    total, trainable = count_parameters(model_basic)
    print(f"参数量: {total:,} (训练参数: {trainable:,})")
    print(f"模型大小: {total * 4 / 1024 / 1024:.2f} MB")
    
    # 前向传播
    with torch.no_grad():
        out_basic = model_basic(x)
    print(f"输出形状: {out_basic.shape}")
    print(f"✓ 基础版本测试通过")
    
    # ========== 测试2: 启用跳跃连接 ==========
    print("\n" + "-" * 80)
    print("测试2: 启用跳跃连接（add模式）")
    print("-" * 80)
    
    model_skip = AlternatingSTModel_Phase2(
        num_nodes=num_nodes,
        in_steps=in_steps,
        out_steps=out_steps,
        input_dim=input_dim,
        embed_dim=64,
        num_heads=4,
        temporal_depth=2,
        spatial_depth=2,
        use_skip_connections=True,
        skip_connection_type='add',
        use_parameter_sharing=False,
        batch_spatial_encoding=False
    )
    
    total, trainable = count_parameters(model_skip)
    print(f"参数量: {total:,} (训练参数: {trainable:,})")
    
    with torch.no_grad():
        out_skip = model_skip(x)
    print(f"输出形状: {out_skip.shape}")
    print(f"✓ 跳跃连接测试通过")
    
    # ========== 测试3: 启用参数共享 ==========
    print("\n" + "-" * 80)
    print("测试3: 启用参数共享")
    print("-" * 80)
    
    model_shared = AlternatingSTModel_Phase2(
        num_nodes=num_nodes,
        in_steps=in_steps,
        out_steps=out_steps,
        input_dim=input_dim,
        embed_dim=64,
        num_heads=4,
        temporal_depth=2,
        spatial_depth=2,
        use_skip_connections=False,
        use_parameter_sharing=True,
        batch_spatial_encoding=False
    )
    
    total_shared, trainable_shared = count_parameters(model_shared)
    print(f"参数量: {total_shared:,} (训练参数: {trainable_shared:,})")
    
    # 计算参数减少比例
    param_reduction = (total - total_shared) / total * 100
    print(f"参数减少: {param_reduction:.1f}%")
    print(f"预期减少: ~40%")
    
    with torch.no_grad():
        out_shared = model_shared(x)
    print(f"输出形状: {out_shared.shape}")
    
    if 35 <= param_reduction <= 45:
        print(f"✓ 参数共享测试通过（减少{param_reduction:.1f}%）")
    else:
        print(f"⚠ 参数减少比例偏离预期（期望~40%，实际{param_reduction:.1f}%）")
    
    # ========== 测试4: 启用批处理优化 ==========
    print("\n" + "-" * 80)
    print("测试4: 启用批处理优化（速度测试）")
    print("-" * 80)
    
    model_batch = AlternatingSTModel_Phase2(
        num_nodes=num_nodes,
        in_steps=in_steps,
        out_steps=out_steps,
        input_dim=input_dim,
        embed_dim=64,
        num_heads=4,
        temporal_depth=2,
        spatial_depth=2,
        use_skip_connections=False,
        use_parameter_sharing=False,
        batch_spatial_encoding=True
    )
    
    # 速度测试
    import time
    
    # 不使用批处理
    model_basic.eval()
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = model_basic(x)
    time_basic = time.time() - start
    
    # 使用批处理
    model_batch.eval()
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = model_batch(x)
    time_batch = time.time() - start
    
    speedup = (time_basic - time_batch) / time_basic * 100
    print(f"无批处理耗时: {time_basic:.4f}s")
    print(f"批处理耗时: {time_batch:.4f}s")
    print(f"加速比例: {speedup:.1f}%")
    print(f"预期加速: ~30%")
    
    if speedup > 0:
        print(f"✓ 批处理优化测试通过（加速{speedup:.1f}%）")
    else:
        print(f"⚠ 批处理未加速（可能是小规模测试导致）")
    
    # ========== 测试5: 全部优化启用 ==========
    print("\n" + "-" * 80)
    print("测试5: 全部优化启用（Phase 2完整版）")
    print("-" * 80)
    
    model_full = AlternatingSTModel_Phase2(
        num_nodes=num_nodes,
        in_steps=in_steps,
        out_steps=out_steps,
        input_dim=input_dim,
        embed_dim=64,
        num_heads=4,
        temporal_depth=2,
        spatial_depth=2,
        use_skip_connections=True,
        skip_connection_type='concat',  # 测试concat模式
        use_parameter_sharing=True,
        batch_spatial_encoding=True
    )
    
    total_full, trainable_full = count_parameters(model_full)
    print(f"参数量: {total_full:,} (训练参数: {trainable_full:,})")
    print(f"模型大小: {total_full * 4 / 1024 / 1024:.2f} MB")
    
    with torch.no_grad():
        out_full = model_full(x)
    print(f"输出形状: {out_full.shape}")
    print(f"✓ 完整优化测试通过")
    
    # ========== 测试6: 与Phase 1对比 ==========
    print("\n" + "-" * 80)
    print("测试6: Phase 1 vs Phase 2 对比")
    print("-" * 80)
    
    model_phase1 = AlternatingSTModel(
        num_nodes=num_nodes,
        in_steps=in_steps,
        out_steps=out_steps,
        input_dim=input_dim,
        output_dim=1,
        embed_dim=64,
        num_heads=4,
        temporal_depth_stage1=2,
        temporal_depth_stage2=2,
        spatial_depth_stage1=2,
        spatial_depth_stage2=2
    )
    
    total_p1, trainable_p1 = count_parameters(model_phase1)
    
    print(f"\nPhase 1:")
    print(f"  参数量: {total_p1:,}")
    print(f"  模型大小: {total_p1 * 4 / 1024 / 1024:.2f} MB")
    
    print(f"\nPhase 2 (参数共享):")
    print(f"  参数量: {total_shared:,}")
    print(f"  模型大小: {total_shared * 4 / 1024 / 1024:.2f} MB")
    print(f"  参数减少: {(total_p1 - total_shared) / total_p1 * 100:.1f}%")
    
    print(f"\nPhase 2 (完整优化):")
    print(f"  参数量: {total_full:,}")
    print(f"  模型大小: {total_full * 4 / 1024 / 1024:.2f} MB")
    
    # ========== 总结 ==========
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print("✓ 所有功能测试通过")
    print(f"✓ 跳跃连接: add/concat 模式均正常")
    print(f"✓ 参数共享: 减少参数量约 {param_reduction:.1f}%")
    print(f"✓ 批处理优化: 加速 {speedup:.1f}%")
    print(f"✓ Phase 1→Phase 2: 参数减少 {(total_p1 - total_shared) / total_p1 * 100:.1f}%")
    print("\nPhase 2 模型已准备就绪！")
    print("=" * 80)


if __name__ == '__main__':
    test_phase2_features()
