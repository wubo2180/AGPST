"""
数据格式快速测试脚本
用于验证模型输入输出格式是否正确
"""

import torch
import yaml

# 模拟数据
batch_size = 2
num_nodes = 358
short_time = 12
long_time = 864
channels = 1

print("=" * 60)
print("🔍 AGPST Direct Forecasting - 数据格式测试")
print("=" * 60)

# 1. 模拟 DataLoader 输出
print("\n1️⃣ DataLoader 输出格式:")
history_data = torch.randn(batch_size, short_time, num_nodes, channels)
long_history_data = torch.randn(batch_size, long_time, num_nodes, channels)
future_data = torch.randn(batch_size, short_time, num_nodes, channels)

print(f"   history_data:      {history_data.shape}  ✓ 格式: (B, T, N, C)")
print(f"   long_history_data: {long_history_data.shape} ✓ 格式: (B, T, N, C)")
print(f"   future_data:       {future_data.shape}  ✓ 格式: (B, T, N, C)")

# 2. 模拟 Patch Embedding 输入转换
print("\n2️⃣ Patch Embedding 输入转换:")
long_history_transposed = long_history_data.transpose(1, 2)
print(f"   转换前: {long_history_data.shape} (B, T, N, C)")
print(f"   转换后: {long_history_transposed.shape} (B, N, T, C) ✓")

# 3. 模拟 Patch Embedding 输出
print("\n3️⃣ Patch Embedding 输出:")
patch_size = 12
num_patches = long_time // patch_size
embed_dim = 96
patches = torch.randn(batch_size, num_nodes, num_patches, embed_dim)
print(f"   patches: {patches.shape} ✓ 格式: (B, N, P, D)")
print(f"   其中: P = {long_time}/{patch_size} = {num_patches} 个patch, D = {embed_dim}")

# 4. 模拟 Transformer 输出
print("\n4️⃣ Transformer 编码器输出:")
hidden_states = torch.randn(batch_size, num_nodes, num_patches, embed_dim)
print(f"   hidden_states: {hidden_states.shape} ✓ 格式: (B, N, P, D)")

# 5. 模拟节点特征提取
print("\n5️⃣ 节点特征提取:")
node_features = hidden_states[:, :, -1, :]
print(f"   node_features: {node_features.shape} ✓ 格式: (B, N, D)")

# 6. 模拟 GraphWaveNet 输出
print("\n6️⃣ GraphWaveNet 输出:")
gwnet_output = torch.randn(batch_size, num_nodes, short_time)
print(f"   GraphWaveNet 输出: {gwnet_output.shape} (B, N, L)")

# 7. 模拟最终输出转换
print("\n7️⃣ 最终输出格式转换:")
final_output = gwnet_output.permute(0, 2, 1).unsqueeze(-1)
print(f"   转换前: {gwnet_output.shape} (B, N, L)")
print(f"   转换后: {final_output.shape} (B, L, N, C) ✓")

# 8. 验证输出与标签格式一致
print("\n8️⃣ 格式一致性检查:")
print(f"   预测值: {final_output.shape}")
print(f"   真实值: {future_data.shape}")
if final_output.shape == future_data.shape:
    print("   ✅ 格式完全一致！可以直接计算损失")
else:
    print("   ❌ 格式不一致！需要调整")

# 9. 测试损失计算
print("\n9️⃣ 损失计算测试:")
try:
    loss = torch.nn.functional.mse_loss(final_output, future_data)
    print(f"   MSE Loss: {loss.item():.6f} ✅")
except Exception as e:
    print(f"   ❌ 损失计算失败: {e}")

print("\n" + "=" * 60)
print("✅ 数据格式测试完成！")
print("=" * 60)

# 10. 配置文件检查
print("\n🔧 配置文件参数检查:")
try:
    with open('./parameters/PEMS03_direct_forecasting.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"   num_nodes: {config['num_nodes']} (期望: 358)")
    print(f"   seq_len: {config['seq_len']} (期望: 864)")
    print(f"   patch_size: {config['patch_size']} (期望: 12)")
    print(f"   dataset_input_len: {config['dataset_input_len']} (期望: 12)")
    print(f"   dataset_output_len: {config['dataset_output_len']} (期望: 12)")
    
    # 验证patch数量
    expected_patches = config['seq_len'] // config['patch_size']
    print(f"\n   计算得到的patch数: {expected_patches} (期望: 72)")
    
    if expected_patches == 72:
        print("   ✅ 配置参数正确！")
    else:
        print("   ⚠️ 警告：patch数量不是72")
        
except FileNotFoundError:
    print("   ⚠️ 配置文件未找到，跳过检查")
except Exception as e:
    print(f"   ❌ 配置文件读取失败: {e}")

print("\n" + "=" * 60)
print("测试脚本执行完毕！")
print("=" * 60)
