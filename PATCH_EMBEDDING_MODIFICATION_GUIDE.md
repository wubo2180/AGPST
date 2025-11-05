"""
PatchEmbedding 修改说明文档
===============================

原始问题：
- 输入数据格式：(B=4, L=864, N=358, C=1) 
- 原代码期望：(B, L, N, K, C) 其中K是topK邻居维度
- 需要适配到实际数据格式

修改方案：
=========

1. 输入维度处理
--------------
原代码:
```python
B, L, N, K, C = long_term_history.shape
long_term_history = long_term_history.permute(0, 4, 1, 2, 3)
```

修改后:
```python
B, L, N, C = long_term_history.shape
# 添加K=1维度以适配Conv3d
long_term_history = long_term_history.unsqueeze(3)  # (B, L, N, 1, C)
long_term_history = long_term_history.permute(0, 4, 1, 2, 3)  # (B, C, L, N, 1)
K = 1  # 明确K维度
```

2. Conv3d kernel_size修改
------------------------
原代码:
```python
kernel_size=(p_size, 1, topK)
```

修改后:
```python
kernel_size=(p_size, 1, 1)  # K维度为1
```

3. 输出维度处理
--------------
原输出: (B, C, L/P, N, K)
新输出: (B, embed_dim, P, N)

- 移除了不必要的K维度
- 更适合后续的AdaptiveGraphLearner处理

4. 多尺度处理优化
---------------
- 使用1D插值而不是2D插值对齐patch数量
- 简化了维度变换过程
- 保持了多尺度融合功能

数据流程示例：
=============

输入: (4, 864, 358, 1)  # B=4, L=864, N=358, C=1
     ↓ 添加K维度
(4, 864, 358, 1, 1)     # B, L, N, K=1, C
     ↓ 维度重排
(4, 1, 864, 358, 1)     # B, C, L, N, K
     ↓ Conv3d (patch_size=12)
(4, 96, 72, 358, 1)     # B, embed_dim, P=L//patch_size, N, K
     ↓ 移除K维度
(4, 96, 72, 358)        # B, embed_dim, P, N
     ↓ 为图学习重排
(4, 72, 358, 96)        # B, P, N, D (用于AdaptiveGraphLearner)

优势：
=====
1. ✅ 完全适配您的数据格式 (B, L, N, C)
2. ✅ 保持原有的多尺度功能
3. ✅ 输出适合AdaptiveGraphLearner使用  
4. ✅ 时间复杂度优化：从864时间步压缩到72个patch
5. ✅ 内存效率提升：减少不必要的维度

在模型中的使用：
===============

```python
# 在 pretrain_model 中使用
def encoding(self, long_term_history, epoch, adp, mask=True):
    # Step 1: Patch Embedding
    patches = self.patch_embedding(long_term_history)  # (B, embed_dim, P, N)
    
    # Step 2: 转换为图学习格式
    B, D, P, N = patches.shape
    patches_for_graph = patches.permute(0, 2, 3, 1)  # (B, P, N, D)
    
    # Step 3: 动态图学习 (这里集成我们之前创建的模块)
    enhanced_patches, learned_adj = self.dynamic_graph_conv(patches_for_graph)
    
    # Step 4: 转回Transformer格式
    enhanced_patches = enhanced_patches.permute(0, 3, 1, 2)  # (B, D, P, N)
    
    # 继续后续处理...
```

测试方法：
=========

1. 创建测试数据：
```python
test_data = torch.randn(4, 864, 358, 1)  # 您的实际数据格式
```

2. 创建PatchEmbedding：
```python
patch_layer = PatchEmbedding(
    patch_size=12,
    in_channel=1,
    embed_dim=96,
    num_nodes=358,
    topK=6,  # 这个参数现在不影响输出维度
    norm_layer=None
)
```

3. 测试输出：
```python
output = patch_layer(test_data)
print(output.shape)  # 应该是 (4, 96, 72, 358)
```

注意事项：
=========
1. topK参数现在主要用于向后兼容，实际的K维度固定为1
2. 输出格式变为(B, embed_dim, P, N)，需要相应调整后续代码
3. 多尺度功能完全保留，可以使用patch_sizes=[6,12,24]
4. 建议在实际使用前进行充分测试

这样修改后，PatchEmbedding就完全适配您的(B=4, L=864, N=358, C=1)数据格式了！
"""