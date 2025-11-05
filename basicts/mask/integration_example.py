# ========================================================================
# 在AGPST模型中集成PostPatch AdaptiveGraphLearner的完整示例
# ========================================================================

"""
集成方案：在patch embedding之后，GNN encoder之前使用动态图学习

数据流程：
(B=4, L=864, N=358, C=1) 
-> PatchEmbedding -> (B=4, P=72, N=358, D=96)  # patch_size=12
-> PostPatchAdaptiveGraphLearner -> 学习动态邻接矩阵 (B, N, N)
-> 使用动态图进行GNN encoding
"""

import torch
import torch.nn as nn
from .post_patch_adaptive_graph import PostPatchDynamicGraphConv

class ImprovedPretrainModel(nn.Module):
    def __init__(self, num_nodes, dim, topK, adaptive, epochs, patch_size, 
                 in_channel, embed_dim, num_heads, mlp_ratio, 
                 dropout, mask_ratio, encoder_depth, decoder_depth,
                 patch_sizes=None, mode="pre-train"):
        super().__init__()
        
        # ... 其他初始化代码保持不变 ...
        
        # === 核心改进：添加动态图学习模块 ===
        self.use_dynamic_graph = True
        if self.use_dynamic_graph:
            self.dynamic_graph_conv = PostPatchDynamicGraphConv(
                embed_dim=embed_dim,
                num_nodes=num_nodes,
                node_dim=dim,
                num_heads=4,
                topk=topK,
                dropout=dropout
            )
            print(f"✅ 启用动态图学习: {num_nodes}节点, Top-{topK}稀疏化")
        
        # 其他组件保持不变
        self.patch_embedding = PatchEmbedding(patch_size, in_channel, embed_dim, 
                                              num_nodes, topK, norm_layer=None, 
                                              patch_sizes=patch_sizes)
        # ... 其余组件 ...

    def encoding(self, long_term_history, epoch, adp, mask=True):
        """改进的编码过程，集成动态图学习"""
        
        if mask:
            # Step 1: Patch Embedding (维度变换)
            # (B=4, L=864, N=358, C=1) -> (B=4, C=96, P=72, N=358, K=6)
            patches = self.patch_embedding(long_term_history)
            batch_size, num_dim, num_time, num_nodes, khop = patches.shape
            
            # 调整维度为 (B, P, N, D)
            patches = patches.squeeze(-1).permute(0, 2, 3, 1)  # (B, P, N, D)
            print(f"📊 Patch嵌入后维度: {patches.shape}")
            
            # Step 2: 位置编码 (保持原有逻辑)
            patches, self.pos_mat = self.positional_encoding(patches.permute(0, 3, 1, 2))  # 临时调整维度
            patches = patches.permute(0, 2, 3, 1)  # 调回 (B, P, N, D)
            
            # Step 3: 🎯 核心改进 - 动态图学习与图卷积
            if self.use_dynamic_graph:
                # 使用动态图进行编码
                patches_enhanced, learned_adjs = self.dynamic_graph_conv(patches)
                print(f"🔗 学习到的动态图: {learned_adjs.shape}")
                
                # 可选：可视化学到的图结构 (调试时使用)
                if torch.rand(1).item() < 0.01:  # 1%概率打印
                    avg_adj = learned_adjs.mean(0)  # (N, N)
                    sparsity = (avg_adj > 0.01).float().mean().item()
                    print(f"📈 图稀疏度: {sparsity:.3f}, 平均度数: {avg_adj.sum(1).mean():.2f}")
                
                patches = patches_enhanced
            else:
                # 使用原有的静态图 (备选方案)
                patches = patches.permute(0, 3, 1, 2)  # (B, D, P, N) for original GNN
                patches, _ = self.GNN_encoder((patches, adp))
                patches = patches.permute(0, 2, 3, 1)  # 调回 (B, P, N, D)
            
            # Step 4: Transformer编码 (保持原有逻辑)
            patches = patches.permute(0, 2, 1, 3)  # (B, N, P, D) for transformer
            
            # 自适应mask ratio
            if self.adaptive:
                mask_ratio = self.mask_ratio * math.pow((epoch+1) / self.epochs, self.lamda)
            else:
                mask_ratio = self.mask_ratio
                
            # Masking
            Maskg = MaskGenerator(patches.shape[2], mask_ratio)
            unmasked_token_index, masked_token_index = Maskg.uniform_rand()
            
            encoder_input = patches[:, :, unmasked_token_index, :]
            hidden_states_unmasked = self.encoder(encoder_input)
            hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked)
            
        else:
            # 推理模式 (不使用mask)
            # ... 类似的流程，但不进行masking ...
            pass
            
        return hidden_states_unmasked, unmasked_token_index, masked_token_index

    def forward(self, long_term_history, epoch):
        """前向传播"""
        
        # 🔧 改进：构建动态邻接矩阵
        # 注意：这里我们让动态图学习在encoding内部完成
        # 不需要预先构建adp，而是让模型自己学习
        
        # 静态图作为备选 (可选)
        if hasattr(self, 'nodevec1') and hasattr(self, 'nodevec2'):
            static_adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        else:
            static_adp = None
        
        # 编码
        hidden_states_unmasked, unmasked_token_index, masked_token_index = self.encoding(
            long_term_history, epoch, static_adp, mask=True
        )
        
        # 解码
        reconstruction_full = self.decoding(hidden_states_unmasked, masked_token_index, static_adp)
        
        # 提取masked tokens
        reconstruction_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_tokens(
            reconstruction_full, long_term_history, unmasked_token_index, masked_token_index
        )
        
        return reconstruction_masked_tokens, label_masked_tokens


# ========================================================================
# 使用示例和配置建议
# ========================================================================

def create_improved_model(config):
    """创建改进版AGPST模型"""
    
    model = ImprovedPretrainModel(
        num_nodes=config['num_nodes'],      # 358
        dim=config['dim'],                  # 10
        topK=config['topK'],               # 6
        adaptive=config['adaptive'],        # True
        epochs=config['pretrain_epochs'],   # 100
        patch_size=config['patch_size'],    # 12
        in_channel=config['in_channel'],    # 1
        embed_dim=config['embed_dim'],      # 96
        num_heads=config['num_heads'],      # 4
        mlp_ratio=config['mlp_ratio'],      # 4
        dropout=config['dropout'],          # 0.1
        mask_ratio=config['mask_ratio'],    # 0.25
        encoder_depth=config['encoder_depth'],  # 4
        decoder_depth=config['decoder_depth'],  # 1
        patch_sizes=config['patch_sizes']   # [6, 12, 24]
    )
    
    return model


# ========================================================================
# 性能分析和调试工具
# ========================================================================

class GraphAnalyzer:
    """分析动态图学习效果的工具类"""
    
    @staticmethod
    def analyze_learned_graphs(model, dataloader, num_samples=5):
        """分析模型学到的图结构"""
        model.eval()
        graph_stats = []
        
        with torch.no_grad():
            for i, (_, history_data) in enumerate(dataloader):
                if i >= num_samples:
                    break
                    
                # 获取patch特征
                patches = model.patch_embedding(history_data)
                patches = patches.squeeze(-1).permute(0, 2, 3, 1)  # (B, P, N, D)
                
                # 获取动态图
                if hasattr(model, 'dynamic_graph_conv'):
                    _, learned_adjs = model.dynamic_graph_conv(patches)
                    
                    # 统计图属性
                    avg_adj = learned_adjs.mean(0)  # (N, N)
                    sparsity = (avg_adj > 0.01).float().mean().item()
                    avg_degree = avg_adj.sum(1).mean().item()
                    max_degree = avg_adj.sum(1).max().item()
                    
                    graph_stats.append({
                        'sparsity': sparsity,
                        'avg_degree': avg_degree,
                        'max_degree': max_degree,
                        'connectivity': (avg_adj.sum(1) > 0).float().mean().item()
                    })
        
        # 打印统计信息
        if graph_stats:
            print("\n📊 动态图学习统计:")
            for key in graph_stats[0].keys():
                values = [stat[key] for stat in graph_stats]
                print(f"  {key}: 均值={np.mean(values):.3f}, 标准差={np.std(values):.3f}")


# ========================================================================
# 配置文件建议修改
# ========================================================================

"""
# parameters/PEMS03_multiscale.yaml 建议修改:

# 1. 增强动态图学习
topK: 8  # 原6 -> 8 (稍微增加连接)
adaptive: True  # 保持自适应

# 2. 优化embedding维度
embed_dim: 128  # 原96 -> 128 (增加表达能力)

# 3. 调整训练参数
lr: 0.0015  # 原0.002 -> 0.0015 (更稳定)
mask_ratio: 0.4  # 原0.25 -> 0.4 (更强自监督)

# 4. 增加模型深度
encoder_depth: 6  # 原4 -> 6
decoder_depth: 2  # 原1 -> 2
"""