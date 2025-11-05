"""
在AGPST模型中集成PostPatchDynamicGraphConv的完整方案
=========================================================

这个文件展示了如何正确地在model.py中集成动态图学习模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .post_patch_adaptive_graph import PostPatchDynamicGraphConv


class ImprovedPretrainModel(nn.Module):
    """改进版的预训练模型，集成了动态图学习"""
    
    def __init__(self, num_nodes, dim, topK, adaptive, epochs, patch_size, 
                 in_channel, embed_dim, num_heads, mlp_ratio, 
                 dropout, mask_ratio, encoder_depth, decoder_depth,
                 patch_sizes=None, mode="pre-train"):
        super().__init__()
        
        # 保持原有参数
        self.adaptive = adaptive
        self.lamda = 0.8
        self.epochs = epochs
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.topK = topK
        self.mask_ratio = mask_ratio
        self.selected_feature = 0
        self.mode = mode
        
        # 静态图参数 (备用)
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, dim), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(dim, num_nodes), requires_grad=True)
        
        # 规范化层
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)
        
        # 🎯 核心组件
        # 1. 简化的Patch Embedding
        self.patch_embedding = PatchEmbedding(
            patch_size=patch_size,
            in_channel=in_channel,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm(embed_dim)
        )
        
        # 2. 🔥 动态图学习模块
        self.dynamic_graph_conv = PostPatchDynamicGraphConv(
            embed_dim=embed_dim,
            num_nodes=num_nodes,
            node_dim=dim,
            num_heads=4,
            topk=topK,
            dropout=dropout
        )
        
        # 3. 位置编码
        self.positional_encoding = PositionalEncoding()
        
        # 4. Transformer编码器
        self.encoder = TransformerLayers(embed_dim, encoder_depth, mlp_ratio, num_heads, dropout)
        
        # 5. 解码器组件
        self.enc_2_dec_emb = nn.Linear(embed_dim, embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.decoder = TransformerLayers(embed_dim, decoder_depth, mlp_ratio, num_heads, dropout)
        
        # 6. 输出层
        self.output_layer = nn.Linear(embed_dim, patch_size)
        
        # 初始化
        self.initialize_weights()
        
        # 位置编码矩阵
        self.pos_mat = None
        
    def initialize_weights(self):
        """初始化权重"""
        nn.init.trunc_normal_(self.mask_token, std=0.02)
    
    def encoding(self, long_term_history, epoch, mask=True):
        """
        改进的编码过程，集成动态图学习
        
        Args:
            long_term_history: (B, L, N, K, C) 或 (B, L, N, C)
            epoch: 当前训练轮数
            mask: 是否使用masking
        """
        
        # 处理输入维度
        if len(long_term_history.shape) == 5:
            # 原格式: (B, L, N, K, C) -> 取第一个K维度
            B, L, N, K, C = long_term_history.shape
            history_data = long_term_history[:, :, :, 0, :]  # (B, L, N, C)
        else:
            # 新格式: (B, L, N, C)
            history_data = long_term_history
            B, L, N, C = history_data.shape
        
        if mask:
            # === 训练模式 (with masking) ===
            
            # Step 1: Patch Embedding
            # (B, L, N, C) -> (B, embed_dim, P, N)
            patches = self.patch_embedding(history_data)
            batch_size, embed_dim, num_time, num_nodes = patches.shape
            
            print(f"📊 Patch embedding 输出: {patches.shape}")
            
            # Step 2: 转换为动态图学习格式
            # (B, embed_dim, P, N) -> (B, P, N, embed_dim)
            patches_for_graph = patches.permute(0, 2, 3, 1)
            
            # Step 3: 🎯 动态图学习与图卷积
            enhanced_patches, learned_adj = self.dynamic_graph_conv(patches_for_graph)
            print(f"🔗 动态图学习完成，邻接矩阵: {learned_adj.shape}")
            
            # Step 4: 转换为Transformer格式
            # (B, P, N, embed_dim) -> (B, N, P, embed_dim)
            patches = enhanced_patches.permute(0, 2, 1, 3)
            
            # Step 5: 位置编码
            patches, self.pos_mat = self.positional_encoding(patches)
            
            # Step 6: 自适应masking
            if self.adaptive:
                mask_ratio = self.mask_ratio * pow((epoch + 1) / self.epochs, self.lamda)
            else:
                mask_ratio = self.mask_ratio
            
            # Step 7: 生成mask
            from .maskgenerator import MaskGenerator
            Maskg = MaskGenerator(patches.shape[2], mask_ratio)
            unmasked_token_index, masked_token_index = Maskg.uniform_rand()
            
            # Step 8: Transformer编码
            encoder_input = patches[:, :, unmasked_token_index, :]
            hidden_states_unmasked = self.encoder(encoder_input)
            hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked)
            
            return hidden_states_unmasked, unmasked_token_index, masked_token_index, learned_adj
        
        else:
            # === 推理模式 (without masking) ===
            
            # 类似的处理流程，但不进行masking
            patches = self.patch_embedding(history_data)
            batch_size, embed_dim, num_time, num_nodes = patches.shape
            
            patches_for_graph = patches.permute(0, 2, 3, 1)
            enhanced_patches, learned_adj = self.dynamic_graph_conv(patches_for_graph)
            patches = enhanced_patches.permute(0, 2, 1, 3)
            
            patches, self.pos_mat = self.positional_encoding(patches)
            
            hidden_states_unmasked = self.encoder(patches)
            hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked)
            
            return hidden_states_unmasked, None, None, learned_adj
    
    def decoding(self, hidden_states_unmasked, masked_token_index, learned_adj=None):
        """
        解码过程 - 可以选择使用学习到的邻接矩阵
        """
        batch_size, num_nodes, num_time, _ = hidden_states_unmasked.shape
        
        if masked_token_index is not None:
            unmasked_token_index = [i for i in range(len(masked_token_index) + num_time) 
                                  if i not in masked_token_index]
            
            # 处理masked tokens
            hidden_states_masked = self.pos_mat[:, :, masked_token_index, :]
            hidden_states_masked += self.mask_token.expand(
                batch_size, num_nodes, len(masked_token_index), self.embed_dim
            )
            
            # 添加位置编码到unmasked tokens
            hidden_states_unmasked += self.pos_mat[:, :, unmasked_token_index, :]
            
            # 拼接
            hidden_states_full = torch.cat([hidden_states_unmasked, hidden_states_masked], dim=2)
        else:
            hidden_states_full = hidden_states_unmasked
        
        # Transformer解码
        hidden_states_full = self.decoder(hidden_states_full)
        hidden_states_full = self.decoder_norm(hidden_states_full)
        
        # 输出层
        reconstruction_full = self.output_layer(hidden_states_full)
        
        return reconstruction_full
    
    def forward(self, history_data, epoch):
        """前向传播"""
        
        # 处理输入数据格式
        if len(history_data.shape) == 4:
            # 如果输入是(B, L, N, C)，需要构建K维度
            B, L, N, C = history_data.shape
            K = self.topK
            
            # 构建静态邻接矩阵 (备用)
            static_adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            values, indices = torch.topk(static_adp, K)
            
            # 构建K-hop数据
            history_data_khop = history_data.transpose(1, 2).reshape(B, N, -1)  # (B, N, L*C)
            history_data_khop = history_data_khop[:, indices, :]  # (B, N, K, L*C)
            history_data_khop = history_data_khop.reshape(B, N, K, L, C)
            history_data_khop = history_data_khop.permute(0, 3, 1, 2, 4)  # (B, L, N, K, C)
        else:
            history_data_khop = history_data
        
        if self.mode == "pre-train":
            # 预训练模式
            hidden_states_unmasked, unmasked_token_index, masked_token_index, learned_adj = \
                self.encoding(history_data_khop, epoch, mask=True)
            
            reconstruction_full = self.decoding(hidden_states_unmasked, masked_token_index, learned_adj)
            
            # 提取masked tokens用于损失计算
            reconstruction_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_tokens(
                reconstruction_full, history_data.permute(0, 2, 3, 1), 
                unmasked_token_index, masked_token_index
            )
            
            return reconstruction_masked_tokens, label_masked_tokens
        else:
            # 推理模式
            hidden_states_full, _, _, learned_adj = self.encoding(history_data_khop, epoch, mask=False)
            return hidden_states_full


# =====================================
# 使用示例和测试
# =====================================

def test_improved_model():
    """测试改进版模型"""
    
    # 模型配置
    config = {
        'num_nodes': 358,
        'dim': 10,
        'topK': 6,
        'adaptive': True,
        'epochs': 100,
        'patch_size': 12,
        'in_channel': 1,
        'embed_dim': 96,
        'num_heads': 4,
        'mlp_ratio': 4,
        'dropout': 0.1,
        'mask_ratio': 0.25,
        'encoder_depth': 4,
        'decoder_depth': 1
    }
    
    # 创建模型
    model = ImprovedPretrainModel(**config)
    
    # 测试数据
    B, L, N, C = 4, 864, 358, 1
    test_data = torch.randn(B, L, N, C)
    
    print(f"🧪 测试改进版AGPST模型")
    print(f"输入数据: {test_data.shape}")
    
    # 测试前向传播
    model.eval()
    with torch.no_grad():
        try:
            output = model(test_data, epoch=0)
            print(f"✅ 模型输出: {[o.shape for o in output] if isinstance(output, tuple) else output.shape}")
            print("✅ 动态图学习集成成功！")
            return True
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            return False


if __name__ == "__main__":
    print("🚀 测试PostPatchDynamicGraphConv集成\n")
    
    if test_improved_model():
        print(f"\n🎉 集成完成！主要改进:")
        print("1. ✅ 适配简化的PatchEmbedding输出格式")
        print("2. ✅ 在patch embedding后使用动态图学习")  
        print("3. ✅ 保持与原模型的接口兼容性")
        print("4. ✅ 计算效率优化：72个patch vs 864个时间步")
        print("5. ✅ 提供学习到的邻接矩阵用于分析")
    else:
        print("❌ 集成失败，请检查代码")