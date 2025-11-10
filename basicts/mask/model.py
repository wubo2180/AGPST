import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicts.data import SCALER_REGISTRY
from easytorch.utils.dist import master_only
from timm.models.vision_transformer import trunc_normal_
from .patch import PatchEmbedding
from .maskgenerator import MaskGenerator
from .positional_encoding import PositionalEncoding
from .transformer_layers import TransformerLayers
from ..graphwavenet import GraphWaveNet
from .post_patch_adaptive_graph import PostPatchDynamicGraphConv


class pretrain_model(nn.Module):
    def __init__(self,num_nodes, 
                 dim, topK, 
                 adaptive, 
                 epochs, patch_size, 
                 in_channel, embed_dim, 
                 num_heads, graph_heads, mlp_ratio, 
                 dropout,  mask_ratio, 
                 encoder_depth, decoder_depth, mode="pre-train") -> None:
        super().__init__()
        assert topK < num_nodes
        self.adaptive = adaptive

        self.lamda = 0.8
        self.epochs = epochs
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.topK = topK
        self.mask_ratio = mask_ratio
        self.selected_feature = 0
        self.mode = mode
        
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, dim), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(dim, num_nodes), requires_grad=True)
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.pos_mat = None
        
        self.patch_embedding = PatchEmbedding(patch_size, in_channel, embed_dim, 
                                              norm_layer=None)
        self.positional_encoding = PositionalEncoding(num_feat=embed_dim)
        self.dynamic_graph_conv = PostPatchDynamicGraphConv(
            embed_dim=embed_dim,
            num_nodes=num_nodes, 
            node_dim=dim,
            graph_heads=graph_heads,
            topk=topK,
            dropout=dropout
        )

        # self.GNN_encoder = nn.Sequential(GIN_layer(nn.Linear(embed_dim, 10)),
        #                                 GIN_layer(nn.Linear(10, embed_dim)))
        
        self.encoder = TransformerLayers(embed_dim, encoder_depth, mlp_ratio, num_heads, dropout)

        self.enc_2_dec_emb = nn.Linear(embed_dim, embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        self.decoder = TransformerLayers(embed_dim, decoder_depth, mlp_ratio, num_heads, dropout)
        # self.GNN_decoder = nn.Sequential(GIN_layer(nn.Linear(embed_dim, 10)),
        #                                 GIN_layer(nn.Linear(10, embed_dim)))

        self.output_layer = nn.Linear(embed_dim, patch_size)
        self.initialize_weights()

    def initialize_weights(self):
        trunc_normal_(self.mask_token, std=.02)

    def encoding(self, long_term_history, epoch, mask=True):
        long_term_history = long_term_history.transpose(1, 2)
        
        # B, L, N, C
        if mask:
            batch_size, num_nodes, num_time, C = long_term_history.shape
            
            patches = self.patch_embedding(long_term_history)  # 输出: (B, N, P, d)
            
            # 🎯 动态图学习 (直接兼容 (B, N, P, D) 格式)
            graph_output = self.dynamic_graph_conv(patches)
            if len(graph_output) == 3:
                patches, learned_adj, contrastive_loss = graph_output
                self.contrastive_loss = contrastive_loss
            else:
                patches, learned_adj = graph_output
                self.contrastive_loss = None
            
            # 位置编码
            patches, self.pos_mat = self.positional_encoding(patches)
 
            if self.adaptive:
                mask_ratio = self.mask_ratio * math.pow(epoch+1 / self.epochs, self.lamda)
            else:
                mask_ratio = self.mask_ratio
                
            Maskg = MaskGenerator(patches.shape[2], mask_ratio)
            unmasked_token_index, masked_token_index = Maskg.uniform_rand()
            
            encoder_input = patches[:, :, unmasked_token_index, :]
            
            hidden_states_unmasked = self.encoder(encoder_input)
            
            hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(batch_size, num_nodes, -1, self.embed_dim)
        else:
            # 推理模式 (不使用mask)
            batch_size, num_nodes, num_time, C = long_term_history.shape
            
            patches = self.patch_embedding(long_term_history)  # (B, N, P, d)

            # 🎯 动态图学习 (直接兼容 (B, N, P, D) 格式)
            graph_output = self.dynamic_graph_conv(patches)
            if len(graph_output) == 3:
                patches, learned_adj, contrastive_loss = graph_output
                self.contrastive_loss = contrastive_loss
            else:
                patches, learned_adj = graph_output
                self.contrastive_loss = None
            
            # 位置编码
            patches, self.pos_mat = self.positional_encoding(patches)
            
            unmasked_token_index, masked_token_index = None, None
            encoder_input = patches
            hidden_states_unmasked = self.encoder(encoder_input)
            hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(batch_size, num_nodes, -1, self.embed_dim)
        return hidden_states_unmasked, unmasked_token_index, masked_token_index
    
    def decoding(self, hidden_states_unmasked, masked_token_index):
        batch_size, num_nodes, num_time, _ = hidden_states_unmasked.shape
        
        if masked_token_index is not None:
            # 训练模式 - 处理masked tokens
            unmasked_token_index = [i for i in range(0, len(masked_token_index)+num_time) if i not in masked_token_index]
            
            # 确保pos_mat不为None
            if self.pos_mat is not None:
                hidden_states_masked = self.pos_mat[:, :, masked_token_index, :]
                hidden_states_masked += self.mask_token.expand(batch_size, num_nodes, len(masked_token_index), hidden_states_unmasked.shape[-1])
                hidden_states_unmasked += self.pos_mat[:, :, unmasked_token_index, :]
                hidden_states_full = torch.cat([hidden_states_unmasked, hidden_states_masked], dim=-2)
            else:
                # 如果pos_mat为None，直接使用unmasked states
                hidden_states_full = hidden_states_unmasked
        else:
            # 推理模式 - 直接使用所有states
            hidden_states_full = hidden_states_unmasked
        
        hidden_states_full = self.decoder(hidden_states_full)
        hidden_states_full = self.decoder_norm(hidden_states_full)

        reconstruction_full = self.output_layer(hidden_states_full.view(batch_size, num_nodes, -1, self.embed_dim))

        return reconstruction_full
    def get_reconstructed_masked_tokens(self, reconstruction_full, real_value_full, unmasked_token_index,
                                        masked_token_index):
        """Get reconstructed masked tokens and corresponding ground-truth for subsequent loss computing.

        Args:
            reconstruction_full (torch.Tensor): reconstructed full tokens.
            real_value_full (torch.Tensor): ground truth full tokens.
            unmasked_token_index (list): unmasked token index.
            masked_token_index (list): masked token index.

        Returns:
            torch.Tensor: reconstructed masked tokens.
            torch.Tensor: ground truth masked tokens.
        """
        batch_size, num_nodes, num_time, _ = reconstruction_full.shape
        reconstruction_masked_tokens = reconstruction_full[:, :, len(unmasked_token_index):, :]
        reconstruction_masked_tokens = reconstruction_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2)
        label_full = real_value_full.permute(0, 3, 1, 2).unfold(1, self.patch_size, self.patch_size)[:, :, :, self.selected_feature, :].transpose(1, 2)
        label_masked_tokens = label_full[:, :, masked_token_index, :].contiguous()
        label_masked_tokens = label_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2)

        return reconstruction_masked_tokens, label_masked_tokens
        
    def forward(self, history_data: torch.Tensor, epoch):
        if self.mode == "pre-train":
            hidden_states_unmasked, unmasked_token_index, masked_token_index = self.encoding(history_data, epoch)
            reconstruction_full = self.decoding(hidden_states_unmasked, masked_token_index)
            
            # 确保unmasked_token_index不为None才进行重建
            if unmasked_token_index is not None and masked_token_index is not None:
                reconstruction_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_tokens(
                    reconstruction_full, history_data.permute(0, 2, 3, 1), 
                    unmasked_token_index, masked_token_index
                )
                # 返回重建结果和对比学习损失
                contrastive_loss = getattr(self, 'contrastive_loss', None)
                return reconstruction_masked_tokens, label_masked_tokens, contrastive_loss
            else:
                # 如果没有mask，返回完整重建
                contrastive_loss = getattr(self, 'contrastive_loss', None)
                return reconstruction_full, history_data.permute(0, 2, 3, 1), contrastive_loss
        else:
            hidden_states_full, _, _ = self.encoding(history_data, epoch, mask=False)
            return hidden_states_full
    def get_mask_ratio(self):
        print(self.mask_ratio)


class finetune_model(nn.Module):

    def __init__(self, pre_trained_path, mask_args, backend_args):
        super().__init__()
        self.pre_trained_path = pre_trained_path
        self.pretrain_model = pretrain_model(**mask_args)
        self.backend = GraphWaveNet(**backend_args)
        if pre_trained_path and os.path.exists(pre_trained_path):
            self.load_pre_trained_model()
        else:
            print(f"Warning: Pre-trained model path '{pre_trained_path}' not found. Using random initialization.")

    def load_pre_trained_model(self):
        """Load pre-trained model with compatibility for both single-scale and multi-scale checkpoints"""
        # checkpoint_dict = torch.load(self.pre_trained_path)
        state_dict = torch.load(self.pre_trained_path, map_location='cpu')  # or 'cuda:0'
        self.pretrain_model.load_state_dict(state_dict)
        print("Pre-trained model loaded successfully")

    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, **kwargs) -> torch.Tensor:
        """Feed forward of STDMAE.

        Args:
            history_data (torch.Tensor): Short-term historical data. shape: [B, L, N, 3]
            long_history_data (torch.Tensor): Long-term historical data. shape: [B, L * P, N, 3]

        Returns:
            torch.Tensor: prediction with shape [B, N, L].
        """
        short_term_history = history_data
        batch_size, _, num_nodes, _ = history_data.shape
        hidden_states = self.pretrain_model(long_history_data, epoch)
        out_len = 1
        hidden_states = hidden_states[:, :, -out_len, :]
        y_hat = self.backend(short_term_history, hidden_states=hidden_states).transpose(1, 2).unsqueeze(-1)

        # 传递对比学习损失给调用者
        self.contrastive_loss = getattr(self.pretrain_model, 'contrastive_loss', None)
        
        return y_hat
