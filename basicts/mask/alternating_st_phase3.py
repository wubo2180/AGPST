"""
交替时空编码解码架构 - Phase 3 多阶段循环版本
Alternating Spatio-Temporal Architecture - Phase 3 Multi-Stage Recurrent

核心创新:
1. 多阶段循环 (Multi-Stage Recurrence): 3-5 个编码-解码循环
2. 多尺度特征金字塔 (Multi-Scale Pyramid): 不同 patch size 捕获多分辨率依赖
3. 跨阶段注意力 (Cross-Stage Attention): 每阶段关注所有历史阶段特征

架构对比:
    Phase 1: Input → Stage1 → Decoder → Stage2 → Output (固定2阶段)
    Phase 2: + 跳跃连接 / 参数共享 (失败 ❌)
    Phase 3: Input → [Stage1...StageN] × N (循环) → Output (自适应N)

实验证明 Phase 2 的优化无效 (初始MAE 6.0+ vs Phase 1 的 5.4)
因此直接实现 Phase 3 的革命性改进
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from .alternating_st import (
    TemporalEncoder, 
    SpatialEncoder, 
    FusionLayer, 
    STDecoder
)


class MultiScaleEncoder(nn.Module):
    """
    多尺度编码器：使用不同 patch size 捕获多分辨率时空依赖
    
    Patch sizes: [1, 2, 4]
    - Patch 1: 原始时间步 (细粒度，短期依赖)
    - Patch 2: 每2步聚合 (中粒度，中期依赖)  
    - Patch 4: 每4步聚合 (粗粒度，长期依赖)
    """
    def __init__(
        self,
        num_nodes: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        patch_sizes: List[int] = [1, 2, 4],
        dropout: float = 0.1
    ):
        super().__init__()
        self.patch_sizes = patch_sizes
        self.num_scales = len(patch_sizes)
        
        # 为每个尺度创建独立的编码器
        self.temporal_encoders = nn.ModuleList([
            TemporalEncoder(d_model, num_heads, num_layers, dropout)
            for _ in patch_sizes
        ])
        
        self.spatial_encoders = nn.ModuleList([
            SpatialEncoder(num_nodes, d_model, num_heads, num_layers, dropout)
            for _ in patch_sizes
        ])
        
        # Patch 投影层 (将 patch_size*D 投影回 D)
        self.patch_projections = nn.ModuleDict({
            str(ps): nn.Linear(d_model * ps, d_model) if ps > 1 else nn.Identity()
            for ps in patch_sizes
        })
        
        # 多尺度特征融合
        self.scale_fusion = nn.Sequential(
            nn.Linear(d_model * self.num_scales, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
    
    def _patchify(self, x: torch.Tensor, patch_size: int) -> torch.Tensor:
        """
        将输入分patch并聚合
        Args:
            x: (B, N, T, D)
            patch_size: patch大小
        Returns:
            patched: (B, N, T//patch_size, D)
        """
        if patch_size == 1:
            return x
        
        B, N, T, D = x.shape
        
        # 填充到 patch_size 的倍数
        if T % patch_size != 0:
            pad_size = patch_size - (T % patch_size)
            x = F.pad(x, (0, 0, 0, pad_size), mode='replicate')
            T = T + pad_size
        
        # 重塑: (B, N, T, D) → (B, N, T//patch_size, patch_size, D)
        x = x.reshape(B, N, T // patch_size, patch_size, D)
        
        # 聚合: (B, N, T//patch_size, patch_size, D) → (B, N, T//patch_size, patch_size*D)
        x = x.reshape(B, N, T // patch_size, patch_size * D)
        
        # 投影回 D 维: (B, N, T//patch_size, patch_size*D) → (B, N, T//patch_size, D)
        x = self.patch_projections[str(patch_size)](x)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, T, D)
        Returns:
            multi_scale_features: (B, N, T, D)
        """
        scale_features = []
        
        for patch_size, temp_enc, spat_enc in zip(
            self.patch_sizes, 
            self.temporal_encoders, 
            self.spatial_encoders
        ):
            # Patchify
            x_patched = self._patchify(x, patch_size) if patch_size > 1 else x
            
            # 时空编码
            temp_feat = temp_enc(x_patched)
            spat_feat = spat_enc(x_patched)
            
            # 简单融合
            fused = (temp_feat + spat_feat) / 2
            
            # 如果需要，插值回原始时间步数
            if patch_size > 1:
                B, N, T_patch, D = fused.shape
                target_T = x.shape[2]  # 目标时间步数
                
                # 重塑: (B, N, T_patch, D) → (B*N, D, T_patch)
                fused_flat = fused.reshape(B * N, T_patch, D).permute(0, 2, 1)
                
                # 插值: (B*N, D, T_patch) → (B*N, D, T)
                fused_interp = F.interpolate(
                    fused_flat,
                    size=target_T,
                    mode='linear',
                    align_corners=True
                )
                
                # 重塑回: (B*N, D, T) → (B, N, T, D)
                fused = fused_interp.permute(0, 2, 1).reshape(B, N, target_T, D)
            
            scale_features.append(fused)
        
        # 拼接多尺度特征
        multi_scale = torch.cat(scale_features, dim=-1)  # (B, N, T, D*num_scales)
        
        # 融合
        output = self.scale_fusion(multi_scale)  # (B, N, T, D)
        
        return output


class CrossStageAttention(nn.Module):
    """
    跨阶段注意力：当前阶段关注所有历史阶段的特征
    
    相比简单跳跃连接的优势:
    - 动态权重分配 (而非固定相加)
    - 可以选择性关注重要的历史阶段
    - 支持长距离依赖 (类似 Transformer 的 self-attention)
    """
    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, current: torch.Tensor, history: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            current: (B, N, T, D) - 当前阶段特征
            history: [(B, N, T, D), ...] - 历史阶段特征列表
        Returns:
            attended: (B, N, T, D) - 融合历史信息后的特征
        """
        if not history:
            return current
        
        B, N, T, D = current.shape
        
        # 将空间-时间维度展平: (B, N, T, D) → (B, N*T, D)
        current_flat = current.reshape(B, N * T, D)
        
        # 拼接所有历史特征作为 key/value
        history_flat = torch.stack([h.reshape(B, N * T, D) for h in history], dim=1)
        history_flat = history_flat.reshape(B, len(history) * N * T, D)
        
        # 跨阶段注意力
        attended, _ = self.attention(
            query=current_flat,
            key=history_flat,
            value=history_flat
        )
        
        # 残差连接
        attended = self.norm(current_flat + attended)
        
        # 重塑回原始形状
        attended = attended.reshape(B, N, T, D)
        
        return attended


class AlternatingSTModel_Phase3(nn.Module):
    """
    Phase 3: 多阶段循环交替时空模型
    
    创新点:
    1. num_stages: 可配置的循环阶段数 (3-5)
    2. 每个阶段: 多尺度编码 → 解码 → 跨阶段注意力
    3. 自适应深度: 可根据验证集表现调整阶段数
    
    架构流程:
        Input (B, T, N, C)
          ↓ Embedding
        for stage in 1..N:
            ├─ Multi-Scale Encoding (patch_sizes: [1,2,4])
            ├─ Decoder
            └─ Cross-Stage Attention (attend to stage 1..stage-1)
          ↓
        Output Projection
    """
    
    def __init__(
        self,
        num_nodes: int,
        in_steps: int = 12,
        out_steps: int = 12,
        input_dim: int = 1,
        embed_dim: int = 96,
        num_heads: int = 4,
        num_layers: int = 2,
        num_stages: int = 3,  # Phase 3 核心: 多阶段循环
        patch_sizes: List[int] = [1, 2, 4],  # 多尺度 patch
        dropout: float = 0.1,
        use_cross_stage_attention: bool = True
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.embed_dim = embed_dim
        self.num_stages = num_stages
        self.use_cross_stage_attention = use_cross_stage_attention
        
        print(f"{'='*70}")
        print(f"Phase 3 多阶段循环交替架构")
        print(f"{'='*70}")
        print(f"阶段数: {num_stages}")
        print(f"多尺度 Patch: {patch_sizes}")
        print(f"跨阶段注意力: {use_cross_stage_attention}")
        print(f"{'='*70}\n")
        
        # ============ 输入嵌入 ============
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # ============ 多阶段编码器 ============
        self.multi_scale_encoders = nn.ModuleList([
            MultiScaleEncoder(
                num_nodes=num_nodes,
                d_model=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                patch_sizes=patch_sizes,
                dropout=dropout
            )
            for _ in range(num_stages)
        ])
        
        # ============ 解码器 ============
        self.decoders = nn.ModuleList([
            STDecoder(d_model=embed_dim, dropout=dropout)
            for _ in range(num_stages - 1)  # 最后一个阶段不需要解码器
        ])
        
        # ============ 跨阶段注意力 ============
        if use_cross_stage_attention:
            self.cross_stage_attentions = nn.ModuleList([
                CrossStageAttention(d_model=embed_dim, num_heads=num_heads)
                for _ in range(num_stages)
            ])
        
        # ============ 输出投影 ============
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, input_dim)
        )
    
    def forward(self, history_data: torch.Tensor) -> torch.Tensor:
        """
        Phase 3 多阶段循环前向传播
        
        Args:
            history_data: (B, T, N, C)
        Returns:
            prediction: (B, T_out, N, C)
        """
        B, T, N, C = history_data.shape
        
        # ============ 输入嵌入 ============
        # (B, T, N, C) → (B, N, T, C)
        x = history_data.permute(0, 2, 1, 3)
        x = self.input_embedding(x)  # (B, N, T, D)
        
        # ============ 多阶段循环编码 ============
        stage_features = []  # 保存每个阶段的特征
        
        for stage_idx in range(self.num_stages):
            # 1. 多尺度编码
            encoded = self.multi_scale_encoders[stage_idx](x)
            
            # 2. 跨阶段注意力 (关注历史阶段)
            if self.use_cross_stage_attention and stage_idx > 0:
                encoded = self.cross_stage_attentions[stage_idx](encoded, stage_features)
            
            # 保存当前阶段特征
            stage_features.append(encoded)
            
            # 3. 解码 (除了最后一个阶段)
            if stage_idx < self.num_stages - 1:
                temporal_comp, spatial_comp = self.decoders[stage_idx](encoded)
                # 组合解码结果作为下一阶段输入
                x = (temporal_comp + spatial_comp) / 2
            else:
                # 最后阶段直接用编码结果
                x = encoded
        
        # ============ 输出投影 ============
        output = self.output_projection(x)  # (B, N, T, C)
        
        # 重塑: (B, N, T, C) → (B, T, N, C)
        prediction = output.permute(0, 2, 1, 3)
        
        # 时间步调整
        if self.out_steps != self.in_steps:
            prediction = F.interpolate(
                prediction.squeeze(-1).permute(0, 2, 1),  # (B, N, T)
                size=self.out_steps,
                mode='linear',
                align_corners=True
            ).permute(0, 2, 1).unsqueeze(-1)  # (B, T_out, N, C)
        
        return prediction
    
    def count_parameters(self):
        """统计参数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n{'='*60}")
        print(f"Phase 3 模型参数:")
        print(f"  总参数: {total:,}")
        print(f"  可训练: {trainable:,}")
        print(f"  模型大小: {total * 4 / 1024 / 1024:.2f} MB")
        
        # 对比 Phase 1
        phase1_params = 1_230_000  # 约 1.23M
        increase = (total - phase1_params) / phase1_params * 100
        print(f"  vs Phase 1: {'+' if increase > 0 else ''}{increase:.1f}%")
        print(f"{'='*60}\n")
        
        return total, trainable
