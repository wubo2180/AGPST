"""
改进的学习率调度器实现

问题分析:
1. 当前使用 ReduceLROnPlateau(factor=0.5, patience=5)
   - 每 5 epochs 不改进就减半 lr
   - 导致 ~20 epoch 时 lr 骤降到接近 0
   - 后续训练停滞

优化方案:
1. Warmup + Cosine Annealing (推荐)
2. Multi-Step LR with Warmup
3. ReduceLROnPlateau 但更保守的参数
"""

import torch
import torch.optim as optim
import math


class WarmupCosineScheduler:
    """
    Warmup + Cosine Annealing 学习率调度器
    
    阶段 1 (Warmup): 线性增长 0 → base_lr (前 warmup_epochs)
    阶段 2 (Cosine): 余弦衰减 base_lr → min_lr (剩余 epochs)
    
    优势:
    - Warmup 防止初期震荡
    - Cosine 平滑衰减,不会突然降到 0
    - 后期保持小但非零的学习率,持续优化
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
    
    def step(self):
        """更新学习率"""
        if self.current_epoch < self.warmup_epochs:
            # Warmup 阶段: 线性增长
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine Annealing 阶段
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr
    
    def get_last_lr(self):
        """返回当前学习率"""
        return [group['lr'] for group in self.optimizer.param_groups]


class WarmupMultiStepLR:
    """
    Warmup + Multi-Step 学习率调度器
    
    阶段 1 (Warmup): 线性增长 0 → base_lr
    阶段 2 (Multi-Step): 在指定 epoch 降低学习率
    
    优势:
    - 更可控的衰减时机
    - 适合已知最佳衰减点的场景
    """
    def __init__(self, optimizer, warmup_epochs, milestones, gamma, base_lr):
        """
        Args:
            warmup_epochs: Warmup 轮数
            milestones: 衰减的 epoch 列表,如 [30, 60, 90]
            gamma: 衰减因子,如 0.1
            base_lr: 基础学习率
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.milestones = milestones
        self.gamma = gamma
        self.base_lr = base_lr
        self.current_epoch = 0
    
    def step(self):
        """更新学习率"""
        if self.current_epoch < self.warmup_epochs:
            # Warmup 阶段
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Multi-Step 阶段
            lr = self.base_lr
            for milestone in self.milestones:
                if self.current_epoch >= milestone:
                    lr *= self.gamma
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


class ImprovedReduceLROnPlateau:
    """
    改进的 ReduceLROnPlateau
    
    改进:
    1. 添加 Warmup
    2. 更保守的 patience (10-15)
    3. 更温和的 factor (0.5 → 0.7)
    4. 添加 min_lr 限制
    """
    def __init__(self, optimizer, warmup_epochs, patience=10, factor=0.7, 
                 min_lr=1e-6, base_lr=0.001):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
        
        # ReduceLROnPlateau (warmup 后启用)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=True
        )
    
    def step(self, val_loss=None):
        """更新学习率"""
        if self.current_epoch < self.warmup_epochs:
            # Warmup 阶段
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.current_epoch += 1
            return lr
        else:
            # ReduceLROnPlateau 阶段
            if val_loss is not None:
                self.scheduler.step(val_loss)
            self.current_epoch += 1
            return self.get_last_lr()[0]
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


# ============================================================================
# 使用示例
# ============================================================================

def get_scheduler(optimizer, config):
    """
    根据配置返回学习率调度器
    
    Args:
        optimizer: PyTorch 优化器
        config: 配置字典
    
    Returns:
        scheduler: 学习率调度器
    """
    scheduler_type = config.get('scheduler_type', 'warmup_cosine')
    base_lr = config['lr']
    total_epochs = config['epochs']
    
    if scheduler_type == 'warmup_cosine':
        # 推荐: Warmup + Cosine Annealing
        warmup_epochs = config.get('warmup_epochs', 5)
        min_lr = config.get('min_lr', 1e-6)
        
        scheduler = WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            base_lr=base_lr,
            min_lr=min_lr
        )
        
    elif scheduler_type == 'warmup_multistep':
        # 备选: Warmup + Multi-Step
        warmup_epochs = config.get('warmup_epochs', 5)
        milestones = config.get('milestones', [30, 60, 90])
        gamma = config.get('gamma', 0.1)
        
        scheduler = WarmupMultiStepLR(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            milestones=milestones,
            gamma=gamma,
            base_lr=base_lr
        )
        
    elif scheduler_type == 'reduce_on_plateau':
        # 改进的 ReduceLROnPlateau
        warmup_epochs = config.get('warmup_epochs', 5)
        patience = config.get('patience', 10)
        factor = config.get('factor', 0.7)
        min_lr = config.get('min_lr', 1e-6)
        
        scheduler = ImprovedReduceLROnPlateau(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            patience=patience,
            factor=factor,
            min_lr=min_lr,
            base_lr=base_lr
        )
        
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler
