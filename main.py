import yaml
import argparse
import logging
import numpy as np
from tqdm import tqdm
import os
import random
import functools

import torch
import torch.optim as optim
from torch.utils.data import  DataLoader
from basicts.utils import load_adj, load_pkl
from basicts.utils.lr_scheduler import get_scheduler  # 🔥 新增学习率调度器

from basicts.data import BasicTSForecastingDataset
from basicts.mask.alternating_st import AlternatingSTModel
from basicts.graphwavenet import GraphWaveNet
from basicts.scaler import ZScoreScaler
# from basicts.lib.data_prepare import get_dataloaders_from_index_data
from basicts.metrics import masked_mae, masked_rmse, masked_mape
metrics = {"MAE": masked_mae, "RMSE": masked_rmse, "MAPE": masked_mape}


def count_model_parameters(model, verbose=True):
    """
    Count model parameters and print detailed statistics
    
    Args:
        model: PyTorch model
        verbose: Whether to print detailed module-wise statistics
    
    Returns:
        dict: Statistics including total, trainable, and frozen parameters
    """
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    
    module_stats = {}
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        
        if param.requires_grad:
            trainable_params += num_params
        else:
            frozen_params += num_params
        
        # Group by module
        module_name = name.split('.')[0]
        if module_name not in module_stats:
            module_stats[module_name] = {'total': 0, 'trainable': 0}
        module_stats[module_name]['total'] += num_params
        if param.requires_grad:
            module_stats[module_name]['trainable'] += num_params
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"{'📊 Model Parameters Statistics':^80}")
        print(f"{'='*80}")
        print(f"{'Module':<40} {'Parameters':>15} {'Trainable':>15} {'%':>8}")
        print(f"{'-'*80}")
        
        for module_name, stats in sorted(module_stats.items(), key=lambda x: x[1]['total'], reverse=True):
            percentage = stats['total'] / total_params * 100
            print(f"{module_name:<40} {stats['total']:>15,} {stats['trainable']:>15,} {percentage:>7.2f}%")
        
        print(f"{'-'*80}")
        print(f"{'TOTAL':<40} {total_params:>15,} {trainable_params:>15,} {'100.00%':>8}")
        print(f"{'='*80}")
        print(f"  Total Parameters:     {total_params:>15,}")
        print(f"  Trainable Parameters: {trainable_params:>15,}")
        print(f"  Frozen Parameters:    {frozen_params:>15,}")
        print(f"  Model Size (FP32):    {total_params * 4 / 1024 / 1024:>14.2f} MB")
        print(f"  Model Size (FP16):    {total_params * 2 / 1024 / 1024:>14.2f} MB")
        print(f"{'='*80}\n")
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params,
        'modules': module_stats
    }

try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    print("Warning: swanlab not installed. Logging features will be disabled.")

def metric_forward(metric_func, args):
    """Computing metrics.

    Args:
        metric_func (function, functools.partial): metric function.
        args (list): arguments for metrics computation.
    """

    if isinstance(metric_func, functools.partial) and list(metric_func.keywords.keys()) == ["null_val"]:
        # support partial(metric_func, null_val = something)
        metric_item = metric_func(*args)
    elif callable(metric_func):
        # is a function
        metric_item = metric_func(*args)
    else:
        raise TypeError("Unknown metric type: {0}".format(type(metric_func)))
    return metric_item


def validate(val_data_loader, model, adj_mx, cfg, scaler, epoch, args):
    """Validate model"""
    model.eval()
    
    prediction = []
    real_value = []
    
    with torch.no_grad():
        for idx, data in enumerate(tqdm(val_data_loader, desc='Validation', ncols=100)):
            history_data = data["inputs"].unsqueeze(-1)
            future_data = data["targets"].unsqueeze(-1)

            history_data = scaler.transform(history_data)

            labels = future_data.to(args.device)
            history_data = history_data.to(args.device)
            
            preds = model(history_data, adj_mx)
            preds = scaler.inverse_transform(preds)

            prediction.append(preds.detach().cpu())
            real_value.append(labels.detach().cpu())
            
        
        prediction = torch.cat(prediction, dim=0)
        real_value = torch.cat(real_value, dim=0)
        metric_results = {}
        for metric_name, metric_func in metrics.items():
            metric_item = metric_forward(metric_func, [prediction, real_value])
            metric_results[metric_name] = metric_item.item()
        
        val_loss = metric_results["MAE"]
        
        if SWANLAB_AVAILABLE:
            swanlab.log({
                "val/MAE": metric_results["MAE"],
                "val/RMSE": metric_results["RMSE"],
                "val/MAPE": metric_results["MAPE"]
            }, step=epoch)
        
        print(f"Val MAE: {metric_results['MAE']:.4f}, Val RMSE: {metric_results['RMSE']:.4f}, Val MAPE: {metric_results['MAPE']:.4f}")
        logging.info(f"Val MAE: {metric_results['MAE']:.4f}, Val RMSE: {metric_results['RMSE']:.4f}, Val MAPE: {metric_results['MAPE']:.4f}")
    
    return val_loss


def test(test_data_loader, model, adj_mx, cfg, scaler, epoch, args):
    """Test model"""
    model.eval()
    
    prediction = []
    real_value = []
    
    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_data_loader, desc='Testing', ncols=100)):
            history_data = data["inputs"].unsqueeze(-1)
            future_data = data["targets"].unsqueeze(-1)

            history_data = scaler.transform(history_data)

            labels = future_data.to(args.device)
            history_data = history_data.to(args.device)

            preds = model(history_data, adj_mx)
            preds = scaler.inverse_transform(preds)
            
            prediction.append(preds.detach().cpu())
            real_value.append(labels.detach().cpu())
        
        prediction = torch.cat(prediction, dim=0)
        real_value = torch.cat(real_value, dim=0)
        
        for i in range(cfg['evaluation_horizons']):
            pred = prediction[:, i, :, :]
            real = real_value[:, i, :, :]
            
            metric_results = {}
            for metric_name, metric_func in metrics.items():
                metric_item = metric_forward(metric_func, [pred, real])
                metric_results[metric_name] = metric_item.item()
            
            print(f"Horizon {i+1:2d} - MAE: {metric_results['MAE']:.4f}, RMSE: {metric_results['RMSE']:.4f}, MAPE: {metric_results['MAPE']:.4f}")
            logging.info(f"Horizon {i+1:2d} - MAE: {metric_results['MAE']:.4f}, RMSE: {metric_results['RMSE']:.4f}, MAPE: {metric_results['MAPE']:.4f}")
        
        # Calculate overall metrics
        metric_results = {}
        for metric_name, metric_func in metrics.items():
            metric_item = metric_forward(metric_func, [prediction, real_value])
            metric_results[metric_name] = metric_item.item()
        
        if SWANLAB_AVAILABLE:
            swanlab.log({
                "test/MAE": metric_results["MAE"],
                "test/RMSE": metric_results["RMSE"],
                "test/MAPE": metric_results["MAPE"]
            }, step=epoch)
        
        print(f"Overall - Test MAE: {metric_results['MAE']:.4f}, Test RMSE: {metric_results['RMSE']:.4f}, Test MAPE: {metric_results['MAPE']:.4f}")
        logging.info(f"Overall - Test MAE: {metric_results['MAE']:.4f}, Test RMSE: {metric_results['RMSE']:.4f}, Test MAPE: {metric_results['MAPE']:.4f}")


def train(cfg, args):
    """
    End-to-end training with adaptive graph learning
    """
    print('### Start Training with Adaptive Graph ... ###')
    adj_mx, _ = load_adj(cfg['dataset_name'], "doubletransition")
    adj_mx = torch.tensor(adj_mx[0], dtype=torch.float32).to(args.device)
    print("adj_mx.shape:", adj_mx.shape)
    train_dataset = BasicTSForecastingDataset(
        dataset_name=cfg['dataset_name'],
        input_len=cfg['input_len'],
        output_len=cfg['output_len'],
        mode='train'
    )
    val_dataset = BasicTSForecastingDataset(
        dataset_name=cfg['dataset_name'],
        input_len=cfg['input_len'],
        output_len=cfg['output_len'],
        mode='val'
    )
    test_dataset = BasicTSForecastingDataset(
        dataset_name=cfg['dataset_name'],
        input_len=cfg['input_len'],
        output_len=cfg['output_len'],
        mode='test'
    )
    scaler = ZScoreScaler(norm_each_channel=cfg['norm_each_channel'], rescale=cfg['rescale'])
    scaler.fit(train_dataset.data)

    train_data_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], num_workers=8, shuffle=True, pin_memory=True)
    val_data_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], num_workers=8, shuffle=False, pin_memory=True)
    test_data_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], num_workers=8, shuffle=False, pin_memory=True)

    # New alternating spatio-temporal architecture
    print(f"\n{'='*60}")
    print("🚀 Using Alternating Spatio-Temporal Architecture!")
    print(f"{'='*60}")
    model = AlternatingSTModel(
        num_nodes=cfg['num_nodes'],
        in_steps=cfg['input_len'],
        out_steps=cfg['output_len'],
        input_dim=cfg['in_channel'],
        embed_dim=cfg.get('embed_dim', 96),
        num_heads=cfg.get('num_heads', 4),
        temporal_depth_1=cfg.get('temporal_depth_1', 2),
        spatial_depth_1=cfg.get('spatial_depth_1', 2),
        temporal_depth_2=cfg.get('temporal_depth_2', 2),
        spatial_depth_2=cfg.get('spatial_depth_2', 2),
        fusion_type=cfg.get('fusion_type', 'gated'),
        dropout=cfg.get('dropout', 0.05),
        use_spatial_encoder=cfg.get('use_spatial_encoder', True),
        spatial_encoder_type=cfg.get('spatial_encoder_type', 'gcn'),
        use_decomposition = cfg.get('use_decomposition', True),
        decomp_type = cfg.get('decomp_type', 'moving_avg'),
        decomp_kernel_size = cfg.get('decomp_kernel_size', 25),
        # gnn_K=cfg.get('gnn_K', 3),
        use_temporal_encoder=cfg.get('use_temporal_encoder', True),
        use_stage2=cfg.get('use_stage2', True),
        use_denoising=cfg.get('use_denoising', True),
        denoise_type=cfg.get('denoise_type', 'conv'),
    )
    model = model.to(args.device)
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"📊 Model Parameters Statistics")
    print(f"{'='*60}")
    print(f"  Total Parameters:     {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Model Size:           {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    print(f"{'='*60}\n")
    # Optimizer
    optimizer = optim.Adam(model.parameters(), cfg['lr'], weight_decay=cfg['weight_decay'], eps=cfg['eps'])

    # Learning rate scheduler
    # 🔥 优化版 ReduceLROnPlateau (更保守的参数)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=cfg.get('lr_decay_factor', 0.5),      # 衰减因子 (默认 0.5)
        patience=cfg.get('lr_patience', 10),          # 耐心值 (默认 10,原来是 5)
        min_lr=cfg.get('min_lr', 1e-6)               # 最小学习率
    )
    
    print(f"\n{'='*60}")
    print(f"📊 Learning Rate Scheduler: ReduceLROnPlateau")
    print(f"  - Initial LR: {cfg['lr']}")
    print(f"  - Patience: {cfg.get('lr_patience', 10)} epochs")
    print(f"  - Decay Factor: {cfg.get('lr_decay_factor', 0.5)}")
    print(f"  - Min LR: {cfg.get('min_lr', 1e-6)}")
    print(f"{'='*60}\n")
    
    best_val_loss = float('inf')
    
    for epoch in range(cfg['epochs']):
        print(f'============ Epoch {epoch}/{cfg["epochs"]} ============')
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        for idx, data in enumerate(tqdm(train_data_loader, desc=f'Epoch {epoch}', ncols=100)):
            history_data = data["inputs"].unsqueeze(-1)
            future_data = data["targets"].unsqueeze(-1)
            history_data = scaler.transform(history_data)
            future_data = future_data.to(args.device)
            history_data = history_data.to(args.device)
            preds = model(history_data, adj_mx)
            preds = scaler.inverse_transform(preds)
            loss = metric_forward(masked_mae, [preds, future_data])
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
        
        # Calculate average loss
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        
        print(f"Epoch {epoch} - Train Loss: {avg_loss:.6f}")

        # Log to SwanLab
        log_dict = {
            "train/loss": avg_loss,
            "train/lr": optimizer.param_groups[0]['lr']
        }

        if SWANLAB_AVAILABLE:
            swanlab.log(log_dict, step=epoch)
        
        # Validation
        print('============ Validation ============')
        val_loss = validate(val_data_loader, model, adj_mx, cfg, scaler, epoch, args)
        
        # Learning rate scheduling
        # ReduceLROnPlateau 根据验证损失自适应调整
        scheduler.step(val_loss)
        
        # 打印学习率变化
        current_lr = optimizer.param_groups[0]['lr']
        if epoch % 10 == 0:
            print(f"📊 Current Learning Rate: {current_lr:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if cfg.get("save_model", False):
                model_save_path = cfg.get("model_save_path", "checkpoints/")
                os.makedirs(model_save_path, exist_ok=True)
                best_model_path = os.path.join(model_save_path, "best_model.pt")
                torch.save(model.state_dict(), best_model_path)
                print(f"✅ Best model saved with val loss: {val_loss:.6f}")
        
        # Test
        print('============ Test ============')
        test(test_data_loader, model, adj_mx, cfg, scaler, epoch, args)
    
    print(f"\n🎉 Training completed! Best validation loss: {best_val_loss:.6f}")


def main(cfg, args):
    # Set experiment name based on model architecture
    model_name = cfg.get('model_name', 'AGPST')
    dataset_name = cfg['dataset_name']
    
    fusion_type = cfg.get('fusion_type', 'gated')
    experiment_name = f"{dataset_name}_AlternatingST_{fusion_type}_lr{cfg['lr']}_bs{cfg['batch_size']}"
    if SWANLAB_AVAILABLE:
        # Build comprehensive cfg for SwanLab
        swanlab_cfg = {
            # Basic info
            "model_name": model_name,
            "dataset": dataset_name,
            "mode": "train",
            
            # Dataset cfg
            "num_nodes": cfg['num_nodes'],
            "input_len": cfg['input_len'],
            "output_len": cfg['output_len'],
            
            # Training cfg
            "epochs": cfg.get('epochs', 100),
            "batch_size": cfg['batch_size'],
            "learning_rate": cfg['lr'],
            "weight_decay": cfg.get('weight_decay', 1.0e-5),
            
            # Model architecture
            "embed_dim": cfg.get('embed_dim', 96),
            "num_heads": cfg.get('num_heads', 4),
            "dropout": cfg.get('dropout', 0.05),
            
            # Optimization
            "use_denoising": cfg.get('use_denoising', True),
        }
        
        # Add architecture-specific cfg
        swanlab_cfg.update({
            "temporal_depth_1": cfg.get('temporal_depth_1', 2),
            "spatial_depth_1": cfg.get('spatial_depth_1', 2),
            "temporal_depth_2": cfg.get('temporal_depth_2', 2),
            "spatial_depth_2": cfg.get('spatial_depth_2', 2),
            "fusion_type": cfg.get('fusion_type', 'gated'),
        })
    
        swanlab.init(
            project="AGPST-forecasting",
            experiment_name=experiment_name,
            cfg=swanlab_cfg,
            mode=args.swanlab_mode,
        )
    
    train(cfg, args)

    if SWANLAB_AVAILABLE:
        swanlab.finish()

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--cfg', default='./parameters/METR-LA_alternating.yaml', type=str, help='Path to the YAML cfg file')
    parser.add_argument('--device', default='cpu', type=str, help='device')
    parser.add_argument('--swanlab_mode', default='disabled', type=str, help='swanlab mode: online or disabled')

    args = parser.parse_args()
    
    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    seed_torch(seed=0)
    main(cfg, args)