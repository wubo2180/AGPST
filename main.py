import yaml
import argparse
import logging
import numpy as np
from tqdm import tqdm
import functools
import os
import random

try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    print("Warning: swanlab not installed. Logging features will be disabled.")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel

from easytorch.utils.dist import master_only
from basicts.stgcn_arch import STGCN
from basicts.utils import load_adj, load_pkl
from basicts.data import TimeSeriesForecastingDataset
from data import PretrainingDataset
from data import ForecastingDataset

from basicts.data.transform import maskTransforms
from basicts.data import SCALER_REGISTRY
from basicts.losses import sce_loss
from basicts.mask.model import AGPSTModel
from basicts.metrics import masked_mae,masked_mape,masked_rmse
from basicts.utils.utils import metric_forward, select_input_features, select_target_features

metrics = {"MAE": masked_mae, "RMSE": masked_rmse, "MAPE": masked_mape}


def validate(val_data_loader, model, config, scaler, epoch, args):
    """验证模型"""
    model.eval()
    
    prediction = []
    real_value = []
    
    with torch.no_grad():
        for idx, data in enumerate(tqdm(val_data_loader, desc='Validation')):
            future_data, history_data, long_history_data = data
            batch_size = future_data.shape[0]
            
            long_history_data = select_input_features(long_history_data, config['froward_features'])
            history_data = select_input_features(history_data, config['target_features'])
            future_data = select_input_features(future_data, config['target_features'])
            
            labels = future_data.to(args.device)
            history_data = history_data.to(args.device)
            long_history_data = long_history_data.to(args.device)
            
            preds = model(history_data, long_history_data, future_data, batch_size, epoch)
            
            prediction.append(preds.detach().cpu())
            real_value.append(labels.detach().cpu())
            
            # 测试模式：只处理一个batch
            if config.get('test_mode', False):
                print(f"Val test mode: Only processing batch {idx+1}")
                break
        
        prediction = torch.cat(prediction, dim=0)
        real_value = torch.cat(real_value, dim=0)
        
        # 反归一化
        prediction_rescaled = SCALER_REGISTRY.get(scaler["func"])(prediction, **scaler["args"])
        real_value_rescaled = SCALER_REGISTRY.get(scaler["func"])(real_value, **scaler["args"])
        
        # 计算指标
        metric_results = {}
        for metric_name, metric_func in metrics.items():
            metric_item = metric_forward(metric_func, [prediction_rescaled, real_value_rescaled])
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


def test(test_data_loader, model, config, scaler, epoch, args):
    """测试模型"""
    model.eval()
    
    prediction = []
    real_value = []
    
    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_data_loader, desc='Testing')):
            future_data, history_data, long_history_data = data
            batch_size = future_data.shape[0]
            
            long_history_data = select_input_features(long_history_data, config['froward_features'])
            history_data = select_input_features(history_data, config['target_features'])
            future_data = select_input_features(future_data, config['target_features'])
            
            labels = future_data.to(args.device)
            history_data = history_data.to(args.device)
            long_history_data = long_history_data.to(args.device)
            
            preds = model(history_data, long_history_data, future_data, batch_size, epoch)
            
            prediction.append(preds.detach().cpu())
            real_value.append(labels.detach().cpu())
            
            # 测试模式：只处理一个batch
            if config.get('test_mode', False):
                print(f"Test mode: Only processing batch {idx+1}")
                break
        
        prediction = torch.cat(prediction, dim=0)
        real_value = torch.cat(real_value, dim=0)
        
        # 反归一化
        prediction = SCALER_REGISTRY.get(scaler["func"])(prediction, **scaler["args"])
        real_value = SCALER_REGISTRY.get(scaler["func"])(real_value, **scaler["args"])
        
        # 计算每个时间步的指标
        for i in range(config['evaluation_horizons']):
            pred = prediction[:, i, :, :]
            real = real_value[:, i, :, :]
            
            metric_results = {}
            for metric_name, metric_func in metrics.items():
                metric_item = metric_forward(metric_func, [pred, real])
                metric_results[metric_name] = metric_item.item()
            
            print(f"Horizon {i+1:2d} - MAE: {metric_results['MAE']:.4f}, RMSE: {metric_results['RMSE']:.4f}, MAPE: {metric_results['MAPE']:.4f}")
            logging.info(f"Horizon {i+1:2d} - MAE: {metric_results['MAE']:.4f}, RMSE: {metric_results['RMSE']:.4f}, MAPE: {metric_results['MAPE']:.4f}")
        
        # 计算总体指标
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


def train(config, args):
    """
    端到端训练，使用自适应图学习
    """
    print('### Start Training with Adaptive Graph ... ###')
    adj_mx, _ = load_adj(config['adj_dir'], "doubletransition")
    
    config['backend_args']['supports'] = [torch.tensor(i) for i in adj_mx]
    scaler = load_pkl(config['scaler_dir'])
    
    # 数据加载
    train_dataset = ForecastingDataset(config['dataset_dir'], config['dataset_index_dir'], 'train', config['seq_len'])
    val_dataset = ForecastingDataset(config['dataset_dir'], config['dataset_index_dir'], 'valid', config['seq_len'])
    test_dataset = ForecastingDataset(config['dataset_dir'], config['dataset_index_dir'], 'test', config['seq_len'])
    
    train_data_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=8, shuffle=True, pin_memory=True)
    val_data_loader = DataLoader(val_dataset, batch_size=config['batch_size'], num_workers=8, shuffle=False, pin_memory=True)
    test_data_loader = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=8, shuffle=False, pin_memory=True)
    
    # 创建模型
    model = AGPSTModel(
        num_nodes=config['num_nodes'],
        dim=config['dim'],
        topK=config['topK'],
        patch_size=config['patch_size'],
        in_channel=config['in_channel'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        graph_heads=config['graph_heads'],
        mlp_ratio=config['mlp_ratio'],
        dropout=config['dropout'],
        encoder_depth=config['encoder_depth'],
        backend_args=config['backend_args']
    )
    model = model.to(args.device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), config['lr'], weight_decay=1.0e-5, eps=1.0e-8)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    
    for epoch in range(config['epochs']):
        print(f'============ Epoch {epoch}/{config["epochs"]} ============')
        model.train()
        
        epoch_loss = 0.0
        epoch_contrastive_loss = 0.0
        num_batches = 0
        
        for idx, data in enumerate(tqdm(train_data_loader, desc=f'Epoch {epoch}')):
            future_data, history_data, long_history_data = data
            batch_size = future_data.shape[0]
            
            # 数据格式验证（仅第一个epoch的第一个batch）
            if idx == 0 and epoch == 0:
                print("=" * 60)
                print("📊 Data Shape Verification:")
                print(f"  history_data (short-term):     {history_data.shape}  -> Expected: (B, 12, 358, 1)")
                print(f"  long_history_data (long-term): {long_history_data.shape} -> Expected: (B, 864, 358, 1)")
                print(f"  future_data (labels):          {future_data.shape}  -> Expected: (B, 12, 358, 1)")
                print("=" * 60)
            
            # 特征选择
            long_history_data = select_input_features(long_history_data, config['froward_features'])
            history_data = select_input_features(history_data, config['target_features'])
            future_data = select_input_features(future_data, config['target_features'])
            
            # 转移到设备
            labels = future_data.to(args.device)
            history_data = history_data.to(args.device)
            long_history_data = long_history_data.to(args.device)
            
            # 前向传播
            preds = model(history_data, long_history_data, future_data, batch_size, epoch)
            
            # 输出格式验证（仅第一个epoch的第一个batch）
            if idx == 0 and epoch == 0:
                print(f"  predictions (model output):    {preds.shape}    -> Expected: (B, 12, 358, 1)")
                print("=" * 60)
            
            # 反归一化
            prediction_rescaled = SCALER_REGISTRY.get(scaler["func"])(preds, **scaler["args"])
            real_value_rescaled = SCALER_REGISTRY.get(scaler["func"])(labels, **scaler["args"])
            
            # 计算主损失
            loss = metric_forward(masked_mae, [prediction_rescaled, real_value_rescaled])
            
            # 添加对比学习损失（如果存在）
            total_loss = loss
            if hasattr(model, 'contrastive_loss') and model.contrastive_loss is not None:
                contrastive_weight = config.get('contrastive_weight', 0.1)
                if isinstance(model.contrastive_loss, torch.Tensor):
                    total_loss = loss + contrastive_weight * model.contrastive_loss
                    epoch_contrastive_loss += model.contrastive_loss.item()
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # 测试模式：只处理一个batch
            if args.test_mode:
                print(f"Test mode: Only processing batch {idx+1}")
                break
        
        # 计算平均损失
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        avg_contrastive = epoch_contrastive_loss / num_batches if num_batches > 0 else 0.0
        
        print(f"Epoch {epoch} - Train Loss: {avg_loss:.6f}, Contrastive Loss: {avg_contrastive:.6f}")
        
        # 记录到SwanLab
        log_dict = {
            "train/loss": avg_loss,
            "train/lr": optimizer.param_groups[0]['lr']
        }
        if avg_contrastive > 0:
            log_dict["train/contrastive_loss"] = avg_contrastive
        if SWANLAB_AVAILABLE:
            swanlab.log(log_dict, step=epoch)
        
        # 验证
        print('============ Validation ============')
        val_loss = validate(val_data_loader, model, config, scaler, epoch, args)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if config.get("save_model", False):
                model_save_path = config.get("model_save_path", "checkpoints/")
                os.makedirs(model_save_path, exist_ok=True)
                best_model_path = os.path.join(model_save_path, "best_model.pt")
                torch.save(model.state_dict(), best_model_path)
                print(f"✅ Best model saved with val loss: {val_loss:.6f}")
        
        # 测试
        print('============ Test ============')
        test(test_data_loader, model, config, scaler, epoch, args)
    
    print(f"\n🎉 Training completed! Best validation loss: {best_val_loss:.6f}")


def main(config, args):
    # 设置实验名称
    experiment_name = f"{config['dataset_name']}_AGPST_patch{config.get('patch_size', 'none')}_topK{config['topK']}"
    
    if SWANLAB_AVAILABLE:
        swanlab.init(
            project="AGPST-TrafficForecasting",
            experiment_name=experiment_name,
            config={
                "mode": "train",
                "dataset": config['dataset_name'],
                "num_nodes": config['num_nodes'],
                "patch_size": config.get('patch_size', 'none'),
                "embed_dim": config['embed_dim'],
                "encoder_depth": config['encoder_depth'],
                "epochs": config.get('epochs', config.get('finetune_epochs', 100)),
                "batch_size": config['batch_size'],
                "learning_rate": config['lr'],
                "topK": config['topK'],
                "graph_heads": config.get('graph_heads', 1),
                "contrastive_weight": config.get('contrastive_weight', 0.0),
            },
            mode=args.swanlab_mode,
        )

    if args.mode == 'train':
        train(config, args)
    else:
        print(f"Error: Unknown mode '{args.mode}'. Only 'train' mode is supported.")
    
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
    parser.add_argument('--config', default='./parameters/PEMS03_direct_forecasting.yaml', type=str, help='Path to the YAML config file')
    parser.add_argument('--device', default='cuda', type=str, help='device')
    parser.add_argument('--test_mode', default=0, type=int, help='test mode (1) or not (0)')
    parser.add_argument('--swanlab_mode', default='disabled', type=str, help='swanlab mode: online or disabled')
    parser.add_argument('--mode', default='train', type=str, help='training mode (only "train" supported)')
    parser.add_argument('--device_ids', default=0, type=int, help='Number of GPUs available')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    seed_torch(seed=0)
    main(config, args)