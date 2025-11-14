import yaml
import argparse
import logging
import numpy as np
from tqdm import tqdm
import os
import random
import functools

try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    print("Warning: swanlab not installed. Logging features will be disabled.")
from scaler import ZScoreScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader
from easytorch.utils.dist import master_only
from basicts.utils import load_adj, load_pkl

from basicts.data import BasicTSForecastingDataset
from basicts.losses import sce_loss
from basicts.mask.model import AGPSTModel
# from basicts.metrics import masked_mae,masked_mape,masked_rmse
from basicts.metrics import masked_mae, masked_rmse, masked_mape
metrics = {"MAE": masked_mae, "RMSE": masked_rmse, "MAPE": masked_mape}


def collate_fn(batch):
    """
    Custom collate function to add channel dimension to data.
    Converts (B, L, N) to (B, L, N, 1)
    """
    inputs = np.array([item['inputs'] for item in batch])
    targets = np.array([item['targets'] for item in batch])
    
    # Add channel dimension if needed
    if inputs.ndim == 3:  # (B, L, N)
        inputs = inputs[..., np.newaxis]  # (B, L, N, 1)
    if targets.ndim == 3:  # (B, L, N)
        targets = targets[..., np.newaxis]  # (B, L, N, 1)
    
    result = {
        'inputs': torch.from_numpy(inputs).float(),
        'targets': torch.from_numpy(targets).float()
    }
    
    # Handle timestamps if present
    if 'inputs_timestamps' in batch[0]:
        inputs_timestamps = np.array([item['inputs_timestamps'] for item in batch])
        targets_timestamps = np.array([item['targets_timestamps'] for item in batch])
        result['inputs_timestamps'] = torch.from_numpy(inputs_timestamps)
        result['targets_timestamps'] = torch.from_numpy(targets_timestamps)
    
    return result


# def select_input_features(data: torch.Tensor,forward_features) -> torch.Tensor:
#     """Select input features.

#     Args:
#         data (torch.Tensor): input history data, shape [B, L, N, C]

#     Returns:
#         torch.Tensor: reshaped data
#     """

#     # select feature using self.forward_features
#     if forward_features is not None:
#         data = data[:, :, :, forward_features]
#     return data

# def select_target_features(data: torch.Tensor,target_features) -> torch.Tensor:
#     """Select target feature.

#     Args:
#         data (torch.Tensor): prediction of the model with arbitrary shape.

#     Returns:
#         torch.Tensor: reshaped data with shape [B, L, N, C]
#     """

#     # select feature using self.target_features
#     data = data[:, :, :, target_features]
#     return data

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


def validate(val_data_loader, model, config, scaler, epoch, args):
    """验证模型"""
    model.eval()
    
    prediction = []
    real_value = []
    
    with torch.no_grad():
        for idx, data in enumerate(tqdm(val_data_loader, desc='Validation')):
            history_data = data["inputs"]
            future_data = data["targets"]
            history_data = scaler.transform(history_data)
            future_data = scaler.transform(future_data)
            

            
            labels = future_data.to(args.device)
            history_data = history_data.to(args.device)
            
            preds = model(history_data)

            prediction.append(preds.detach().cpu())
            real_value.append(labels.detach().cpu())
            
            # 测试模式：只处理一个batch
            if config.get('test_mode', False):
                print(f"Val test mode: Only processing batch {idx+1}")
                break
        
        prediction = torch.cat(prediction, dim=0)
        real_value = torch.cat(real_value, dim=0)
        
        # 反归一化
        prediction_rescaled = scaler.inverse_transform(prediction)
        real_value_rescaled = scaler.inverse_transform(real_value)

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
            history_data = data["inputs"]
            future_data = data["targets"]
            history_data = scaler.transform(history_data)
            future_data = scaler.transform(future_data)
            
            labels = future_data.to(args.device)
            history_data = history_data.to(args.device)

            preds = model(history_data)

            prediction.append(preds.detach().cpu())
            real_value.append(labels.detach().cpu())
            
            # 测试模式：只处理一个batch
            if config.get('test_mode', False):
                print(f"Test mode: Only processing batch {idx+1}")
                break
        
        prediction = torch.cat(prediction, dim=0)
        real_value = torch.cat(real_value, dim=0)
        
        # 反归一化
        prediction = scaler.inverse_transform(prediction)
        real_value = scaler.inverse_transform(real_value)

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
    adj_mx, _ = load_adj(config['dataset_name'], "doubletransition")
    
    config['backend_args']['supports'] = [torch.tensor(i) for i in adj_mx]
    
    # Initialize scaler
    train_scaler = ZScoreScaler(norm_each_channel=config['norm_each_channel'], rescale=config['rescale'])

    val_scaler = ZScoreScaler(norm_each_channel=config['norm_each_channel'], rescale=config['rescale'])

    test_scaler = ZScoreScaler(norm_each_channel=config['norm_each_channel'], rescale=config['rescale'])

    # train_dataset = BasicTSForecastingDataset(
    #     dataset_name=config['dataset_name'],
    #     input_len=config['input_len'],
    #     output_len=config['output_len'],
    #     mode='train',
    #     use_timestamps=True,
    #     local=config.get('local', True)
    # )
    # print(f'Train data shape: {train_dataset.data.shape}')
    # dd
    # Data loading - use correct file paths
    train_dataset = BasicTSForecastingDataset(
        dataset_name=config['dataset_name'],
        input_len=config['input_len'],
        output_len=config['output_len'],
        mode='train'
    )
    
    val_dataset = BasicTSForecastingDataset(
        dataset_name=config['dataset_name'],
        input_len=config['input_len'],
        output_len=config['output_len'],
        mode='val'
    )
    
    test_dataset = BasicTSForecastingDataset(
        dataset_name=config['dataset_name'],
        input_len=config['input_len'],
        output_len=config['output_len'],
        mode='test'
    )
    # print(f'Train data shape: {train_dataset.data.shape}')
    # Fit scaler on training data
    print('Fitting scaler on training data...')
    # Add channel dimension before fitting: (T, N) -> (T, N, 1)
    train_data = train_dataset.data
    if train_data.ndim == 2:
        train_data = train_data[..., np.newaxis]
    train_scaler.fit(train_data)
    print(f'Train scaler - Mean shape: {train_scaler.stats["mean"].shape}, Std shape: {train_scaler.stats["std"].shape}')
    print(f'Train scaler - Mean: {train_scaler.stats["mean"].mean().item():.4f}, Std: {train_scaler.stats["std"].mean().item():.4f}')
    
    # Fit scaler on validation data
    print('Fitting scaler on validation data...')
    val_data = val_dataset.data
    if val_data.ndim == 2:
        val_data = val_data[..., np.newaxis]
    val_scaler.fit(val_data)
    print(f'Val scaler - Mean shape: {val_scaler.stats["mean"].shape}, Std shape: {val_scaler.stats["std"].shape}')
    print(f'Val scaler - Mean: {val_scaler.stats["mean"].mean().item():.4f}, Std: {val_scaler.stats["std"].mean().item():.4f}')
    
    # Fit scaler on test data
    print('Fitting scaler on test data...')
    test_data = test_dataset.data
    if test_data.ndim == 2:
        test_data = test_data[..., np.newaxis]
    test_scaler.fit(test_data)
    print(f'Test scaler - Mean shape: {test_scaler.stats["mean"].shape}, Std shape: {test_scaler.stats["std"].shape}')
    print(f'Test scaler - Mean: {test_scaler.stats["mean"].mean().item():.4f}, Std: {test_scaler.stats["std"].mean().item():.4f}')
    
    train_data_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=8, shuffle=True, pin_memory=True, collate_fn=collate_fn)
    val_data_loader = DataLoader(val_dataset, batch_size=config['batch_size'], num_workers=8, shuffle=False, pin_memory=True, collate_fn=collate_fn)
    test_data_loader = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=8, shuffle=False, pin_memory=True, collate_fn=collate_fn)
    
    # 创建模型
    model = AGPSTModel(
        num_nodes=config['num_nodes'],
        dim=config['dim'],
        topK=config['topK'],
        in_channel=config['in_channel'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
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

            history_data = data["inputs"]
            future_data = data["targets"]
            # print(f"Original history_data shape: {history_data.shape}, future_data shape: {future_data.shape}")
            history_data = train_scaler.transform(history_data)
            future_data = train_scaler.transform(future_data)

            labels = future_data.to(args.device)
            history_data = history_data.to(args.device)
            # print(f"history_data shape: {history_data.shape}, labels shape: {labels.shape}")
            # 前向传播
            preds = model(history_data)
            
            # 输出格式验证（仅第一个epoch的第一个batch）
            # if idx == 0 and epoch == 0:
            #     print(f"  predictions (model output):    {preds.shape}    -> Expected: (B, 12, 358, 1)")
            #     print("=" * 60)
            prediction_rescaled = train_scaler.inverse_transform(preds)
            # 反归一化
            real_value_rescaled = train_scaler.inverse_transform(labels)
            
            # 计算主损失
            # loss = masked_mae(prediction_rescaled, real_value_rescaled)
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
        val_loss = validate(val_data_loader, model, config, val_scaler, epoch, args)
        
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
        test(test_data_loader, model, config, test_scaler, epoch, args)
    
    print(f"\n🎉 Training completed! Best validation loss: {best_val_loss:.6f}")


def main(config, args):
    # 设置实验名称
    experiment_name = f"{config['dataset_name']}_AGPST_topK{config['topK']}"
    
    if SWANLAB_AVAILABLE:
        swanlab.init(
            project="AGPST-forecasting",
            experiment_name=experiment_name,
            config={
                "mode": "train",
                "dataset": config['dataset_name'],
                "num_nodes": config['num_nodes'],
                "embed_dim": config['embed_dim'],
                "encoder_depth": config['encoder_depth'],
                "epochs": config.get('epochs', config.get('finetune_epochs', 100)),
                "batch_size": config['batch_size'],
                "learning_rate": config['lr'],
                "topK": config['topK'],
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
    parser.add_argument('--config', default='./parameters/PEMS03_v3.yaml', type=str, help='Path to the YAML config file')
    parser.add_argument('--device', default='cpu', type=str, help='device')
    parser.add_argument('--test_mode', default=0, type=int, help='test mode (1) or not (0)')
    parser.add_argument('--swanlab_mode', default='disabled', type=str, help='swanlab mode: online or disabled')
    parser.add_argument('--mode', default='train', type=str, help='training mode (only "train" supported)')
    parser.add_argument('--device_ids', default=0, type=int, help='Number of GPUs available')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    seed_torch(seed=0)
    main(config, args)