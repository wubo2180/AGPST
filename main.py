import yaml
import argparse
import logging
import numpy as np
from tqdm import tqdm
import functools
import os
import random
import swanlab

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
from basicts.mask.model import pretrain_model, finetune_model
from basicts.metrics import masked_mae,masked_mape,masked_rmse
from basicts.utils.utils import metric_forward, select_input_features, select_target_features

metrics = {"MAE": masked_mae, "RMSE": masked_rmse, "MAPE": masked_mape, "SCE": sce_loss}


def val(val_data_loader, model, config, scaler, epoch):
    model.eval()
    
    prediction = []
    real_value = []
    with torch.no_grad():
        for idx, data in enumerate(tqdm(val_data_loader)):
            future_data, history_data, long_history_data = data
            batch_size = future_data.shape[0]
            long_history_data = select_input_features(long_history_data, config['froward_features'])
            history_data = select_input_features(history_data, config['target_features'])
            future_data = select_input_features(future_data, config['target_features'])
            
            labels = future_data.to(config['device'])
            history_data = history_data.to(config['device'])
            long_history_data = long_history_data.to(config['device'])
            
            preds = model(history_data, long_history_data, future_data, batch_size, epoch)

            prediction.append(preds.detach().cpu())        # preds = forward_return[0]
            real_value.append(labels.detach().cpu())        # testy = forward_return[1]
            
            # 测试模式：只处理一个batch
            if config.get('test_mode', False):
                print(f"Val test mode: Only processing batch {idx+1}")
                break

        prediction = torch.cat(prediction, dim=0)
        real_value = torch.cat(real_value, dim=0)
        # re-scale data
        prediction_rescaled = SCALER_REGISTRY.get(scaler["func"])(prediction, **scaler["args"])
        real_value_rescaled = SCALER_REGISTRY.get(scaler["func"])(real_value, **scaler["args"])
        # print(real_value_rescaled)
        # dd
        metric_results = {}
        for metric_name, metric_func in metrics.items():
            metric_item = metric_forward(metric_func, [prediction_rescaled, real_value_rescaled])
            metric_results[metric_name] = metric_item.item()
        
        swanlab.log({
            "val/MAE": metric_results["MAE"],
            "val/RMSE": metric_results["RMSE"],
            "val/MAPE": metric_results["MAPE"]
        })
        
        print("Evaluate val data" + \
                    "val MAE: {:.4f}, val RMSE: {:.4f}, val MAPE: {:.4f}".format(metric_results["MAE"], metric_results["RMSE"], metric_results["MAPE"]))
        logging.info("Evaluate val data" + \
                    "val MAE: {:.4f}, val RMSE: {:.4f}, val MAPE: {:.4f}".format(metric_results["MAE"], metric_results["RMSE"], metric_results["MAPE"]))
def test(test_data_loader, model, config, scaler, epoch):
    """Evaluate the model.

    Args:
        train_epoch (int, optional): current epoch if in training process.
    """
    model.eval()
    # test loop
    

    prediction = []
    real_value = []
    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_data_loader)):
            future_data, history_data, long_history_data = data
            batch_size = future_data.shape[0]
            long_history_data = select_input_features(long_history_data, config['froward_features'])
            history_data = select_input_features(history_data, config['target_features'])
            future_data = select_input_features(future_data, config['target_features'])
            
            labels = future_data.to(config['device'])
            history_data = history_data.to(config['device'])
            long_history_data = long_history_data.to(config['device'])
            
            preds = model(history_data, long_history_data, future_data, batch_size, epoch)
            
            prediction.append(preds.detach().cpu())        # preds = forward_return[0]
            real_value.append(labels.detach().cpu())        # testy = forward_return[1]s
            
            # 测试模式：只处理一个batch
            if config.get('test_mode', False):
                print(f"Test mode: Only processing batch {idx+1}")
                break

        prediction = torch.cat(prediction, dim=0)
        real_value = torch.cat(real_value, dim=0)
        # re-scale data
        prediction = SCALER_REGISTRY.get(scaler["func"])(prediction, **scaler["args"])
        real_value = SCALER_REGISTRY.get(scaler["func"])(real_value, **scaler["args"])

        for i in range(config['evaluation_horizons']):
            # For horizon i, only calculate the metrics **at that time** slice here.
            pred = prediction[:, i, :, :]
            real = real_value[:, i, :, :]

            metric_results = {}
            for metric_name, metric_func in metrics.items():
                metric_item = metric_forward(metric_func, [pred, real])
                metric_results[metric_name] = metric_item.item()

            print("Evaluate best model on test data for horizon " + \
                "{:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}".format(i+1,metric_results["MAE"], metric_results["RMSE"], metric_results["MAPE"]))
            logging.info("Evaluate best model on test data for horizon " + \
                "{:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}".format(i+1,metric_results["MAE"], metric_results["RMSE"], metric_results["MAPE"]))
            
        metric_results = {}
        for metric_name, metric_func in metrics.items():
            metric_item = metric_forward(metric_func, [prediction, real_value])
            metric_results[metric_name] = metric_item.item()
        
        swanlab.log({
            "test/MAE": metric_results["MAE"],
            "test/RMSE": metric_results["RMSE"],
            "test/MAPE": metric_results["MAPE"]
        })
        
        print("Evaluate val data" + \
                    "val MAE: {:.4f}, val RMSE: {:.4f}, val MAPE: {:.4f}".format(metric_results["MAE"], metric_results["RMSE"], metric_results["MAPE"]))
        logging.info("Evaluate val data" + \
                    "val MAE: {:.4f}, val RMSE: {:.4f}, val MAPE: {:.4f}".format(metric_results["MAE"], metric_results["RMSE"], metric_results["MAPE"]))

def finetune(config, args):
    print('### start finetune ... ###')
    adj_mx, _ = load_adj(config['adj_dir'], "doubletransition")

    config['backend_args']['supports'] = [torch.tensor(i) for i in adj_mx]
    scaler = load_pkl(config['scaler_dir'])
    
    train_dataset = ForecastingDataset(config['dataset_dir'],config['dataset_index_dir'],'train',config['seq_len'])
    val_dataset = ForecastingDataset(config['dataset_dir'],config['dataset_index_dir'],'valid',config['seq_len'])
    test_dataset = ForecastingDataset(config['dataset_dir'],config['dataset_index_dir'],'test',config['seq_len'])
    
    train_data_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers = 8, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=config['batch_size'], num_workers = 8, shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers = 8, shuffle=False)
    model = finetune_model(config['pre_trained_path'], config['mask_args'], config['backend_args'])
    model = model.to(args.device)
    optimizer = optim.Adam(model.parameters(), config['lr'], weight_decay=1.0e-5,eps=1.0e-8)
    
    for epoch in range(config['finetune_epochs']):
        print('============ epoch {:d} ============'.format(epoch))
        for idx, data in enumerate(tqdm(train_data_loader)):

            future_data, history_data, long_history_data = data
            batch_size = future_data.shape[0]
            long_history_data = select_input_features(long_history_data, config['froward_features'])
            history_data = select_input_features(history_data, config['target_features'])
            future_data = select_input_features(future_data, config['target_features'])
            
            labels = future_data.to(args.device)
            history_data = history_data.to(args.device)
            long_history_data = long_history_data.to(args.device)

            preds = model(history_data, long_history_data, future_data, batch_size, epoch)

            prediction_rescaled = SCALER_REGISTRY.get(scaler["func"])(preds, **scaler["args"])
            real_value_rescaled = SCALER_REGISTRY.get(scaler["func"])(labels, **scaler["args"])
            
            # 主损失
            loss = metric_forward(masked_mae, [prediction_rescaled, real_value_rescaled])
            
            # 添加对比学习损失（如果存在）
            if hasattr(model, 'contrastive_loss') and model.contrastive_loss is not None:
                contrastive_weight = config.get('contrastive_weight', 0.1)  # 从配置文件读取权重
                if isinstance(model.contrastive_loss, torch.Tensor):
                    loss = loss + contrastive_weight * model.contrastive_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 测试模式：只处理一个batch
            if args.test_mode:
                print(f"Test mode: Only processing batch {idx+1}")
                break
            
        # 记录损失
        log_dict = {"finetune/train_loss": loss.item()}
        if hasattr(model, 'contrastive_loss') and model.contrastive_loss is not None:
            if isinstance(model.contrastive_loss, torch.Tensor):
                log_dict["finetune/contrastive_loss"] = model.contrastive_loss.item()
        swanlab.log(log_dict, step=epoch)
        
        print('============ val and test ============')
        # val(val_data_loader, model, config, scaler, epoch)
        test(test_data_loader, model, config, scaler, epoch)
        
@torch.no_grad()
@master_only
def preTrain_test(data_loader, model, scaler, mode='val'):
    """Evaluate the model.

    Args:
        train_epoch (int, optional): current epoch if in training process.
    """
    model.eval()
    
    prediction = []
    real_value = []
    MAE, RMSE, MAPE = 0.0, 0.0, 0.0
    with torch.no_grad():
        for idx, data in enumerate(tqdm(data_loader)):
            future_data, history_data = data
            
            history_data = select_input_features(history_data, config['froward_features'])
            history_data = history_data.to(config['device'])
            reconstruction_masked_tokens, label_masked_tokens, _ = model(history_data, 0)

            prediction = reconstruction_masked_tokens.detach().cpu()
            real_value = label_masked_tokens.detach().cpu()
            prediction_rescaled = SCALER_REGISTRY.get(scaler["func"])(prediction, **scaler["args"])
            real_value_rescaled = SCALER_REGISTRY.get(scaler["func"])(real_value, **scaler["args"])
            
            metric_results = {}
            for metric_name, metric_func in metrics.items():
                metric_item = metric_forward(metric_func, [prediction_rescaled, real_value_rescaled])
                metric_results[metric_name] = metric_item.item()
            metric_1, metric_2, metric_3 = metric_results["MAE"], metric_results["RMSE"], metric_results["MAPE"]
            MAE += metric_1
            RMSE += metric_2
            MAPE += metric_3
            
            # 测试模式：只处理一个batch
            if config.get('test_mode', False):
                print(f"Test mode: Only processing batch {idx+1}")
                break
        MAE = MAE / (idx + 1)
        RMSE = RMSE / (idx + 1)
        MAPE = MAPE / (idx + 1)
        print("Evaluate {} data MAE: {:.4f},  RMSE: {:.4f},  MAPE: {:.4f}".format(mode, metric_results["MAE"], metric_results["RMSE"], metric_results["MAPE"]))

def pretrain(config, args):
    print('### start pre-training ... ###')
    adj_mx, _ = load_adj(config['adj_dir'], "doubletransition")
    scaler = load_pkl(config['preTrain_scaler_dir'])
    preTrain_train_dataset = PretrainingDataset(config['preTrain_dataset_dir'], config['preTrain_dataset_index_dir'],'train', config['device'])
    preTrain_val_dataset = PretrainingDataset(config['preTrain_dataset_dir'], config['preTrain_dataset_index_dir'],'valid', config['device'])
    preTrain_test_dataset = PretrainingDataset(config['preTrain_dataset_dir'], config['preTrain_dataset_index_dir'],'test', config['device'])


    # 自适应设置数据加载参数
    use_cuda = torch.cuda.is_available() and args.device == 'cuda'
    pin_memory = use_cuda
    cpu_count = os.cpu_count() or 4  # 处理None的情况
    num_workers = 8
    
    train_data_loader = DataLoader(preTrain_train_dataset, 
                                  batch_size=config['preTrain_batch_size'], 
                                  num_workers=num_workers, 
                                  shuffle=True, 
                                  pin_memory=pin_memory, 
                                  persistent_workers=True if num_workers > 0 else False,
                                  prefetch_factor=4 if num_workers > 0 else None)
    val_data_loader = DataLoader(preTrain_val_dataset, 
                                batch_size=config['preTrain_batch_size'], 
                                num_workers=num_workers//2, 
                                shuffle=False, 
                                pin_memory=pin_memory, 
                                persistent_workers=True if num_workers > 0 else False,
                                prefetch_factor=2 if num_workers > 0 else None)
    test_data_loader = DataLoader(preTrain_test_dataset, 
                                 batch_size=config['preTrain_batch_size'], 
                                 num_workers=num_workers//2, 
                                 shuffle=False, 
                                 pin_memory=pin_memory, 
                                 persistent_workers=True if num_workers > 0 else False,
                                 prefetch_factor=2 if num_workers > 0 else None)

    model = pretrain_model(config['num_nodes'], config['dim'],
                           config['topK'], config['adaptive'],
                           config['pretrain_epochs'], config['patch_size'],
                           config['in_channel'], config['embed_dim'], 
                           config['num_heads'], config['graph_heads'], 
                           config['mlp_ratio'], config['dropout'],
                           config['mask_ratio'], config['encoder_depth'],
                           config['decoder_depth'])
    device_ids = list(range(torch.cuda.device_count()))
    # if device_ids:
    #     model = DataParallel(model, device_ids=device_ids).to(device_ids[0]) 
    # else:
    model = model.to(args.device)
    
    optimizer = optim.Adam(model.parameters(), config['lr'], weight_decay=1.0e-5, eps=1.0e-8)
    if args.lossType == 'mae':
        lossType = masked_mae
    elif args.lossType == 'sce':
        lossType = sce_loss
    print("config['pretrain_epochs']:", config['pretrain_epochs'])
    
    # 预分配变量以减少重复创建
    loss_accumulator = 0.0
    batch_count = 0
    
    for epoch in range(config['pretrain_epochs']):
        print('============ epoch {:d} ============'.format(epoch))
        model.train()  # 确保训练模式
        loss_accumulator = 0.0
        batch_count = 0
        
        for idx, data in enumerate(tqdm(train_data_loader)):
            future_data, history_data = data
            
            history_data = select_input_features(history_data, config['froward_features'])
            history_data = history_data.to(args.device, non_blocking=True)

            # 获取模型输出
            model_output = model(history_data, epoch)
            
            # 处理返回值（可能是2个或3个值）
            if len(model_output) == 3:
                reconstruction_masked_tokens, label_masked_tokens, contrastive_loss = model_output
            else:
                reconstruction_masked_tokens, label_masked_tokens = model_output
                contrastive_loss = None

            # 主损失（重构损失）
            loss = metric_forward(lossType, [reconstruction_masked_tokens, label_masked_tokens])
            
            # 添加对比学习损失（如果存在）
            total_loss = loss
            if contrastive_loss is not None:
                contrastive_weight = config.get('contrastive_weight', 0.1)
                if isinstance(contrastive_loss, torch.Tensor):
                    total_loss = loss + contrastive_weight * contrastive_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 累积损失，减少.item()调用频率
            loss_accumulator += total_loss.detach()
            batch_count += 1
            
            # 测试模式：只处理一个batch
            if config.get('test_mode', False):
                print(f"Test mode: Only processing batch {idx+1}")
                break
        
        # epoch结束时计算平均损失
        if batch_count > 0:
            if isinstance(loss_accumulator, torch.Tensor):
                avg_loss = (loss_accumulator / batch_count).item()
            else:
                avg_loss = loss_accumulator / batch_count
        else:
            avg_loss = 0.0
        print(f"preTrain loss: {avg_loss:.6f}")
        
        # 记录预训练损失
        log_dict = {"pretrain/loss": avg_loss}
        # 如果有对比学习损失，也记录下来（使用最后一个batch的值作为示例）
        if contrastive_loss is not None and isinstance(contrastive_loss, torch.Tensor):
            log_dict["pretrain/contrastive_loss"] = contrastive_loss.item()
        swanlab.log(log_dict, step=epoch)
        
        if config["save_model"]:
            print("Saving Model ...")
            pre_trained_path = config["model_save_path"] + "checkpoint_" + str(epoch) + ".pt"
            state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(state_dict, pre_trained_path)
        config['pre_trained_path'] = pre_trained_path
        # finetune(config, args)

    return model


def main(config, args):
    swanlab.init(
        project="AGPST-TrafficForecasting",
        experiment_name=f"{config['dataset_name']}_patch{config['patch_size']}_mask{config['mask_ratio']}",
        config={
            "dataset": config['dataset_name'],
            "num_nodes": config['num_nodes'],
            "patch_size": config['patch_size'],
            "mask_ratio": config['mask_ratio'],
            "embed_dim": config['embed_dim'],
            "encoder_depth": config['encoder_depth'],
            "decoder_depth": config['decoder_depth'],
            "pretrain_epochs": config['pretrain_epochs'],
            "finetune_epochs": config['finetune_epochs'],
            "preTrain_batch_size": config['preTrain_batch_size'],
            "batch_size": config['batch_size'],
            "learning_rate": config['lr'],
            "topK": config['topK'],
            "adaptive": config['adaptive'],
        },
        mode=args.swanlab_mode,
    )

    if args.mode == 'pretrain':
        model = pretrain(config, args)
        model = model.cpu()
    elif args.mode == 'forecasting':
        finetune(config, args)
    else:
        print("mode error")
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
    parser.add_argument('--config', default='./parameters/PEMS03_v2.yaml', type=str, help='Path to the YAML config file')
    parser.add_argument('--device', default='cuda', type=str, help='device')
    parser.add_argument('--lossType', default='mae', type=str, help='pre-training loss type and default is mae. {mae, sce}')
    parser.add_argument('--test_mode', default=0, type=int, help='test or not')
    parser.add_argument('--swanlab_mode', default='online', type=str, help='swanlab mode online or disabled')
    parser.add_argument('--mode', default='forecasting', type=str, help='pretrain or forecasting')
    # parser.add_argument('--preTrainVal', default="true", type=str, help='pre-training validate or not')
    parser.add_argument('--device_ids', default=0, type=int, help='Number of GPUs available')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    seed_torch(seed=0)
    main(config, args)