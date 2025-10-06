from omegaconf import DictConfig
from torch._prims_common import corresponding_complex_dtype
from model.gcn import get_gcn_model, get_edge_mlp_model
from model.gat import get_gat_model
from model.sage import get_sage_model
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time
import os
import wandb
from datetime import datetime
import omegaconf
from datasets.cora_dataset import CoraDataModule
import tqdm
from utils import get_optimiser, get_cross_entropy_loss, setup_wandb
from model.process import process, eval_process
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

logger = logging.getLogger(__name__)



def train(cfg: DictConfig) -> None:
    BASE_PATH = 'gnn-edge-classifier'
    os.makedirs(BASE_PATH, exist_ok=True)
    logger.info("Training started")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Using model: {cfg.gnn_model.name}")
    if cfg.gnn_model.name == "gcn":
        gnn_model = get_gcn_model(cfg).to(device)
        edge_classifier_model = get_edge_mlp_model(cfg).to(device)
    elif cfg.gnn_model.name == "gat":
        gnn_model = get_gat_model(cfg).to(device)
        edge_classifier_model = get_edge_mlp_model(cfg).to(device)
    elif cfg.gnn_model.name == "sage":
        gnn_model = get_sage_model(cfg).to(device)
        edge_classifier_model = get_edge_mlp_model(cfg).to(device)
    else:
        raise ValueError(f"Unknown model: {cfg.gnn_model.name}")
    
    dataset_name = cfg.dataset.name
    logger.info(f"Using dataset: {dataset_name}")
    
    if dataset_name == "Cora":
        datamodule = CoraDataModule(cfg)
        dataset = datamodule.graphs
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    cfg = setup_wandb(cfg)
    optimizer = get_optimiser(cfg, gnn_model, edge_classifier_model)
    criterion = get_cross_entropy_loss(cfg)
    logger.info("Training started, for {} epochs".format(cfg.train.n_epochs))
    time.sleep(1)
    start_time = time.time()
    accuracy = 0
    loss = 0
    epoch = 0
    eval_loss = 0
    eval_accuracy = 0
    
    train_dataset, val_dataset = train_test_split(datamodule.graphs.data, test_size=0.05, random_state=42)
    val_batches = val_dataset
    
    for epoch in tqdm.tqdm(range(1, cfg.train.n_epochs + 1), desc=f'Training: epoch: {epoch}, loss: {loss:.4f}, accuracy: {accuracy:.4f}'):
        total_loss = 0
        total_correct = 0
        total_samples = 0

        # iid shuffle of the dataset
        # random.shuffle(dataset)
        batches = [train_dataset[i : i + cfg.dataset.batch_size] for i in range(0, len(train_dataset), cfg.dataset.batch_size)]
        
        for batch in batches:
            (gnn_model, 
             edge_classifier_model, 
             loss, 
             preds, 
             edge_labels_batch) = process(batch, gnn_model, edge_classifier_model, device, optimizer, criterion)
            
            total_loss += loss
            total_correct += (preds == edge_labels_batch).sum().item()
            total_samples += edge_labels_batch.size(0)  
            
        avg_loss = total_loss / len(batches)
        accuracy = total_correct / total_samples
        
        if epoch % cfg.general.check_val_every_n_epochs == 0:
            _, _, loss, preds, edge_labels_batch = eval_process(val_batches, gnn_model, edge_classifier_model, device, optimizer, criterion)
            preds_cpu = preds.cpu().numpy()
            labels_cpu = edge_labels_batch.cpu().numpy()
            precision = precision_score(labels_cpu, preds_cpu, average='macro')
            recall = recall_score(labels_cpu, preds_cpu, average='macro')
            f1 = f1_score(labels_cpu, preds_cpu, average='macro')
            cm = confusion_matrix(labels_cpu, preds_cpu)
            eval_accuracy = accuracy_score(labels_cpu, preds_cpu)
            logger.info(f"Evaluation | Loss: {eval_loss:.4f} | Accuracy: {eval_accuracy:.4f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': gnn_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
                }, "{}/{}_{}.pt".format(BASE_PATH, cfg.gnn_model.name, epoch))
            torch.save({
                'epoch': epoch,
                'model_state_dict': edge_classifier_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
                }, "{}/{}_{}.pt".format(BASE_PATH, cfg.edge_classifier_model.name, epoch))

            wandb.log({
            'val/loss': loss,
            'val/accuracy': eval_accuracy,
            'val/precision': precision,
            'val/recall': recall,
            'val/f1': f1,
            'val/cm': cm
            })
            
        # logger.info(f"Epoch {epoch:02d}/{cfg.train.n_epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")
        wandb.log({
            'train/epoch': epoch,
            'train/loss': avg_loss,
            'train/accuracy': accuracy,

        })
    wandb.finish()
    
    