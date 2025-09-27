from omegaconf import DictConfig
from model.gcn import get_gcn_model, get_edge_mlp_model
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time
import os
from datasets.cora_dataset import CoraDataModule

logger = logging.getLogger(__name__)

BASE_PATH = 'output/gnn-edge-classifier'
os.makedirs(BASE_PATH, exist_ok=True)

def get_optimiser(cfg: DictConfig, gnn_model: nn.Module, edge_mlp: nn.Module) -> optim.Optimizer:
    if cfg.train.optimizer == "adam":
        return optim.Adam(list(gnn_model.parameters()) + list(edge_mlp.parameters()), lr=cfg.train.lr)
    elif cfg.train.optimizer == "sgd":
        return optim.SGD(list(gnn_model.parameters()) + list(edge_mlp.parameters()), lr=cfg.train.lr)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.train.optimizer}")
    
def get_cross_entropy_loss(cfg: DictConfig) -> nn.CrossEntropyLoss:
    return nn.CrossEntropyLoss()

def train(cfg: DictConfig) -> None:
    logger.info("Training started")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if cfg.gnn_model.name == "gcn":
        gnn_model = get_gcn_model(cfg).to(device)
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
    
    optimizer = get_optimiser(cfg, gnn_model, edge_classifier_model)
    criterion = get_cross_entropy_loss(cfg)
    logger.info("Training started, for {} epochs".format(cfg.train.n_epochs))
    start_time = time.time()
    for epoch in range(1, cfg.train.n_epochs + 1):
        total_loss = 0
        total_correct = 0
        total_samples = 0

        # iid shuffle of the dataset
        # random.shuffle(dataset)
        batches = [dataset[i : i + cfg.dataset.batch_size] for i in range(0, len(dataset), cfg.dataset.batch_size)]

        for batch in batches:
            num_nodes_all = [graph.x.shape[0] for graph in batch]
            node_features_all = [graph.x for graph in batch]
            node_feature_batch = torch.cat(node_features_all, dim=0).to(device)
            
            # Get edge indices (already in COO format: [2, num_edges])
            edge_index_all = []
            for graph in batch:
                edge_index_all.append(graph.edge_index)
            
            edge_labels_all = []
            for graph in batch:
                edge_labels_all.append(torch.argmax(graph.edge_attr, dim=1))
            edge_labels_batch = torch.cat(edge_labels_all, dim=0).to(device)

            # Update the edge indices to reflect the new node ordering in the batched tensor
            # We do this by adding the cumulative sum of the number of nodes from previous graphs
            edge_index_batch = []
            num_nodes_offset = 0
            for i, edge_index in enumerate(edge_index_all):
                edge_index_offset = edge_index + num_nodes_offset
                edge_index_batch.append(edge_index_offset)
                num_nodes_offset += num_nodes_all[i]

            # Concatenate all edge indices along the edge dimension (dim=1)
            edge_index_batch = torch.cat(edge_index_batch, dim=1).to(device)

            # Forward pass through GCN to get node embeddings
            node_embeddings_out = gnn_model(node_feature_batch, edge_index_batch)

            # Prepare edge features by concatenating the embeddings of the head and tail nodes
            edge_embeddings = torch.cat(
                [
                    node_embeddings_out[edge_index_batch[0]],
                    node_embeddings_out[edge_index_batch[1]],
                ],
                dim=1,
            )

            # Forward pass through MLP for edge classification
            out = edge_classifier_model(edge_embeddings)

            # Compute loss
            loss = criterion(out, edge_labels_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            preds = out.argmax(dim=1)
            total_correct += (preds == edge_labels_batch).sum().item()
            total_samples += edge_labels_batch.size(0)
        avg_loss = total_loss / len(batches)
        accuracy = total_correct / total_samples
        
        if epoch % cfg.general.check_val_every_n_epochs == 0:
            # eval_loss, eval_accuracy = eval(gnn_model, edge_classifier_model, criterion)
            # logger.info(f"Evaluation | Loss: {eval_loss:.4f} | Accuracy: {eval_accuracy:.4f}")
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

        logger.info(f"Epoch {epoch:02d}/{cfg.train.n_epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")
        
