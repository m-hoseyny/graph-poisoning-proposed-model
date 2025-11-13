from omegaconf import DictConfig
from model.gcn import get_gcn_model, get_edge_mlp_model
from model.gat import get_gat_model
from model.sage import get_sage_model
from model.regressor import get_edge_regressor_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
import os
import wandb
from datetime import datetime
import omegaconf
from datasets.load_datasets import CoraDataModule
import tqdm
from utils import get_cross_entropy_loss, setup_wandb, get_mse_loss
from model.process import eval_process, eval_process_regressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from scipy import stats

logger = logging.getLogger(__name__)


def load_model_gnn(cfg: DictConfig) -> nn.Module:
    '''
    Load the GNN model from checkpoint
    '''
    # First create the model with the same architecture
    # if cfg.gnn_model.name == "gcn":
    #     model = get_gcn_model(cfg)
    # elif cfg.gnn_model.name == "gat":
    #     model = get_gcn_model(cfg)
    # elif cfg.gnn_model.name == "sage":
    #     # TODO: Be careful! The model has been changed!
    #     model = get_gcn_model(cfg)
    # else:
    #     raise ValueError(f"Unknown model: {cfg.gnn_model.name}")
    model = get_gcn_model(cfg)
    # Then load the state dict from checkpoint
    checkpoint = torch.load(cfg.general.gnn_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded GNN model from {cfg.general.gnn_model_path}, epoch {checkpoint['epoch']}")
    return model


def load_model_edge_classifier(cfg: DictConfig) -> nn.Module:
    '''
    Load the edge classifier model from checkpoint
    '''
    # Create the model with the same architecture
    model = get_edge_mlp_model(cfg)
    
    # Load the state dict from checkpoint
    checkpoint = torch.load(cfg.general.edge_classifier_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded edge classifier model from {cfg.general.edge_classifier_model_path}, epoch {checkpoint['epoch']}")
    return model


def load_model_edge_regressor(cfg: DictConfig) -> nn.Module:
    '''
    Load the edge classifier model from checkpoint
    '''
    # Create the model with the same architecture
    model = get_edge_regressor_model(cfg, model_type=cfg.general.edge_model)
    
    # Load the state dict from checkpoint
    checkpoint = torch.load(cfg.general.edge_classifier_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded edge classifier model from {cfg.general.edge_classifier_model_path}, epoch {checkpoint['epoch']}")
    return model


# def merge


def test_once(cfg: DictConfig, test_number: int = 0, datamodule=None) -> None:
    logger.info("Testing started")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Check if checkpoint paths are provided
    if cfg.general.gnn_model_path is None or cfg.general.edge_classifier_model_path is None:
        raise ValueError("Please provide paths to both GNN and edge classifier model checkpoints")
    
    # Load models from checkpoints
    logger.info(f"Loading models from checkpoints")
    gnn_model = load_model_gnn(cfg)
    gnn_model.to(device)
    gnn_model.eval()  # Set model to evaluation mode
    
    edge_classifier_model = load_model_edge_classifier(cfg)
    edge_classifier_model.to(device)
    edge_classifier_model.eval()  # Set model to evaluation mode
    
    # Load dataset
    dataset_name = cfg.dataset.name
    logger.info(f"Using dataset: {dataset_name}")
    
    # if dataset_name == "Cora":
    if datamodule is None:
        datamodule = CoraDataModule(cfg)
    test_dataset = datamodule.graphs.test_dataset
    # else:
    #     raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Setup wandb for logging

    
    # Define loss function
    criterion = get_cross_entropy_loss(cfg)
    
    # Test the model
    logger.info("Starting evaluation on test dataset")
    start_time = time.time()
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    # Create batches
    # batches = [test_dataset[i : i + cfg.dataset.batch_size] for i in range(0, len(test_dataset), cfg.dataset.batch_size)]
    batches = [[test_dataset[i]] for i in range(len(test_dataset))]
    # Create lists to store samples with predicted edge attributes
    samples_with_predictions = []
    
    # Disable gradient computation for evaluation
    sample_id = 0
    # Save samples with predictions
    output_dir = os.path.join('outputs', 'predictions')
    os.makedirs(output_dir, exist_ok=True)
    for batch_idx, batch in enumerate(batches):
        logger.info(f"Processing batch {batch_idx+1}/{len(batches)}")
              
        # Track edge offsets for mapping predictions back to the correct edges
        edge_offsets = [0]
        for i, graph in enumerate(batch[:-1]):
            edge_offsets.append(edge_offsets[i] + graph.edge_index.shape[1])
        
        # Process the batch using the same function as in training
        _, _, loss, preds, edge_labels_batch = eval_process(
            batch, 
            gnn_model, 
            edge_classifier_model, 
            device, 
            None,  # No optimizer during testing
            criterion
        )
        
        # Accumulate metrics
        preds_cpu = preds.cpu().numpy()
        labels_cpu = edge_labels_batch.cpu().numpy()
        precision = precision_score(labels_cpu, preds_cpu, average='macro')
        recall = recall_score(labels_cpu, preds_cpu, average='macro')
        f1 = f1_score(labels_cpu, preds_cpu, average='macro')
        cm = confusion_matrix(labels_cpu, preds_cpu)
        accuracy = accuracy_score(labels_cpu, preds_cpu)
        
        logger.info(f"Batch {batch_idx+1} metrics:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision}")
        logger.info(f"  Recall: {recall}")
        logger.info(f"  F1: {f1}")
        logger.info(f"  Confusion Matrix:\n{cm}")
        
        # Create new samples with predicted edge attributes
        for i, graph in enumerate(batch):
            # Get predictions for this graph's edges
            start_idx = edge_offsets[i]
            end_idx = edge_offsets[i] + graph.edge_index.shape[1] if i < len(batch) - 1 else len(preds)
            graph_preds = preds[start_idx:end_idx]
            
            # Create one-hot encoded edge attributes from predictions
            num_classes = graph.edge_attr.shape[1]
            print(graph_preds)
            # pred_edge_attr = F.one_hot(graph_preds, num_classes=num_classes).float()
            
            # Create a new graph with predicted edge attributes
            from torch_geometric.data import Data
            pred_graph = Data(
                x=graph.x.clone(),
                edge_index=graph.edge_index.clone(),
                edge_attr=graph_preds.to('cpu'),
                # y=graph.y.clone() if hasattr(graph, 'y') else None,
                original_edge_attr=graph.edge_attr.clone(),  # Store original for comparison
                original_cora_edge_index=graph.original_edge_index.clone(),
                pred_edge_class=graph_preds.clone(),  # Store raw predictions,
                target_node_id=graph.target_node_id
            )
            
            # Store the sample
            torch.save(pred_graph, os.path.join(output_dir, f'sample_t{test_number}_s{graph.target_node_id}.pt'))
            samples_with_predictions.append(pred_graph)
        
        total_loss += loss
        total_correct += (preds == edge_labels_batch).sum().item()
        total_samples += edge_labels_batch.size(0)
    
        # Calculate final metrics
        avg_loss = total_loss / len(batches) if len(batches) > 0 else 0
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        # Log overall results
        logger.info(f"Overall Test Results | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")
        
        # Log to wandb
        wandb.log({
            'test/accuracy': accuracy,
            'test/loss': avg_loss,
            'test/precision': precision,
            'test/recall': recall,
            'test/f1': f1
        })
        
    
    # Calculate and log execution time
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Testing completed in {execution_time:.2f} seconds")
    

def ego_task(n, graph, radius, per_node_samples, subgraph_size):
    ego_net = nx.ego_graph(graph.to_undirected(), n, radius=radius)
    temp_initial_size = len(list(ego_net.nodes()))
    samples = []
    sizes = []
    for _ in range(per_node_samples):
        temp_ego = copy.deepcopy(ego_net)
        temp_size = temp_initial_size
        final_sample = list(temp_ego.nodes())
        while temp_size > subgraph_size:
            n_nodes_to_burn = int(temp_size / 2)
            temp_burning = copy.deepcopy(list(temp_ego.nodes()))
            temp_burning.remove(n)
            burned_nodes = list(random.choices(temp_burning, k=n_nodes_to_burn))
            temp_ego.remove_nodes_from(burned_nodes)
            final_sample = list(set(list(max(nx.connected_components(temp_ego), key=len)) + [n]))
            temp_size = len(final_sample)
        samples.append(final_sample)
        sizes.append(temp_size)
    return samples, sizes


def test_once_regressor(cfg: DictConfig, test_number: int = 0, datamodule=None) -> None:
    logger.info("Testing started")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Check if checkpoint paths are provided
    if cfg.general.gnn_model_path is None or cfg.general.edge_classifier_model_path is None:
        raise ValueError("Please provide paths to both GNN and edge classifier model checkpoints")
    
    # Load models from checkpoints
    logger.info(f"Loading models from checkpoints")
    gnn_model = load_model_gnn(cfg)
    gnn_model.to(device)
    gnn_model.eval()  # Set model to evaluation mode
    
    edge_classifier_model = load_model_edge_regressor(cfg)
    edge_classifier_model.to(device)
    edge_classifier_model.eval()  # Set model to evaluation mode
    
    # Load dataset
    dataset_name = cfg.dataset.name
    logger.info(f"Using dataset: {dataset_name}")
    
    # if dataset_name == "Cora":
    if datamodule is None:
        datamodule = CoraDataModule(cfg)
    test_dataset = datamodule.graphs.test_dataset
    # else:
    #     raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Setup wandb for logging

    
    # Define loss function
    criterion = get_mse_loss(cfg)
    
    # Test the model
    logger.info("Starting evaluation on test dataset")
    start_time = time.time()
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    # Create batches
    # batches = [test_dataset[i : i + cfg.dataset.batch_size] for i in range(0, len(test_dataset), cfg.dataset.batch_size)]
    batches = [[test_dataset[i]] for i in range(len(test_dataset))]
    # Create lists to store samples with predicted edge attributes
    samples_with_predictions = []
    
    # Disable gradient computation for evaluation
    sample_id = 0
    # Save samples with predictions
    output_dir = os.path.join('outputs', 'predictions')
    os.makedirs(output_dir, exist_ok=True)
    for batch_idx, batch in enumerate(batches):
        logger.info(f"Processing batch {batch_idx+1}/{len(batches)}")
              
        # Track edge offsets for mapping predictions back to the correct edges
        edge_offsets = [0]
        for i, graph in enumerate(batch[:-1]):
            edge_offsets.append(edge_offsets[i] + graph.edge_index.shape[1])
        
        # Process the batch using the same function as in training
        _, _, loss, preds, edge_labels_batch = eval_process_regressor(
            batch, 
            gnn_model, 
            edge_classifier_model, 
            device, 
            None,  # No optimizer during testing
            criterion
        )
        
        # Accumulate metrics
        preds_cpu = preds.cpu().numpy()
        labels_cpu = edge_labels_batch.cpu().numpy()
        
        pearson_corr, pearson_pvalue = stats.pearsonr(preds_cpu, labels_cpu)
        spearman_tau, spearman_pvalue = stats.spearmanr(preds_cpu, labels_cpu)
        mae = mean_absolute_error(labels_cpu, preds_cpu)
        rmse = mean_squared_error(labels_cpu, preds_cpu)
        
        logger.info(f"Batch {batch_idx+1} metrics:")
        logger.info(f"  Pearson: {pearson_corr:.4f}")
        logger.info(f"  Spearman: {spearman_tau}")
        logger.info(f"  MAE: {mae}")
        logger.info(f"  RMSE: {rmse}")
        
        # Create new samples with predicted edge attributes
        for i, graph in enumerate(batch):
            # Get predictions for this graph's edges
            start_idx = edge_offsets[i]
            end_idx = edge_offsets[i] + graph.edge_index.shape[1] if i < len(batch) - 1 else len(preds)
            graph_preds = preds[start_idx:end_idx]
            
            # Create one-hot encoded edge attributes from predictions
            # num_classes = graph.edge_attr.shape[1]
            print(graph_preds.detach().cpu().numpy())
            print(graph.edge_attr.detach().cpu().numpy())
            # pred_edge_attr = F.one_hot(graph_preds, num_classes=num_classes).float()
            
            # Create a new graph with predicted edge attributes
            from torch_geometric.data import Data
            pred_graph = Data(
                x=graph.x.clone(),
                edge_index=graph.edge_index.clone(),
                edge_attr=graph_preds.to('cpu'),
                # y=graph.y.clone() if hasattr(graph, 'y') else None,
                original_edge_attr=graph.edge_attr.clone(),  # Store original for comparison
                original_cora_edge_index=graph.original_edge_index.clone(),
                pred_edge_class=graph_preds.clone(),  # Store raw predictions,
                target_node_id=graph.target_node_id
            )
            
            # Store the sample
            torch.save(pred_graph, os.path.join(output_dir, f'sample_t{test_number}_s{graph.target_node_id}.pt'))
            samples_with_predictions.append(pred_graph)
        
        total_loss += loss
    
        # Calculate final metrics
        avg_loss = total_loss / len(batches) if len(batches) > 0 else 0
        
        # Log overall results
        logger.info(f"Evaluation | Loss: {loss:.4f} | Pearson: {pearson_corr:.4f} | Spearman: {spearman_tau:.4f}")
        
        # Log to wandb
        wandb.log({
            'val/loss': loss,
            'val/pearson': pearson_corr,
            'val/kendal': spearman_tau,
            'val/mae': mae,
            'val/rmse': rmse
            })
        
    
    # Calculate and log execution time
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Testing completed in {execution_time:.2f} seconds")


def test_once_ego_sample(cfg: DictConfig, test_number: int = 0) -> None:
    logger.info("Testing started")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Check if checkpoint paths are provided
    if cfg.general.gnn_model_path is None or cfg.general.edge_classifier_model_path is None:
        raise ValueError("Please provide paths to both GNN and edge classifier model checkpoints")
    
    # Load models from checkpoints
    logger.info(f"Loading models from checkpoints")
    gnn_model = load_model_gnn(cfg)
    gnn_model.to(device)
    gnn_model.eval()  # Set model to evaluation mode
    
    edge_classifier_model = load_model_edge_classifier(cfg)
    edge_classifier_model.to(device)
    edge_classifier_model.eval()  # Set model to evaluation mode
    
    # Load dataset
    dataset_name = cfg.dataset.name
    logger.info(f"Using dataset: {dataset_name}")
    
    # if dataset_name == "Cora":
    datamodule = CoraDataModule(cfg)
    test_dataset = datamodule.graphs.test_dataset
    # else:
    #     raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Setup wandb for logging
    cfg = setup_wandb(cfg)
    
    # Define loss function
    criterion = get_cross_entropy_loss(cfg)
    
    # Test the model
    logger.info("Starting evaluation on test dataset")
    start_time = time.time()
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    # Create batches
    # Each barch is a sample! Like we have 122 samples for Cora 
    batches = [[test_dataset[i]] for i in range(len(test_dataset))]
    # Create lists to store samples with predicted edge attributes
    samples_with_predictions = []
    
    # Save samples with predictions
    output_dir = os.path.join('outputs', 'predictions')
    os.makedirs(output_dir, exist_ok=True)
    for batch_idx, batch in enumerate(batches):
        logger.info(f"Processing batch {batch_idx+1}/{len(batches)}")
              
        # Track edge offsets for mapping predictions back to the correct edges
        edge_offsets = [0]
        for i, graph in enumerate(batch[:-1]):
            edge_offsets.append(edge_offsets[i] + graph.edge_index.shape[1])
        
        # Process the batch using the same function as in training
        _, _, loss, preds, edge_labels_batch = eval_process(
            batch, 
            gnn_model, 
            edge_classifier_model, 
            device, 
            None,  # No optimizer during testing
            criterion
        )
        
        # Accumulate metrics
        preds_cpu = preds.cpu().numpy()
        labels_cpu = edge_labels_batch.cpu().numpy()
        precision = precision_score(labels_cpu, preds_cpu, average=None)
        recall = recall_score(labels_cpu, preds_cpu, average=None)
        f1 = f1_score(labels_cpu, preds_cpu, average=None)
        cm = confusion_matrix(labels_cpu, preds_cpu)
        accuracy = accuracy_score(labels_cpu, preds_cpu)
        
        logger.info(f"Batch {batch_idx+1} metrics:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision}")
        logger.info(f"  Recall: {recall}")
        logger.info(f"  F1: {f1}")
        logger.info(f"  Confusion Matrix:\n{cm}")
        
        # Create new samples with predicted edge attributes
        for i, graph in enumerate(batch):
            # Get predictions for this graph's edges
            start_idx = edge_offsets[i]
            end_idx = edge_offsets[i] + graph.edge_index.shape[1] if i < len(batch) - 1 else len(preds)
            graph_preds = preds[start_idx:end_idx]
            
            # Create one-hot encoded edge attributes from predictions
            num_classes = graph.edge_attr.shape[1]
            print(graph_preds)
            # pred_edge_attr = F.one_hot(graph_preds, num_classes=num_classes).float()
            
            # Create a new graph with predicted edge attributes
            from torch_geometric.data import Data
            pred_graph = Data(
                x=graph.x.clone(),
                edge_index=graph.edge_index.clone(),
                edge_attr=graph_preds.to('cpu'),
                # y=graph.y.clone() if hasattr(graph, 'y') else None,
                original_edge_attr=graph.edge_attr.clone(),  # Store original for comparison
                original_cora_edge_index=graph.original_edge_index.clone(),
                pred_edge_class=graph_preds.clone(),  # Store raw predictions,
                target_node_id=graph.target_node_id
            )
            
            # Store the sample
            torch.save(pred_graph, os.path.join(output_dir, f'sample_t{test_number}_s{graph.target_node_id}.pt'))
            samples_with_predictions.append(pred_graph)
        
        total_loss += loss
        total_correct += (preds == edge_labels_batch).sum().item()
        total_samples += edge_labels_batch.size(0)
    
        # Calculate final metrics
        avg_loss = total_loss / len(batches) if len(batches) > 0 else 0
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        # Log overall results
        logger.info(f"Overall Test Results | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")
        
        # Log to wandb
        wandb.log({
            'test/accuracy': accuracy,
            'test/loss': avg_loss
        })
        
    
    # Calculate and log execution time
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Testing completed in {execution_time:.2f} seconds")
    
    wandb.finish()
    
    
def test(cfg: DictConfig):
    cfg = setup_wandb(cfg)
    datamodule = CoraDataModule(cfg)
    for i in range(cfg.general.number_of_tests):
        print('Trial {}'.format(i))
        if cfg.general.edge_model in ('classifier', 'binary_classifier'):
            test_once(cfg, test_number=i, datamodule=datamodule)
        else:
            test_once_regressor(cfg, test_number=i, datamodule=datamodule)
    
    wandb.finish()
    