# Graph Poisoning for Node Rank Manipulation

This repository contains the implementation of a black-box graph poisoning attack for degrading target node rankings in graph-based retrieval systems, as described in _Graph Poisoning for Node Rank Manipulation_.

## Repository Structure

```
adverserial-attack/
├── configs/                      # Configuration files (Hydra-based)
│   ├── config.yaml              # Main configuration file
│   ├── dataset/                 # Dataset-specific configurations
│   ├── gnn_model/               # GNN model configurations (GCN, GAT, SAGE)
│   ├── edge_classifier_model/   # Edge scorer model configurations
│   ├── train/                   # Training hyperparameters
│   └── general/                 # General experiment settings
├── src/                         # Source code
│   ├── main.py                  # Main entry point
│   ├── train.py                 # Training logic for edge scorer
│   ├── test.py                  # Testing and inference logic
│   ├── utils.py                 # Utility functions
│   ├── datasets/                # Dataset loading and preprocessing
│   │   ├── load_datasets.py    # Dataset module
│   │   ├── load_cora.py        # Cora dataset loader
│   │   └── abstract_dataset.py # Abstract dataset class
│   └── model/                   # Model implementations
│       ├── gcn.py              # GCN model
│       ├── gat.py              # GAT model
│       ├── sage.py             # GraphSAGE model
│       ├── regressor.py        # Edge scorer (regressor)
│       └── process.py          # Training/evaluation processing
├── datasets/                    # Raw dataset files
├── outputs/                     # Training outputs and checkpoints
├── data/                        # Processed data
└── examples/                    # Example scripts and notebooks

```

## Key Components

### 1. **Edge Scorer Training**
The core of our approach is training a local scorer that predicts context-sensitive edge effects:
- **GNN Encoder**: Encodes node features and graph structure (GCN/GAT/SAGE)
- **Edge Scorer**: Predicts the utility change from removing each edge
- **Training Data**: Generated from ego-network samples with empirical edge ablation measurements

### 2. **Context-Dependent Sampling**
- Samples multiple ego-networks around target nodes
- Supports multiple sampling strategies: uniform, random walk, ego-based
- Configurable subgraph sizes and sampling parameters

### 3. **Attack Execution**
- Aggregates predictions across sampled subgraphs
- Selects edges for deletion based on predicted impact
- Model-agnostic: works with any graph-based ranking system

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- PyTorch 1.10+
- PyTorch Geometric

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd adverserial-attack

# Install dependencies
pip install torch torchvision torchaudio
pip install torch-geometric
pip install hydra-core wandb tqdm scikit-learn scipy networkx
```

## Configuration System

This project uses [Hydra](https://hydra.cc/) for configuration management. All configurations are stored in the `configs/` directory with a hierarchical structure.

### Main Configuration (`configs/config.yaml`)
The main config file specifies defaults for all components:
```yaml
defaults:
    - gnn_model: gcn
    - edge_classifier_model: edge_mlp
    - train: train_default
    - dataset: cora
    - general: general_default
```

### Dataset Configurations (`configs/dataset/`)
Configure dataset-specific parameters:
- **Available datasets**: Cora, CiteSeer, PubMed, CoraFull, Amazon-Photo, Wikipedia
- **Key parameters**:
  - `name`: Dataset name
  - `directed`: Whether to use directed graphs
  - `batch_size`: Training batch size
  - `sampling_method`: Sampling strategy (uniform, random_walk, ego, mix)
  - `per_node_samples_*`: Number of subgraphs per node for each sampling method
  - `subgraph_size`: Target size for sampled subgraphs
  - `edge_attribute_classes`: Number of edge utility classes
  - `edge_attribute_mode`: 'classifier' for classification or 'regression' for regression

**Example** (`configs/dataset/cora.yaml`):
```yaml
name: 'Cora'
directed: True
batch_size: 16
sampling_method: 'uniform'
per_node_samples_unif: 100000
subgraph_size: 40
edge_attribute_classes: 10
edge_attribute_mode: 'classifier'
```

### GNN Model Configurations (`configs/gnn_model/`)
Configure the GNN encoder architecture:
- **Available models**: GCN, GAT, GraphSAGE
- **Parameters**: Hidden dimensions, number of layers, activation functions

### Edge Scorer Configurations (`configs/edge_classifier_model/`)
Configure the edge scorer model that predicts edge utility:
- MLP-based scorer architecture
- Hidden dimensions and output classes

### Training Configurations (`configs/train/`)
Training hyperparameters:
```yaml
n_epochs: 2000
lr: 0.001
optimizer: adam
weight_decay: 1e-12
```

### General Configurations (`configs/general/`)
Experiment-level settings:
- **Logging**: WandB integration (online/offline/disabled)
- **Testing**: `test_only`, `number_of_tests`
- **Edge model type**: 'classifier', 'basic', 'dotproduct', 'advanced'
- **Victim model**: Target GNN model to attack (e.g., 'epagcl', 'gca')

### Overriding Configurations
You can override any configuration parameter from the command line:
```bash
# Override dataset
python src/main.py dataset=citeseer

# Override multiple parameters
python src/main.py dataset=cora train.n_epochs=1000 train.lr=0.0001

# Override nested parameters
python src/main.py dataset.batch_size=32 general.edge_model=advanced
```

## Usage

### Training Phase

Train the edge scorer model to learn context-dependent edge effects:

```bash
# Basic training with default configuration (Cora dataset, GCN model)
python src/main.py

# Train with specific dataset
python src/main.py dataset=citeseer

# Train with custom hyperparameters
python src/main.py dataset=cora train.n_epochs=1000 train.lr=0.0001

# Train with different GNN encoder
python src/main.py gnn_model=gat dataset=pubmed

# Train with regression-based edge scorer
python src/main.py general.edge_model=advanced dataset.edge_attribute_mode=regression
```

**Training Output**:
- Model checkpoints saved to `gnn-edge-classifier/` or `gnn-edge-regressor/`
- Training logs and metrics logged to WandB
- Checkpoints saved every `check_val_every_n_epochs` epochs

### Testing Phase

Evaluate the trained model and generate edge deletion predictions:

```bash
# Test with trained model
python src/main.py general.test_only=True \
    general.gnn_model_path=gnn-edge-classifier/gcn_2000.pt \
    general.edge_classifier_model_path=gnn-edge-classifier/edge_mlp_2000.pt

# Test with multiple trials
python src/main.py general.test_only=True \
    general.number_of_tests=10 \
    general.gnn_model_path=<path_to_gnn_checkpoint> \
    general.edge_classifier_model_path=<path_to_edge_scorer_checkpoint>

# Test on different dataset
python src/main.py dataset=citeseer general.test_only=True \
    general.gnn_model_path=<checkpoint_path> \
    general.edge_classifier_model_path=<checkpoint_path>
```

**Testing Output**:
- Predictions saved to `outputs/predictions/`
- Each sample saved as `sample_t{test_number}_s{target_node_id}.pt`
- Contains predicted edge utilities and deletion rankings
- Evaluation metrics logged to WandB

### Model Types

The framework supports two edge scoring approaches:

1. **Classification-based** (`edge_model=classifier`):
   - Predicts discrete utility classes for edges
   - Uses cross-entropy loss
   - Suitable for categorical edge importance

2. **Regression-based** (`edge_model=advanced`):
   - Predicts continuous utility scores
   - Uses MSE loss
   - Provides fine-grained edge ranking
   - Evaluated with Pearson/Spearman correlation

## Datasets

Supported datasets:
- **Cora**: Citation network (2,708 nodes, 10,556 edges)
- **CiteSeer**: Citation network (3,327 nodes, 9,104 edges)
- **PubMed**: Citation network (19,717 nodes, 88,648 edges)
- **CoraFull**: Extended Cora dataset
- **Amazon-Photo**: Co-purchase network
- **Wikipedia**: Hyperlink network

Each dataset configuration includes:
- Sampling parameters for ego-network generation
- Edge attribute settings
- Preprocessing options

## Monitoring and Logging

The project integrates with [Weights & Biases](https://wandb.ai/) for experiment tracking:

- **Training metrics**: Loss, accuracy, precision, recall, F1-score
- **Validation metrics**: Evaluated every `check_val_every_n_epochs`
- **Regression metrics**: Pearson correlation, Spearman rank correlation, MAE, RMSE
- **Configuration**: All hyperparameters logged automatically

Configure WandB mode in `configs/general/general_default.yaml`:
```yaml
wandb: 'online'  # online | offline | disabled
```

## Output Structure

```
outputs/
├── predictions/                 # Test predictions
│   └── sample_t{test}_s{node}.pt
├── {date}/                      # Timestamped experiment runs
│   └── {experiment_name}/
│       ├── wandb/              # WandB logs
│       └── checkpoints/        # Model checkpoints
└── ...
```
