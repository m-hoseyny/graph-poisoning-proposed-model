# Define the GCN model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from omegaconf import DictConfig

class GCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        """Init function for the GCN model.

        Args:
            input_dim: Dimension of the input features
            hidden_dim: Dimension of the hidden layer
        """
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward function for the GCN model.

        Args:
            x: Input node features
            edge_index: Graph edge indices (COO)

        Returns:
            Output (updated) node features with message passing.
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


# Define the MLP module for edge classification
class EdgeMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        """Init function for the EdgeMLP model.

        Args:
            input_dim: Dimension of the input features (concatenated embeddings)
            hidden_dim: Dimension of the hidden layer
            output_dim: Dimension of the output (number of classes)
        """
        super(EdgeMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for the EdgeMLP model.

        Args:
            x: Input edge features (concatenated node embeddings)

        Returns:
            Output edge class logits
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
def get_gcn_model(cfg: DictConfig) -> GCN:
    return GCN(input_dim=cfg.dataset.embedding_dim, hidden_dim=cfg.gnn_model.hidden_dim)

def get_edge_mlp_model(cfg: DictConfig) -> EdgeMLP:
    return EdgeMLP(input_dim=2 * cfg.gnn_model.hidden_dim, 
                   hidden_dim=cfg.edge_classifier_model.hidden_dim, 
                   output_dim=cfg.dataset.edge_attribute_classes)