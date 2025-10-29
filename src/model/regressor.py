# Edge Attribute Regression Models for GNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class EdgeRegressor(nn.Module):
    """Basic MLP-based edge attribute regressor.
    
    Takes concatenated node embeddings (src, dst) and predicts a continuous edge attribute.
    This is suitable when the process.py prepares edge_embeddings as concatenated node features.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1) -> None:
        """Initialize the EdgeRegressor model.

        Args:
            input_dim: Dimension of the input features (concatenated node embeddings: 2 * node_dim)
            hidden_dim: Dimension of the hidden layer
            output_dim: Dimension of the output (default 1 for scalar regression)
        """
        super(EdgeRegressor, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Initialize weights
        self.weights_init()
        
    def weights_init(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for edge attribute regression.

        Args:
            x: Input edge features (concatenated node embeddings) [num_edges, input_dim]

        Returns:
            Predicted edge attributes [num_edges, output_dim] or [num_edges] if output_dim=1
        """
        out = self.mlp(x)
        # Squeeze last dimension if output_dim is 1 for scalar regression
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out


class DotProductRegressor(nn.Module):
    """DGL-style dot product predictor for edge attribute regression.
    
    Computes dot product between source and destination node embeddings.
    Note: This requires node embeddings to be passed separately (not concatenated).
    This is more suitable if you modify process.py to pass embeddings differently.
    """
    
    def __init__(self, node_dim: int, use_mlp: bool = True, hidden_dim: int = 64) -> None:
        """Initialize the DotProductRegressor.
        
        Args:
            node_dim: Dimension of node embeddings
            use_mlp: Whether to use MLP after dot product for non-linear transformation
            hidden_dim: Hidden dimension for optional MLP
        """
        super(DotProductRegressor, self).__init__()
        self.use_mlp = use_mlp
        
        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(node_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            self.weights_init()
    
    def weights_init(self):
        """Initialize weights using Xavier initialization."""
        if self.use_mlp:
            for m in self.mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using dot product.
        
        Args:
            x: Concatenated node embeddings [num_edges, 2 * node_dim]
            
        Returns:
            Edge scores [num_edges]
        """
        # Split concatenated embeddings
        node_dim = x.shape[1] // 2
        src_h = x[:, :node_dim]
        dst_h = x[:, node_dim:]
        
        if self.use_mlp:
            # Use MLP on concatenated embeddings
            score = self.mlp(x).squeeze(-1)
        else:
            # Simple dot product
            score = (src_h * dst_h).sum(dim=-1)
        
        return score


class EdgeAttrRegressor(nn.Module):
    """Advanced edge attribute regressor that can incorporate edge features.
    
    Combines source node, destination node, and optional edge features
    to predict continuous edge attributes.
    """
    
    def __init__(self, node_dim: int, edge_feat_dim: int = 0, 
                 hidden_dim: int = 64, output_dim: int = 1) -> None:
        """Initialize the EdgeAttrRegressor.
        
        Args:
            node_dim: Dimension of node embeddings
            edge_feat_dim: Dimension of edge features (0 if not using edge features)
            hidden_dim: Hidden dimension for MLP
            output_dim: Output dimension (1 for scalar regression)
        """
        super(EdgeAttrRegressor, self).__init__()
        self.edge_feat_dim = edge_feat_dim
        
        # Input is: source_embedding + dest_embedding + edge_features
        input_dim = node_dim * 2 + edge_feat_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.weights_init()
    
    def weights_init(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor = None) -> torch.Tensor:
        """Forward pass for edge attribute regression.
        
        Args:
            x: Concatenated node embeddings [num_edges, 2 * node_dim]
            edge_attr: Optional edge features [num_edges, edge_feat_dim]
            
        Returns:
            Predicted edge attributes [num_edges, output_dim] or [num_edges] if output_dim=1
        """
        if edge_attr is not None and self.edge_feat_dim > 0:
            # Concatenate node embeddings with edge features
            x = torch.cat([x, edge_attr], dim=-1)
        
        out = self.mlp(x)
        
        # Squeeze last dimension if output_dim is 1 for scalar regression
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
        
        return out


def get_edge_regressor_model(cfg: DictConfig, model_type: str = 'basic') -> nn.Module:
    """Factory function to get edge regressor model.
    
    Args:
        cfg: Configuration object
        model_type: Type of regressor ('basic', 'dotproduct', 'advanced')
        
    Returns:
        Edge regressor model
    """
    node_hidden_dim = cfg.gnn_model.hidden_dim
    regressor_hidden_dim = cfg.edge_classifier_model.hidden_dim
    
    if model_type == 'basic':
        return EdgeRegressor(
            input_dim=2 * node_hidden_dim,
            hidden_dim=regressor_hidden_dim,
            output_dim=1
        )
    elif model_type == 'dotproduct':
        return DotProductRegressor(
            node_dim=node_hidden_dim,
            use_mlp=True,
            hidden_dim=regressor_hidden_dim
        )
    elif model_type == 'advanced':
        # For advanced model, you might want to add edge_feat_dim to config
        edge_feat_dim = getattr(cfg.dataset, 'edge_feat_dim', 0)
        return EdgeAttrRegressor(
            node_dim=node_hidden_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=regressor_hidden_dim,
            output_dim=1
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'basic', 'dotproduct', or 'advanced'")