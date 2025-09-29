import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from omegaconf import DictConfig


class GraphSAGE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        aggregator: str = "mean",
    ) -> None:
        """Two-layer GraphSAGE model.

        Args:
            input_dim: Dimension of input node features.
            hidden_dim: Hidden/embedding dimension to output.
            dropout: Dropout rate applied between layers.
            aggregator: Aggregation method ("mean", "max", "sum", or "lstm").
        """
        super().__init__()

        # First SAGE layer
        self.sage1 = SAGEConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            normalize=True,
            aggr=aggregator,
        )

        # Second SAGE layer
        self.sage2 = SAGEConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            normalize=True,
            aggr=aggregator,
        )

        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features [num_nodes, input_dim].
            edge_index: Edge list in COO format [2, num_edges].

        Returns:
            Node embeddings of shape [num_nodes, hidden_dim].
        """
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.sage2(x, edge_index)
        return x


def get_sage_model(cfg: DictConfig) -> GraphSAGE:
    """Factory to build a GraphSAGE model from config.

    Expected cfg fields:
      - cfg.dataset.embedding_dim: input_dim
      - cfg.gnn_model.hidden_dim: hidden_dim
      - optional cfg.gnn_model.dropout: float
      - optional cfg.gnn_model.aggregator: str ("mean", "max", "sum", or "lstm")
    """
    input_dim = cfg.dataset.embedding_dim
    hidden_dim = cfg.gnn_model.hidden_dim
    dropout = getattr(cfg.gnn_model, "dropout", 0.0)
    aggregator = getattr(cfg.gnn_model, "aggregator", "mean")
    return GraphSAGE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        aggregator=aggregator,
    )
