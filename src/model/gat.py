import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from omegaconf import DictConfig


class GAT(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        """Two-layer Graph Attention Network (GAT).

        The model outputs node embeddings of size `hidden_dim` so it can be
        concatenated.

        Args:
            input_dim: Dimension of input node features.
            hidden_dim: Hidden/embedding dimension to output.
            heads: Number of attention heads in the first layer.
            dropout: Dropout applied inside attention layers.
        """
        super().__init__()

        # First GAT layer: outputs hidden_dim for each head, concatenated
        self.gat1 = GATConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            heads=heads,
            dropout=dropout,
            concat=True,
        )

        # Second GAT layer: aggregate to hidden_dim (no concat) -> final size hidden_dim
        self.gat2 = GATConv(
            in_channels=hidden_dim * heads,
            out_channels=hidden_dim,
            heads=1,
            dropout=dropout,
            concat=False,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features [num_nodes, input_dim].
            edge_index: Edge list in COO format [2, num_edges].

        Returns:
            Node embeddings of shape [num_nodes, hidden_dim].
        """
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        return x


def get_gat_model(cfg: DictConfig) -> GAT:
    """Factory to build a GAT model from config.

    Expected cfg fields:
      - cfg.dataset.embedding_dim: input_dim
      - cfg.gnn_model.hidden_dim: hidden_dim
      - optional cfg.gnn_model.heads: int
      - optional cfg.gnn_model.dropout: float
    """
    input_dim = cfg.dataset.embedding_dim
    hidden_dim = cfg.gnn_model.hidden_dim
    heads = getattr(cfg.gnn_model, "heads", 8)
    dropout = getattr(cfg.gnn_model, "dropout", 0.0)
    return GAT(input_dim=input_dim, hidden_dim=hidden_dim, heads=heads, dropout=dropout)

