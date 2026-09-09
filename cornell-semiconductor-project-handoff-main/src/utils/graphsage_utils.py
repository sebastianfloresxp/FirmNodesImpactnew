"""
graphsage_utils.py
------------------
Shared utilities for GraphSAGE training and evaluation.

This module contains common functions and classes used by both regular and temporal GraphSAGE trainers.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor
from torch_geometric.nn import SAGEConv


def sample_hard_negatives(
    adj: torch.sparse.FloatTensor,  # type: ignore[attr-defined]
    pos_batch: Tensor,
    num_neg: int,
    device: torch.device | None = None,
) -> Tensor:
    """
    Hard negatives: sample node pairs biased by node degree.

    Parameters
    ----------
    adj : torch.sparse.FloatTensor
        Sparse adjacency matrix
    pos_batch : Tensor
        Positive edge batch (2 x N)
    num_neg : int
        Number of negative samples to generate
    device : torch.device, optional
        Device to place tensors on

    Returns
    -------
    Tensor
        Negative edge indices (2 x num_neg)
    """
    if device is None:
        device = pos_batch.device

    deg = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1)
    probs = deg / deg.sum()
    u = torch.multinomial(probs, num_neg, replacement=True)
    v = torch.multinomial(probs, num_neg, replacement=True)
    return torch.stack([u, v], dim=0).to(device)


def dot_decoder(z_src: Tensor, z_dst: Tensor, edge_attr: Tensor | None = None) -> Tensor:
    """
    Dot product decoder for link prediction.

    Parameters
    ----------
    z_src : Tensor
        Source node embeddings
    z_dst : Tensor
        Destination node embeddings
    edge_attr : Tensor, optional
        Edge attributes (not used in simple dot product)

    Returns
    -------
    Tensor
        Edge scores (logits)
    """
    if edge_attr is None:
        return (z_src * z_dst).sum(dim=-1)
    else:
        # Simple concatenation and linear projection for edge attributes
        pair_feat = torch.cat([z_src * z_dst, edge_attr], dim=-1)
        return torch.sum(pair_feat, dim=-1)


def score_edges(z: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None) -> np.ndarray:
    """
    Compute sigmoid scores for a batch of edges.

    Parameters
    ----------
    z : Tensor
        Node embeddings
    edge_index : Tensor
        Edge indices (2 x N)
    edge_attr : Tensor, optional
        Edge attributes

    Returns
    -------
    np.ndarray
        Edge scores (sigmoid probabilities)
    """
    logits = dot_decoder(z[edge_index[0]], z[edge_index[1]], edge_attr)
    return torch.sigmoid(logits).cpu().numpy()


def eval_split(
    model: torch.nn.Module,
    x: Tensor,
    full_edge_index: Tensor,
    pos_idx: Tensor,
    neg_idx: Tensor,
    pos_attr: Tensor | None = None,
    neg_attr: Tensor | None = None,
    adj: torch.sparse.FloatTensor | None = None,  # type: ignore[attr-defined]
    device: torch.device | None = None,
) -> tuple[float, float]:
    """
    Evaluate model on a given positive/negative split.

    Parameters
    ----------
    model : torch.nn.Module
        GraphSAGE model
    x : Tensor
        Node features
    full_edge_index : Tensor
        Full edge index for graph construction
    pos_idx : Tensor
        Positive edge indices
    neg_idx : Tensor
        Negative edge indices
    pos_attr : Tensor, optional
        Positive edge attributes
    neg_attr : Tensor, optional
        Negative edge attributes
    adj : torch.sparse.FloatTensor, optional
        Sparse adjacency matrix (not used in this implementation)
    device : torch.device, optional
        Device to use for computation

    Returns
    -------
    tuple[float, float]
        (ROC-AUC, Average Precision)
    """
    if device is None:
        device = x.device

    model.eval()
    with torch.no_grad():
        z = model(x.to(device), full_edge_index.to(device))
        pos_scores = score_edges(
            z, pos_idx.to(device), pos_attr.to(device) if pos_attr is not None else None
        )
        neg_scores = score_edges(
            z, neg_idx.to(device), neg_attr.to(device) if neg_attr is not None else None
        )

        y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
        y_score = np.concatenate([pos_scores, neg_scores])

        # Guard against single-class slices
        if len(np.unique(y_true)) < 2:
            return float("nan"), float("nan")

        return roc_auc_score(y_true, y_score), average_precision_score(y_true, y_score)  # type: ignore[return-value]


class GraphSAGE(torch.nn.Module):
    """
    GraphSAGE encoder with multiple layers.

    Parameters
    ----------
    in_channels : int
        Input feature dimension
    hidden : int
        Hidden dimension
    num_layers : int
        Number of GraphSAGE layers
    dropout : float
        Dropout rate
    """

    def __init__(self, in_channels: int, hidden: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList()

        # Input layer
        self.layers.append(SAGEConv(in_channels, hidden))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden, hidden))

        # Output layer
        self.layers.append(SAGEConv(hidden, hidden))
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Forward pass through the GraphSAGE model.

        Parameters
        ----------
        x : Tensor
            Node features
        edge_index : Tensor
            Edge indices

        Returns
        -------
        Tensor
            Node embeddings
        """
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            if i != len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
