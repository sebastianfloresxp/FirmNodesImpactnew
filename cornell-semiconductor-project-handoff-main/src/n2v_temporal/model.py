#!/usr/bin/env python3
"""
TGNN Model Architecture
=======================

Temporal Graph Neural Network with attention-based aggregation across snapshots.

Architecture:
1. Standard GCN processes each snapshot independently (fixed weights)
2. Temporal aggregation combines snapshot embeddings:
   - "attention": Learned weighting of snapshots (best for long sequences)
   - "mean": Simple average (fast baseline)
   - "last": Only most recent snapshot (ablation)
3. No RNN evolution → full gradient flow through all snapshots

This avoids the gradient flow issues in EvolveGCN while still capturing
temporal patterns.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

LOGGER = logging.getLogger(__name__)


class TemporalGNN(nn.Module):
    """
    Temporal Graph Neural Network with snapshot aggregation.

    Processes multiple graph snapshots and aggregates their embeddings
    using attention, mean, or last-only strategies. Optionally fuses
    per-snapshot Node2Vec embeddings with the GCN hidden state.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        temporal_mode: str = "attention",
        snapshot_batch_size: int = 2,
        checkpoint_snapshots: bool = False,
        checkpoint_threshold: int = 16000,
        temporal_decay: float = 0.08,
        num_score_heads: int = 1,
        n2v_dim: int = 0,
        use_n2v_features: bool = False,
        concat_base_features: bool = False,
    ):
        """Initialize TGNN with optional Node2Vec fusion."""
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.temporal_mode = temporal_mode
        self.snapshot_batch_size = snapshot_batch_size
        self.checkpoint_snapshots = checkpoint_snapshots
        self.checkpoint_threshold = checkpoint_threshold
        self.temporal_decay = temporal_decay
        self.num_score_heads = max(1, int(num_score_heads))
        self.n2v_dim = max(0, int(n2v_dim))
        self.use_n2v_features = bool(use_n2v_features)
        self.concat_base_features = bool(concat_base_features) and self.in_channels > 0

        if self.use_n2v_features and self.n2v_dim <= 0:
            raise ValueError("use_n2v_features=True requires n2v_dim > 0")

        if self.use_n2v_features and out_channels != self.n2v_dim:
            LOGGER.info(
                "Adjusting out_channels from %d to match Node2Vec dim %d for feature mode",
                out_channels,
                self.n2v_dim,
            )
            out_channels = self.n2v_dim

        self.out_channels = out_channels

        self.model_complexity = hidden_channels * self.out_channels * max(1, num_layers)
        prod = self.model_complexity

        self.auto_checkpoint = (
            checkpoint_threshold is not None
            and checkpoint_threshold > 0
            and prod >= checkpoint_threshold
        )

        # Standard GCN layers (same weights for all snapshots)
        self.convs = nn.ModuleList()
        self.convs.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(nn.Linear(hidden_channels, hidden_channels))
        self.out_proj = nn.Linear(hidden_channels, self.out_channels)
        self.residual_proj = nn.Linear(in_channels, self.out_channels, bias=False)
        self.residual_gate = nn.Parameter(torch.tensor(0.0))
        self.out_norm = nn.LayerNorm(self.out_channels)
        self.final_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Temporal aggregation (if using attention)
        if temporal_mode == "attention":
            self.temporal_query = nn.Linear(self.out_channels, self.out_channels)
            self.temporal_key = nn.Linear(self.out_channels, self.out_channels)
            self.temporal_value = nn.Linear(self.out_channels, self.out_channels)
            LOGGER.debug("TGNN initialized with attention aggregation (%dD)", self.out_channels)
        elif temporal_mode == "mean":
            LOGGER.debug("TGNN initialized with mean aggregation")
        elif temporal_mode == "last":
            LOGGER.debug("TGNN initialized with last-snapshot aggregation")
        else:
            raise ValueError(f"Unknown temporal_mode: {temporal_mode}")

        LOGGER.info(
            "TGNN: in=%d, hidden=%d, out=%d, layers=%d, dropout=%.3f, mode=%s, "
            "snapshot_batch=%d, temporal_decay=%.3f, score_heads=%d, n2v_dim=%d",
            in_channels,
            hidden_channels,
            self.out_channels,
            num_layers,
            dropout,
            temporal_mode,
            snapshot_batch_size,
            temporal_decay,
            self.num_score_heads,
            self.n2v_dim,
        )
        if self.checkpoint_snapshots:
            LOGGER.info("  Snapshot gradient checkpointing: forced ON")
        elif self.auto_checkpoint:
            LOGGER.info(
                "  Snapshot gradient checkpointing: auto ON (complexity=%d ≥ %d)",
                prod,
                checkpoint_threshold,
            )
        if temporal_mode == "attention" and temporal_decay > 0:
            LOGGER.info("  Temporal recency bias: exponential decay rate=%.3f", temporal_decay)
        if self.n2v_dim > 0:
            LOGGER.info("  Node2Vec fusion enabled (dim=%d)", self.n2v_dim)
        if self.use_n2v_features:
            LOGGER.info(
                "  Node2Vec feature mode: direct embeddings (concat_tabular=%s)",
                self.concat_base_features,
            )

        self.head_projections = nn.ModuleList()
        for _ in range(self.num_score_heads):
            proj = nn.Linear(self.out_channels, self.out_channels, bias=False)
            nn.init.eye_(proj.weight)
            self.head_projections.append(proj)

        if self.n2v_dim > 0:
            self.n2v_proj = nn.Linear(self.n2v_dim, self.hidden_channels)
            self.n2v_out_proj = nn.Linear(self.n2v_dim, self.out_channels)
        else:
            self.n2v_proj = None
            self.n2v_out_proj = None

        if self.use_n2v_features and self.concat_base_features:
            self.base_feat_proj = nn.Linear(self.in_channels, self.n2v_dim, bias=False)
        else:
            self.base_feat_proj = None

    @staticmethod
    def normalize_adjacency(
        edge_index: Tensor,
        num_nodes: int,
        add_self_loops: bool = True,
    ) -> torch.sparse.Tensor:
        """Return D^{-1/2} (A + I) D^{-1/2} as a sparse tensor."""
        if edge_index.is_sparse:
            return edge_index.coalesce()

        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")

        device = edge_index.device
        edge_index = edge_index.long()
        if add_self_loops:
            diag = torch.arange(num_nodes, device=device, dtype=torch.long)
            self_loops = torch.stack([diag, diag])
            edge_index = torch.cat([edge_index, self_loops], dim=1)

        values = torch.ones(edge_index.size(1), device=device, dtype=torch.float32)
        deg = torch.zeros(num_nodes, device=device, dtype=torch.float32)
        deg.scatter_add_(0, edge_index[0], values)
        deg_inv_sqrt = deg.clamp(min=1e-12).pow_(-0.5)
        norm_values = deg_inv_sqrt[edge_index[0]] * values * deg_inv_sqrt[edge_index[1]]
        adj = torch.sparse_coo_tensor(edge_index, norm_values, (num_nodes, num_nodes))
        return adj.coalesce()

    def gcn_forward(
        self,
        x: Tensor,
        adj: torch.sparse.Tensor,
        n2v: Tensor | None = None,
    ) -> Tensor:
        """Standard GCN forward pass on a single snapshot."""
        with torch.amp.autocast("cuda", enabled=False):
            h0 = x.float() if x.dtype == torch.float16 else x
            h = h0
            n2v_tensor: Tensor | None = None
            if n2v is not None and self.n2v_dim > 0:
                if not torch.is_tensor(n2v):
                    n2v_tensor = torch.as_tensor(n2v, dtype=torch.float32, device=h.device)
                else:
                    n2v_tensor = n2v
                    if n2v_tensor.device != h.device:
                        n2v_tensor = n2v_tensor.to(device=h.device)
                    if n2v_tensor.dtype != torch.float32:
                        n2v_tensor = n2v_tensor.float()

            for layer_idx, conv in enumerate(self.convs):
                aggregated = torch.sparse.mm(adj, h)
                h = conv(aggregated)
                if layer_idx == 0 and n2v_tensor is not None and self.n2v_proj is not None:
                    h = h + self.n2v_proj(n2v_tensor)
                if layer_idx < len(self.convs) - 1:
                    h = F.relu(h)
                    if self.dropout > 0:
                        h = F.dropout(h, p=self.dropout, training=self.training)

            out = self.out_proj(h)
            if n2v_tensor is not None and self.n2v_out_proj is not None:
                out = out + self.n2v_out_proj(n2v_tensor)
            residual = self.residual_proj(h0)
            gate = torch.sigmoid(self.residual_gate)
            out = out + gate * residual
            out = self.out_norm(out)
            out = self.final_dropout(out)
        return out

    def _should_checkpoint(self) -> bool:
        return self.training and (self.checkpoint_snapshots or self.auto_checkpoint)

    def _run_gcn(
        self,
        x: Tensor,
        adj: torch.sparse.Tensor,
        n2v: Tensor | None = None,
    ) -> Tensor:
        if self._should_checkpoint():
            if n2v is not None and self.n2v_dim > 0:
                return checkpoint(  # type: ignore[return-value]
                    lambda inp_x, inp_n: self.gcn_forward(inp_x, adj, inp_n),
                    x,
                    n2v,
                    use_reentrant=False,
                )
            return checkpoint(lambda inp: self.gcn_forward(inp, adj), x, use_reentrant=False)  # type: ignore[return-value]
        return self.gcn_forward(x, adj, n2v)

    def _forward_from_n2v(self, x: Tensor, n2v: Tensor) -> Tensor:
        """Return snapshot embeddings when using pre-computed Node2Vec features."""
        if n2v.device != x.device:
            n2v = n2v.to(device=x.device)
        if n2v.dtype != torch.float32:
            n2v = n2v.float()

        embedding = n2v

        if self.concat_base_features and self.base_feat_proj is not None:
            base = x.float() if x.dtype != torch.float32 else x
            base = self.base_feat_proj(base)
            embedding = embedding + base

        return embedding

    def _encode_snapshot(
        self,
        x: Tensor,
        adj: torch.sparse.Tensor | None,
        n2v: Tensor | None,
    ) -> Tensor:
        if self.use_n2v_features:
            if n2v is None:
                raise ValueError("Node2Vec embeddings required when use_n2v_features=True")
            return self._forward_from_n2v(x, n2v)
        if adj is None:
            raise ValueError("Adjacency tensor required when use_n2v_features=False")
        return self._run_gcn(x, adj, n2v)

    def forward(
        self,
        x: Tensor,
        edge_indices: list[Tensor],
        n2v_embeddings: list[Tensor] | None = None,
        snapshot_weights: Tensor | None = None,
    ) -> Tensor:
        """Forward pass across all snapshots with temporal aggregation."""
        num_nodes = x.size(0)
        device = x.device
        if self.use_n2v_features:
            if n2v_embeddings is None:
                raise ValueError("Node2Vec embeddings must be provided when use_n2v_features=True")
            num_snapshots = len(n2v_embeddings)
        else:
            num_snapshots = len(edge_indices)

        if num_snapshots == 0:
            raise ValueError("Must provide at least one snapshot")

        adjs: list[torch.sparse.Tensor | None]
        if self.use_n2v_features:
            adjs = [None] * num_snapshots
        else:
            adjs = []
            for edge_index in edge_indices:
                if not isinstance(edge_index, torch.Tensor):
                    raise TypeError("edge_indices must be a list of torch.Tensor objects")
                if edge_index.device != device:
                    edge_index = edge_index.to(device)
                adj = self.normalize_adjacency(edge_index, num_nodes)
                adjs.append(adj)

        if self.n2v_dim > 0 or self.use_n2v_features:
            if n2v_embeddings is None:
                raise ValueError(
                    "Node2Vec embeddings must be provided when Node2Vec features are enabled"
                )
            if len(n2v_embeddings) != num_snapshots:
                raise ValueError(
                    f"Expected {num_snapshots} Node2Vec embeddings, got {len(n2v_embeddings)}",
                )
            n2v_list: list[Tensor | None] = []
            target_device = device if not self.use_n2v_features else torch.device("cpu")
            for idx, emb in enumerate(n2v_embeddings):
                if not torch.is_tensor(emb):
                    emb_tensor = torch.as_tensor(emb, dtype=torch.float32)
                else:
                    emb_tensor = emb
                    if emb_tensor.dtype != torch.float32:
                        emb_tensor = emb_tensor.float()
                if emb_tensor.device != target_device:
                    emb_tensor = emb_tensor.to(device=target_device)
                if emb_tensor.shape[0] != num_nodes:
                    raise ValueError(
                        f"Node2Vec embedding {idx} has {emb_tensor.shape[0]} nodes, expected {num_nodes}",
                    )
                if self.n2v_dim > 0 and emb_tensor.shape[1] != self.n2v_dim:
                    raise ValueError(
                        f"Node2Vec embedding {idx} has dim {emb_tensor.shape[1]}, expected {self.n2v_dim}",
                    )
                n2v_list.append(emb_tensor)
        else:
            if n2v_embeddings is not None:
                LOGGER.debug(
                    "Ignoring supplied Node2Vec embeddings because n2v_dim == 0 and use_n2v_features=False"
                )
            n2v_list = [None] * num_snapshots

        if self.temporal_mode == "last":
            last_adj = adjs[-1]
            return self._encode_snapshot(x, last_adj, n2v_list[-1])

        if self.temporal_mode == "attention":
            last_adj = adjs[-1]
            n2v_last = n2v_list[-1]
            last_emb = self._encode_snapshot(x, last_adj, n2v_last)

            with torch.amp.autocast("cuda", enabled=False):
                last_emb_fp32 = last_emb.float() if last_emb.dtype == torch.float16 else last_emb
                query = self.temporal_query(last_emb_fp32)

            accum_dtype = torch.float32 if last_emb.dtype == torch.float16 else last_emb.dtype
            query_accum = query.to(accum_dtype)
            log_sum = torch.full((num_nodes,), float("-inf"), device=device, dtype=accum_dtype)
            weighted_sum = torch.zeros(
                num_nodes, self.out_channels, device=device, dtype=accum_dtype
            )

            if self.temporal_decay > 0:
                time_indices = torch.arange(
                    num_snapshots - 1, -1, -1, device=device, dtype=accum_dtype
                )
                recency_bias = (-self.temporal_decay * time_indices).to(accum_dtype)
            else:
                recency_bias = None

            model_size = self.model_complexity
            batch_size = 1 if model_size > 10000 else max(1, self.snapshot_batch_size)
            for batch_start in range(0, num_snapshots, batch_size):
                batch_end = min(batch_start + batch_size, num_snapshots)
                for idx in range(batch_start, batch_end):
                    adj = adjs[idx]
                    n2v_snap = n2v_list[idx]
                    if idx == num_snapshots - 1:
                        z = last_emb
                    else:
                        z = self._encode_snapshot(x, adj, n2v_snap)

                    if torch.isnan(z).any():
                        LOGGER.warning(
                            "NaN in snapshot %d embeddings! Skipping this snapshot.", idx
                        )
                        continue

                    with torch.amp.autocast("cuda", enabled=False):
                        z_fp32 = z.float() if z.dtype == torch.float16 else z
                        key = self.temporal_key(z_fp32).to(accum_dtype)
                        value = self.temporal_value(z_fp32).to(accum_dtype)

                    if torch.isnan(key).any() or torch.isnan(value).any():
                        LOGGER.warning(
                            "NaN in key/value for snapshot %d! Skipping this snapshot.", idx
                        )
                        continue

                    score = (query_accum * key).sum(dim=1) / math.sqrt(self.out_channels)
                    score = torch.nan_to_num(score, nan=0.0, posinf=80.0, neginf=-80.0)
                    score = torch.clamp(score, min=-80.0, max=80.0)

                    if recency_bias is not None:
                        score = score + recency_bias[idx]

                    if snapshot_weights is not None:
                        weight = snapshot_weights[idx]
                        if not torch.is_tensor(weight):
                            weight = torch.tensor(weight, device=device, dtype=accum_dtype)
                        else:
                            weight = weight.to(device=device, dtype=accum_dtype)
                        score = score + weight

                    new_log_sum = torch.logaddexp(log_sum, score)
                    weight_prev = torch.exp(log_sum - new_log_sum)
                    weight_new = torch.exp(score - new_log_sum)
                    weighted_sum.mul_(weight_prev.unsqueeze(1)).add_(
                        weight_new.unsqueeze(1) * value
                    )
                    log_sum = new_log_sum

                    del key, value, score, new_log_sum, weight_prev, weight_new

                if batch_end < num_snapshots and device.type == "cuda":
                    torch.cuda.empty_cache()

            z_final = weighted_sum
            if torch.isnan(z_final).any():
                LOGGER.warning(
                    "NaN detected in attention output! Using fallback (last snapshot only)"
                )
                z_final = last_emb
            elif z_final.dtype != query.dtype:
                z_final = z_final.to(query.dtype)

            return z_final

        if self.temporal_mode == "mean":
            z_sum = torch.zeros(num_nodes, self.out_channels, device=device)
            batch_size = self.snapshot_batch_size

            for batch_start in range(0, num_snapshots, batch_size):
                batch_end = min(batch_start + batch_size, num_snapshots)
                for idx in range(batch_start, batch_end):
                    adj = adjs[idx]
                    n2v_snap = n2v_list[idx]
                    z = self._encode_snapshot(x, adj, n2v_snap)
                    z_sum += z

                if batch_end < num_snapshots and device.type == "cuda":
                    torch.cuda.empty_cache()

            return z_sum / num_snapshots

        raise ValueError(f"Unknown temporal_mode: {self.temporal_mode}")

    def reset_parameters(self) -> None:
        """Reset all learnable parameters."""
        for conv in self.convs:
            if hasattr(conv, "reset_parameters"):
                conv.reset_parameters()  # type: ignore[operator]
        if hasattr(self.out_proj, "reset_parameters"):
            self.out_proj.reset_parameters()
        if hasattr(self.residual_proj, "reset_parameters"):
            self.residual_proj.reset_parameters()

        if self.temporal_mode == "attention":
            for module in (self.temporal_query, self.temporal_key, self.temporal_value):
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()

        for proj in self.head_projections:
            nn.init.eye_(proj.weight)

        if self.n2v_proj is not None:
            nn.init.xavier_uniform_(self.n2v_proj.weight)
            if self.n2v_proj.bias is not None:
                nn.init.zeros_(self.n2v_proj.bias)
        if self.n2v_out_proj is not None:
            nn.init.xavier_uniform_(self.n2v_out_proj.weight)
            if self.n2v_out_proj.bias is not None:
                nn.init.zeros_(self.n2v_out_proj.bias)
        if self.base_feat_proj is not None:
            nn.init.zeros_(self.base_feat_proj.weight)

        with torch.no_grad():
            self.residual_gate.fill_(0.0)
        if hasattr(self.out_norm, "reset_parameters"):
            self.out_norm.reset_parameters()

    def project_heads(self, z: Tensor) -> list[Tensor]:
        return [proj(z) for proj in self.head_projections]


class TemporalGNNIncremental(nn.Module):
    """
    Memory-efficient incremental TGNN.

    Processes snapshots one at a time with exponential decay of previous embeddings.
    Useful for very long sequences where storing all snapshot embeddings is prohibitive.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        decay_factor: float = 0.9,
    ):
        """
        Initialize Incremental TGNN.

        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden layer dimension
            out_channels: Output embedding dimension
            num_layers: Number of GCN layers
            dropout: Dropout probability
            decay_factor: Exponential decay for previous embeddings (0-1)
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.decay_factor = decay_factor

        # Standard GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(nn.Linear(hidden_channels, hidden_channels))
        self.out_proj = nn.Linear(hidden_channels, out_channels)

        # Learnable combination of current + accumulated
        self.combine = nn.Linear(out_channels * 2, out_channels)

        LOGGER.info(
            f"Incremental TGNN: in={in_channels}, hidden={hidden_channels}, out={out_channels}, "
            f"layers={num_layers}, dropout={dropout:.3f}, decay={decay_factor:.3f}"
        )

    def gcn_forward(self, x: Tensor, adj: torch.sparse.Tensor) -> Tensor:
        """Standard GCN forward pass."""
        h = x
        for _i, conv in enumerate(self.convs):
            h = conv(h)
            h = torch.sparse.mm(adj, h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        out = self.out_proj(h)
        out = torch.sparse.mm(adj, out)
        return out

    def forward_incremental(
        self,
        x: Tensor,
        edge_indices: list[Tensor],
    ) -> Tensor:
        """
        Process snapshots incrementally with accumulation.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_indices: List of edge indices for each snapshot

        Returns:
            Accumulated node embeddings [num_nodes, out_channels]
        """
        num_nodes = x.size(0)
        device = x.device

        # Initialize accumulated embeddings
        z_accum = torch.zeros(num_nodes, self.out_channels, device=device)

        for i, edge_index in enumerate(edge_indices):
            # Build sparse adjacency
            val = torch.ones(edge_index.size(1), device=device, dtype=torch.float32)
            adj = torch.sparse_coo_tensor(edge_index, val, (num_nodes, num_nodes)).coalesce()

            # Get embeddings for current snapshot
            z_curr = self.gcn_forward(x, adj)

            # Combine with accumulated (decayed) embeddings
            if i == 0:
                z_accum = z_curr
            else:
                # Decay previous accumulation and add current
                z_combined = torch.cat([z_accum * self.decay_factor, z_curr], dim=1)
                z_accum = self.combine(z_combined)
                z_accum = F.relu(z_accum)

        return z_accum


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    print("Testing TGNN architecture...")
    num_nodes = 100
    in_channels = 16
    hidden_channels = 32
    out_channels = 16
    num_snapshots = 5

    # Create random data
    x = torch.randn(num_nodes, in_channels)
    edge_indices = [torch.randint(0, num_nodes, (2, 200)) for _ in range(num_snapshots)]

    # Test attention mode
    model = TemporalGNN(in_channels, hidden_channels, out_channels, temporal_mode="attention")
    z = model(x, edge_indices)
    print(f"[OK] Attention mode: {z.shape}")

    # Test backward
    loss = z.mean()
    loss.backward()
    print("[OK] Backward pass successful")

    print("\n[PASS] All tests passed!")
