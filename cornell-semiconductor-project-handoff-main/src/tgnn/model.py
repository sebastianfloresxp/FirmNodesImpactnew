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
    using attention, mean, or last-only strategies.
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
    ):
        """
        Initialize TGNN.

        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden layer dimension
            out_channels: Output embedding dimension
            num_layers: Number of GCN layers
            dropout: Dropout probability
            temporal_mode: How to aggregate snapshots ("attention", "mean", "last")
            snapshot_batch_size: Max snapshots to process at once (memory control)
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.temporal_mode = temporal_mode
        self.snapshot_batch_size = snapshot_batch_size
        self.checkpoint_snapshots = checkpoint_snapshots
        self.checkpoint_threshold = checkpoint_threshold
        self.temporal_decay = temporal_decay
        self.num_score_heads = max(1, int(num_score_heads))
        self.model_complexity = hidden_channels * out_channels * max(1, num_layers)
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
        self.out_proj = nn.Linear(hidden_channels, out_channels)
        self.residual_proj = nn.Linear(in_channels, out_channels, bias=False)
        self.residual_gate = nn.Parameter(torch.tensor(0.0))
        self.out_norm = nn.LayerNorm(out_channels)
        self.final_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.skip_proj = nn.Linear(in_channels, out_channels, bias=False)
        self.head_skip_gates = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.0)) for _ in range(self.num_score_heads)]
        )

        # Temporal aggregation (if using attention)
        if temporal_mode == "attention":
            self.temporal_query = nn.Linear(out_channels, out_channels)
            self.temporal_key = nn.Linear(out_channels, out_channels)
            self.temporal_value = nn.Linear(out_channels, out_channels)
            LOGGER.debug(f"TGNN initialized with attention aggregation ({out_channels}D)")
        elif temporal_mode == "mean":
            LOGGER.debug("TGNN initialized with mean aggregation")
        elif temporal_mode == "last":
            LOGGER.debug("TGNN initialized with last-snapshot aggregation")
        else:
            raise ValueError(f"Unknown temporal_mode: {temporal_mode}")

        LOGGER.info(
            f"TGNN: in={in_channels}, hidden={hidden_channels}, out={out_channels}, "
            f"layers={num_layers}, dropout={dropout:.3f}, mode={temporal_mode}, "
            f"snapshot_batch={snapshot_batch_size}, temporal_decay={temporal_decay:.3f}, "
            f"score_heads={self.num_score_heads}"
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
            LOGGER.info(f"  Temporal recency bias: exponential decay rate={temporal_decay:.3f}")

        self.head_projections = nn.ModuleList()
        for _head_idx in range(self.num_score_heads):
            proj = nn.Linear(out_channels, out_channels, bias=False)
            nn.init.eye_(proj.weight)
            self.head_projections.append(proj)

    @staticmethod
    def normalize_adjacency(
        edge_index: Tensor,
        num_nodes: int,
        add_self_loops: bool = True,
    ) -> torch.sparse.Tensor:
        """Return D^-1/2 (A + I) D^-1/2 as a sparse tensor."""
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

    def gcn_forward(self, x: Tensor, adj: torch.sparse.Tensor) -> Tensor:
        """
        Standard GCN forward pass on a single snapshot.

        Sparse operations don't support float16, so we disable autocast here.

        Args:
            x: Node features [num_nodes, in_channels]
            adj: Sparse adjacency matrix [num_nodes, num_nodes]

        Returns:
            Node embeddings [num_nodes, out_channels]
        """
        # Disable autocast for sparse operations (not supported in fp16)
        with torch.amp.autocast("cuda", enabled=False):
            h0 = x.float() if x.dtype == torch.float16 else x
            h = h0
            for i, conv in enumerate(self.convs):
                aggregated = torch.sparse.mm(adj, h)
                h = conv(aggregated)
                if i < len(self.convs) - 1:
                    h = F.relu(h)
                    if self.dropout > 0:
                        h = F.dropout(h, p=self.dropout, training=self.training)

            out = self.out_proj(h)
            residual = self.residual_proj(h0)
            gate = torch.sigmoid(self.residual_gate)
            out = out + gate * residual
            out = self.out_norm(out)
            out = self.final_dropout(out)
        return out

    def _should_checkpoint(self) -> bool:
        return self.training and (self.checkpoint_snapshots or self.auto_checkpoint)

    def _run_gcn(self, x: Tensor, adj: torch.sparse.Tensor) -> Tensor:
        if self._should_checkpoint():

            def run(inp: Tensor) -> Tensor:
                return self.gcn_forward(inp, adj)

            return checkpoint(run, x, use_reentrant=False)  # type: ignore[return-value]
        return self.gcn_forward(x, adj)

    def forward(
        self,
        x: Tensor,
        edge_indices: list[Tensor],
        snapshot_weights: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass across all snapshots with temporal aggregation.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_indices: List of edge indices [2, num_edges] for each snapshot
            snapshot_weights: Optional weights for each snapshot [num_snapshots]

        Returns:
            Aggregated node embeddings [num_nodes, out_channels]
        """
        num_nodes = x.size(0)
        device = x.device
        num_snapshots = len(edge_indices)

        if num_snapshots == 0:
            raise ValueError("Must provide at least one snapshot")

        # Convert input to normalized sparse adjacencies when necessary
        adjs: list[torch.sparse.Tensor] = []
        for edge_index in edge_indices:
            if not isinstance(edge_index, torch.Tensor):
                raise TypeError("edge_indices must be a list of torch.Tensor objects")

            if edge_index.device != device:
                edge_index = edge_index.to(device)

            adj = self.normalize_adjacency(edge_index, num_nodes)
            adjs.append(adj)

        # Process snapshots in batches to control memory usage
        # This prevents OOM when num_snapshots is large (e.g., 62 quarterly snapshots)

        # Optimization: if only using last snapshot, don't store all embeddings
        if self.temporal_mode == "last":
            # Only process and keep the last snapshot
            last_adj = adjs[-1]
            z_final = self._run_gcn(x, last_adj)
            return z_final

        # For attention mode with many snapshots, use memory-efficient batched processing
        if self.temporal_mode == "attention":
            # First, get the query from the last snapshot
            last_adj = adjs[-1]
            last_emb = self._run_gcn(x, last_adj)

            # Apply temporal query projection with autocast disabled for stability
            with torch.amp.autocast("cuda", enabled=False):
                last_emb_fp32 = last_emb.float() if last_emb.dtype == torch.float16 else last_emb
                query = self.temporal_query(last_emb_fp32)  # [num_nodes, out_channels]

            # Streaming softmax accumulators (per node)
            accum_dtype = torch.float32 if last_emb.dtype == torch.float16 else last_emb.dtype
            query_accum = query.to(accum_dtype)
            log_sum = torch.full((num_nodes,), float("-inf"), device=device, dtype=accum_dtype)
            weighted_sum = torch.zeros(
                num_nodes, self.out_channels, device=device, dtype=accum_dtype
            )

            # Compute temporal recency bias (exponential decay from oldest to newest)
            # time_indices: [num_snapshots-1, ..., 1, 0] (0 = most recent)
            # recency_bias: log-space weights, more negative for older snapshots
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
                    # Reuse last embedding if this is the last snapshot (avoid recomputing)
                    is_last = idx == num_snapshots - 1
                    if is_last:
                        z = last_emb
                    else:
                        z = self._run_gcn(x, adj)

                    # Check for NaN in embeddings before attention
                    if torch.isnan(z).any():
                        LOGGER.warning(f"NaN in snapshot {idx} embeddings! Skipping this snapshot.")
                        continue

                    # Apply temporal projections with autocast disabled for stability
                    with torch.amp.autocast("cuda", enabled=False):
                        z_fp32 = z.float() if z.dtype == torch.float16 else z
                        key = self.temporal_key(z_fp32).to(accum_dtype)
                        value = self.temporal_value(z_fp32).to(accum_dtype)

                    # Check for NaN after linear projections
                    if torch.isnan(key).any() or torch.isnan(value).any():
                        LOGGER.warning(
                            f"NaN in key/value for snapshot {idx}! Skipping this snapshot."
                        )
                        continue

                    score = (query_accum * key).sum(dim=1) / math.sqrt(self.out_channels)
                    score = torch.nan_to_num(score, nan=0.0, posinf=80.0, neginf=-80.0)
                    score = torch.clamp(score, min=-80.0, max=80.0)

                    # Apply temporal recency bias (exponential decay)
                    if recency_bias is not None:
                        score = score + recency_bias[idx]

                    # Optional external weights (log-space) if provided
                    if snapshot_weights is not None:
                        weight = snapshot_weights[idx]
                        if not torch.is_tensor(weight):
                            weight = torch.tensor(weight, device=device, dtype=accum_dtype)
                        else:
                            weight = weight.to(device=device, dtype=accum_dtype)
                        score = score + weight

                    # Streaming softmax update via log-sum-exp
                    new_log_sum = torch.logaddexp(log_sum, score)
                    weight_prev = torch.exp(log_sum - new_log_sum)
                    weight_new = torch.exp(score - new_log_sum)
                    weighted_sum.mul_(weight_prev.unsqueeze(1)).add_(
                        weight_new.unsqueeze(1) * value
                    )
                    log_sum = new_log_sum

                    # Encourage GC to drop intermediates
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

        # For mean mode (if ever re-enabled), process in batches
        elif self.temporal_mode == "mean":
            z_sum = torch.zeros(num_nodes, self.out_channels, device=device)
            batch_size = self.snapshot_batch_size

            for batch_start in range(0, num_snapshots, batch_size):
                batch_end = min(batch_start + batch_size, num_snapshots)
                for idx in range(batch_start, batch_end):
                    adj = adjs[idx]
                    z = self._run_gcn(x, adj)
                    z_sum += z

                if batch_end < num_snapshots and device.type == "cuda":
                    torch.cuda.empty_cache()

            z_final = z_sum / num_snapshots
            return z_final
        else:
            raise ValueError(f"Unknown temporal_mode: {self.temporal_mode}")

    def reset_parameters(self):
        """Reset all learnable parameters."""
        for conv in self.convs:
            if hasattr(conv, "reset_parameters"):
                conv.reset_parameters()  # type: ignore[operator]
        if hasattr(self.out_proj, "reset_parameters"):
            self.out_proj.reset_parameters()
        if hasattr(self.residual_proj, "reset_parameters"):
            self.residual_proj.reset_parameters()
        with torch.no_grad():
            self.residual_gate.fill_(0.0)
        if hasattr(self.out_norm, "reset_parameters"):
            self.out_norm.reset_parameters()
        if hasattr(self.skip_proj, "reset_parameters"):
            self.skip_proj.reset_parameters()
        for gate in self.head_skip_gates:
            with torch.no_grad():
                gate.fill_(0.0)

        if self.temporal_mode == "attention":
            for module in [self.temporal_query, self.temporal_key, self.temporal_value]:
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()

        for proj in self.head_projections:
            nn.init.eye_(proj.weight)

    def project_heads(self, z: Tensor, skip: Tensor | None = None) -> list[Tensor]:
        outputs: list[Tensor] = []
        for idx, proj in enumerate(self.head_projections):
            head = proj(z)
            if skip is not None:
                gate = torch.sigmoid(self.head_skip_gates[idx])
                head = head + gate * skip
            outputs.append(head)
        return outputs


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
