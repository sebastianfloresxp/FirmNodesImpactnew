"""
TGNN (Temporal Graph Neural Network) Package
=============================================

Snapshot-based temporal GNN with attention aggregation for link prediction.

This package implements a temporal graph neural network that:
1. Processes quarterly/annual graph snapshots independently with GCN
2. Aggregates embeddings across time using attention mechanism
3. Predicts future links based on temporally-enriched node representations

Compared to the baseline `tgnn_temporal` fork, this variant fuses
per-snapshot Node2Vec embeddings into the encoder and output projections.

Key differences from EvolveGCN:
- Evolves embeddings (not GCN weights) → no RNN gradient issues
- Aggregates across ALL snapshots simultaneously → learns global temporal patterns
- Full gradient flow → can actually learn from history

Modules:
- model: TGNN architecture (attention, mean, last aggregation modes)
- run_n2v_temporal_eval: Core training and evaluation logic
- 01_n2v_temporal_hpo_optuna: Phase A hyperparameter optimization
- 02_n2v_temporal_finalize_hpo: Multi-seed robust evaluation
- 00_n2v_temporal_hpo_two_phase: Full HPO orchestrator
- 04_build_n2v_embeddings: Utility to generate per-snapshot Node2Vec features
- 03_n2v_temporal_train_eval: Production CLI wrapper
"""

__version__ = "1.0.0"
__author__ = "Semiconductor Supply Chain Team"
