#!/usr/bin/env python3
"""
Comprehensive DoD Supply Chain Analysis and Visualization
========================================================

Publication-ready analysis of the DoD observed subgraph with:
- Rich metadata integration (sectors, industries, geography)
- Multi-tier analysis with entity samples
- Critical node identification
- Publication-quality static visualizations
- Interactive exploration interface
- Sector/industry distribution analysis

This serves as the baseline before adding predicted links.
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import json
from datetime import datetime
import warnings
from typing import Dict, List, Tuple, Any, Optional
import logging
from collections import Counter, defaultdict
import random

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style for publication-ready plots
plt.style.use('default')
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

class ComprehensiveDoDAnalyzer:
    """Comprehensive analysis and visualization of DoD supply chain network."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.output_dir = Path("../results/dod_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.edges_df = None
        self.tiers_df = None
        self.contractors_df = None
        self.node_features_df = None
        self.G = None
        
        # Analysis results
        self.network_stats = {}
        self.critical_nodes = None
        self.tier_samples = {}
        self.sector_analysis = {}
        
    def load_all_data(self):
        """Load all required data including node features."""
        logger.info("[DATA] Loading comprehensive DoD network data...")
        
        # Load DoD subgraph data
        self.edges_df = pd.read_parquet(self.data_dir / "dod_subgraph_edges.parquet")
        self.tiers_df = pd.read_parquet(self.data_dir / "dod_subgraph_tiers.parquet")
        self.contractors_df = pd.read_parquet(self.data_dir / "contractor_all_matches.parquet")
        
        # Load comprehensive node features
        node_features_file = Path("../data/processed/nodes/nodes_comprehensive_factset.parquet")
        if node_features_file.exists():
            self.node_features_df = pd.read_parquet(node_features_file)
            logger.info(f"[OK] Loaded {len(self.node_features_df):,} entities with features")
        else:
            logger.warning("[WARN] Node features file not found, using basic analysis")
            
        # Filter node features to DoD entities only
        if self.node_features_df is not None:
            dod_entities = set(self.tiers_df['factset_entity_id'])
            self.node_features_df = self.node_features_df[
                self.node_features_df['factset_entity_id'].isin(dod_entities)
            ]
            logger.info(f"[OK] Filtered to {len(self.node_features_df):,} DoD entities with features")
        
        logger.info(f"[OK] Data loaded: {len(self.edges_df):,} edges, {len(self.tiers_df):,} entities")
    
    def build_network_efficiently(self):
        """Build NetworkX graph with sampling for large network operations."""
        logger.info("[FIX] Building NetworkX graph...")
        
        # Build full graph
        self.G = nx.DiGraph()
        
        # Add edges in batches for efficiency
        edges_to_add = [(row['supplier_factset_entity_id'], row['customer_factset_entity_id']) 
                       for _, row in self.edges_df.iterrows()]
        self.G.add_edges_from(edges_to_add)
        
        # Add tier attributes
        tier_dict = dict(zip(self.tiers_df['factset_entity_id'], self.tiers_df['tier']))
        nx.set_node_attributes(self.G, tier_dict, 'tier')
        
        logger.info(f"[OK] Built graph: {self.G.number_of_nodes():,} nodes, {self.G.number_of_edges():,} edges")
        return self.G
    
    def calculate_network_statistics(self):
        """Calculate comprehensive network statistics with efficient sampling."""
        logger.info("[CHART] Calculating network statistics...")
        
        # Basic statistics
        self.network_stats = {
            'total_nodes': self.G.number_of_nodes(),
            'total_edges': self.G.number_of_edges(),
            'density': nx.density(self.G),
            'timestamp': datetime.now().isoformat()
        }
        
        # Degree statistics
        in_degrees = [d for n, d in self.G.in_degree()]
        out_degrees = [d for n, d in self.G.out_degree()]
        
        self.network_stats.update({
            'avg_in_degree': np.mean(in_degrees),
            'avg_out_degree': np.mean(out_degrees),
            'max_in_degree': max(in_degrees),
            'max_out_degree': max(out_degrees),
            'std_in_degree': np.std(in_degrees),
            'std_out_degree': np.std(out_degrees)
        })
        
        # Component analysis
        self.network_stats['weakly_connected_components'] = nx.number_weakly_connected_components(self.G)
        self.network_stats['strongly_connected_components'] = nx.number_strongly_connected_components(self.G)
        
        # Sample for expensive computations
        if self.G.number_of_nodes() > 50000:
            sample_nodes = random.sample(list(self.G.nodes()), 10000)
            sample_G = self.G.subgraph(sample_nodes)
            logger.info("Using sample of 10,000 nodes for clustering analysis")
        else:
            sample_G = self.G
        
        self.network_stats['avg_clustering'] = nx.average_clustering(sample_G)
        
        # Tier distribution
        tier_counts = self.tiers_df['tier'].value_counts().sort_index()
        self.network_stats['tier_distribution'] = tier_counts.to_dict()
        
        logger.info("[OK] Network statistics calculated")
        return self.network_stats
    
    def identify_critical_nodes(self, sample_size: int = 5000):
        """Identify critical nodes using centrality measures on a sample."""
        logger.info(f"[TARGET] Identifying critical nodes (sample size: {sample_size:,})...")
        
        # Sample nodes for centrality analysis
        if self.G.number_of_nodes() > sample_size:
            sample_nodes = random.sample(list(self.G.nodes()), sample_size)
            sample_G = self.G.subgraph(sample_nodes)
        else:
            sample_G = self.G
            sample_nodes = list(self.G.nodes())
        
        # Calculate centrality measures
        logger.info("Computing centrality measures...")
        betweenness = nx.betweenness_centrality(sample_G, k=min(1000, len(sample_nodes)))
        closeness = nx.closeness_centrality(sample_G)
        in_degree_centrality = nx.in_degree_centrality(sample_G)
        out_degree_centrality = nx.out_degree_centrality(sample_G)
        
        # Create centrality dataframe
        centrality_data = []
        for node in sample_nodes:
            centrality_data.append({
                'factset_entity_id': node,
                'betweenness': betweenness.get(node, 0),
                'closeness': closeness.get(node, 0),
                'in_degree_centrality': in_degree_centrality.get(node, 0),
                'out_degree_centrality': out_degree_centrality.get(node, 0),
                'tier': self.tiers_df[self.tiers_df['factset_entity_id'] == node]['tier'].iloc[0] 
                       if len(self.tiers_df[self.tiers_df['factset_entity_id'] == node]) > 0 else None
            })
        
        self.critical_nodes = pd.DataFrame(centrality_data)
        
        # Calculate composite criticality score
        self.critical_nodes['criticality_score'] = (
            self.critical_nodes['betweenness'] * 0.4 +
            self.critical_nodes['closeness'] * 0.3 +
            self.critical_nodes['in_degree_centrality'] * 0.15 +
            self.critical_nodes['out_degree_centrality'] * 0.15
        )
        
        # Sort by criticality
        self.critical_nodes = self.critical_nodes.sort_values('criticality_score', ascending=False)
        
        logger.info(f"[OK] Identified {len(self.critical_nodes)} critical nodes")
        return self.critical_nodes
    
    def sample_entities_by_tier(self, samples_per_tier: int = 10):
        """Sample representative entities from each tier with metadata."""
        logger.info(f"[RAND] Sampling {samples_per_tier} entities per tier...")
        
        for tier in sorted(self.tiers_df['tier'].unique()):
            tier_entities = self.tiers_df[self.tiers_df['tier'] == tier]['factset_entity_id'].tolist()
            
            # Sample entities
            sample_size = min(samples_per_tier, len(tier_entities))
            sampled_entities = random.sample(tier_entities, sample_size)
            
            # Get metadata for sampled entities
            tier_sample_data = []
            for entity_id in sampled_entities:
                entity_data = {'factset_entity_id': entity_id, 'tier': tier}
                
                # Add node features if available
                if self.node_features_df is not None:
                    features = self.node_features_df[
                        self.node_features_df['factset_entity_id'] == entity_id
                    ]
                    if len(features) > 0:
                        entity_data.update(features.iloc[0].to_dict())
                
                # Add contractor information if available
                if self.contractors_df is not None:
                    contractor_info = self.contractors_df[
                        self.contractors_df['factset_id'] == entity_id
                    ]
                    if len(contractor_info) > 0:
                        entity_data['contractor_name'] = contractor_info.iloc[0]['contractor_name']
                        entity_data['match_type'] = contractor_info.iloc[0]['match_type']
                        entity_data['match_score'] = contractor_info.iloc[0]['match_score']
                
                tier_sample_data.append(entity_data)
            
            self.tier_samples[tier] = pd.DataFrame(tier_sample_data)
        
        logger.info(f"[OK] Sampled entities across {len(self.tier_samples)} tiers")
        return self.tier_samples
    
    def analyze_sector_distribution(self):
        """Analyze sector and industry distribution across tiers."""
        logger.info("[FACTORY] Analyzing sector distribution...")
        
        if self.node_features_df is None:
            logger.warning("[WARN] No node features available for sector analysis")
            return {}
        
        # Merge tiers with node features
        tier_features = self.tiers_df.merge(
            self.node_features_df, 
            on='factset_entity_id', 
            how='left'
        )
        
        # Analyze by different classification systems
        classification_systems = {
            'primary_sic_code': 'SIC Codes',
            'industry_code': 'FactSet Industry',
            'sector_code': 'FactSet Sector',
            'rbics_l1_id': 'RBICS L1',
            'rbics_l2_id': 'RBICS L2',
            'rbics_l3_id': 'RBICS L3'
        }
        
        self.sector_analysis = {}
        
        for code_col, name in classification_systems.items():
            if code_col in tier_features.columns:
                # Overall distribution
                code_dist = tier_features[code_col].value_counts().head(20)
                
                # Distribution by tier
                tier_dist = tier_features.groupby(['tier', code_col]).size().unstack(fill_value=0)
                
                # Coverage analysis
                total_entities = len(tier_features)
                entities_with_code = tier_features[code_col].notna().sum()
                coverage_pct = (entities_with_code / total_entities) * 100
                
                self.sector_analysis[code_col] = {
                    'name': name,
                    'coverage_percent': coverage_pct,
                    'top_codes': code_dist.to_dict(),
                    'tier_distribution': tier_dist.to_dict(),
                    'unique_codes': tier_features[code_col].nunique()
                }
        
        logger.info(f"[OK] Analyzed {len(self.sector_analysis)} classification systems")
        return self.sector_analysis
    
    def create_publication_visualizations(self):
        """Create comprehensive publication-ready visualizations."""
        logger.info("[VIZ] Creating publication-ready visualizations...")
        
        # 1. Network Overview Dashboard
        self._create_network_overview()
        
        # 2. Tier Analysis Plots
        self._create_tier_analysis()
        
        # 3. Sector Distribution Analysis
        self._create_sector_analysis()
        
        # 4. Critical Nodes Analysis
        self._create_critical_nodes_analysis()
        
        # 5. Network Structure Analysis
        self._create_network_structure_analysis()
        
        # 6. Interactive Network Visualization
        self._create_interactive_visualizations()
        
        logger.info("[OK] All visualizations created")
    
    def _create_network_overview(self):
        """Create high-level network overview figure."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('DoD Supply Chain Network Overview', fontsize=24, fontweight='bold')
        
        # 1. Tier Distribution
        tier_counts = pd.Series(self.network_stats['tier_distribution'])
        tier_labels = [f'DoD' if i == 0 else f'Tier {i}' for i in tier_counts.index]
        
        bars = ax1.bar(range(len(tier_counts)), tier_counts.values, 
                      color=plt.cm.viridis(np.linspace(0, 1, len(tier_counts))))
        ax1.set_xlabel('Supply Chain Tier', fontweight='bold')
        ax1.set_ylabel('Number of Entities', fontweight='bold')
        ax1.set_title('Entity Distribution by Tier', fontweight='bold')
        ax1.set_xticks(range(len(tier_counts)))
        ax1.set_xticklabels(tier_labels, rotation=45)
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Degree Distribution
        in_degrees = [d for n, d in self.G.in_degree()]
        out_degrees = [d for n, d in self.G.out_degree()]
        
        ax2.hist(in_degrees, bins=50, alpha=0.6, label='In-Degree', color='skyblue')
        ax2.hist(out_degrees, bins=50, alpha=0.6, label='Out-Degree', color='lightcoral')
        ax2.set_xlabel('Degree', fontweight='bold')
        ax2.set_ylabel('Frequency', fontweight='bold')
        ax2.set_title('Degree Distribution', fontweight='bold')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Network Statistics Table
        ax3.axis('off')
        stats_data = [
            ['Total Entities', f"{self.network_stats['total_nodes']:,}"],
            ['Total Relationships', f"{self.network_stats['total_edges']:,}"],
            ['Network Density', f"{self.network_stats['density']:.2e}"],
            ['Avg In-Degree', f"{self.network_stats['avg_in_degree']:.1f}"],
            ['Avg Out-Degree', f"{self.network_stats['avg_out_degree']:.1f}"],
            ['Max In-Degree', f"{self.network_stats['max_in_degree']:,}"],
            ['Max Out-Degree', f"{self.network_stats['max_out_degree']:,}"],
            ['Weak Components', f"{self.network_stats['weakly_connected_components']:,}"],
            ['Strong Components', f"{self.network_stats['strongly_connected_components']:,}"]
        ]
        
        table = ax3.table(cellText=stats_data, 
                         colLabels=['Metric', 'Value'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)
        ax3.set_title('Network Statistics', fontweight='bold', pad=20)
        
        # 4. Tier Connectivity Heatmap
        tier_connectivity = self._calculate_tier_connectivity_matrix()
        im = ax4.imshow(tier_connectivity, cmap='YlOrRd', aspect='auto')
        
        tier_labels_short = ['DoD', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']
        ax4.set_xticks(range(len(tier_labels_short)))
        ax4.set_yticks(range(len(tier_labels_short)))
        ax4.set_xticklabels(tier_labels_short)
        ax4.set_yticklabels(tier_labels_short)
        ax4.set_xlabel('Customer Tier', fontweight='bold')
        ax4.set_ylabel('Supplier Tier', fontweight='bold')
        ax4.set_title('Tier Connectivity (log scale)', fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('Log10(Connections + 1)', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'network_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _calculate_tier_connectivity_matrix(self):
        """Calculate tier-to-tier connectivity matrix."""
        max_tier = self.tiers_df['tier'].max()
        connectivity = np.zeros((max_tier + 1, max_tier + 1))
        
        # Create tier lookup
        tier_lookup = dict(zip(self.tiers_df['factset_entity_id'], self.tiers_df['tier']))
        
        # Count connections between tiers
        for _, edge in self.edges_df.iterrows():
            src_tier = tier_lookup.get(edge['supplier_factset_entity_id'], -1)
            dst_tier = tier_lookup.get(edge['customer_factset_entity_id'], -1)
            
            if src_tier >= 0 and dst_tier >= 0:
                connectivity[src_tier, dst_tier] += 1
        
        # Apply log transformation for visualization
        return np.log10(connectivity + 1)
    
    def _create_tier_analysis(self):
        """Create detailed tier analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('DoD Supply Chain Tier Analysis', fontsize=24, fontweight='bold')
        
        # 1. Tier Growth Pattern
        tier_counts = pd.Series(self.network_stats['tier_distribution'])
        cumulative_counts = tier_counts.cumsum()
        
        ax1 = axes[0, 0]
        ax1.bar(range(len(tier_counts)), tier_counts.values, alpha=0.7, label='Tier Entities')
        ax1_twin = ax1.twinx()
        ax1_twin.plot(range(len(cumulative_counts)), cumulative_counts.values, 
                     'ro-', linewidth=3, markersize=8, label='Cumulative')
        
        ax1.set_xlabel('Tier', fontweight='bold')
        ax1.set_ylabel('Number of Entities', fontweight='bold', color='blue')
        ax1_twin.set_ylabel('Cumulative Entities', fontweight='bold', color='red')
        ax1.set_title('Supply Chain Expansion by Tier', fontweight='bold')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # 2. Within-Tier vs Cross-Tier Connections
        within_tier_edges = []
        cross_tier_edges = []
        tier_lookup = dict(zip(self.tiers_df['factset_entity_id'], self.tiers_df['tier']))
        
        for tier in sorted(tier_counts.index):
            within_count = 0
            cross_count = 0
            
            for _, edge in self.edges_df.iterrows():
                src_tier = tier_lookup.get(edge['supplier_factset_entity_id'])
                dst_tier = tier_lookup.get(edge['customer_factset_entity_id'])
                
                if src_tier == tier:
                    if dst_tier == tier:
                        within_count += 1
                    else:
                        cross_count += 1
            
            within_tier_edges.append(within_count)
            cross_tier_edges.append(cross_count)
        
        ax2 = axes[0, 1]
        x = np.arange(len(tier_counts))
        width = 0.35
        
        ax2.bar(x - width/2, within_tier_edges, width, label='Within Tier', alpha=0.8)
        ax2.bar(x + width/2, cross_tier_edges, width, label='Cross Tier', alpha=0.8)
        
        ax2.set_xlabel('Tier', fontweight='bold')
        ax2.set_ylabel('Number of Connections', fontweight='bold')
        ax2.set_title('Within-Tier vs Cross-Tier Connectivity', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'T{i}' if i > 0 else 'DoD' for i in tier_counts.index])
        ax2.legend()
        ax2.set_yscale('log')
        
        # 3. Average Degree by Tier
        avg_in_degree_by_tier = []
        avg_out_degree_by_tier = []
        
        for tier in sorted(tier_counts.index):
            tier_nodes = set(self.tiers_df[self.tiers_df['tier'] == tier]['factset_entity_id'])
            
            in_degrees = [self.G.in_degree(node) for node in tier_nodes if node in self.G]
            out_degrees = [self.G.out_degree(node) for node in tier_nodes if node in self.G]
            
            avg_in_degree_by_tier.append(np.mean(in_degrees) if in_degrees else 0)
            avg_out_degree_by_tier.append(np.mean(out_degrees) if out_degrees else 0)
        
        ax3 = axes[1, 0]
        x = np.arange(len(tier_counts))
        
        ax3.plot(x, avg_in_degree_by_tier, 'o-', linewidth=3, markersize=8, label='Avg In-Degree')
        ax3.plot(x, avg_out_degree_by_tier, 's-', linewidth=3, markersize=8, label='Avg Out-Degree')
        
        ax3.set_xlabel('Tier', fontweight='bold')
        ax3.set_ylabel('Average Degree', fontweight='bold')
        ax3.set_title('Average Connectivity by Tier', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'T{i}' if i > 0 else 'DoD' for i in tier_counts.index])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Critical Nodes by Tier
        if self.critical_nodes is not None:
            ax4 = axes[1, 1]
            
            # Count critical nodes by tier (top 10% of criticality scores)
            threshold = self.critical_nodes['criticality_score'].quantile(0.9)
            top_critical = self.critical_nodes[self.critical_nodes['criticality_score'] >= threshold]
            
            critical_by_tier = top_critical['tier'].value_counts().sort_index()
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(critical_by_tier)))
            ax4.pie(critical_by_tier.values, 
                   labels=[f'Tier {i}' if i > 0 else 'DoD' for i in critical_by_tier.index],
                   autopct='%1.1f%%',
                   colors=colors,
                   startangle=90)
            ax4.set_title('Distribution of Critical Nodes\n(Top 10% by Criticality)', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'tier_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_sector_analysis(self):
        """Create sector and industry analysis plots."""
        if not self.sector_analysis:
            logger.warning("[WARN] No sector analysis data available")
            return
        
        # Create multi-panel sector analysis
        n_systems = len(self.sector_analysis)
        n_cols = 2
        n_rows = (n_systems + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows))
        fig.suptitle('DoD Supply Chain Sector Analysis', fontsize=24, fontweight='bold')
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (code_col, analysis) in enumerate(self.sector_analysis.items()):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]
            
            # Plot top categories
            top_codes = pd.Series(analysis['top_codes']).head(10)
            
            bars = ax.barh(range(len(top_codes)), top_codes.values)
            ax.set_yticks(range(len(top_codes)))
            ax.set_yticklabels([str(code)[:15] + ('...' if len(str(code)) > 15 else '') 
                               for code in top_codes.index])
            ax.set_xlabel('Number of Entities', fontweight='bold')
            ax.set_title(f'{analysis["name"]}\n(Coverage: {analysis["coverage_percent"]:.1f}%)', 
                        fontweight='bold')
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                       f'{int(width):,}', ha='left', va='center', fontweight='bold')
        
        # Remove empty subplots
        for idx in range(n_systems, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sector_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_critical_nodes_analysis(self):
        """Create critical nodes analysis visualization."""
        if self.critical_nodes is None or len(self.critical_nodes) == 0:
            logger.warning("[WARN] No critical nodes data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('DoD Supply Chain Critical Nodes Analysis', fontsize=24, fontweight='bold')
        
        # 1. Top Critical Nodes
        top_20 = self.critical_nodes.head(20)
        
        ax1 = axes[0, 0]
        bars = ax1.barh(range(len(top_20)), top_20['criticality_score'])
        ax1.set_yticks(range(len(top_20)))
        ax1.set_yticklabels([f"Node {i+1}" for i in range(len(top_20))])
        ax1.set_xlabel('Criticality Score', fontweight='bold')
        ax1.set_title('Top 20 Critical Nodes', fontweight='bold')
        
        # Color by tier
        tier_colors = plt.cm.viridis(top_20['tier'] / top_20['tier'].max())
        for bar, color in zip(bars, tier_colors):
            bar.set_color(color)
        
        # 2. Centrality Measures Comparison
        ax2 = axes[0, 1]
        centrality_cols = ['betweenness', 'closeness', 'in_degree_centrality', 'out_degree_centrality']
        
        x = np.arange(len(centrality_cols))
        means = [top_20[col].mean() for col in centrality_cols]
        stds = [top_20[col].std() for col in centrality_cols]
        
        bars = ax2.bar(x, means, yerr=stds, capsize=5, alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Betweenness', 'Closeness', 'In-Degree', 'Out-Degree'], rotation=45)
        ax2.set_ylabel('Average Centrality', fontweight='bold')
        ax2.set_title('Centrality Measures for Top Critical Nodes', fontweight='bold')
        
        # 3. Critical Nodes by Tier Distribution
        ax3 = axes[1, 0]
        
        tier_dist = self.critical_nodes['tier'].value_counts().sort_index()
        colors = plt.cm.Set3(np.linspace(0, 1, len(tier_dist)))
        
        wedges, texts, autotexts = ax3.pie(tier_dist.values, 
                                          labels=[f'Tier {i}' if i > 0 else 'DoD' for i in tier_dist.index],
                                          autopct='%1.1f%%',
                                          colors=colors,
                                          startangle=90)
        
        ax3.set_title('Critical Nodes Distribution by Tier', fontweight='bold')
        
        # 4. Criticality Score Distribution
        ax4 = axes[1, 1]
        
        ax4.hist(self.critical_nodes['criticality_score'], bins=30, alpha=0.7, edgecolor='black')
        ax4.axvline(self.critical_nodes['criticality_score'].mean(), 
                   color='red', linestyle='--', linewidth=2, 
                   label=f"Mean: {self.critical_nodes['criticality_score'].mean():.3f}")
        ax4.axvline(self.critical_nodes['criticality_score'].quantile(0.9), 
                   color='orange', linestyle='--', linewidth=2,
                   label=f"90th %ile: {self.critical_nodes['criticality_score'].quantile(0.9):.3f}")
        
        ax4.set_xlabel('Criticality Score', fontweight='bold')
        ax4.set_ylabel('Frequency', fontweight='bold')
        ax4.set_title('Distribution of Criticality Scores', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'critical_nodes_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_network_structure_analysis(self):
        """Create network structure analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('DoD Supply Chain Network Structure Analysis', fontsize=24, fontweight='bold')
        
        # 1. Path Length Analysis (sample-based)
        ax1 = axes[0, 0]
        
        # Sample paths from DoD node
        dod_node = 'DOD_CENTRAL'
        if dod_node in self.G:
            # Sample 1000 nodes for path analysis
            sample_nodes = random.sample([n for n in self.G.nodes() if n != dod_node], 
                                       min(1000, self.G.number_of_nodes() - 1))
            
            path_lengths = []
            for target in sample_nodes:
                try:
                    if nx.has_path(self.G, dod_node, target):
                        path_length = nx.shortest_path_length(self.G, dod_node, target)
                        path_lengths.append(path_length)
                except:
                    continue
            
            if path_lengths:
                ax1.hist(path_lengths, bins=range(1, max(path_lengths) + 2), 
                        alpha=0.7, edgecolor='black')
                ax1.set_xlabel('Path Length from DoD', fontweight='bold')
                ax1.set_ylabel('Number of Nodes', fontweight='bold')
                ax1.set_title('Distribution of Path Lengths from DoD', fontweight='bold')
                ax1.axvline(np.mean(path_lengths), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(path_lengths):.1f}')
                ax1.legend()
        
        # 2. Degree Distribution (log-log plot)
        ax2 = axes[0, 1]
        
        degrees = [d for n, d in self.G.degree()]
        degree_counts = Counter(degrees)
        
        x = list(degree_counts.keys())
        y = list(degree_counts.values())
        
        ax2.loglog(x, y, 'bo', alpha=0.6, markersize=4)
        ax2.set_xlabel('Degree (log scale)', fontweight='bold')
        ax2.set_ylabel('Frequency (log scale)', fontweight='bold')
        ax2.set_title('Degree Distribution (Log-Log)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Component Size Distribution
        ax3 = axes[1, 0]
        
        # Weakly connected components
        wcc_sizes = [len(c) for c in nx.weakly_connected_components(self.G)]
        wcc_counts = Counter(wcc_sizes)
        
        # Plot only components of size > 1
        filtered_sizes = [(size, count) for size, count in wcc_counts.items() if size > 1]
        if filtered_sizes:
            sizes, counts = zip(*filtered_sizes)
            ax3.bar(range(len(sizes)), counts)
            ax3.set_xticks(range(len(sizes)))
            ax3.set_xticklabels([f'{s}' for s in sizes], rotation=45)
            ax3.set_xlabel('Component Size', fontweight='bold')
            ax3.set_ylabel('Number of Components', fontweight='bold')
            ax3.set_title('Connected Component Size Distribution', fontweight='bold')
            ax3.set_yscale('log')
        
        # 4. In-Degree vs Out-Degree Scatter
        ax4 = axes[1, 1]
        
        in_degrees = [self.G.in_degree(n) for n in self.G.nodes()]
        out_degrees = [self.G.out_degree(n) for n in self.G.nodes()]
        
        # Sample for visualization if too many points
        if len(in_degrees) > 10000:
            sample_indices = random.sample(range(len(in_degrees)), 10000)
            in_degrees = [in_degrees[i] for i in sample_indices]
            out_degrees = [out_degrees[i] for i in sample_indices]
        
        ax4.scatter(in_degrees, out_degrees, alpha=0.5, s=1)
        ax4.set_xlabel('In-Degree', fontweight='bold')
        ax4.set_ylabel('Out-Degree', fontweight='bold')
        ax4.set_title('In-Degree vs Out-Degree', fontweight='bold')
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        
        # Add diagonal line
        max_degree = max(max(in_degrees), max(out_degrees))
        ax4.plot([1, max_degree], [1, max_degree], 'r--', alpha=0.5, label='Equal Degrees')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'network_structure_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_interactive_visualizations(self):
        """Create interactive visualizations using Plotly."""
        logger.info("[GLOBE] Creating interactive visualizations...")
        
        # 1. Interactive Tier Distribution
        tier_counts = pd.Series(self.network_stats['tier_distribution'])
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f'Tier {i}' if i > 0 else 'DoD' for i in tier_counts.index],
            y=tier_counts.values,
            text=[f'{v:,}' for v in tier_counts.values],
            textposition='auto',
            marker_color=px.colors.qualitative.Set3
        ))
        
        fig.update_layout(
            title='DoD Supply Chain Tier Distribution',
            xaxis_title='Supply Chain Tier',
            yaxis_title='Number of Entities',
            showlegend=False,
            height=600
        )
        
        fig.write_html(self.output_dir / 'interactive_tier_distribution.html')
        
        # 2. Interactive Sector Analysis
        if self.sector_analysis:
            self._create_interactive_sector_dashboard()
        
        # 3. Interactive Network Sample
        self._create_interactive_network_sample()
        
        logger.info("[OK] Interactive visualizations created")
    
    def _create_interactive_sector_dashboard(self):
        """Create interactive sector analysis dashboard."""
        # Create subplot figure
        n_systems = len(self.sector_analysis)
        fig = make_subplots(
            rows=(n_systems + 1) // 2, cols=2,
            subplot_titles=[analysis['name'] for analysis in self.sector_analysis.values()],
            specs=[[{"type": "bar"}] * 2] * ((n_systems + 1) // 2)
        )
        
        for idx, (code_col, analysis) in enumerate(self.sector_analysis.items()):
            row = (idx // 2) + 1
            col = (idx % 2) + 1
            
            top_codes = pd.Series(analysis['top_codes']).head(10)
            
            fig.add_trace(
                go.Bar(
                    x=top_codes.values,
                    y=[str(code)[:20] for code in top_codes.index],
                    orientation='h',
                    name=analysis['name'],
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title='DoD Supply Chain Sector Analysis Dashboard',
            height=300 * ((n_systems + 1) // 2),
            showlegend=False
        )
        
        fig.write_html(self.output_dir / 'interactive_sector_dashboard.html')
    
    def _create_interactive_network_sample(self):
        """Create interactive network visualization of a sample."""
        # Sample nodes for visualization
        sample_size = min(1000, self.G.number_of_nodes())
        sample_nodes = random.sample(list(self.G.nodes()), sample_size)
        sample_G = self.G.subgraph(sample_nodes)
        
        # Get positions
        try:
            pos = nx.spring_layout(sample_G, k=2, iterations=50)
        except:
            pos = nx.random_layout(sample_G)
        
        # Create edge trace
        edge_x, edge_y = [], []
        for edge in sample_G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node trace
        node_x, node_y, node_text, node_color = [], [], [], []
        tier_lookup = dict(zip(self.tiers_df['factset_entity_id'], self.tiers_df['tier']))
        
        for node in sample_G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            tier = tier_lookup.get(node, -1)
            node_color.append(tier)
            
            # Add node information
            info = f'Entity: {node}<br>Tier: {tier}<br>'
            info += f'In-Degree: {sample_G.in_degree(node)}<br>'
            info += f'Out-Degree: {sample_G.out_degree(node)}'
            node_text.append(info)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                size=8,
                color=node_color,
                colorbar=dict(
                    thickness=15,
                    title='Tier',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2
            )
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=f'DoD Supply Chain Network Sample ({sample_size:,} nodes)',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[dict(
                    text="Interactive network visualization of DoD supply chain entities",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color="grey", size=12)
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=800
            )
        )
        
        fig.write_html(self.output_dir / 'interactive_network_sample.html')
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report."""
        logger.info("[LIST] Generating comprehensive analysis report...")
        
        # Compile all analysis results
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'analysis_version': '1.0',
                'data_source': 'DoD Observed Subgraph'
            },
            'network_overview': self.network_stats,
            'tier_analysis': {
                'tier_distribution': self.network_stats['tier_distribution'],
                'samples_per_tier': {
                    tier: len(sample_df) for tier, sample_df in self.tier_samples.items()
                }
            },
            'sector_analysis': self.sector_analysis,
            'critical_nodes': {
                'total_analyzed': len(self.critical_nodes) if self.critical_nodes is not None else 0,
                'top_10_critical': (self.critical_nodes.head(10).to_dict('records') 
                                  if self.critical_nodes is not None else [])
            },
            'data_quality': {
                'node_features_coverage': (len(self.node_features_df) / len(self.tiers_df) * 100 
                                         if self.node_features_df is not None else 0),
                'contractor_matches': len(self.contractors_df) if self.contractors_df is not None else 0
            }
        }
        
        # Save JSON report with proper type conversion
        def convert_numpy_types(obj):
            """Convert numpy types to native Python types for JSON serialization."""
            if isinstance(obj, dict):
                return {str(k): convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Convert report to JSON-serializable format
        report_serializable = convert_numpy_types(report)
        
        with open(self.output_dir / 'comprehensive_analysis_report.json', 'w') as f:
            json.dump(report_serializable, f, indent=2)
        
        # Generate markdown summary
        self._generate_comprehensive_markdown_report(report)
        
        # Create tier sample files
        self._create_tier_sample_files()
        

        
        logger.info(f"[OK] Comprehensive report generated: {self.output_dir}")
        return report
    
    def _generate_comprehensive_markdown_report(self, report):
        """Generate comprehensive markdown report."""
        md_content = f"""# DoD Supply Chain Comprehensive Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Analysis Version:** {report['metadata']['analysis_version']}  
**Data Source:** {report['metadata']['data_source']}

## Executive Summary

This comprehensive analysis of the DoD observed supply chain network provides baseline insights before incorporating predicted links. The analysis covers network topology, tier structure, sector distribution, and critical node identification.

## Network Overview

### Scale and Structure
- **Total Entities:** {report['network_overview']['total_nodes']:,}
- **Total Relationships:** {report['network_overview']['total_edges']:,}
- **Network Density:** {report['network_overview']['density']:.2e}
- **Average In-Degree:** {report['network_overview']['avg_in_degree']:.2f}
- **Average Out-Degree:** {report['network_overview']['avg_out_degree']:.2f}

### Connectivity
- **Weakly Connected Components:** {report['network_overview']['weakly_connected_components']:,}
- **Strongly Connected Components:** {report['network_overview']['strongly_connected_components']:,}
- **Maximum In-Degree:** {report['network_overview']['max_in_degree']:,}
- **Maximum Out-Degree:** {report['network_overview']['max_out_degree']:,}

## Tier Analysis

### Distribution Across Supply Chain Tiers
"""
        
        for tier, count in report['tier_analysis']['tier_distribution'].items():
            tier_name = 'DoD (Central Node)' if tier == 0 else f'Tier {tier}'
            md_content += f"- **{tier_name}:** {count:,} entities\n"
        
        md_content += f"""
### Entity Samples by Tier
"""
        
        for tier, sample_count in report['tier_analysis']['samples_per_tier'].items():
            tier_name = 'DoD' if tier == 0 else f'Tier {tier}'
            md_content += f"- **{tier_name}:** {sample_count} sample entities (see `tier_{tier}_sample.csv`)\n"
        
        if report['sector_analysis']:
            md_content += f"""
## Sector and Industry Analysis

### Classification System Coverage
"""
            
            for code_col, analysis in report['sector_analysis'].items():
                md_content += f"""
#### {analysis['name']}
- **Coverage:** {analysis['coverage_percent']:.1f}% of entities
- **Unique Codes:** {analysis['unique_codes']:,}
- **Top Categories:** {', '.join([str(k) for k in list(analysis['top_codes'].keys())[:5]])}
"""
        
        md_content += f"""
## Critical Nodes Analysis

### Summary
- **Total Nodes Analyzed:** {report['critical_nodes']['total_analyzed']:,}
- **Critical Node Identification:** Based on composite centrality measures
- **Centrality Measures Used:** Betweenness, Closeness, In-Degree, Out-Degree

### Top 10 Critical Nodes
"""
        
        for i, node in enumerate(report['critical_nodes']['top_10_critical']):
            tier_name = 'DoD' if node.get('tier', -1) == 0 else f"Tier {node.get('tier', 'Unknown')}"
            md_content += f"{i+1}. **{node['factset_entity_id']}** ({tier_name}) - Criticality Score: {node['criticality_score']:.4f}\n"
        
        md_content += f"""
## Data Quality Assessment

### Feature Coverage
- **Node Features Coverage:** {report['data_quality']['node_features_coverage']:.1f}% of entities have complete feature data
- **Contractor Matches:** {report['data_quality']['contractor_matches']:,} DoD contractors successfully matched to FactSet entities

### Data Completeness
The analysis incorporates multiple data sources:
- **FactSet Supply Chain Data:** Primary relationship data
- **USAspending Contract Data:** DoD contractor identification
- **FactSet Entity Features:** Industry, sector, and geographic metadata

## Generated Outputs

### Publication-Ready Visualizations
1. **`network_overview.png`** - High-level network statistics and structure
2. **`tier_analysis.png`** - Detailed tier distribution and connectivity analysis
3. **`sector_analysis.png`** - Industry and sector distribution across tiers
4. **`critical_nodes_analysis.png`** - Critical node identification and analysis
5. **`network_structure_analysis.png`** - Network topology and structural properties

### Interactive Visualizations
1. **`interactive_tier_distribution.html`** - Interactive tier distribution chart
2. **`interactive_sector_dashboard.html`** - Interactive sector analysis dashboard
3. **`interactive_network_sample.html`** - Interactive network visualization

### Data Exports
1. **`tier_X_sample.csv`** - Sample entities for each tier with metadata
2. **`comprehensive_analysis_report.json`** - Complete analysis results in JSON format

## Key Insights

### Network Characteristics
1. **Massive Scale:** The DoD supply chain network contains {report['network_overview']['total_nodes']:,} entities across {len(report['tier_analysis']['tier_distribution'])} tiers, demonstrating the extensive reach of defense procurement.

2. **Hierarchical Structure:** Clear tier-based organization with {report['tier_analysis']['tier_distribution'].get(1, 0):,} direct contractors (Tier 1) expanding to hundreds of thousands of downstream suppliers.

3. **Complex Connectivity:** With {report['network_overview']['total_edges']:,} relationships, the network shows rich interconnections that enable comprehensive supply chain analysis.

### Strategic Implications
1. **Supply Chain Depth:** The multi-tier structure reveals deep supply chain dependencies that extend far beyond direct contractors.

2. **Critical Dependencies:** Identification of {report['critical_nodes']['total_analyzed']:,} critical nodes provides insights into potential vulnerabilities and concentration risks.

3. **Industry Diversity:** Sector analysis reveals the breadth of industries involved in DoD supply chains, from traditional defense contractors to civilian suppliers.

## Baseline Establishment

This analysis establishes a comprehensive baseline for the observed DoD supply chain network. Key baseline metrics include:

- **Network Topology:** {report['network_overview']['total_nodes']:,} nodes, {report['network_overview']['total_edges']:,} edges
- **Connectivity Patterns:** Average degree of {report['network_overview']['avg_in_degree']:.1f} (in) / {report['network_overview']['avg_out_degree']:.1f} (out)
- **Tier Distribution:** From DoD to Tier {max(report['tier_analysis']['tier_distribution'].keys())}
- **Critical Infrastructure:** {len(report['critical_nodes']['top_10_critical'])} highly critical nodes identified

## Next Steps

### Recommended Analysis Workflow
1. **Review Generated Visualizations:** Examine all static and interactive plots for insights
2. **Explore Entity Samples:** Review tier sample files for representative entities
3. **Identify Research Questions:** Use baseline to formulate specific supply chain queries
4. **Prepare for Link Prediction:** Use this baseline to evaluate predicted link integration
5. **Plan Vulnerability Assessment:** Leverage critical node analysis for risk evaluation

### Future Enhancements
1. **Temporal Analysis:** Incorporate time-series data for dynamic supply chain analysis
2. **Geographic Analysis:** Add geographic clustering and risk assessment
3. **Financial Analysis:** Integrate contract value and financial metrics
4. **Scenario Modeling:** Develop disruption impact models based on network structure

---

*This analysis provides a comprehensive foundation for understanding the DoD supply chain network structure and prepares the groundwork for advanced analytics including link prediction and vulnerability assessment.*
"""
        
        # Save markdown report
        with open(self.output_dir / 'comprehensive_analysis_report.md', 'w') as f:
            f.write(md_content)
        
        logger.info("[OK] Comprehensive markdown report generated")
    
    def _create_tier_sample_files(self):
        """Create sample entity files for each tier with metadata."""
        logger.info("[LIST] Creating tier sample files...")
        
        # Sample entities for each tier
        for tier in sorted(self.tiers_df['tier'].unique()):
            tier_entities = self.tiers_df[self.tiers_df['tier'] == tier]['factset_entity_id'].tolist()
            
            # Sample entities (up to 10 per tier)
            sample_size = min(10, len(tier_entities))
            sampled_entities = random.sample(tier_entities, sample_size)
            
            # Create sample data
            sample_data = []
            for entity_id in sampled_entities:
                entity_info = {
                    'factset_entity_id': entity_id,
                    'tier': tier,
                    'tier_name': 'DoD (Central Node)' if tier == 0 else f'Tier {tier}'
                }
                
                # Add node features if available
                if self.node_features_df is not None:
                    features = self.node_features_df[
                        self.node_features_df['factset_entity_id'] == entity_id
                    ]
                    if len(features) > 0:
                        feature_row = features.iloc[0]
                        entity_info.update({
                            'iso_country': feature_row.get('iso_country', 'Unknown'),
                            'entity_type': feature_row.get('entity_type', 'Unknown'),
                            'primary_sic_code': feature_row.get('primary_sic_code', 'Unknown'),
                            'industry_code': feature_row.get('industry_code', 'Unknown'),
                            'sector_code': feature_row.get('sector_code', 'Unknown'),
                            'rbics_l1_id': feature_row.get('rbics_l1_id', 'Unknown'),
                            'rbics_l2_id': feature_row.get('rbics_l2_id', 'Unknown'),
                            'rbics_l3_id': feature_row.get('rbics_l3_id', 'Unknown'),
                            'gr_country': feature_row.get('gr_country', 'Unknown'),
                            'gr_region': feature_row.get('gr_region', 'Unknown'),
                            'gr_continent': feature_row.get('gr_continent', 'Unknown')
                        })
                
                # Add contractor information if available
                if self.contractors_df is not None:
                    contractor_info = self.contractors_df[
                        self.contractors_df['factset_id'] == entity_id
                    ]
                    if len(contractor_info) > 0:
                        entity_info.update({
                            'contractor_name': contractor_info.iloc[0]['contractor_name'],
                            'match_type': contractor_info.iloc[0]['match_type'],
                            'match_score': contractor_info.iloc[0]['match_score'],
                            'factset_name': contractor_info.iloc[0]['factset_name']
                        })
                    else:
                        entity_info.update({
                            'contractor_name': 'Not a DoD contractor',
                            'match_type': 'N/A',
                            'match_score': 0,
                            'factset_name': 'Unknown'
                        })
                
                sample_data.append(entity_info)
            
            # Create DataFrame and save
            sample_df = pd.DataFrame(sample_data)
            output_file = self.output_dir / f'tier_{tier}_sample.csv'
            sample_df.to_csv(output_file, index=False)
            
            logger.info(f"[OK] Created {len(sample_df)} sample entities for Tier {tier}")
        
        logger.info(f"[TARGET] Tier sample files created in: {self.output_dir}")

def main():
    """Main execution function."""
    print("[START] Comprehensive DoD Supply Chain Analysis")
    print("=" * 70)
    
    try:
        # Initialize analyzer
        data_dir = Path("../data/processed/dod_analysis")
        analyzer = ComprehensiveDoDAnalyzer(data_dir)
        
        # Load all data
        analyzer.load_all_data()
        
        # Build network
        analyzer.build_network_efficiently()
        
        # Perform comprehensive analysis
        analyzer.calculate_network_statistics()
        analyzer.identify_critical_nodes()
        analyzer.sample_entities_by_tier()
        analyzer.analyze_sector_distribution()
        
        # Create visualizations
        analyzer.create_publication_visualizations()
        
        # Generate comprehensive report
        report = analyzer.generate_comprehensive_report()
        
        print("\n[OK] Comprehensive analysis completed successfully!")
        print(f"\n[TARGET] Key Results:")
        print(f"  - {report['network_overview']['total_nodes']:,} entities analyzed")
        print(f"  - {report['network_overview']['total_edges']:,} relationships mapped")
        print(f"  - {len(report['tier_analysis']['tier_distribution'])} supply chain tiers")
        print(f"  - {report['critical_nodes']['total_analyzed']:,} critical nodes identified")
        print(f"  - {len(report['sector_analysis'])} classification systems analyzed")
        
        print(f"\n[DATA] Generated Outputs:")
        print(f"  - Publication-ready visualizations: {analyzer.output_dir}")
        print(f"  - Interactive dashboards: *.html files")
        print(f"  - Entity samples: tier_*_sample.csv files")
        print(f"  - Comprehensive report: comprehensive_analysis_report.md")
        
        print(f"\n[TARGET] Next Steps:")
        print("1. Review all generated visualizations and reports")
        print("2. Examine entity samples for insights")
        print("3. Use baseline for predicted link integration")
        print("4. Proceed with vulnerability assessment")
        
    except Exception as e:
        logger.error(f"[ERR] Error in comprehensive analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main() 