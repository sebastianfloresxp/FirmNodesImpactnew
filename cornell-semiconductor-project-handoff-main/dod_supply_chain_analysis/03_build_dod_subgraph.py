#!/usr/bin/env python3
"""
Build DoD Supply Chain Subgraph (Tier-n with DoD Node)
=====================================================

Build the observed DoD supply chain subgraph with Tier-n expansion until connections dissipate.
This version adds a central "DoD" node and focuses only on known/observed connections.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import networkx as nx

def load_matched_contractors():
    """Load matched contractors and deduplicate to get unique Tier 1"""
    print("[DATA] Loading matched contractors...")
    
    data_dir = Path("../data/processed")
    
    # Load exact matches
    exact_file = data_dir / "dod_analysis" / "contractor_exact_matches.parquet"
    if exact_file.exists():
        exact_matches_df = pd.read_parquet(exact_file)
        print(f"[OK] Loaded {len(exact_matches_df):,} exact matches")
    else:
        print("[ERR] Exact matches file not found")
        return None
    
    # Try to load fuzzy matches if available
    fuzzy_file = data_dir / "dod_analysis" / "contractor_fuzzy_matches.parquet"
    if fuzzy_file.exists():
        fuzzy_matches_df = pd.read_parquet(fuzzy_file)
        print(f"[OK] Loaded {len(fuzzy_matches_df):,} fuzzy matches")
        all_matches_df = pd.concat([exact_matches_df, fuzzy_matches_df], ignore_index=True)
    else:
        print("[WARN]  Fuzzy matches not yet available, using exact matches only")
        all_matches_df = exact_matches_df
    
    # CRITICAL FIX: Deduplicate to get unique contractors
    print("\n[FIX] Correcting Tier 1 assignment...")
    print(f"  Original matches: {len(all_matches_df):,}")
    print(f"  Unique contractors: {all_matches_df['contractor_name'].nunique():,}")
    print(f"  Unique FactSet entities: {all_matches_df['factset_id'].nunique():,}")
    
    # For each contractor, keep only the best match (highest match score)
    unique_contractors_df = all_matches_df.sort_values('match_score', ascending=False).drop_duplicates(
        subset=['contractor_name'], 
        keep='first'
    )
    
    print(f"  After deduplication: {len(unique_contractors_df):,} unique contractor matches")
    print(f"  Unique FactSet entities after dedup: {unique_contractors_df['factset_id'].nunique():,}")
    
    return unique_contractors_df

def load_factset_supply_chain():
    """Load FactSet supply chain relationships"""
    print("\n[DATA] Loading FactSet supply chain data...")
    
    data_dir = Path("../data/processed/edges")
    
    # Load supply chain edges
    edges_file = data_dir / "edges_positive_full.parquet"
    if edges_file.exists():
        edges_df = pd.read_parquet(edges_file)
        print(f"[OK] Loaded {len(edges_df):,} supply chain relationships")
    else:
        print("[ERR] FactSet supply chain edges file not found")
        return None
    
    return edges_df

def build_dod_subgraph_tier_n(matched_contractors_df, factset_edges_df, max_iterations=20, min_new_entities=10):
    """Build the DoD supply chain subgraph with Tier-n expansion until connections dissipate"""
    print(f"\n[BUILD]  Building DoD supply chain subgraph (Tier-n until dissipation)...")
    
    # Get Tier 1 contractor IDs (unique USAspending contractors)
    tier1_contractor_ids = set(matched_contractors_df['factset_id'].unique())
    print(f"[TARGET] Tier 1 contractors (unique USAspending): {len(tier1_contractor_ids):,}")
    
    # Initialize tier assignments
    tier_assignments = {}
    tier_counts = {}
    
    # Tier 1: Direct DoD contractors (unique USAspending contractors)
    for entity_id in tier1_contractor_ids:
        tier_assignments[entity_id] = 1
    tier_counts[1] = len(tier1_contractor_ids)
    
    print(f"  Tier 1 assigned: {tier_counts[1]:,} entities")
    
    # Build supply chain depth by iteration until connections dissipate
    current_tier_entities = tier1_contractor_ids
    all_connected_entities = set(tier1_contractor_ids)
    
    for depth in range(2, max_iterations + 1):
        print(f"\n[SEARCH] Building Tier {depth}...")
        
        # Find suppliers to current tier entities
        suppliers_to_current = factset_edges_df[
            factset_edges_df['customer_factset_entity_id'].isin(current_tier_entities)
        ]
        
        # Find customers of current tier entities
        customers_of_current = factset_edges_df[
            factset_edges_df['supplier_factset_entity_id'].isin(current_tier_entities)
        ]
        
        # Get new entities at this depth
        new_suppliers = set(suppliers_to_current['supplier_factset_entity_id'].unique())
        new_customers = set(customers_of_current['customer_factset_entity_id'].unique())
        
        # Only assign entities that haven't been assigned to earlier tiers
        new_entities = (new_suppliers | new_customers) - all_connected_entities
        
        if len(new_entities) == 0:
            print(f"  No new entities found at depth {depth}, stopping")
            break
        
        # Check if connections are dissipating (fewer than min_new_entities)
        if len(new_entities) < min_new_entities:
            print(f"  Only {len(new_entities)} new entities at depth {depth} (below threshold of {min_new_entities}), stopping")
            break
        
        # Assign tier to new entities
        for entity_id in new_entities:
            tier_assignments[entity_id] = depth
        
        tier_counts[depth] = len(new_entities)
        all_connected_entities.update(new_entities)
        
        print(f"  Tier {depth}: {len(new_entities):,} new entities")
        print(f"  Total entities so far: {len(all_connected_entities):,}")
        
        # Update current tier entities for next iteration
        current_tier_entities = new_entities
    
    # Get all edges within the subgraph
    subgraph_edges = factset_edges_df[
        (factset_edges_df['supplier_factset_entity_id'].isin(all_connected_entities)) &
        (factset_edges_df['customer_factset_entity_id'].isin(all_connected_entities))
    ]
    
    print(f"\n[DATA] Final Tier Distribution:")
    for tier in sorted(tier_counts.keys()):
        print(f"  Tier {tier}: {tier_counts[tier]:,} entities")
    
    print(f"\n[CHART] Growth Analysis:")
    for tier in sorted(tier_counts.keys())[1:]:
        prev_tier = tier - 1
        if prev_tier in tier_counts:
            growth_rate = tier_counts[tier] / tier_counts[prev_tier]
            print(f"  Tier {prev_tier} → Tier {tier}: {growth_rate:.1f}x growth")
    
    return {
        'tier1_contractors': tier1_contractor_ids,
        'all_connected_entities': all_connected_entities,
        'subgraph_edges': subgraph_edges,
        'tier_assignments': tier_assignments,
        'tier_counts': tier_counts
    }

def add_dod_node_to_subgraph(subgraph_data, matched_contractors_df):
    """Add a central 'DoD' node and connect it to all Tier 1 contractors"""
    print("\n[TARGET] Adding DoD central node...")
    
    # Create DoD node ID (use a special identifier)
    dod_node_id = "DOD_CENTRAL"
    
    # Create edges from DoD to all Tier 1 contractors
    dod_edges = []
    for contractor_id in subgraph_data['tier1_contractors']:
        dod_edges.append({
            'supplier_factset_entity_id': dod_node_id,
            'customer_factset_entity_id': contractor_id,
            'relationship_type': 'DOD_CONTRACT',
            'source': 'DOD_CENTRAL'
        })
    
    # Convert to DataFrame
    dod_edges_df = pd.DataFrame(dod_edges)
    
    # Combine with existing edges
    all_edges = pd.concat([subgraph_data['subgraph_edges'], dod_edges_df], ignore_index=True)
    
    # Add DoD node to tier assignments (Tier 0)
    tier_assignments_with_dod = subgraph_data['tier_assignments'].copy()
    tier_assignments_with_dod[dod_node_id] = 0
    
    # Update tier counts
    tier_counts_with_dod = subgraph_data['tier_counts'].copy()
    tier_counts_with_dod[0] = 1  # DoD node
    
    print(f"  [OK] Added DoD central node with {len(dod_edges):,} connections to Tier 1 contractors")
    
    return {
        'tier1_contractors': subgraph_data['tier1_contractors'],
        'all_connected_entities': subgraph_data['all_connected_entities'] | {dod_node_id},
        'subgraph_edges': all_edges,
        'tier_assignments': tier_assignments_with_dod,
        'tier_counts': tier_counts_with_dod,
        'dod_node_id': dod_node_id
    }

def analyze_supply_chain_depth(subgraph_data, factset_edges_df):
    """Analyze the supply chain depth and connectivity"""
    print("\n[DATA] Analyzing supply chain depth...")
    
    # Create NetworkX graph for analysis
    G = nx.DiGraph()
    
    # Add edges
    for _, row in subgraph_data['subgraph_edges'].iterrows():
        G.add_edge(row['supplier_factset_entity_id'], row['customer_factset_entity_id'])
    
    print(f"[CHART] Graph Statistics:")
    print(f"  Nodes: {G.number_of_nodes():,}")
    print(f"  Edges: {G.number_of_edges():,}")
    
    # Analyze paths from DoD to other tiers
    dod_node_id = subgraph_data.get('dod_node_id', 'DOD_CENTRAL')
    
    if dod_node_id in G.nodes():
        # Find longest paths from DoD
        max_path_lengths = []
        try:
            # Find all reachable nodes from DoD
            reachable = nx.descendants(G, dod_node_id)
            
            # Find longest path to each reachable node
            for target in list(reachable)[:1000]:  # Sample for performance
                try:
                    path_length = nx.shortest_path_length(G, dod_node_id, target)
                    max_path_lengths.append(path_length)
                except nx.NetworkXNoPath:
                    continue
        except:
            pass
        
        if max_path_lengths:
            print(f"  Average path length from DoD: {np.mean(max_path_lengths):.1f}")
            print(f"  Maximum path length from DoD: {max(max_path_lengths)}")
            print(f"  Path length distribution: {np.percentile(max_path_lengths, [25, 50, 75])}")
    
    # Analyze connectivity between tiers
    tier_connectivity = {}
    for tier in sorted(subgraph_data['tier_counts'].keys()):
        tier_entities = [e for e, t in subgraph_data['tier_assignments'].items() if t == tier]
        
        # Count relationships within this tier
        within_tier_edges = subgraph_data['subgraph_edges'][
            (subgraph_data['subgraph_edges']['supplier_factset_entity_id'].isin(tier_entities)) &
            (subgraph_data['subgraph_edges']['customer_factset_entity_id'].isin(tier_entities))
        ]
        
        # Count relationships to other tiers
        to_other_tiers = subgraph_data['subgraph_edges'][
            (subgraph_data['subgraph_edges']['supplier_factset_entity_id'].isin(tier_entities)) &
            (~subgraph_data['subgraph_edges']['customer_factset_entity_id'].isin(tier_entities))
        ]
        
        tier_connectivity[tier] = {
            'entities': len(tier_entities),
            'within_tier_edges': len(within_tier_edges),
            'to_other_tiers_edges': len(to_other_tiers)
        }
    
    print(f"\n[LINK] Tier Connectivity Analysis:")
    for tier, stats in tier_connectivity.items():
        tier_name = "DoD" if tier == 0 else f"Tier {tier}"
        print(f"  {tier_name}: {stats['entities']:,} entities")
        print(f"    Within-tier relationships: {stats['within_tier_edges']:,}")
        print(f"    To other tiers: {stats['to_other_tiers_edges']:,}")
    
    return G, tier_connectivity

def analyze_subgraph_characteristics(subgraph_data, matched_contractors_df):
    """Analyze the characteristics of the DoD subgraph"""
    print("\n[DATA] Analyzing subgraph characteristics...")
    
    # Basic statistics
    total_entities = len(subgraph_data['all_connected_entities'])
    total_edges = len(subgraph_data['subgraph_edges'])
    tier1_count = len(subgraph_data['tier1_contractors'])
    
    print(f"[CHART] Subgraph Statistics:")
    print(f"  Total entities: {total_entities:,}")
    print(f"  Total relationships: {total_edges:,}")
    print(f"  Tier 1 contractors: {tier1_count:,}")
    print(f"  Average relationships per entity: {total_edges/total_entities:.1f}")
    
    # Top Tier 1 contractors by match score
    top_contractors = matched_contractors_df.sort_values('match_score', ascending=False).head(10)
    
    print(f"\n[BEST] Top 10 Tier 1 Contractors by Match Score:")
    for i, row in top_contractors.iterrows():
        name = row['contractor_name']
        match_score = row['match_score']
        match_type = row['match_type']
        factset_name = row['factset_name']
        print(f"  {i+1:2d}. {name} -> {factset_name}")
        print(f"      Score: {match_score:.1f} ({match_type})")
    
    return {
        'total_entities': total_entities,
        'total_edges': total_edges,
        'tier1_count': tier1_count
    }

def save_subgraph_data(subgraph_data, analysis_results, tier_connectivity):
    """Save the DoD subgraph data"""
    print("\n[SAVE] Saving subgraph data...")
    
    data_dir = Path("../data/processed")
    
    # Create output directory if it doesn't exist
    output_dir = data_dir / "dod_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save subgraph edges
    edges_file = output_dir / "dod_subgraph_edges.parquet"
    subgraph_data['subgraph_edges'].to_parquet(edges_file, index=False, compression='snappy')
    print(f"[OK] Saved {len(subgraph_data['subgraph_edges']):,} subgraph edges to {edges_file}")
    
    # Save tier assignments
    tier_assignments_df = pd.DataFrame([
        {'factset_entity_id': entity_id, 'tier': tier}
        for entity_id, tier in subgraph_data['tier_assignments'].items()
    ])
    tier_file = output_dir / "dod_subgraph_tiers.parquet"
    tier_assignments_df.to_parquet(tier_file, index=False, compression='snappy')
    print(f"[OK] Saved {len(tier_assignments_df):,} tier assignments to {tier_file}")
    
    # Save analysis results
    analysis_file = output_dir / "dod_subgraph_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump({
            **analysis_results,
            'tier_connectivity': tier_connectivity,
            'tier_counts': subgraph_data['tier_counts'],
            'dod_node_id': subgraph_data.get('dod_node_id', 'DOD_CENTRAL')
        }, f, indent=2)
    print(f"[OK] Saved analysis to {analysis_file}")
    
    # Create summary file
    summary_file = output_dir / "dod_subgraph_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("DoD Supply Chain Subgraph Summary (Tier-n with DoD Node)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total entities: {analysis_results['total_entities']:,}\n")
        f.write(f"Total relationships: {analysis_results['total_edges']:,}\n")
        f.write(f"Tier 1 contractors: {analysis_results['tier1_count']:,}\n")
        f.write(f"DoD node ID: {subgraph_data.get('dod_node_id', 'DOD_CENTRAL')}\n\n")
        f.write("Tier distribution:\n")
        for tier, count in subgraph_data['tier_counts'].items():
            tier_name = "DoD" if tier == 0 else f"Tier {tier}"
            f.write(f"  {tier_name}: {count:,} entities\n")
        f.write(f"\nTier connectivity:\n")
        for tier, stats in tier_connectivity.items():
            tier_name = "DoD" if tier == 0 else f"Tier {tier}"
            f.write(f"  {tier_name}: {stats['entities']:,} entities, {stats['within_tier_edges']:,} within, {stats['to_other_tiers_edges']:,} to others\n")
    
    print(f"[OK] Saved summary to {summary_file}")

def main():
    print("[START] Building DoD Supply Chain Subgraph (Tier-n with DoD Node)")
    print("=" * 70)
    
    # Load matched contractors
    matched_contractors_df = load_matched_contractors()
    if matched_contractors_df is None:
        print("[ERR] Failed to load matched contractors")
        return
    
    # Load FactSet supply chain data
    factset_edges_df = load_factset_supply_chain()
    if factset_edges_df is None:
        print("[ERR] Failed to load FactSet supply chain data")
        return
    
    # Build DoD subgraph with Tier-n expansion
    subgraph_data = build_dod_subgraph_tier_n(matched_contractors_df, factset_edges_df, max_iterations=20, min_new_entities=10)
    
    # Add DoD central node
    subgraph_data = add_dod_node_to_subgraph(subgraph_data, matched_contractors_df)
    
    # Analyze supply chain depth
    graph, tier_connectivity = analyze_supply_chain_depth(subgraph_data, factset_edges_df)
    
    # Analyze subgraph characteristics
    analysis_results = analyze_subgraph_characteristics(subgraph_data, matched_contractors_df)
    
    # Save subgraph data
    save_subgraph_data(subgraph_data, analysis_results, tier_connectivity)
    
    print("\n[OK] DoD subgraph built successfully!")
    print(f"\n[TARGET] Key Results:")
    print(f"  - {analysis_results['total_entities']:,} entities in DoD supply chain")
    print(f"  - {analysis_results['total_edges']:,} observed relationships")
    print(f"  - {analysis_results['tier1_count']:,} Tier 1 contractors (unique USAspending)")
    print(f"  - {len(subgraph_data['tier_counts'])} tiers (including DoD)")
    print(f"  - DoD central node: {subgraph_data.get('dod_node_id', 'DOD_CENTRAL')}")
    print(f"\nNext steps:")
    print("1. Visualize the DoD supply chain network")
    print("2. Analyze supply chain characteristics and vulnerabilities")
    print("3. Run ML link prediction on missing connections")

if __name__ == "__main__":
    main() 