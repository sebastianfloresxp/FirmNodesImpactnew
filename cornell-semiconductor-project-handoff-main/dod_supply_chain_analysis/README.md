# DoD Supply Chain Analysis Pipeline

> ** DEPRECATED — Do not use for new work.** These scripts are preserved as historical reference only. The canonical Chapter 3 pipeline is [`src/ch3/`](../src/ch3/) — see its `README.md` for the reproducible run order and output paths.

This directory contains early exploratory scripts for building and analyzing the Department of Defense (DoD) supply chain using USAspending data and FactSet entities.

## Objective

Illuminate the DoD supply chain by:
1. Processing USAspending DoD contract data
2. Matching DoD contractors to FactSet entities
3. Building a comprehensive DoD supply chain network
4. Analyzing network structure and generating publication-ready visualizations

## Pipeline Structure

```
dod_supply_chain_analysis/
├── 00_prepare_data.py # Prepare FactSet and USAspending data
├── 01_process_usaspending_data.py # Process USAspending DoD contracts
├── 02_match_contractors.py # Match contractors to FactSet entities
├── 03_build_dod_subgraph.py # Build observed DoD supply chain subgraph
├── 04_analyze_dod_network.py # Comprehensive analysis and visualization
└── README.md # This file
```

## Quick Start

Run the entire pipeline:
```bash
cd dod_supply_chain_analysis
python 00_prepare_data.py
python 01_process_usaspending_data.py
python 02_match_contractors.py
python 03_build_dod_subgraph.py
python 04_analyze_dod_network.py
```

Or run the comprehensive analysis (requires previous steps):
```bash
python 04_analyze_dod_network.py
```

## Data Requirements

The pipeline expects the following data files in the `../data/` directory:

### Input Files:
- `FY2024*.csv` - USAspending FY2024 contract data
- `FY2025*.csv` - USAspending FY2025 contract data
- `sym_entity_table.csv` - FactSet entity table
- `edges_positive_full.parquet` - FactSet supply chain edges (in `data/processed/edges/`)

### Output Files:
- `usaspending_dod_combined.parquet` - Processed DoD contracts
- `dod_contractors_combined_summary.parquet` - Contractor summary
- `contractor_exact_matches.parquet` - Exact contractor matches
- `contractor_fuzzy_matches.parquet` - Fuzzy contractor matches
- `contractor_all_matches.parquet` - All contractor matches
- `dod_subgraph_edges.parquet` - Observed supply chain relationships
- `dod_subgraph_tiers.parquet` - Tier assignments

## Analysis Outputs

The comprehensive analysis (`04_analyze_dod_network.py`) generates:

### Publication-Ready Visualizations:
- `network_overview.png` - 4-panel network dashboard
- `tier_analysis.png` - Tier distribution and connectivity
- `sector_analysis.png` - Industry/sector distribution
- `critical_nodes_analysis.png` - Critical node analysis
- `network_structure_analysis.png` - Network topology

### Interactive Visualizations:
- `interactive_tier_distribution.html` - Interactive tier chart
- `interactive_sector_dashboard.html` - Sector analysis dashboard
- `interactive_network_sample.html` - Network visualization

### Data Exports:
- `comprehensive_analysis_report.json` - Complete analysis results
- `comprehensive_analysis_report.md` - Executive summary
- `tier_X_sample.csv` - Entity samples for each tier (0-7)

### Output Location:
All outputs are saved to: `../results/dod_analysis/`

## Key Findings

### Current Status:
- **341,441 entities** in DoD supply chain network
- **1,222,450 relationships** mapped
- **22,594 contractor matches** (exact + fuzzy)
- **8 supply chain tiers** (DoD + 7 tiers)
- **99.9% node feature coverage** with complete metadata

### Network Characteristics:
- **Tier 1:** 22,139 direct contractors
- **Tier 2:** 26,190 suppliers
- **Tier 3:** 197,125 suppliers (massive expansion)
- **Tier 4:** 90,399 suppliers
- **Tier 5:** 4,961 suppliers
- **Tier 6:** 586 suppliers
- **Tier 7:** 40 suppliers

### Data Quality:
- **Complete metadata** for all sampled entities
- **Industry classifications** (SIC, FactSet, RBICS)
- **Geographic data** (country, region, continent)
- **Contractor information** with match confidence scores

## Next Steps

1. **Review Generated Visualizations**
 - Examine all static and interactive plots
 - Review tier sample files for representative entities
 - Use baseline for predicted link integration

2. **Vulnerability Assessment**
 - Leverage critical node analysis for risk evaluation
 - Identify supply chain concentration risks
 - Map strategic dependencies

3. **Link Prediction Integration**
 - Use baseline for comparative analysis
 - Integrate predicted links with observed network
 - Generate comprehensive supply chain map

## Performance Metrics

- **Processing Speed**: ~8.1M contracts processed
- **Memory Usage**: Optimized for large datasets
- **Match Quality**: 59% exact match rate
- **Coverage**: $1.96T in matched obligations

## Technical Details

### Dependencies:
- pandas, numpy, networkx, matplotlib, seaborn, plotly
- fuzzywuzzy (for fuzzy matching)
- pathlib, json, datetime

### Data Processing:
- Chunked processing for large CSV files
- Parallel processing for contractor matching
- Efficient graph operations with sampling
- Publication-ready visualization generation

## Script Descriptions

### `00_prepare_data.py`
- Sets up database and processes FactSet entities
- Creates contractor matching database structure
- Prepares data for downstream analysis

### `01_process_usaspending_data.py`
- Processes DoD contract data from USAspending
- Extracts contractor information and obligations
- Creates clean contractor dataset

### `02_match_contractors.py`
- Performs exact and fuzzy matching of contractors
- Uses smart fuzzy matching with validation layers
- Generates confidence scores for matches

### `03_build_dod_subgraph.py`
- Builds Tier-n supply chain until connections dissipate
- Adds DoD central node for visualization
- Creates comprehensive supply chain network

### `04_analyze_dod_network.py`
- Comprehensive network analysis and visualization
- Publication-ready static and interactive plots
- Critical node identification and sector analysis 