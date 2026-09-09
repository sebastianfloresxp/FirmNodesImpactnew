#!/usr/bin/env python3
"""
Process USAspending DoD Data
============================

Process comprehensive USAspending DoD contract data for supply chain analysis.
This is the first step in the DoD supply chain pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import gc
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

def process_usaspending_combined():
    """Process combined USAspending DoD contract data (2024-2025)"""
    print("[DATA] Processing Combined USAspending DoD Data (2024-2025)...")
    
    data_dir = Path("../data/raw")
    
    # Find all CSV files for 2024 and 2025
    csv_files_2024 = list(data_dir.glob("FY2024*.csv"))
    csv_files_2025 = list(data_dir.glob("FY2025*.csv"))
    all_csv_files = csv_files_2024 + csv_files_2025
    
    print(f"[DIR] Found {len(all_csv_files):,} CSV files to process")
    print(f"  - 2024 files: {len(csv_files_2024):,}")
    print(f"  - 2025 files: {len(csv_files_2025):,}")
    
    if not all_csv_files:
        print("[ERR] No CSV files found in data directory")
        return None
    
    # Process each file in chunks with larger chunk size for better performance
    all_data = []
    total_records = 0
    
    for i, csv_file in enumerate(all_csv_files, 1):
        print(f"\n[FILE] Processing file {i}/{len(all_csv_files)}: {csv_file.name}")
        
        # Read CSV in larger chunks for better performance
        chunk_size = 500000  # Increased from 100,000
        file_records = 0
        
        for chunk_num, chunk in enumerate(pd.read_csv(csv_file, chunksize=chunk_size, low_memory=False)):
            # Basic cleaning
            chunk = chunk.copy()
            
            # Convert date columns
            date_columns = ['action_date', 'period_of_performance_start_date', 'period_of_performance_end_date']
            for col in date_columns:
                if col in chunk.columns:
                    chunk[col] = pd.to_datetime(chunk[col], errors='coerce')
            
            # Convert numeric columns
            numeric_columns = ['federal_action_obligation', 'base_and_all_options_value', 'base_and_exercised_options_value']
            for col in numeric_columns:
                if col in chunk.columns:
                    chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
            
            # Clean recipient name
            if 'recipient_name' in chunk.columns:
                chunk['recipient_name'] = chunk['recipient_name'].astype(str).str.strip()
                chunk['recipient_name'] = chunk['recipient_name'].replace('', np.nan)
            
            # Clean DUNS number
            if 'recipient_duns' in chunk.columns:
                chunk['recipient_duns'] = chunk['recipient_duns'].astype(str).str.strip()
                chunk['recipient_duns'] = chunk['recipient_duns'].replace('', np.nan)
            
            # Handle problematic columns for Parquet conversion
            dtype_overrides = {
                'parent_award_agency_id': str,
                'awarding_agency_code': str,
                'funding_agency_code': str,
                'recipient_fax_number': str,
                'recipient_phone_number': str,
                'recipient_email': str
            }
            
            for col, dtype in dtype_overrides.items():
                if col in chunk.columns:
                    chunk[col] = chunk[col].astype(dtype)
            
            all_data.append(chunk)
            file_records += len(chunk)
            
            if chunk_num % 5 == 0:  # Reduced logging frequency
                print(f"    Processed chunk {chunk_num + 1} ({file_records:,} records so far)")
        
        total_records += file_records
        print(f"[OK] Processed {file_records:,} records from {csv_file.name}")
        
        # Clear memory more aggressively
        del chunk
        gc.collect()
    
    print(f"\n[SYNC] Combining all data...")
    df = pd.concat(all_data, ignore_index=True)
    print(f"[OK] Combined dataset: {len(df):,} records")
    
    # Remove duplicates
    initial_count = len(df)
    df = df.drop_duplicates()
    print(f"[OK] Removed {initial_count - len(df):,} duplicate records")
    
    # Basic filtering for DoD contracts
    print(f"\n[TARGET] Filtering for DoD contracts...")
    
    # Filter for DoD contracts (multiple possible agency names)
    dod_indicators = [
        'defense', 'dod', 'department of defense', 'military', 'army', 'navy', 'air force',
        'marine corps', 'coast guard', 'defense logistics', 'defense advanced'
    ]
    
    # Check various agency columns
    agency_columns = ['awarding_agency_name', 'funding_agency_name', 'awarding_toptier_agency_name', 'funding_toptier_agency_name']
    
    dod_mask = pd.Series([False] * len(df))
    for col in agency_columns:
        if col in df.columns:
            col_mask = df[col].astype(str).str.lower().str.contains('|'.join(dod_indicators), na=False)
            dod_mask = dod_mask | col_mask
    
    df_dod = df[dod_mask].copy()
    print(f"[OK] Filtered to {len(df_dod):,} DoD contracts")
    
    # Save combined data
    output_file = Path("../data/processed/dod_analysis/usaspending/usaspending_dod_combined.parquet")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_dod.to_parquet(output_file, index=False, compression='snappy')  # Added compression
    print(f"[OK] Saved combined DoD data to {output_file}")
    
    return df_dod

def extract_dod_contractors_combined(df):
    """Extract and clean DoD contractor information from combined data"""
    print("\n[TARGET] Extracting DoD contractor information...")
    
    # Group by contractor with optimized aggregation
    contractor_cols = ['recipient_name', 'recipient_duns']
    if 'recipient_duns' not in df.columns:
        contractor_cols = ['recipient_name']
    
    # Use more efficient aggregation
    contractors = df.groupby('recipient_name', observed=True).agg({
        'federal_action_obligation': ['sum', 'count'],
        'action_date': ['min', 'max'],
        'recipient_duns': 'first',
        'awarding_agency_name': lambda x: list(x.unique()),
        'naics_code': lambda x: list(x.unique()),
        'product_or_service_code': lambda x: list(x.unique())
    }).reset_index()
    
    # Flatten column names
    contractors.columns = [
        'contractor_name', 'total_obligation', 'contract_count',
        'first_contract_date', 'last_contract_date', 'duns_number',
        'awarding_agencies', 'naics_codes', 'product_codes'
    ]
    
    # Clean up data types
    contractors['total_obligation'] = pd.to_numeric(contractors['total_obligation'], errors='coerce').fillna(0)
    contractors['contract_count'] = pd.to_numeric(contractors['contract_count'], errors='coerce').fillna(0)
    
    # Calculate years active
    contractors['first_contract_date'] = pd.to_datetime(contractors['first_contract_date'], errors='coerce')
    contractors['last_contract_date'] = pd.to_datetime(contractors['last_contract_date'], errors='coerce')
    contractors['years_active'] = (
        (contractors['last_contract_date'] - contractors['first_contract_date']).dt.days / 365.25
    ).fillna(0)
    
    # Add multi-year indicator
    contractors['is_multi_year'] = contractors['years_active'] > 1
    
    # Sort by total obligation
    contractors = contractors.sort_values('total_obligation', ascending=False)
    
    print(f"[OK] Extracted {len(contractors):,} unique contractors")
    print(f"[DATA] Total obligation: ${contractors['total_obligation'].sum():,.0f}")
    print(f"[DATA] Average contracts per contractor: {contractors['contract_count'].mean():.1f}")
    
    # Save contractor summary
    output_file = Path("../data/processed/dod_analysis/contractors/dod_contractors_combined_summary.parquet")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    contractors.to_parquet(output_file, index=False, compression='snappy')
    print(f"[OK] Saved contractor summary to {output_file}")
    
    # Show top contractors
    print(f"\n[BEST] Top 20 DoD Contractors by Total Obligation:")
    for i, row in contractors.head(20).iterrows():
        name = row['contractor_name']
        obligation = row['total_obligation']
        contracts = row['contract_count']
        years = row['years_active']
        print(f"  {i+1:2d}. {name}")
        print(f"      ${obligation:,.0f} ({contracts:,} contracts, {years:.1f} years)")
    
    return contractors

def main():
    print("[START] Process Combined USAspending DoD Data (2024-2025)")
    print("=" * 60)
    
    # Process combined data
    df = process_usaspending_combined()
    
    if df is not None:
        # Extract contractors
        contractors = extract_dod_contractors_combined(df)
        
        print(f"\n[TARGET] We now have {len(df):,} DoD contracts with {len(contractors):,} unique contractors!")
        print(f"\nNext steps:")
        print("1. Use this comprehensive dataset for DoD supply chain analysis")
        print("2. Match contractors to FactSet entities")
        print("3. Build the complete DoD supply chain network")
    else:
        print("[ERR] Failed to process USAspending data")

if __name__ == "__main__":
    main() 