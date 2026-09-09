#!/usr/bin/env python3
"""
00_prepare_data.py
------------------
Step 0: Prepare FactSet and USAspending Data
This is the first step in the DoD supply chain pipeline.

This script:
1. Extracts USAspending zip files and processes them
2. Ensures sym_entity_table.csv is available
3. Prepares all required data files for the pipeline
"""

import zipfile
import pandas as pd
from pathlib import Path
import subprocess
import sys
import os

def check_factset_data():
    """Check if FactSet entity table is available."""
    print("[SEARCH] Checking FactSet data...")
    
    factset_file = Path("../data/raw/sym_entity_table.csv")
    
    if factset_file.exists():
        size_mb = factset_file.stat().st_size / (1024*1024)
        print(f"[OK] FactSet entity table found: {factset_file} ({size_mb:.1f} MB)")
        return True
    else:
        print("[ERR] FactSet entity table not found")
        print("   Run the data processing pipeline to extract FactSet data")
        return False

def extract_usaspending_zip_files():
    """Extract USAspending zip files."""
    print("\n[PKG] Extracting USAspending zip files...")
    
    data_dir = Path("../data/raw")
    csv_files = list(data_dir.glob("FY*.csv"))
    
    if not csv_files:
        print("[ERR] No CSV files found in data/raw directory")
        return False
    
    print(f"Found {len(csv_files)} CSV files:")
    for csv_file in csv_files:
        print(f"  - {csv_file.name}")
    
    print("[OK] CSV files are already extracted and ready for processing")
    return True
    
    return True

def process_usaspending_data():
    """Process USAspending data using the existing script."""
    print("\n[SYNC] Processing USAspending data...")
    
    # Import and run the processing function
    sys.path.append(str(Path(__file__).parent))
    import importlib.util
    spec = importlib.util.spec_from_file_location("process_usaspending", Path(__file__).parent / "01_process_usaspending_data.py")
    process_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(process_module)
    
    try:
        # Process the data
        df = process_module.process_usaspending_combined()
        if df is not None:
            # Extract contractors
            contractors = process_module.extract_dod_contractors_combined(df)
            print("[OK] USAspending data processing completed")
            return True
        else:
            print("[ERR] Failed to process USAspending data")
            return False
    except Exception as e:
        print(f"[ERR] Error processing USAspending data: {e}")
        return False

def check_required_files():
    """Check if all required files are available."""
    print("\n[SEARCH] Checking required files...")
    
    data_raw_dir = Path("../data/raw")
    data_processed_dir = Path("../data/processed")
    required_files = [
        ("sym_entity_table.csv", data_raw_dir),
        ("edges_positive_full.parquet", data_processed_dir),
        ("usaspending_dod_combined.parquet", data_processed_dir / "dod_analysis" / "usaspending"),
        ("dod_contractors_combined_summary.parquet", data_processed_dir / "dod_analysis" / "contractors")
    ]
    
    missing_files = []
    for file_name, file_dir in required_files:
        file_path = file_dir / file_name
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024*1024)
            print(f"[OK] {file_name} ({size_mb:.1f} MB)")
        else:
            print(f"[ERR] {file_name} - MISSING")
            missing_files.append(file_name)
    
    if missing_files:
        print(f"\n[WARN] Missing files: {missing_files}")
        return False
    else:
        print("\n[OK] All required files are available!")
        return True

def main():
    """Main function to prepare all data."""
    print("[START] Prepare FactSet and USAspending Data")
    print("=" * 50)
    
    # Step 1: Check FactSet data
    if not check_factset_data():
        print("\n[ERR] Please ensure FactSet data is available before proceeding")
        return
    
    # Step 2: Extract USAspending zip files
    if not extract_usaspending_zip_files():
        print("\n[ERR] Failed to extract USAspending zip files")
        return
    
    # Step 3: Process USAspending data
    if not process_usaspending_data():
        print("\n[ERR] Failed to process USAspending data")
        return
    
    # Step 4: Check all required files
    if not check_required_files():
        print("\n[ERR] Some required files are missing")
        return
    
    print("\n[DONE] Data preparation completed successfully!")
    print("Ready to run the DoD supply chain analysis pipeline.")

if __name__ == "__main__":
    main() 