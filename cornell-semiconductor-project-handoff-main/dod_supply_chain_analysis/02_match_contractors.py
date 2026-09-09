#!/usr/bin/env python3
"""
COMPREHENSIVE DoD Contractor to FactSet Matching (V7 - Complete Solution)
Handles both exact matching and improved fuzzy matching in a single script
"""

import pandas as pd
import numpy as np
import sqlite3
import time
import logging
import traceback
from collections import Counter
from fuzzywuzzy import fuzz, process
import re
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path

# Configuration - SMART THRESHOLDS
MAX_CPUS = 32
CHUNK_SIZE = 10000
BATCH_SIZE = 500
FUZZY_THRESHOLD_HIGH = 91  # Very high confidence (updated from 90)
FUZZY_THRESHOLD_MEDIUM = 91  # High confidence (updated from 85)
FUZZY_THRESHOLD_LOW = 91  # Medium confidence (updated from 80) - now only accepting 91%+

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/dod_analysis/contractor_matching.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_system_info():
    """Get system information for optimization"""
    cpu_count = mp.cpu_count()
    memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3)
    return cpu_count, memory

def clean_company_name_smart(name):
    """Smart cleaning with better normalization"""
    if pd.isna(name) or name == '':
        return ''
    
    # Convert to string and uppercase
    name = str(name).upper().strip()
    
    # Remove common suffixes but keep some context
    suffixes = [' INC', ' CORP', ' CORPORATION', ' LLC', ' LP', ' CO', ' COMPANY', ' LTD', ' LIMITED']
    
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    
    # Remove punctuation but keep spaces
    name = re.sub(r'[^\w\s]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name

def extract_keywords(name):
    """Extract key identifying words from company name"""
    if not name:
        return []
    
    # Split into words and filter out common/generic words
    words = name.split()
    
    # Remove common words that don't help with identification
    common_words = {'THE', 'AND', 'OF', 'FOR', 'WITH', 'IN', 'ON', 'AT', 'TO', 'FROM', 'BY', 'AS', 'OR', 'BUT', 'SO', 'IF', 'THEN', 'ELSE', 'WHEN', 'WHERE', 'WHY', 'HOW', 'WHAT', 'WHO', 'WHICH', 'THAT', 'THIS', 'THESE', 'THOSE', 'A', 'AN'}
    
    keywords = [word for word in words if word not in common_words and len(word) > 2]
    
    return keywords

def calculate_similarity_score(contractor_name, factset_name):
    """Calculate multiple similarity scores and combine them intelligently"""
    
    # Basic ratio score
    ratio_score = fuzz.ratio(contractor_name.lower(), factset_name.lower())
    
    # Partial ratio for substring matching
    partial_score = fuzz.partial_ratio(contractor_name.lower(), factset_name.lower())
    
    # Token sort for word order variations
    token_sort_score = fuzz.token_sort_ratio(contractor_name.lower(), factset_name.lower())
    
    # Token set for subset matching
    token_set_score = fuzz.token_set_ratio(contractor_name.lower(), factset_name.lower())
    
    # Keyword matching
    contractor_keywords = extract_keywords(contractor_name)
    factset_keywords = extract_keywords(factset_name)
    
    if contractor_keywords and factset_keywords:
        keyword_overlap = len(set(contractor_keywords) & set(factset_keywords))
        keyword_total = len(set(contractor_keywords) | set(factset_keywords))
        keyword_score = (keyword_overlap / keyword_total) * 100 if keyword_total > 0 else 0
    else:
        keyword_score = 0
    
    # Weighted combination
    final_score = (
        ratio_score * 0.3 +
        partial_score * 0.2 +
        token_sort_score * 0.2 +
        token_set_score * 0.2 +
        keyword_score * 0.1
    )
    
    return final_score

def validate_match(contractor_name, factset_name, score):
    """Additional validation checks for matches"""
    
    # Length similarity check
    len_diff = abs(len(contractor_name) - len(factset_name))
    max_len = max(len(contractor_name), len(factset_name))
    length_similarity = ((max_len - len_diff) / max_len) * 100 if max_len > 0 else 0
    
    # First word similarity check
    contractor_words = contractor_name.split()
    factset_words = factset_name.split()
    
    if contractor_words and factset_words:
        first_word_similarity = fuzz.ratio(contractor_words[0], factset_words[0])
    else:
        first_word_similarity = 0
    
    # Common character check
    contractor_chars = set(contractor_name.lower().replace(' ', ''))
    factset_chars = set(factset_name.lower().replace(' ', ''))
    
    if contractor_chars and factset_chars:
        char_overlap = len(contractor_chars & factset_chars)
        char_total = len(contractor_chars | factset_chars)
        char_similarity = (char_overlap / char_total) * 100 if char_total > 0 else 0
    else:
        char_similarity = 0
    
    # Validation rules
    validation_score = 0
    
    # High confidence if all checks pass
    if (length_similarity >= 70 and 
        first_word_similarity >= 80 and 
        char_similarity >= 60):
        validation_score = 100
    # Medium confidence if most checks pass
    elif (length_similarity >= 60 and 
          first_word_similarity >= 70 and 
          char_similarity >= 50):
        validation_score = 80
    # Low confidence if some checks pass
    elif (length_similarity >= 50 and 
          first_word_similarity >= 60 and 
          char_similarity >= 40):
        validation_score = 60
    else:
        validation_score = 0
    
    # Combine fuzzy score with validation
    final_score = (score * 0.7) + (validation_score * 0.3)
    
    return final_score, validation_score

def exact_match_batch_parallel(batch_data):
    """Find exact matches for a batch (parallel version)"""
    try:
        db_path, contractor_batch = batch_data
        conn = sqlite3.connect(db_path, timeout=60.0)
        cursor = conn.cursor()
        
        cursor.execute('PRAGMA journal_mode=WAL')
        cursor.execute('PRAGMA synchronous=NORMAL')
        cursor.execute('PRAGMA cache_size=10000')
        cursor.execute('PRAGMA temp_store=MEMORY')
        
        matches = []
        total_contractors = len(contractor_batch)
        processed_count = 0
        
        logger.info(f"[SYNC] Processing exact match batch: {total_contractors} contractors")
        
        for idx, contractor in contractor_batch.iterrows():
            try:
                contractor_name = contractor['clean_name']
                processed_count += 1
                
                if processed_count % 100 == 0:
                    logger.info(f"   Exact batch: Processed {processed_count}/{total_contractors} contractors")
                
                # Find exact matches
                cursor.execute('''
                    SELECT factset_entity_id, entity_proper_name, entity_type 
                    FROM factset_entities 
                    WHERE clean_name = ?
                ''', (contractor_name,))
                
                exact_matches = cursor.fetchall()
                
                if exact_matches:
                    # Take the first exact match
                    factset_id, factset_name, entity_type = exact_matches[0]
                    matches.append({
                        'contractor_name': contractor['contractor_name'],
                        'factset_id': factset_id,
                        'factset_name': factset_name,
                        'entity_type': entity_type,
                        'match_score': 100,
                        'match_type': 'exact'
                    })
                    
            except Exception as e:
                logger.warning(f"[WARN] Error processing contractor {idx}: {str(e)}")
                continue
        
        conn.close()
        return matches
        
    except Exception as e:
        logger.error(f"[ERR] Error in exact match batch: {str(e)}")
        return []

def smart_fuzzy_match_batch(batch_data):
    """Smart fuzzy matching with multiple validation layers"""
    try:
        db_path, contractor_batch = batch_data
        conn = sqlite3.connect(db_path, timeout=60.0)
        cursor = conn.cursor()
        
        cursor.execute('PRAGMA journal_mode=WAL')
        cursor.execute('PRAGMA synchronous=NORMAL')
        cursor.execute('PRAGMA cache_size=10000')
        cursor.execute('PRAGMA temp_store=MEMORY')
        
        # Pre-load all FactSet entities
        cursor.execute('SELECT clean_name, factset_entity_id, entity_proper_name, entity_type FROM factset_entities')
        factset_entities = cursor.fetchall()
        
        factset_names = []
        factset_lookup = {}
        for clean_name, factset_id, entity_name, entity_type in factset_entities:
            factset_names.append(clean_name)
            if clean_name not in factset_lookup:
                factset_lookup[clean_name] = (factset_id, entity_name, entity_type)
        
        matches = []
        total_contractors = len(contractor_batch)
        
        for idx, (_, contractor) in enumerate(contractor_batch.iterrows()):
            clean_name = contractor['clean_name']
            
            if (idx + 1) % 50 == 0:
                logger.info(f"   [SYNC] Processing contractor {idx + 1}/{total_contractors} in batch")
            
            best_match = None
            best_score = 0
            best_validation_score = 0
            
            # SMART CANDIDATE SELECTION
            contractor_len = len(clean_name)
            contractor_keywords = extract_keywords(clean_name)
            
            candidates = []
            
            # Strategy 1: Length-based filtering (±30% length difference)
            for factset_name in factset_names:
                if abs(len(factset_name) - contractor_len) <= max(3, contractor_len * 0.3):
                    candidates.append(factset_name)
            
            # Strategy 2: Keyword-based filtering
            if len(candidates) > 1000:
                keyword_candidates = []
                for factset_name in candidates:
                    factset_keywords = extract_keywords(factset_name)
                    if set(contractor_keywords) & set(factset_keywords):
                        keyword_candidates.append(factset_name)
                
                if len(keyword_candidates) > 100:
                    candidates = keyword_candidates[:500]  # Limit to top keyword matches
                else:
                    candidates = candidates[:1000]  # Limit overall candidates
            
            # Strategy 3: First character matching
            if len(candidates) > 500:
                first_char_candidates = []
                first_char = clean_name[0] if clean_name else ''
                for factset_name in candidates:
                    if factset_name and factset_name[0] == first_char:
                        first_char_candidates.append(factset_name)
                
                if len(first_char_candidates) > 50:
                    candidates = first_char_candidates[:300]
                else:
                    candidates = candidates[:500]
            
            # Limit final candidates
            if len(candidates) > 300:
                candidates = candidates[:300]
            
            # SMART FUZZY MATCHING
            for candidate in candidates:
                try:
                    # Calculate similarity score
                    similarity_score = calculate_similarity_score(clean_name, candidate)
                    
                    # Apply validation
                    final_score, validation_score = validate_match(clean_name, candidate, similarity_score)
                    
                    # Only consider high-quality matches
                    if final_score >= FUZZY_THRESHOLD_LOW:
                        if final_score > best_score:
                            best_score = final_score
                            best_match = candidate
                            best_validation_score = validation_score
                            
                except Exception as e:
                    continue
            
            # QUALITY CHECKS
            if best_match:
                # Additional quality checks
                contractor_words = clean_name.split()
                factset_words = best_match.split()
                
                # Check if at least one significant word matches
                significant_match = False
                for cw in contractor_words:
                    for fw in factset_words:
                        if len(cw) > 3 and len(fw) > 3:
                            if fuzz.ratio(cw, fw) >= 85:
                                significant_match = True
                                break
                    if significant_match:
                        break
                
                # Only accept if we have significant word matches
                if significant_match and best_match in factset_lookup:
                    factset_id, factset_name, entity_type = factset_lookup[best_match]
                    
                    # Determine match type based on score (all matches are now 91%+)
                    if best_score >= FUZZY_THRESHOLD_HIGH:
                        match_type = 'fuzzy_high'
                    else:
                        # This shouldn't happen with the new thresholds, but keeping for safety
                        match_type = 'fuzzy_high'
                    
                    matches.append({
                        'contractor_name': contractor['contractor_name'],
                        'factset_id': factset_id,
                        'factset_name': factset_name,
                        'entity_type': entity_type,
                        'match_score': best_score,
                        'match_type': match_type
                    })
        
        conn.close()
        return matches
        
    except Exception as e:
        logger.error(f"[ERR] Error in smart fuzzy match batch: {str(e)}")
        return []

def save_matches_to_database(matches, db_path):
    """Save matches to database"""
    try:
        if not matches:
            return
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Insert matches using the existing schema
        for match in matches:
            cursor.execute('''
                INSERT INTO matches (contractor_name, factset_id, factset_name, entity_type, match_score, match_type)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (match['contractor_name'], match['factset_id'], match['factset_name'], 
                  match['entity_type'], match['match_score'], match['match_type']))
        
        conn.commit()
        conn.close()
        logger.info(f"[OK] Saved {len(matches)} matches to database")
        
    except Exception as e:
        logger.error(f"[ERR] Error saving matches: {str(e)}")

def main():
    """Main execution with comprehensive matching"""
    start_time = time.time()
    
    # Get system info
    cpu_count, memory = get_system_info()
    
    logger.info("[START] STARTING COMPREHENSIVE DoD CONTRACTOR MATCHING (V7 - COMPLETE SOLUTION)")
    logger.info("=" * 80)
    logger.info(f"[DATA] System: {cpu_count} CPUs, {memory:.1f}GB RAM")
    logger.info(f"[CFG] Configuration: MAX_CPUS={MAX_CPUS}, BATCH_SIZE={BATCH_SIZE}")
    logger.info(f"[TARGET] Smart thresholds: HIGH={FUZZY_THRESHOLD_HIGH}, MEDIUM={FUZZY_THRESHOLD_MEDIUM}, LOW={FUZZY_THRESHOLD_LOW}")
    
    try:
        # Setup paths - use existing database
        db_path = '../data/processed/dod_analysis/contractor_matching.db'
        
        # Check if database exists and has data
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='contractors'")
        if not cursor.fetchone():
            logger.error("[ERR] Contractors table not found in database")
            conn.close()
            return
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='factset_entities'")
        if not cursor.fetchone():
            logger.error("[ERR] FactSet entities table not found in database")
            conn.close()
            return
        
        # Get counts
        cursor.execute('SELECT COUNT(*) FROM contractors')
        total_contractors = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM factset_entities')
        total_factset = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT contractor_name) FROM matches WHERE match_type = "exact"')
        existing_exact_matches = cursor.fetchone()[0]
        
        conn.close()
        
        logger.info(f"[DATA] Found {total_contractors:,} contractors and {total_factset:,} FactSet entities")
        logger.info(f"[DATA] Existing exact matches: {existing_exact_matches:,}")
        
        # PHASE 1: EXACT MATCHING (skip if already done)
        if existing_exact_matches == 0:
            logger.info("[TARGET] PHASE 1: Finding exact matches...")
            
            # Process contractors in batches
            contractor_batches = []
            for i in range(0, total_contractors, BATCH_SIZE):
                conn = sqlite3.connect(db_path)
                batch = pd.read_sql(f'SELECT * FROM contractors LIMIT {BATCH_SIZE} OFFSET {i}', conn)
                conn.close()
                contractor_batches.append(batch)
            
            exact_matches = []
            num_processes = min(MAX_CPUS, mp.cpu_count(), len(contractor_batches))
            
            logger.info(f"[SYNC] Starting parallel exact processing with {num_processes} processes...")
            
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                future_to_batch = {
                    executor.submit(exact_match_batch_parallel, (db_path, batch)): i 
                    for i, batch in enumerate(contractor_batches)
                }
                
                for future in as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]
                    try:
                        batch_matches = future.result()
                        exact_matches.extend(batch_matches)
                        logger.info(f"   [OK] Exact batch {batch_idx + 1}/{len(contractor_batches)} completed: {len(batch_matches)} matches")
                    except Exception as e:
                        logger.error(f"   [ERR] Exact batch {batch_idx + 1} failed: {str(e)}")
            
            logger.info(f"[OK] PHASE 1 COMPLETED: {len(exact_matches)} exact matches found")
            
            # Save exact matches
            save_matches_to_database(exact_matches, db_path)
        else:
            logger.info("[OK] PHASE 1: Exact matches already exist, skipping...")
            exact_matches = []
        
        # PHASE 2: SMART FUZZY MATCHING
        logger.info("[SYNC] PHASE 2: Smart fuzzy matching with validation...")
        
        # Get unmatched contractors
        conn = sqlite3.connect(db_path)
        unmatched_contractors = pd.read_sql('''
            SELECT c.* FROM contractors c 
            WHERE c.contractor_name NOT IN (SELECT DISTINCT contractor_name FROM matches)
        ''', conn)
        conn.close()
        
        if len(unmatched_contractors) > 0:
            logger.info(f"[DATA] Processing {len(unmatched_contractors):,} unmatched contractors")
            
            fuzzy_batches = [unmatched_contractors[i:i + BATCH_SIZE] 
                           for i in range(0, len(unmatched_contractors), BATCH_SIZE)]
            
            fuzzy_matches = []
            num_processes = min(MAX_CPUS, mp.cpu_count(), len(fuzzy_batches))
            
            logger.info(f"[SYNC] Starting parallel smart fuzzy processing with {num_processes} processes...")
            
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                future_to_batch = {
                    executor.submit(smart_fuzzy_match_batch, (db_path, batch)): i 
                    for i, batch in enumerate(fuzzy_batches)
                }
                
                for future in as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]
                    try:
                        batch_matches = future.result()
                        fuzzy_matches.extend(batch_matches)
                        logger.info(f"   [OK] Smart fuzzy batch {batch_idx + 1}/{len(fuzzy_batches)} completed: {len(batch_matches)} matches")
                    except Exception as e:
                        logger.error(f"   [ERR] Smart fuzzy batch {batch_idx + 1} failed: {str(e)}")
            
            logger.info(f"[OK] PHASE 2 COMPLETED: {len(fuzzy_matches)} smart fuzzy matches found")
            
            # Save fuzzy matches
            save_matches_to_database(fuzzy_matches, db_path)
        else:
            fuzzy_matches = []
            logger.info("[OK] No contractors need fuzzy matching")
        
        # Final summary
        total_matches = len(exact_matches) + len(fuzzy_matches)
        execution_time = time.time() - start_time
        
        logger.info("")
        logger.info("COMPREHENSIVE CONTRACTOR MATCHING RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total Contractors: {total_contractors:,}")
        logger.info(f"Exact Matches: {len(exact_matches):,} ({(len(exact_matches)/total_contractors*100):.1f}%)")
        logger.info(f"Smart Fuzzy Matches: {len(fuzzy_matches):,} ({(len(fuzzy_matches)/total_contractors*100):.1f}%)")
        logger.info(f"Total Matches: {total_matches:,} ({(total_matches/total_contractors*100):.1f}%)")
        logger.info(f"Unmatched: {total_contractors - total_matches:,} ({((total_contractors - total_matches)/total_contractors*100):.1f}%)")
        logger.info(f"Execution Time: {execution_time:.1f} seconds")
        logger.info(f"Database: {db_path}")
        logger.info(f"Parallel Processing: {num_processes} processes")
        
        # Export results to parquet files for downstream scripts
        logger.info("")
        logger.info("[EXPORT] EXPORTING RESULTS TO PARQUET FILES...")
        try:
            output_dir = Path("../data/processed/dod_analysis")
            if not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
            
            # Connect to database to export results
            conn = sqlite3.connect(db_path)
            
            # Export exact matches
            exact_matches_df = pd.read_sql('''
                SELECT * FROM matches WHERE match_type = 'exact'
            ''', conn)
            
            if len(exact_matches_df) > 0:
                exact_file = output_dir / "contractor_exact_matches.parquet"
                exact_matches_df.to_parquet(exact_file, index=False, compression='snappy')
                logger.info(f"[OK] Exported {len(exact_matches_df):,} exact matches to {exact_file}")
            else:
                logger.warning("[WARN]  No exact matches found")
            
            # Export fuzzy matches
            fuzzy_matches_df = pd.read_sql('''
                SELECT * FROM matches WHERE match_type LIKE 'fuzzy_%'
            ''', conn)
            
            if len(fuzzy_matches_df) > 0:
                fuzzy_file = output_dir / "contractor_fuzzy_matches.parquet"
                fuzzy_matches_df.to_parquet(fuzzy_file, index=False, compression='snappy')
                logger.info(f"[OK] Exported {len(fuzzy_matches_df):,} fuzzy matches to {fuzzy_file}")
            else:
                logger.warning("[WARN]  No fuzzy matches found")
            
            # Export all matches combined
            all_matches_df = pd.read_sql('''
                SELECT * FROM matches
            ''', conn)
            
            if len(all_matches_df) > 0:
                all_file = output_dir / "contractor_all_matches.parquet"
                all_matches_df.to_parquet(all_file, index=False, compression='snappy')
                logger.info(f"[OK] Exported {len(all_matches_df):,} total matches to {all_file}")
            else:
                logger.warning("[WARN]  No matches found")
            
            conn.close()
            logger.info("[DONE] Parquet export completed successfully!")
            
        except Exception as e:
            logger.error(f"[ERR] Error during parquet export: {str(e)}")
            logger.info("[NOTE] Parquet export failed, but results are still available in the database")
        
        logger.info("")
        logger.info("[DONE] COMPREHENSIVE MATCHING COMPLETED!")
        
    except Exception as e:
        logger.error(f"[ERR] Error in main execution: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 