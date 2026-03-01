import sys
import pandas as pd
from pathlib import Path
import os
import argparse

# Insert root
sys.path.insert(0, str(Path(__file__).parent))
from CEEMD_Baselines.EC.utils.metrics import print_metrics
from CEEMD_Baselines.EC.utils.decomposition import get_or_create_imfs
from CEEMD_Baselines.EC.utils.data_loader import load_raw_data
from CEEMD_Baselines.EC.config import DATA_CONFIG, DATA_DIR

def main():
    parser = argparse.ArgumentParser(description='Generate CEEMDAN data cache')
    parser.add_argument('--target', '-t', default='EC', choices=['EC', 'pH'],
                        help='Target variable')
    parser.add_argument('--force', action='store_true', help='Force recompute')
    args = parser.parse_args()

    project_dir = Path(__file__).parent
    
    for t in ['EC', 'pH'] if args.target == 'all' else [args.target]:
        data_file = str(DATA_DIR / DATA_CONFIG['data_file'])
        
        print(f"\n======================================")
        print(f"Generating CEEMDAN Cache for Target {t}")
        print(f"======================================")

        try:
            df, data = load_raw_data(data_file, t)
            print(f"Loaded {len(data)} samples from {data_file}")
            
            # Use CEEMD_Baselines cache directory
            cache_dir = project_dir / "CEEMD_Baselines" / t / "cache"
            
            print(f"Starting Decomposition (saving to {cache_dir})...")
            # Force recompute if requested so that it actually runs the DLL/EMD
            result = get_or_create_imfs(data, cache_dir, prefix=t.lower(), n_imfs=12, force_recompute=args.force)
            
            imfs, residue = result['imfs'], result['residue']
            print(f"SUCCESS: Generated {len(imfs)} IMFs and 1 residue.")
            
        except Exception as e:
            print(f"FAILED: {str(e)}")

if __name__ == "__main__":
    main()
