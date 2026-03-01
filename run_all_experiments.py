"""
Run All Experiments
Chạy tất cả experiments cho cả EC và pH
"""

import subprocess
import sys
import time
from pathlib import Path
import argparse


ROOT_DIR = Path(__file__).parent

EXPERIMENTS = {}
TARGETS = ['EC', 'pH', 'Temp', 'Flow', 'DO', 'Turbidity']

for t in TARGETS:
    # Proposed Models (run from Proposed_Models/ target folder but using ../main.py)
    EXPERIMENTS[f'Proposed_Models/{t}'] = {
        'models': ['SpikeDLinear'],
        'script': '../main.py',
        'arg_format': f'--target {t} --horizon {{horizon}} --site {{site}}',
    }
    
    # CEEMD Baselines
    EXPERIMENTS[f'CEEMD_Baselines/{t}'] = {
        'models': ['dlinear', 'nlinear'],
        'script': 'main.py',
        'arg_format': '--model {model} --horizon {horizon} --site {site}',
    }

    # Deep Baselines
    EXPERIMENTS[f'Deep_Baselines/{t}'] = {
        'models': ['lstm', 'patchtst', 'transformer'],
        'script': 'main.py',
        'arg_format': '--model {model} --horizon {horizon} --site {site}',
    }


HORIZONS = [6, 12, 24, 48, 96, 168]


def run_experiment(folder: str, model: str, horizon: int, site: int, n_jobs: int = 1, verbose: bool = True):
    """Run a single experiment."""
    exp_dir = ROOT_DIR / folder
    script = EXPERIMENTS[folder]['script']
    arg_format = EXPERIMENTS[folder]['arg_format']

    cmd = f"python {script} {arg_format.format(model=model, horizon=horizon, site=site)}"
    
    if "CEEMD_Baselines" in folder and n_jobs > 1:
        cmd += f" --n-jobs {n_jobs}"

    if verbose:
        if 'tqdm' in sys.modules:
            from tqdm import tqdm
            tqdm.write(f"\n{'='*60}")
            tqdm.write(f"Running: {folder} | {model} | h={horizon} | Site: {site}")
            tqdm.write(f"Command: {cmd}")
            tqdm.write(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print(f"Running: {folder} | {model} | h={horizon} | Site: {site}")
            print(f"Command: {cmd}")
            print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=exp_dir,
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes timeout per experiment
        )

        if result.returncode != 0:
            print(f"ERROR: \n{result.stderr}\n")
            return False

        if verbose:
            # Print last few lines of output
            lines = result.stdout.strip().split('\n')
            for line in lines[-5:]:
                if 'tqdm' in sys.modules:
                    from tqdm import tqdm
                    tqdm.write(f"  {line}")
                else:
                    print(f"  {line}")

        return True

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {folder}/{model}/h{horizon}")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def run_all(
    targets: list = None, # Defaults to TARGETS if None
    models: list = None,
    horizons: list = None,
    folders: list = None,
    site: int = 1463500,
    n_jobs: int = 1,
    verbose: bool = True
):
    """
    Run all experiments.

    Args:
        targets: ['EC', 'pH'] or None for all
        models: ['dlinear', 'lstm', etc.] or None for all
        horizons: [24, 48] or None for all
        folders: ['Proposed_Model', 'CEEMD_Baselines', 'Deep_Baselines'] or None for all
    """
    if targets is None:
        targets = TARGETS
    if horizons is None:
        horizons = HORIZONS
    if folders is None:
        folders = ['Proposed_Models', 'CEEMD_Baselines', 'Deep_Baselines']

    experiments_to_run = []
    
    # If site is a list, we loop over sites. If it's a single int, we wrap it in a list.
    site_list = site if isinstance(site, list) else [site]

    for current_site in site_list:
        for folder, config in EXPERIMENTS.items():
            # Filter by folder
            folder_base = folder.split('/')[0]
            if folder_base not in folders:
                continue

            # Filter by target
            target = folder.split('/')[1]
            if target not in targets:
                continue

            # Get models for this folder
            exp_models = config['models']
            if models is not None:
                exp_models = [m for m in exp_models if m in models]

            for model in exp_models:
                for horizon in horizons:
                    experiments_to_run.append((folder, model, horizon, current_site))

    total_experiments = len(experiments_to_run)
    successful = 0
    failed = []

    start_time = time.time()

    print("\n" + "#" * 70)
    print("RUNNING ALL EXPERIMENTS")
    print("#" * 70)
    print(f"Targets: {targets}")
    print(f"Horizons: {horizons}")
    print(f"Folders: {folders}")
    print(f"Sites: {site_list}")
    print(f"Total Configurations: {total_experiments}")
    print("#" * 70 + "\n")

    from tqdm import tqdm
    pbar = tqdm(experiments_to_run, desc="Overall Progress", unit="exp")
    
    for folder, model, horizon, current_site in pbar:
        # Update progress bar description
        pbar.set_description(f"Site {current_site} | {folder.split('/')[1]} | {model} | h={horizon}")
        
        # We set verbose=False to keep the progress bar clean, 
        # unless debug verbosity is specifically requested if needed
        success = run_experiment(folder, model, horizon, current_site, n_jobs=n_jobs, verbose=verbose)

        if success:
            successful += 1
        else:
            failed.append(f"Site {current_site}: {folder}/{model}/h{horizon}")

    elapsed = time.time() - start_time

    print("\n" + "#" * 70)
    print("SUMMARY")
    print("#" * 70)
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(failed)}")
    print(f"Time: {elapsed/60:.1f} minutes")

    if failed:
        print("\nFailed experiments:")
        for f in failed:
            print(f"  - {f}")

    print("#" * 70)


def main():
    parser = argparse.ArgumentParser(description='Run All Experiments')

    parser.add_argument('--target', '-t', type=str, nargs='+',
                        choices=TARGETS, default=None,
                        help='Target(s) to run (default: all)')

    parser.add_argument('--model', '-m', type=str, nargs='+', default=None,
                        help='Model(s) to run (default: all)')

    parser.add_argument('--horizon', '-H', type=int, nargs='+', default=None,
                        help='Horizon(s) to run (default: all)')

    parser.add_argument('--folder', '-f', type=str, nargs='+',
                        choices=['Proposed_Models', 'CEEMD_Baselines', 'Deep_Baselines'],
                        default=None, help='Folder(s) to run (default: all)')

    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Quiet mode (less output)')

    parser.add_argument('--site', '-s', type=str, default='1463500',
                        help='USGS Site Number to train on, or "all" to train on all sites (default: 1463500)')
                        
    parser.add_argument('--n-jobs', '-j', type=int, default=1,
                        help='Number of parallel processes for CEEMD IMF training (default: 1)')

    # Quick presets
    parser.add_argument('--proposed-only', action='store_true',
                        help='Run only Proposed Models')
    parser.add_argument('--baselines-only', action='store_true',
                        help='Run only baselines (CEEMD + Deep)')
    parser.add_argument('--ec-only', action='store_true',
                        help='Run only EC target')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: only horizon 24')

    args = parser.parse_args()

    # Apply presets
    targets = args.target
    models = args.model
    horizons = args.horizon
    folders = args.folder

    if args.proposed_only:
        folders = ['Proposed_Models']
    if args.baselines_only:
        folders = ['CEEMD_Baselines', 'Deep_Baselines']
    if args.ec_only:
        targets = ['EC']
    if args.quick:
        horizons = [24]

    target_sites = []
    if args.site.lower() == 'all':
        import pandas as pd
        from pathlib import Path
        data_path = Path(__file__).parent / "CEEMD_Baselines" / "data" / "USGs" / "water_data_2021_2025_clean.csv"
        if data_path.exists():
            df = pd.read_csv(data_path)
            if 'site_no' in df.columns:
                target_sites = df['site_no'].unique().tolist()
                print(f"Found {len(target_sites)} unique sites in dataset: {target_sites}")
            else:
                print("Error: 'site_no' column not found in dataset.")
                target_sites = [1463500]
        else:
            print(f"Error: Dataset not found at {data_path}")
            target_sites = [1463500]
    else:
        target_sites = [int(args.site)]

    print(f"\n================ PREPARING EXPERIMENTS FOR {len(target_sites)} SITES ================\n")
    # Pass the entire list of target sites to run_all to handle the global progress bar
    run_all(
        targets=targets,
        models=models,
        horizons=horizons,
        folders=folders,
        site=target_sites,
        n_jobs=args.n_jobs,
        verbose=False # Set global verbosity to False to prevent tqdm glitches
    )


if __name__ == "__main__":
    main()
