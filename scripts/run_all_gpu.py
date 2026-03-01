"""
=======================================================================
  run_all_gpu.py - RTX 3090 Optimized Training Pipeline
  Runs ALL models x ALL sites x ALL targets (EC, pH) x ALL horizons
=======================================================================

Features:
  - CUDA availability check with GPU info display
  - tqdm global progress bar with ETA
  - Automatic site discovery from dataset
  - Ordered execution: GPU-heavy models first (maximize GPU utilization)
  - Per-experiment timeout protection  
  - Summary report with success/failure tracking
  - Compatible with both Linux (vast.ai) and Windows
"""

import subprocess
import sys
import os
import time
import argparse
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================
ROOT_DIR = Path(__file__).parent

EXPERIMENTS = {
    # --- Proposed Models (SpikeDLinear) ---
    'Proposed_Models/EC': {
        'models': ['SpikeDLinear'],
        'script': 'main.py',
        'arg_format': '--model {model} --horizon {horizon} --site {site}',
        'group': 'Proposed',
        'timeout': 600,
    },
    'Proposed_Models/pH': {
        'models': ['SpikeDLinear'],
        'script': 'main.py',
        'arg_format': '--model {model} --horizon {horizon} --site {site}',
        'group': 'Proposed',
        'timeout': 600,
    },
    # --- CEEMD Baselines (DLinear, NLinear) ---
    'CEEMD_Baselines/EC': {
        'models': ['dlinear', 'nlinear'],
        'script': 'main.py',
        'arg_format': '--model {model} --horizon {horizon} --site {site}',
        'group': 'CEEMD',
        'timeout': 900,
    },
    'CEEMD_Baselines/pH': {
        'models': ['dlinear', 'nlinear'],
        'script': 'main.py',
        'arg_format': '--model {model} --horizon {horizon} --site {site}',
        'group': 'CEEMD',
        'timeout': 900,
    },
    # --- Deep Baselines (LSTM, PatchTST, Transformer) ---
    'Deep_Baselines/EC': {
        'models': ['lstm', 'patchtst', 'transformer'],
        'script': 'main.py',
        'arg_format': '--model {model} --horizon {horizon} --site {site}',
        'group': 'Deep',
        'timeout': 1200,
    },
    'Deep_Baselines/pH': {
        'models': ['lstm', 'patchtst', 'transformer'],
        'script': 'main.py',
        'arg_format': '--model {model} --horizon {horizon} --site {site}',
        'group': 'Deep',
        'timeout': 1200,
    },
}

HORIZONS = [6, 12, 24, 48, 96, 168]


# ============================================================================
# GPU CHECK
# ============================================================================
def check_cuda():
    """Check CUDA availability and print GPU info."""
    try:
        import torch
        print("=" * 70)
        print("  GPU DIAGNOSTICS")
        print("=" * 70)

        if not torch.cuda.is_available():
            print("  [WARNING] CUDA is NOT available!")
            print("  Training will fall back to CPU (MUCH slower)")
            print("=" * 70)
            return False

        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            mem_total = torch.cuda.get_device_properties(i).total_mem / (1024**3)
            print(f"  GPU {i}: {name}")
            print(f"         VRAM: {mem_total:.1f} GB")

        # PyTorch optimizations for Ampere GPUs (RTX 3090)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        print(f"\n  PyTorch: {torch.__version__}")
        print(f"  CUDA:    {torch.version.cuda}")
        print(f"  cuDNN:   {torch.backends.cudnn.version()}")
        print(f"  TF32:    Enabled (Ampere acceleration)")
        print(f"  cuDNN Benchmark: Enabled")
        print("=" * 70)
        return True

    except ImportError:
        print("  [ERROR] PyTorch is not installed!")
        return False


# ============================================================================
# SITE DISCOVERY
# ============================================================================
def discover_sites():
    """Automatically discover all unique sites from the dataset."""
    import pandas as pd
    data_path = ROOT_DIR / "CEEMD_Baselines" / "data" / "USGs" 
    csv_files = list(data_path.glob("*.csv"))
    if not csv_files:
        # Fallback: try other data directories
        data_path = ROOT_DIR / "Proposed_Models" / "data" / "USGs"
        csv_files = list(data_path.glob("*.csv"))

    if not csv_files:
        print("  [WARNING] No data files found. Using default site.")
        return [1463500]

    df = pd.read_csv(csv_files[0])
    if 'site_no' in df.columns:
        sites = sorted(df['site_no'].unique().tolist())
        return sites
    return [1463500]


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================
def run_experiment(folder, model, horizon, site, timeout=600):
    """Run a single experiment via subprocess."""
    exp_dir = ROOT_DIR / folder
    script = EXPERIMENTS[folder]['script']
    arg_format = EXPERIMENTS[folder]['arg_format']
    cmd = f"python {script} {arg_format.format(model=model, horizon=horizon, site=site)}"

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=exp_dir,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if result.returncode != 0:
            return False, result.stderr[:300]
        return True, None
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)[:200]


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="RTX 3090 Optimized Training Pipeline - All Models/Sites/Targets"
    )
    parser.add_argument('--site', '-s', type=str, default='all',
                        help='Site number or "all" (default: all)')
    parser.add_argument('--target', '-t', nargs='+', default=['EC', 'pH'],
                        choices=['EC', 'pH'], help='Targets (default: EC pH)')
    parser.add_argument('--folder', '-f', nargs='+', default=None,
                        choices=['Proposed_Models', 'CEEMD_Baselines', 'Deep_Baselines'],
                        help='Folders to run (default: all)')
    parser.add_argument('--horizon', '-H', type=int, nargs='+', default=None,
                        help='Horizons (default: all)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: horizon 24 only')
    parser.add_argument('--skip-cuda-check', action='store_true',
                        help='Skip CUDA diagnostics')
    args = parser.parse_args()

    # --- Header ---
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#   RTX 3090 OPTIMIZED TRAINING PIPELINE                             #")
    print("#   All Models x All Sites x EC + pH                                 #")
    print("#" + " " * 68 + "#")
    print("#" * 70 + "\n")

    # --- CUDA Check ---
    if not args.skip_cuda_check:
        has_cuda = check_cuda()
        if not has_cuda:
            response = input("\n  Continue without GPU? (y/N): ").strip().lower()
            if response != 'y':
                print("  Aborted.")
                return
    else:
        print("  [SKIP] CUDA diagnostics skipped.\n")

    # --- Discover Sites ---
    if args.site.lower() == 'all':
        sites = discover_sites()
    else:
        sites = [int(args.site)]

    horizons = [24] if args.quick else (args.horizon or HORIZONS)
    targets = args.target
    folder_filter = args.folder

    # --- Build Experiment Queue ---
    # Order: Deep (GPU-heavy) -> Proposed -> CEEMD (CPU-bound)
    # This maximizes GPU utilization by front-loading heavy workloads
    group_order = {'Deep': 0, 'Proposed': 1, 'CEEMD': 2}

    experiments = []
    for folder, config in sorted(EXPERIMENTS.items(), key=lambda x: group_order.get(x[1]['group'], 99)):
        # Filter by target
        target = folder.split('/')[1]  # EC or pH
        if target not in targets:
            continue
        # Filter by folder
        folder_prefix = folder.split('/')[0]
        if folder_filter and folder_prefix not in folder_filter:
            continue

        for site in sites:
            for model in config['models']:
                for h in horizons:
                    experiments.append({
                        'folder': folder,
                        'model': model,
                        'horizon': h,
                        'site': site,
                        'group': config['group'],
                        'timeout': config['timeout'],
                    })

    total = len(experiments)

    print("=" * 70)
    print(f"  Sites:    {sites}")
    print(f"  Targets:  {targets}")
    print(f"  Horizons: {horizons}")
    print(f"  Folders:  {folder_filter or ['ALL']}")
    print(f"  Total Experiments: {total}")
    print("=" * 70 + "\n")

    if total == 0:
        print("  No experiments matched the filter. Exiting.")
        return

    # --- Import tqdm ---
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False
        print("  [INFO] Install tqdm for a progress bar: pip install tqdm\n")

    # --- Execute ---
    successful = 0
    failed = []
    results_log = []
    start_time = time.time()

    if has_tqdm:
        pbar = tqdm(experiments, desc="Training", unit="exp",
                    bar_format="{l_bar}{bar:30}{r_bar}",
                    ncols=120)
    else:
        pbar = experiments

    for i, exp in enumerate(pbar):
        exp_start = time.time()

        if has_tqdm:
            pbar.set_description(
                f"Site {exp['site']} | {exp['group']:8s} | {exp['model']:14s} | h={exp['horizon']:3d}"
            )

        success, error = run_experiment(
            exp['folder'], exp['model'], exp['horizon'], exp['site'], exp['timeout']
        )

        exp_time = time.time() - exp_start

        if success:
            successful += 1
            results_log.append({
                'status': 'OK',
                'folder': exp['folder'],
                'model': exp['model'],
                'horizon': exp['horizon'],
                'site': exp['site'],
                'time': f"{exp_time:.1f}s",
            })
        else:
            tag = f"Site {exp['site']} | {exp['folder']} | {exp['model']} | h={exp['horizon']}"
            failed.append(tag)
            results_log.append({
                'status': 'FAIL',
                'folder': exp['folder'],
                'model': exp['model'],
                'horizon': exp['horizon'],
                'site': exp['site'],
                'error': error,
            })
            if has_tqdm:
                tqdm.write(f"  [FAIL] {tag}: {error}")
            else:
                print(f"  [{i+1}/{total}] FAIL: {tag}: {error}")

        if not has_tqdm:
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (total - i - 1)
            print(f"  [{i+1}/{total}] {'OK' if success else 'FAIL'} | "
                  f"Site {exp['site']} | {exp['model']} h={exp['horizon']} | "
                  f"{exp_time:.1f}s | ETA: {eta/60:.1f}m")

    total_time = time.time() - start_time

    # --- Summary ---
    print("\n" + "#" * 70)
    print("#  TRAINING COMPLETE - SUMMARY")
    print("#" * 70)
    print(f"  Total Experiments: {total}")
    print(f"  Successful:        {successful}")
    print(f"  Failed:            {len(failed)}")
    print(f"  Total Time:        {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    if successful > 0:
        print(f"  Avg per Exp:       {total_time/successful:.1f}s")

    if failed:
        print(f"\n  {'='*60}")
        print(f"  FAILED EXPERIMENTS ({len(failed)}):")
        print(f"  {'='*60}")
        for f in failed:
            print(f"    X  {f}")

    print(f"\n  {'='*60}")
    print(f"  NEXT STEPS:")
    print(f"  {'='*60}")
    print(f"  1. Download 'Proposed_Models/', 'CEEMD_Baselines/', 'Deep_Baselines/'")
    print(f"     results folders to your local machine.")
    print(f"  2. Run 'run_local_dashboard.bat' to view on Streamlit.")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
