"""
CEEMD Baselines - Direct In-Process Runner
Optimized for Intel Core i5-14600KF (14 Cores / 20 Threads)

This script eliminates ALL subprocess overhead by:
1. Importing PyTorch ONCE (no repeated subprocess.run() calls)
2. Caching IMFs and data in RAM across experiments
3. Running sequentially but letting PyTorch use ALL CPU threads internally
   for matrix operations (OpenMP/MKL parallelism)

On Windows, ProcessPoolExecutor uses 'spawn' which re-imports PyTorch
in every worker (~5s each). For lightweight models like DLinear/NLinear,
this overhead EXCEEDS the actual training time. Sequential + full threads
is the optimal strategy.
"""

import sys
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch

ROOT_DIR = Path(__file__).parent

# ============================================================================
# Import from CEEMD_Baselines modules directly
# ============================================================================
sys.path.insert(0, str(ROOT_DIR / "CEEMD_Baselines" / "EC"))
from config import (
    DATA_DIR, DATA_CONFIG as EC_DATA_CONFIG,
    TRAIN_CONFIG, DECOMPOSITION_CONFIG, HORIZONS,
    CACHE_DIR as EC_CACHE_DIR, RESULTS_DIR as EC_RESULTS_DIR,
)
from models import DLinear, NLinear
from utils import get_or_create_imfs, create_dataloaders, calculate_all_metrics, print_metrics, plot_prediction
from utils.data_loader import load_raw_data

# pH paths
PH_RESULTS_DIR = ROOT_DIR / "CEEMD_Baselines" / "pH" / "results"
PH_CACHE_DIR = ROOT_DIR / "CEEMD_Baselines" / "pH" / "cache"

DEVICE = torch.device('cpu')


def train_imf_fast(idx, comp, model_type, seq_len, horizon, train_config, batch_size):
    """Train a single IMF model. PyTorch uses all CPU threads internally."""
    train_ld, val_ld, test_ld, _ = create_dataloaders(
        comp, seq_len, horizon, batch_size=batch_size
    )

    model = DLinear(seq_len, horizon) if model_type == 'dlinear' else NLinear(seq_len, horizon)
    model = model.to(DEVICE)
    
    lr = train_config.get('learning_rate', 0.001)
    epochs = train_config.get('epochs', 50)
    patience = train_config.get('early_stopping_patience', 10)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    
    best_val = float('inf')
    patience_cnt = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for x, y in train_ld:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_ld:
                val_loss += criterion(model(x), y).item()
        val_loss /= max(len(val_ld), 1)

        if val_loss < best_val:
            best_val = val_loss
            patience_cnt = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for x, y in test_ld:
            preds.append(model(x).numpy())
            actuals.append(y.numpy())

    preds = np.concatenate(preds).flatten()
    actuals = np.concatenate(actuals).flatten()
    
    valid_len = (len(preds) // horizon) * horizon
    n_samples = valid_len // horizon

    return idx, preds[:valid_len].reshape(n_samples, horizon), actuals[:valid_len].reshape(n_samples, horizon)


def run_single_experiment(model_type, horizon, imfs, residue, site, target,
                          results_dir, seq_len, batch_size):
    """Run a single CEEMD experiment sequentially (fastest on Windows for small models)."""
    components = list(imfs) + [residue]

    all_preds = [None] * len(components)
    all_actuals = [None] * len(components)

    for i, comp in enumerate(components):
        idx, preds, actuals = train_imf_fast(
            i, comp, model_type, seq_len, horizon, TRAIN_CONFIG, batch_size
        )
        all_preds[idx] = preds
        all_actuals[idx] = actuals

    # Sum all components -> shape (n_samples, horizon)
    summed_preds = np.array(all_preds).sum(axis=0)
    summed_actuals = np.array(all_actuals).sum(axis=0)

    # Metrics on full 2D flatten
    metrics = calculate_all_metrics(summed_actuals.flatten(), summed_preds.flatten())

    # Series & Plot: last-step forecast ([:, -1])
    final_pred = summed_preds[:, -1]
    final_actual = summed_actuals[:, -1]

    # Save metrics
    (results_dir / "metrics").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{'Model': model_type, 'Horizon': horizon, **metrics}]).to_csv(
        results_dir / "metrics" / f"{model_type}_site{site}_h{horizon}.csv", index=False
    )

    # Save series
    (results_dir / "series").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'Actual': final_actual, 'Predicted': final_pred}).to_csv(
        results_dir / "series" / f"series_{model_type}_site{site}_P{horizon}_{target}.csv", index=False
    )

    # Save plot
    (results_dir / "plots").mkdir(parents=True, exist_ok=True)
    plot_prediction(
        actual=final_actual, predicted=final_pred,
        title=f"CEEMD-{model_type.upper()} Site{site} P{horizon} {target}",
        save_path=results_dir / "plots" / f"series_{model_type}_site{site}_P{horizon}_{target}.png"
    )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="CEEMD Direct Runner (CPU Optimized)")
    parser.add_argument('--site', '-s', type=str, default='all', help='Site number or "all"')
    parser.add_argument('--target', '-t', nargs='+', default=['EC', 'pH'], choices=['EC', 'pH'])
    parser.add_argument('--model', '-m', nargs='+', default=['dlinear', 'nlinear'], choices=['dlinear', 'nlinear'])
    parser.add_argument('--horizon', '-H', type=int, nargs='+', default=None)
    args = parser.parse_args()

    horizons = args.horizon or HORIZONS
    
    n_threads = torch.get_num_threads()
    print("=" * 70)
    print("  CEEMD DIRECT RUNNER - Zero Subprocess Overhead")
    print(f"  Device: CPU | PyTorch threads: {n_threads}")
    print(f"  Strategy: Sequential IMFs + internal OpenMP parallelism")
    print("=" * 70)

    # Resolve sites
    if args.site.lower() == 'all':
        data_file = str(DATA_DIR / EC_DATA_CONFIG['data_file'])
        df_all = pd.read_csv(data_file)
        sites = sorted(df_all['site_no'].unique().tolist())
    else:
        sites = [int(args.site)]

    # Build experiment list
    experiments = []
    for site in sites:
        for target in args.target:
            for model_type in args.model:
                for h in horizons:
                    experiments.append((site, target, model_type, h))

    total = len(experiments)
    print(f"  Sites: {sites}")
    print(f"  Models: {args.model} | Targets: {args.target}")
    print(f"  Horizons: {horizons}")
    print(f"  Total Experiments: {total}")
    print("=" * 70)

    # Cache loaded data & IMFs per (site, target)
    data_cache = {}
    imf_cache = {}

    successful = 0
    failed = []
    global_start = time.time()

    for i, (site, target, model_type, horizon) in enumerate(experiments):
        exp_start = time.time()
        
        cache_key = (site, target)
        if cache_key not in data_cache:
            data_file = str(DATA_DIR / EC_DATA_CONFIG['data_file'])
            _, data = load_raw_data(data_file, target, site_no=site)
            data_cache[cache_key] = data
            print(f"\n  [DATA] Loaded {len(data)} samples for Site {site} / {target}")

        data = data_cache[cache_key]

        if cache_key not in imf_cache:
            cache_dir = EC_CACHE_DIR if target == 'EC' else PH_CACHE_DIR
            result = get_or_create_imfs(
                data, cache_dir, prefix=target.lower(),
                n_imfs=DECOMPOSITION_CONFIG['max_imfs'],
                trials=DECOMPOSITION_CONFIG['trials'],
                noise_width=DECOMPOSITION_CONFIG['noise_width']
            )
            imf_cache[cache_key] = (result['imfs'], result['residue'])

        imfs, residue = imf_cache[cache_key]
        results_dir = EC_RESULTS_DIR if target == 'EC' else PH_RESULTS_DIR
        batch_size = EC_DATA_CONFIG['batch_size']
        seq_len = EC_DATA_CONFIG['seq_len']

        try:
            metrics = run_single_experiment(
                model_type, horizon, imfs, residue, site, target,
                results_dir, seq_len, batch_size
            )
            exp_time = time.time() - exp_start
            elapsed_total = time.time() - global_start
            remaining = (elapsed_total / (i + 1)) * (total - i - 1)
            
            rmse = metrics.get('RMSE', 0)
            r2 = metrics.get('R2', 0)
            print(f"  [{i+1:3d}/{total}] Site {site} | {target} | {model_type:8s} | h={horizon:3d} | "
                  f"RMSE={rmse:.4f} R2={r2:.4f} | {exp_time:.1f}s | ETA: {remaining/60:.1f}m")
            successful += 1
        except Exception as e:
            print(f"  [{i+1:3d}/{total}] FAILED: {e}")
            failed.append(f"Site {site}/{target}/{model_type}/h{horizon}")

    total_time = time.time() - global_start
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Total: {total} | Success: {successful} | Failed: {len(failed)}")
    print(f"  Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    if successful > 0:
        print(f"  Avg: {total_time/successful:.1f}s per experiment")
    if failed:
        print(f"\n  Failed:")
        for f in failed:
            print(f"    - {f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
