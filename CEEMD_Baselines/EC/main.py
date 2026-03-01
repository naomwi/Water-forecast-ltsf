"""
CEEMD Baselines - Multi-Channel Architecture
CEEMDAN decomposition → all IMFs as input channels → ONE DLinear/NLinear model → predict target

Matches reference: TrumAIFPTU/Water-quality-prediction/ceemdan_EVloss/main.py
"""

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR, DATA_CONFIG, TRAIN_CONFIG, DECOMPOSITION_CONFIG, HORIZONS, RESULTS_DIR, MODEL_DIR, CACHE_DIR
from models import DLinear, NLinear
from utils import get_or_create_imfs, calculate_all_metrics, print_metrics, plot_prediction
from utils.data_loader import load_raw_data, create_multi_channel_loaders

TARGET_COL = DATA_CONFIG['target_col']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(model, train_loader, val_loader, device, epochs=100, lr=0.001, 
                learning_rate=None, patience=15, early_stopping_patience=None, **kwargs):
    """Train model with early stopping."""
    lr = learning_rate if learning_rate is not None else lr
    patience = early_stopping_patience if early_stopping_patience is not None else patience

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    best_val = float('inf')
    patience_cnt = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item()
        val_loss /= max(len(val_loader), 1)

        if val_loss < best_val:
            best_val = val_loss
            patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_val


def run_experiment(model_type, horizon, raw_data, imfs_data, site, verbose=True):
    """
    Run a single CEEMD experiment with multi-channel input.
    
    Args:
        model_type: 'dlinear' or 'nlinear'
        horizon: prediction horizon
        raw_data: raw target values (1D array)
        imfs_data: dict with 'imfs' and 'residue' from CEEMDAN
        site: site number
    """
    if verbose:
        print(f"\n{'='*50}")
        print(f"CEEMD + {model_type.upper()} (Multi-Channel) | Horizon {horizon}")
        print(f"{'='*50}")

    seq_len = DATA_CONFIG['seq_len']
    
    # Build multi-channel input: [target, IMF_0, IMF_1, ..., IMF_n, Residue]
    imfs = imfs_data['imfs']      # shape: (n_samples, n_imfs) or list of arrays
    residue = imfs_data['residue'] # shape: (n_samples,)
    
    # Ensure proper shapes
    if isinstance(imfs, list):
        imfs = np.array(imfs).T  # (n_imfs, n_samples) → (n_samples, n_imfs)
    if imfs.ndim == 1:
        imfs = imfs.reshape(-1, 1)
    if len(imfs.shape) == 2 and imfs.shape[0] < imfs.shape[1]:
        imfs = imfs.T  # Transpose if (n_imfs, n_samples)
    
    residue = residue.reshape(-1, 1)
    raw = raw_data.reshape(-1, 1)
    
    # Concatenate: [target, IMF_0, IMF_1, ..., Residue]
    data_multi = np.concatenate([raw, imfs, residue], axis=1)
    n_channels = data_multi.shape[1]
    
    if verbose:
        print(f"  Multi-channel input: {n_channels} channels "
              f"(1 target + {imfs.shape[1]} IMFs + 1 residue)")
        print(f"  Data shape: {data_multi.shape}")
    
    # Create dataloaders with StandardScaler
    train_ld, val_ld, test_ld, scaler = create_multi_channel_loaders(
        data_multi, seq_len, horizon, batch_size=DATA_CONFIG['batch_size']
    )
    
    # Create model
    if model_type == 'dlinear':
        model = DLinear(seq_len, horizon, in_channels=n_channels)
    else:
        model = NLinear(seq_len, horizon, in_channels=n_channels)
    
    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")
    
    # Train
    start = time.time()
    model, best_val = train_model(model, train_ld, val_ld, device, **TRAIN_CONFIG)
    elapsed = time.time() - start
    if verbose:
        print(f"  Training done in {elapsed:.1f}s, best_val={best_val:.6f}")
    
    # Test
    model.eval()
    preds_list, trues_list = [], []
    with torch.no_grad():
        for bx, by in test_ld:
            bx = bx.to(device)
            out = model(bx)  # (batch, pred_len, 1)
            
            # Inverse transform: only target column (column 0)
            pred_inv = test_ld.dataset.inverse_target(
                out.detach().cpu().numpy().flatten(), scaler
            )
            true_inv = test_ld.dataset.inverse_target(
                by.numpy().flatten(), scaler
            )
            
            preds_list.append(pred_inv)
            trues_list.append(true_inv)
    
    # Stack: (n_test_samples, pred_len)
    final_preds = np.array(preds_list)
    final_trues = np.array(trues_list)
    
    # Metrics on ALL predictions
    metrics = calculate_all_metrics(final_trues.flatten(), final_preds.flatten())
    if verbose:
        print_metrics(metrics, f"  CEEMD-{model_type.upper()} H{horizon} Results:")
    
    # Series & Plot: step-1 forecast ([:, 0])
    series_pred = final_preds[:, 0]
    series_actual = final_trues[:, 0]
    
    # Paths organized by site
    site_dir = RESULTS_DIR / f"site_{site}"
    metrics_dir = site_dir / "metrics"
    series_dir = site_dir / "series"
    plots_dir = site_dir / "plots"
    
    # Save metrics
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{'Model': model_type, 'Horizon': horizon, **metrics}]).to_csv(
        metrics_dir / f"{model_type}_h{horizon}.csv", index=False
    )
    
    # Save series
    series_dir.mkdir(parents=True, exist_ok=True)
    series_df = pd.DataFrame({'Actual': series_actual, 'Predicted': series_pred})
    series_path = series_dir / f"series_{model_type}_P{horizon}_{TARGET_COL}.csv"
    series_df.to_csv(series_path, index=False)
    
    # Plot comparison
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / f"series_{model_type}_P{horizon}_{TARGET_COL}.png"
    plot_prediction(
        actual=series_actual,
        predicted=series_pred,
        title=f"CEEMD-{model_type.upper()} Site{site} P{horizon} {TARGET_COL}",
        save_path=plot_path
    )
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='dlinear', choices=['dlinear', 'nlinear'])
    parser.add_argument('--horizon', type=int, nargs='+', default=None)
    parser.add_argument('--site', type=int, default=1463500)
    args = parser.parse_args()

    print(f"Device: {device}")

    # Load raw data
    _, raw_data = load_raw_data(
        str(DATA_DIR / DATA_CONFIG['data_file']),
        TARGET_COL,
        site_no=args.site
    )
    print(f"Loaded {len(raw_data)} samples for site {args.site}")

    # CEEMD decomposition (cache per site)
    imfs_data = get_or_create_imfs(
        raw_data, CACHE_DIR, prefix=f"{TARGET_COL.lower()}_{args.site}",
        n_imfs=DECOMPOSITION_CONFIG['max_imfs'],
        trials=DECOMPOSITION_CONFIG['trials'],
        noise_width=DECOMPOSITION_CONFIG.get('noise_width', 0.2)
    )

    # Run experiments
    horizons = args.horizon if args.horizon else HORIZONS
    summary_metrics = []

    for h in horizons:
        metrics = run_experiment(args.model, h, raw_data, imfs_data, args.site)
        summary_metrics.append({'Horizon': h, **metrics})

    # Summary
    if len(summary_metrics) > 1:
        df_sum = pd.DataFrame(summary_metrics)
        print(f"\n{'='*60}")
        print(f"CEEMD-{args.model.upper()} Summary for site {args.site}")
        print(f"{'='*60}")
        print(df_sum.to_string(index=False))


if __name__ == "__main__":
    main()
