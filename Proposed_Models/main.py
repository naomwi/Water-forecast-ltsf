"""
Proposed Model (SpikeDLinear) — Adapted from GitHub:robust-wq-hybrid
Logic: 100% from GitHub repo, only paths/args adapted for local runner.

Usage:
  python Proposed_Models/main.py --target EC --site 1463500 --horizon 24
  python Proposed_Models/main.py --target pH --site 1463500  # runs all horizons
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

# Add Proposed_Models to path for src imports
PROPOSED_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROPOSED_DIR))

from src.layers import SpikeDLinear
from src.loss import SpikeAwareLoss
from src.utils import load_and_preprocess_data, TimeSeriesDataset, plot_results
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# ================= CONFIG (from GitHub, unchanged) =================
DATA_FILE = str(PROPOSED_DIR / "data" / "USGs" / "water_data_2021_2025_clean.csv")
SEQ_LEN = 168
PRED_LENS = [6, 12, 24, 48, 96, 168]
BATCH_SIZE = 64
EPOCHS = 150
PATIENCE = 15
LR = 0.001
TRAIN_RATIO = 0.6

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    except:
        mape = np.nan
    return {'MAE': mae, 'MAPE': mape, 'MSE': mse, 'RMSE': rmse, 'R2': r2}


# ================= MAIN LOOP (from GitHub, paths adapted) =================
def train_and_evaluate(target_col, site, horizons=None):
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: File not found {DATA_FILE}")
        return

    if horizons is None:
        horizons = PRED_LENS

    # Results dir: Proposed_Models/{target}/results/site_{XXXX}/
    results_base = PROPOSED_DIR / target_col / "results" / f"site_{site}"
    metrics_dir = results_base / "metrics"
    series_dir = results_base / "series"
    plots_dir = results_base / "plots"
    models_dir = PROPOSED_DIR / target_col / "saved_models" / f"site_{site}"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    series_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Cache dir: per-target per-site cache
    cache_dir = str(PROPOSED_DIR / "data" / "cache" / target_col)

    # Load Data (from GitHub logic, unchanged)
    print(">> Loading and crunching data...")
    (high_imfs, low_imfs, features, raw_scaled,
     scaler_target, train_size) = load_and_preprocess_data(
        DATA_FILE, target_col, TRAIN_RATIO, site, cache_dir
    )

    val_size = int(len(raw_scaled) * 0.2)
    train_end = train_size
    val_end = train_size + val_size
    n = len(raw_scaled)

    def slice_data(start, end):
        return (high_imfs[start:end], low_imfs[start:end], features[start:end], raw_scaled[start:end])

    train_data = slice_data(0, train_end)
    val_data = slice_data(train_end, val_end)
    test_data = slice_data(val_end, n)

    summary_metrics = []

    for pred_len in horizons:
        print(f"\n{'='*30}\n>>> PROCESSING PRED_LEN = {pred_len}\n{'='*30}")

        train_set = TimeSeriesDataset(*train_data, SEQ_LEN, pred_len)
        val_set = TimeSeriesDataset(*val_data, SEQ_LEN, pred_len)
        test_set = TimeSeriesDataset(*test_data, SEQ_LEN, pred_len)

        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

        # Init Model
        model = SpikeDLinear(
            seq_len=SEQ_LEN, pred_len=pred_len,
            num_high=high_imfs.shape[1],
            num_low=low_imfs.shape[1],
            num_feat=features.shape[1]
        )

        criterion = SpikeAwareLoss(gamma=2.0, penalty_weight=5.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        best_val_loss = float('inf')
        patience_counter = 0
        best_path = str(models_dir / f"best_model_len{pred_len}.pth")

        # --- TRAIN LOOP ---
        if not os.path.exists(best_path):
            for epoch in range(EPOCHS):
                model.train()
                train_loss = 0
                for x_h, x_l, x_f, y_delta, _, _ in train_loader:
                    optimizer.zero_grad()
                    pred = model(x_h, x_l, x_f)
                    loss = criterion(pred, y_delta)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_loss += loss.item()

                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for x_h, x_l, x_f, y_delta, _, _ in val_loader:
                        pred = model(x_h, x_l, x_f)
                        val_loss += criterion(pred, y_delta).item()

                avg_train = train_loss / len(train_loader)
                avg_val = val_loss / len(val_loader)
                scheduler.step(avg_val)

                print(f"\rEp {epoch+1:03d} | Train: {avg_train:.4f} | Val: {avg_val:.4f}", end="")

                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    torch.save(model.state_dict(), best_path)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        print(f" -> Early Stopping!")
                        break

        # --- TEST & RESULTS ---
        print("\n>> Testing & Generating Results...")
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, weights_only=True))
        model.eval()

        preds_list, actuals_list = [], []

        with torch.no_grad():
            for x_h, x_l, x_f, y_delta, last_val, y_true_scaled in test_loader:
                pred_delta = model(x_h, x_l, x_f)
                rec_pred_scaled = last_val.unsqueeze(1) + pred_delta
                rec_real = scaler_target.inverse_transform(rec_pred_scaled.numpy())
                act_real = scaler_target.inverse_transform(y_true_scaled.numpy())

                preds_list.append(rec_real)
                actuals_list.append(act_real)

        preds_array = np.concatenate(preds_list, axis=0)
        actuals_array = np.concatenate(actuals_list, axis=0)

        # Metrics (Step-1)
        preds_step1 = preds_array[:, 0]
        actuals_step1 = actuals_array[:, 0]

        metrics = calculate_metrics(actuals_array, preds_array)
        metrics['Pred_Len'] = f"{pred_len} (Step-1)"
        summary_metrics.append(metrics)

        print(f">> Result (Step-1): MSE={metrics['MSE']:.4f} | R2={metrics['R2']:.4f}")

        # Save in dashboard-compatible format
        model_name = "SpikeDLinear"

        # Metrics CSV
        pd.DataFrame([{'Model': model_name, 'Horizon': pred_len, **metrics}]).to_csv(
            metrics_dir / f"{model_name}_h{pred_len}.csv", index=False
        )

        # Series CSV (step-1 only, matching CEEMD baseline format)
        pd.DataFrame({'Actual': actuals_step1, 'Predicted': preds_step1}).to_csv(
            series_dir / f"series_{model_name}_P{pred_len}_{target_col}.csv", index=False
        )

        # Plot
        plot_results(preds_array, actuals_array, pred_len, str(plots_dir / f"series_{model_name}_P{pred_len}_{target_col}.png"))

    # Final summary
    df_sum = pd.DataFrame(summary_metrics)
    df_sum = df_sum[['Pred_Len', 'MAE', 'MAPE', 'MSE', 'RMSE', 'R2']]
    df_sum.to_csv(metrics_dir / f"{model_name}_final_metrics.csv", index=False)
    print(f"\n>>> ALL DONE! Check {results_base}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Proposed Model (SpikeDLinear)")
    parser.add_argument('--target', type=str, default='EC', choices=['EC', 'pH', 'Temp', 'Flow', 'DO', 'Turbidity'])
    parser.add_argument('--site', type=int, default=1463500)
    parser.add_argument('--horizon', type=int, nargs='+', default=None, help="Specific horizons, e.g. --horizon 6 12 24")
    args = parser.parse_args()

    print('='*20)
    print(f"PREDICTING SITE {args.site}")
    print('='*20)
    train_and_evaluate(args.target, args.site, args.horizon)
