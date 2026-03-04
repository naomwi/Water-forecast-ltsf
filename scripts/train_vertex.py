"""
train_vertex.py - Cloud Training Script for Google Vertex AI
=============================================================
This script is designed to run INSIDE a Vertex AI Custom Container.
It performs the following pipeline steps:
  1. Download new_data.csv + hold_out_test.csv + active model from GCS
  2. Fine-tune SpikeDLinear on the new data
  3. Evaluate new model vs old model on the hold-out test set
  4. Upload new model weights + metrics.json to GCS
  5. Report comparison results back for Streamlit to consume

Usage (inside container):
  python train_vertex.py \
    --gcs_bucket hydropred-bucket \
    --data_uri gs://hydropred-bucket/temp_train/new_data.csv \
    --target EC \
    --site 1463500 \
    --pred_len 24 \
    --epochs 50 \
    --timeout_minutes 30
"""

import os
import sys
import json
import time
import signal
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime, timezone
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Timeout handler (Pipeline V6.0 - Section 2: Timeout mechanism)
# ---------------------------------------------------------------------------
class TrainingTimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TrainingTimeoutError("Training exceeded maximum allowed time.")

# ---------------------------------------------------------------------------
# Google Cloud Storage helpers
# ---------------------------------------------------------------------------
def download_from_gcs(gcs_uri: str, local_path: str):
    """Download a file from GCS to a local path."""
    from google.cloud import storage
    # Parse gs://bucket/path
    parts = gcs_uri.replace("gs://", "").split("/", 1)
    bucket_name, blob_name = parts[0], parts[1]
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)
    print(f"[GCS] Downloaded {gcs_uri} -> {local_path}")


def upload_to_gcs(local_path: str, gcs_uri: str):
    """Upload a local file to GCS."""
    from google.cloud import storage
    parts = gcs_uri.replace("gs://", "").split("/", 1)
    bucket_name, blob_name = parts[0], parts[1]
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    print(f"[GCS] Uploaded {local_path} -> {gcs_uri}")


def append_audit_log(gcs_bucket: str, message: str):
    """Append a line to the audit trail log on GCS."""
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(gcs_bucket)
    blob = bucket.blob("logs/audit_trail.jsonl")
    
    log_entry = json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "vertex_ai_training",
        "message": message
    }) + "\n"
    
    # Download existing, append, re-upload (simple approach for low-volume logs)
    try:
        existing = blob.download_as_text()
    except Exception:
        existing = ""
    
    blob.upload_from_string(existing + log_entry)


# ---------------------------------------------------------------------------
# Model & Training imports (reuse project source code)
# ---------------------------------------------------------------------------
# In the Docker container, the project source is copied to /app/
PROPOSED_DIR = Path("/app/Proposed_Models")
sys.path.insert(0, str(PROPOSED_DIR))

from src.layers import SpikeDLinear
from src.loss import SpikeAwareLoss
from src.utils import load_and_preprocess_data, TimeSeriesDataset


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 6), "MSE": round(mse, 6), "RMSE": round(rmse, 6), "R2": round(r2, 6)}


# ---------------------------------------------------------------------------
# Core Training Function
# ---------------------------------------------------------------------------
def run_training_pipeline(args):
    """
    Full training pipeline with:
    - Fine-tuning on new data
    - Early stopping
    - Model Validation Gate (compare old vs new on hold-out test set)
    - Versioned model upload to GCS
    """
    
    gcs_base = f"gs://{args.gcs_bucket}"
    work_dir = Path("/tmp/vertex_work")
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Optimize for multi-core CPU (n1-standard-16 = 16 vCPU)
    import multiprocessing
    num_cpus = multiprocessing.cpu_count()
    torch.set_num_threads(num_cpus)
    print(f"  CPU cores available: {num_cpus}, PyTorch threads: {num_cpus}")
    
    # =========================================================================
    # STEP 1: Download assets from GCS (Pipeline V6 - Step 10)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 1: Downloading assets from GCS...")
    print("=" * 60)
    
    # 1a. New training data
    local_data = str(work_dir / "new_data.csv")
    download_from_gcs(args.data_uri, local_data)
    
    # 1b. Hold-out test set (fixed, for fair comparison)
    local_test = str(work_dir / "hold_out_test.csv")
    download_from_gcs(f"{gcs_base}/data/hold_out_test.csv", local_test)
    
    # 1c. Current active model config
    local_config = str(work_dir / "active_model.json")
    download_from_gcs(f"{gcs_base}/config/active_model.json", local_config)
    
    with open(local_config) as f:
        active_config = json.load(f)
    
    current_version = active_config.get("active_version", "v0")
    current_model_gcs = active_config.get("path", "")
    next_version_num = int(current_version.replace("v", "")) + 1
    next_version = f"v{next_version_num}"
    
    print(f"  Current active model: {current_version} ({current_model_gcs})")
    print(f"  New model version: {next_version}")
    
    # 1d. Download current active model weights for comparison
    local_old_weights = str(work_dir / f"model_{current_version}.pth")
    if current_model_gcs:
        try:
            download_from_gcs(current_model_gcs, local_old_weights)
        except Exception as e:
            print(f"  [WARN] Could not download old model: {e}. Will skip comparison.")
            local_old_weights = None
    else:
        local_old_weights = None
    
    # =========================================================================
    # STEP 2: Preprocess & Build DataLoaders (Pipeline V6 - Step 11)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Preprocessing data & building DataLoaders...")
    print("=" * 60)
    
    SEQ_LEN = args.seq_len
    PRED_LEN = args.pred_len
    BATCH_SIZE = args.batch_size
    TRAIN_RATIO = 0.6
    
    cache_dir = str(work_dir / "cache" / args.target)
    
    (high_imfs, low_imfs, features, raw_scaled,
     scaler_target, train_size) = load_and_preprocess_data(
        local_data, args.target, TRAIN_RATIO, args.site, cache_dir
    )
    
    val_size = int(len(raw_scaled) * 0.2)
    train_end = train_size
    val_end = train_size + val_size
    n = len(raw_scaled)
    
    def slice_data(start, end):
        return (high_imfs[start:end], low_imfs[start:end],
                features[start:end], raw_scaled[start:end])
    
    train_data = slice_data(0, train_end)
    val_data = slice_data(train_end, val_end)
    
    train_set = TimeSeriesDataset(*train_data, SEQ_LEN, PRED_LEN)
    val_set = TimeSeriesDataset(*val_data, SEQ_LEN, PRED_LEN)
    
    num_workers = min(4, multiprocessing.cpu_count())
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    
    print(f"  Train samples: {len(train_set)}, Val samples: {len(val_set)}")
    print(f"  High IMFs shape: {high_imfs.shape}, Low IMFs shape: {low_imfs.shape}")
    print(f"  Features shape: {features.shape}")
    
    # =========================================================================
    # STEP 3: Fine-tune SpikeDLinear (Pipeline V6 - Step 11)
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"STEP 3: Fine-tuning SpikeDLinear ({args.epochs} epochs max, Early Stopping patience={args.patience})")
    print("=" * 60)
    
    model = SpikeDLinear(
        seq_len=SEQ_LEN, pred_len=PRED_LEN,
        num_high=high_imfs.shape[1],
        num_low=low_imfs.shape[1],
        num_feat=features.shape[1]
    )
    
    # Load old weights as starting point (Transfer Learning / Fine-tune)
    if local_old_weights and os.path.exists(local_old_weights):
        try:
            old_state = torch.load(local_old_weights, map_location="cpu", weights_only=True)
            model.load_state_dict(old_state, strict=False)
            print("  Loaded previous model weights for fine-tuning.")
        except Exception as e:
            print(f"  [WARN] Could not load old weights: {e}. Training from scratch.")
    
    criterion = SpikeAwareLoss(gamma=2.0, penalty_weight=5.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = str(work_dir / f"model_{next_version}.pth")
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        epoch_train_loss = 0
        for x_h, x_l, x_f, y_delta, _, _ in train_loader:
            optimizer.zero_grad()
            pred = model(x_h, x_l, x_f)
            loss = criterion(pred, y_delta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item()
        
        # --- Validate ---
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for x_h, x_l, x_f, y_delta, _, _ in val_loader:
                pred = model(x_h, x_l, x_f)
                epoch_val_loss += criterion(pred, y_delta).item()
        
        avg_train = epoch_train_loss / max(len(train_loader), 1)
        avg_val = epoch_val_loss / max(len(val_loader), 1)
        scheduler.step(avg_val)
        
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        
        print(f"  Epoch {epoch+1:03d}/{args.epochs} | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")
        
        # Early Stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  -> Early Stopping triggered at epoch {epoch+1}!")
                break
    
    print(f"  Best model saved to: {best_model_path}")
    
    # =========================================================================
    # STEP 4: Model Validation Gate (Pipeline V6 - Steps 14-15)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: MODEL VALIDATION GATE — Evaluating on Hold-out Test Set")
    print("=" * 60)
    
    # Preprocess hold-out test data
    test_cache = str(work_dir / "cache_test" / args.target)
    (test_high, test_low, test_feat, test_raw,
     test_scaler, test_train_size) = load_and_preprocess_data(
        local_test, args.target, 0.0, args.site, test_cache
    )
    
    test_set = TimeSeriesDataset(test_high, test_low, test_feat, test_raw, SEQ_LEN, PRED_LEN)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    
    def evaluate_model_on_test(model_instance, loader, scaler):
        """Run inference and return metrics + raw arrays."""
        model_instance.eval()
        preds_list, actuals_list = [], []
        with torch.no_grad():
            for x_h, x_l, x_f, y_delta, last_val, y_true_scaled in loader:
                pred_delta = model_instance(x_h, x_l, x_f)
                rec_pred_scaled = last_val.unsqueeze(1) + pred_delta
                rec_real = scaler.inverse_transform(rec_pred_scaled.numpy())
                act_real = scaler.inverse_transform(y_true_scaled.numpy())
                preds_list.append(rec_real)
                actuals_list.append(act_real)
        
        if len(preds_list) == 0:
            return {"MAE": 999, "MSE": 999, "RMSE": 999, "R2": -999}, None, None
        
        preds = np.concatenate(preds_list, axis=0)
        actuals = np.concatenate(actuals_list, axis=0)
        return calculate_metrics(actuals, preds), preds, actuals
    
    # 4a. Evaluate NEW model (V_next)
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    new_metrics, new_preds, new_actuals = evaluate_model_on_test(model, test_loader, test_scaler)
    print(f"  NEW Model ({next_version}): {new_metrics}")
    
    # 4b. Evaluate OLD model (V_current) if available
    old_metrics = None
    model_is_better = True  # Default: deploy if no old model to compare
    
    if local_old_weights and os.path.exists(local_old_weights):
        try:
            old_model = SpikeDLinear(
                seq_len=SEQ_LEN, pred_len=PRED_LEN,
                num_high=test_high.shape[1],
                num_low=test_low.shape[1],
                num_feat=test_feat.shape[1]
            )
            old_model.load_state_dict(torch.load(local_old_weights, weights_only=True))
            old_metrics, _, _ = evaluate_model_on_test(old_model, test_loader, test_scaler)
            print(f"  OLD Model ({current_version}): {old_metrics}")
            
            # Compare: lower MAE = better
            model_is_better = new_metrics["MAE"] <= old_metrics["MAE"]
            comparison = "BETTER" if model_is_better else "WORSE"
            print(f"\n  >>> VERDICT: New model is {comparison} than old model.")
        except Exception as e:
            print(f"  [WARN] Could not evaluate old model: {e}. Proceeding with new model.")
    else:
        print("  No previous model to compare. New model accepted by default.")
    
    # =========================================================================
    # STEP 5: Upload Results to GCS (Pipeline V6 - Step 16)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 5: Uploading results to GCS...")
    print("=" * 60)
    
    # 5a. Upload new model weights (always, with neutral name)
    model_gcs_path = f"{gcs_base}/models/model_{next_version}.pth"
    upload_to_gcs(best_model_path, model_gcs_path)
    
    # 5b. Create and upload metrics.json
    result_payload = {
        "version": next_version,
        "previous_version": current_version,
        "target": args.target,
        "site": args.site,
        "seq_len": SEQ_LEN,
        "pred_len": PRED_LEN,
        "epochs_trained": len(train_losses),
        "best_val_loss": best_val_loss,
        "new_model_metrics": new_metrics,
        "old_model_metrics": old_metrics,
        "model_is_better": model_is_better,
        "model_gcs_path": model_gcs_path,
        "train_loss_curve": train_losses,
        "val_loss_curve": val_losses,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    
    metrics_local = str(work_dir / "metrics.json")
    with open(metrics_local, "w") as f:
        json.dump(result_payload, f, indent=2)
    
    upload_to_gcs(metrics_local, f"{gcs_base}/results/metrics_{next_version}.json")
    # Also upload as "latest" for Streamlit to easily find
    upload_to_gcs(metrics_local, f"{gcs_base}/results/metrics_latest.json")
    
    # 5c. Save predictions vs actuals CSV for Streamlit plotting
    if new_preds is not None and new_actuals is not None:
        # Save step-1 forecast (first column) for clean visualization
        pred_df = pd.DataFrame({
            "actual": new_actuals[:, 0],
            "predicted": new_preds[:, 0]
        })
        pred_csv_local = str(work_dir / "predictions.csv")
        pred_df.to_csv(pred_csv_local, index=False)
        upload_to_gcs(pred_csv_local, f"{gcs_base}/results/predictions_latest.csv")
        print(f"  Predictions CSV uploaded ({len(pred_df)} samples).")
    
    # 5d. Audit Log
    if model_is_better:
        log_msg = f"Training completed. {next_version} is BETTER than {current_version}. MAE: {new_metrics['MAE']} vs {old_metrics['MAE'] if old_metrics else 'N/A'}"
    else:
        log_msg = f"Training completed. {next_version} is WORSE than {current_version}. MAE: {new_metrics['MAE']} vs {old_metrics['MAE'] if old_metrics else 'N/A'}"
    
    append_audit_log(args.gcs_bucket, log_msg)
    
    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE. Model {next_version} uploaded to {model_gcs_path}")
    print(f"Verdict: {'AUTO-DEPLOY RECOMMENDED' if model_is_better else 'USER DECISION REQUIRED'}")
    print(f"{'=' * 60}")
    
    return result_payload


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vertex AI Training Script for SpikeDLinear")
    
    # GCS Configuration
    parser.add_argument("--gcs_bucket", type=str, required=True, help="GCS bucket name (without gs://)")
    parser.add_argument("--data_uri", type=str, required=True, help="GCS URI for the uploaded training data CSV")
    
    # Model Configuration
    parser.add_argument("--target", type=str, default="EC", choices=["EC", "pH", "Temp", "Flow", "DO", "Turbidity"])
    parser.add_argument("--site", type=int, default=1463500)
    parser.add_argument("--seq_len", type=int, default=168, help="Look-back window (min: 168)")
    parser.add_argument("--pred_len", type=int, default=24, help="Forecast horizon (min: 24)")
    
    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="Maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    
    # Safety
    parser.add_argument("--timeout_minutes", type=int, default=30, help="Max training time in minutes")
    
    args = parser.parse_args()
    
    # Set timeout (Unix only; on Windows this is a no-op in containers)
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(args.timeout_minutes * 60)
    except AttributeError:
        print("[WARN] SIGALRM not available on this OS. Timeout disabled.")
    
    print(f"\n{'#' * 60}")
    print(f"# HydroPred Vertex AI Training Pipeline")
    print(f"# Target: {args.target} | Site: {args.site}")
    print(f"# Seq: {args.seq_len} | Pred: {args.pred_len} | Epochs: {args.epochs}")
    print(f"# Timeout: {args.timeout_minutes} minutes")
    print(f"{'#' * 60}\n")
    
    try:
        result = run_training_pipeline(args)
        # Exit code 0 = success
        sys.exit(0)
    except TrainingTimeoutError:
        print("\n[FATAL] Training TIMEOUT exceeded! Aborting.")
        append_audit_log(args.gcs_bucket, f"TIMEOUT: Training exceeded {args.timeout_minutes} minutes. Job aborted.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FATAL] Training CRASHED: {e}")
        try:
            append_audit_log(args.gcs_bucket, f"CRASH: {str(e)}")
        except:
            pass
        sys.exit(1)
