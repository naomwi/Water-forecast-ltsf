import os
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PyEMD import CEEMDAN
from sklearn.preprocessing import StandardScaler
import multiprocessing
import gc

# --- CẤU HÌNH ---
FIXED_NUM_IMFS = 12
NUM_HIGH_FREQ = 3

def compute_features(series, window=12):
    # Tính toán trên dữ liệu Raw (chưa scale)
    s = pd.Series(series.flatten())
    delta = s.diff().fillna(0)
    roll_std = s.rolling(window).std().fillna(0)
    roll_mean = s.rolling(window).mean().fillna(0)
    
    # Z-score cục bộ
    z_score = (s - roll_mean) / (roll_std + 1e-6)
    z_score = z_score.replace([np.inf, -np.inf], 0).fillna(0)
    
    return np.stack([delta.values, roll_std.values, z_score.values], axis=1)

def load_and_preprocess_data(file_path, target_col, train_ratio=0.7, site=1463500, cache_dir="data/cache"):
    print(f"-> Reading data from {file_path}...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    df = pd.read_csv(file_path)
    if 'site_no' in df.columns: df = df[df['site_no'] == site]
    
    # Lấy dữ liệu RAW (Chưa Scale)
    raw_values = df[target_col].values.astype(float).reshape(-1, 1)
    n_total = len(raw_values)
    train_size = int(n_total * train_ratio)

    # --- BƯỚC 1: CEEMDAN TRÊN DỮ LIỆU RAW (Thay đổi ở đây) ---
    # Lưu file cache với tên khác (thêm _raw) hoặc xóa cache cũ để tránh nhầm lẫn
    os.makedirs(cache_dir, exist_ok=True)
    high_path = os.path.join(cache_dir, f"imfs_{site}_{target_col}_raw_high.npy") # Đổi tên file cache
    low_path = os.path.join(cache_dir, f"imfs_{site}_{target_col}_raw_low.npy")
    
    force_run = False
    if os.path.exists(high_path) and os.path.exists(low_path):
        temp_high = np.load(high_path)
        # Check độ dài khớp với raw data
        if temp_high.shape[0] != n_total: 
            print("-> Cache mismatch. Deleting...")
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            force_run = True

    if os.path.exists(high_path) and os.path.exists(low_path) and not force_run:
        print(f"-> Found Cached Raw IMFs ({FIXED_NUM_IMFS} components). Loading...")
        high_imfs_raw = np.load(high_path)
        low_imfs_raw = np.load(low_path)
    else:
        print(f"-> Running CEEMDAN on Raw Data...")
        # Limit to max 6 cores to prevent ArrayMemoryError on high-thread CPUs
        num_cores = min(6, multiprocessing.cpu_count())
        ceemdan = CEEMDAN(trials=20)
        ceemdan.processes = num_cores
        
        # Flatten raw values để đưa vào CEEMDAN
        imfs = ceemdan(raw_values.flatten(), max_imf=FIXED_NUM_IMFS-1).T
        
        # Padding/Truncating logic giữ nguyên
        current_cnt = imfs.shape[1]
        if current_cnt < FIXED_NUM_IMFS:
            padding = np.zeros((imfs.shape[0], FIXED_NUM_IMFS - current_cnt))
            imfs = np.concatenate([imfs, padding], axis=1)
        elif current_cnt > FIXED_NUM_IMFS:
            imfs = imfs[:, :FIXED_NUM_IMFS]
            
        high_imfs_raw = imfs[:, :NUM_HIGH_FREQ]
        low_imfs_raw = imfs[:, NUM_HIGH_FREQ:]
        
        np.save(high_path, high_imfs_raw)
        np.save(low_path, low_imfs_raw)
        
        # Bắt buộc dọn dẹp RAM vì numpy.array của IMFs ngốn tới vài GB
        del imfs, ceemdan 
        gc.collect()
        
        print("-> Finished CEEMDAN and saved cache.")

    # Tính features trên Raw
    features_raw = compute_features(raw_values)

    # --- BƯỚC 2: SCALING (FIT TRÊN TRAIN, TRANSFORM ALL) ---
    print("-> Normalizing data (Fit on Train, Transform all)...")
    
    # Khởi tạo các Scaler riêng biệt
    scaler_high = StandardScaler()
    scaler_low = StandardScaler()
    scaler_feat = StandardScaler()
    scaler_target = StandardScaler() # Dùng để scale raw_values
    
    # Fit trên tập Train (Chỉ dùng dữ liệu từ 0 -> train_size)
    scaler_high.fit(high_imfs_raw[:train_size])
    scaler_low.fit(low_imfs_raw[:train_size])
    scaler_feat.fit(features_raw[:train_size])
    scaler_target.fit(raw_values[:train_size])
    
    # Transform toàn bộ
    high_imfs_scaled = scaler_high.transform(high_imfs_raw)
    low_imfs_scaled = scaler_low.transform(low_imfs_raw)
    features_scaled = scaler_feat.transform(features_raw)
    raw_scaled = scaler_target.transform(raw_values).flatten()
    
    # Trả về thêm scaler_target để sau này Inverse
    return (high_imfs_scaled, low_imfs_scaled, features_scaled, 
            raw_scaled, scaler_target, train_size)

# ================= 2. DATASET & PLOTTING (GIỮ NGUYÊN CỦA BẠN) =================
class TimeSeriesDataset(Dataset):
    def __init__(self, high, low, feat, raw_scaled, seq_len, pred_len):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.high = torch.FloatTensor(high)
        self.low = torch.FloatTensor(low)
        self.feat = torch.FloatTensor(feat)
        self.raw = torch.FloatTensor(raw_scaled)
        
    def __len__(self):
        return len(self.raw) - self.seq_len - self.pred_len
    
    def __getitem__(self, idx):
        x_high = self.high[idx : idx + self.seq_len]
        x_low = self.low[idx : idx + self.seq_len]
        x_feat = self.feat[idx : idx + self.seq_len]
        
        last_val = self.raw[idx + self.seq_len - 1]
        y_true = self.raw[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        target_delta = y_true - last_val
        
        return x_high, x_low, x_feat, target_delta, last_val, y_true

# Hàm vẽ giữ nguyên theo ý bạn (Zoom-in spike)
def plot_results(preds, actuals, pred_len, save_path):
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(actuals[:, 0], label='Actual', color='blue', alpha=0.6, linewidth=1)
    plt.plot(preds[:, 0], label='Prediction',linestyle='--', color='#d62728', alpha=0.8, linewidth=1)
    plt.title(f'Test Set Performance (Step-1 Forecast) - Horizon {pred_len}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    # Vẽ mẫu spike lớn nhất trong tập test
    max_idx = np.argmax(np.max(actuals, axis=1))
    plt.plot(actuals[max_idx], label='Actual Sequence', marker='o', color='black')
    plt.plot(preds[max_idx], label='Predicted Sequence', marker='x', color='#d62728', linestyle='--')
    plt.title(f'Zoom-in: Highest Spike Sample (Index {max_idx})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()