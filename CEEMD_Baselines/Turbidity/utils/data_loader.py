"""
Data Loader for CEEMD Baselines
Multi-channel dataset: all IMFs + target as input channels
Uses StandardScaler (fit on train) for proper normalization

Architecture matches: TrumAIFPTU/Water-quality-prediction/ceemdan_EVloss/main.py
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional


def load_raw_data(data_path: str, target_col: str, site_no: int = 1463500):
    """
    Load raw data from CSV file and filter by station.
    """
    df = pd.read_csv(data_path)

    if 'site_no' in df.columns and site_no is not None:
        df = df[df['site_no'] == site_no]
        print(f"Filtered to site_no={site_no}: {len(df)} samples")

    if 'Time' in df.columns:
        df['date'] = pd.to_datetime(df['Time'])
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear').bfill().ffill()

    return df, df[target_col].values.astype(np.float64)


class MultiChannelDataset(Dataset):
    """
    Multi-channel dataset for CEEMD baselines.
    
    Input: DataFrame with columns [target, IMF_0, IMF_1, ..., IMF_n]
    - All channels are scaled together using StandardScaler
    - Target is column 0 (first column)
    - y = target values at prediction horizon
    
    Matches reference: ceemdan_EVloss/main.py FeatureDataset
    """
    def __init__(self, data_values, seq_len, pred_len, flag='train', scaler=None):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scaler = scaler

        total_len = len(data_values)
        train_end = int(total_len * 0.6)
        val_end = int(total_len * 0.8)

        if flag == 'train':
            self.data = data_values[:train_end]
        elif flag == 'val':
            # Match Proposed Model: no overlap
            self.data = data_values[train_end:val_end]
        else:
            # Match Proposed Model: no overlap
            self.data = data_values[val_end:]

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_end = index + self.seq_len
        r_end = s_end + self.pred_len

        # x: all channels for input window
        seq_x = self.data[index:s_end]  # (seq_len, n_channels)

        # y: target column only for prediction window
        target = self.data[s_end:r_end, 0:1]  # (pred_len, 1)

        return (
            torch.tensor(seq_x, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32)
        )

    def inverse_target(self, pred, scaler):
        """Inverse transform target column only."""
        dummy = np.zeros((len(pred), scaler.n_features_in_))
        dummy[:, 0] = pred
        inv = scaler.inverse_transform(dummy)
        return inv[:, 0]


def create_multi_channel_loaders(data_values, seq_len, pred_len, batch_size=64):
    """
    Create train/val/test dataloaders for multi-channel CEEMD.
    
    Args:
        data_values: numpy array of shape (n_samples, n_channels)
                     Column 0 = target, columns 1+ = IMFs
        seq_len: input sequence length
        pred_len: prediction horizon
        batch_size: batch size
    
    Returns:
        train_loader, val_loader, test_loader, scaler
    """
    # Scale with StandardScaler (fit on train only)
    train_end = int(len(data_values) * 0.6)
    scaler = StandardScaler()
    scaler.fit(data_values[:train_end])
    data_scaled = scaler.transform(data_values)

    train_ds = MultiChannelDataset(data_scaled, seq_len, pred_len, 'train', scaler)
    val_ds = MultiChannelDataset(data_scaled, seq_len, pred_len, 'val', scaler)
    test_ds = MultiChannelDataset(data_scaled, seq_len, pred_len, 'test', scaler)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=1, shuffle=False),
        scaler
    )


# Keep backward compatibility
def create_dataloaders(imf, seq_len, pred_len, batch_size=64):
    """Legacy single-IMF dataloader (kept for backward compatibility)."""
    from torch.utils.data import Dataset as _Dataset

    class _IMFDataset(_Dataset):
        def __init__(self, data, seq_len, pred_len, flag='train'):
            self.seq_len = seq_len
            self.pred_len = pred_len
            n = len(data)
            n_train = int(n * 0.6)
            n_val = int(n * 0.2)
            border1s = [0, n_train, n_train + n_val]
            border2s = [n_train, n_train + n_val, n]
            idx = {'train': 0, 'val': 1, 'test': 2}[flag]
            self.data = data[border1s[idx]:border2s[idx]].reshape(-1, 1)

        def __len__(self):
            return len(self.data) - self.seq_len - self.pred_len + 1

        def __getitem__(self, index):
            s_end = index + self.seq_len
            r_end = s_end + self.pred_len
            return (
                torch.tensor(self.data[index:s_end], dtype=torch.float32),
                torch.tensor(self.data[s_end:r_end], dtype=torch.float32)
            )

    train_ds = _IMFDataset(imf, seq_len, pred_len, 'train')
    val_ds = _IMFDataset(imf, seq_len, pred_len, 'val')
    test_ds = _IMFDataset(imf, seq_len, pred_len, 'test')

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        None
    )
