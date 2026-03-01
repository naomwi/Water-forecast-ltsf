"""
DLinear and NLinear for CEEMD Baselines
Multi-channel input: all IMFs + target column as input channels
Predicts SINGLE target output (EC or pH)

Architecture matches: TrumAIFPTU/Water-quality-prediction/ceemdan_EVloss/src/ltsf_linear.py
"""

import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=2)
        return self.avg(x)


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    """
    DLinear with multi-channel input.
    Flattens all channels × seq_len into one linear layer → predicts pred_len target values.
    
    Input:  (batch, seq_len, in_channels)
    Output: (batch, pred_len, 1)
    """
    def __init__(self, seq_len: int, pred_len: int, in_channels: int = 1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.decomposition = SeriesDecomp(25)

        # Flatten all channels into linear layer (matches reference)
        self.Linear_Seasonal = nn.Linear(self.seq_len * in_channels, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len * in_channels, self.pred_len)

    def forward(self, x):
        # x: (batch, seq_len, in_channels) → permute to (batch, in_channels, seq_len)
        x = x.permute(0, 2, 1)

        seasonal_init, trend_init = self.decomposition(x)

        # Flatten: (batch, in_channels, seq_len) → (batch, in_channels * seq_len)
        seasonal_init = seasonal_init.reshape(seasonal_init.shape[0], -1)
        trend_init = trend_init.reshape(trend_init.shape[0], -1)

        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.unsqueeze(-1)  # (batch, pred_len, 1)


class NLinear(nn.Module):
    """
    NLinear with multi-channel input.
    Subtracts last value of first channel (target), flattens, linear → pred.
    
    Input:  (batch, seq_len, in_channels)
    Output: (batch, pred_len, 1)
    """
    def __init__(self, seq_len: int, pred_len: int, in_channels: int = 1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.Linear = nn.Linear(self.seq_len * in_channels, self.pred_len)

    def forward(self, x):
        # Subtract last value of target (first channel) for normalization
        seq_last = x[:, -1:, 0:1].detach()  # (batch, 1, 1)
        x_norm = x.clone()
        x_norm[:, :, 0:1] = x_norm[:, :, 0:1] - seq_last

        # Flatten all channels
        x_flat = x_norm.reshape(x_norm.shape[0], -1)
        out = self.Linear(x_flat)

        # Add back the last target value
        out = out + seq_last.squeeze(-1)
        return out.unsqueeze(-1)  # (batch, pred_len, 1)
