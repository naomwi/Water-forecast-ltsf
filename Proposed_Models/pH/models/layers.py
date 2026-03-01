import torch
import torch.nn as nn

class SpikeDLinear(nn.Module):
    def __init__(self, seq_len, pred_len, num_high, num_low, num_feat):
        super(SpikeDLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # --- QUAY VỀ KIẾN TRÚC MLP ĐƠN GIẢN ---
        
        # 1. Nhánh High Freq: Tăng Hidden Size lên 512
        self.high_enc = nn.Sequential(
            # Input: Flatten vector
            # Output Layer 1: 512 (Tăng dung lượng bộ nhớ)
            nn.Linear(seq_len * num_high, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            # Input Layer 2: 512 (Phải khớp với Output Layer 1)
            nn.Linear(512, pred_len)
        )

        # 2. Nhánh Low Freq: Linear (Giữ nguyên)
        self.low_enc = nn.Linear(seq_len * num_low, pred_len)

        # 3. Nhánh Features: MLP
        self.feat_enc = nn.Sequential(
            nn.Linear(seq_len * num_feat, 64),
            nn.ReLU(),
            nn.Linear(64, pred_len)
        )

        # 4. Fusion Layer
        self.fusion = nn.Linear(3, 1)

    def forward(self, x_high, x_low, x_feat):
        b = x_high.shape[0]

        # Flatten
        flat_high = x_high.reshape(b, -1)
        flat_low = x_low.reshape(b, -1)
        flat_feat = x_feat.reshape(b, -1)
        
        # Forward Pass
        out_high = self.high_enc(flat_high)
        out_low = self.low_enc(flat_low)
        out_feat = self.feat_enc(flat_feat)

        # Fusion
        stacked = torch.stack([out_high, out_low, out_feat], dim=-1)
        
        # Squeeze(-1) để đưa về [Batch, Pred_Len]
        final_pred = self.fusion(stacked).squeeze(-1)
        
        return final_pred