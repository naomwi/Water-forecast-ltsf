import torch
import torch.nn as nn

class SpikeAwareLoss(nn.Module):
    def __init__(self, gamma=2.0, penalty_weight=5.0):
        super(SpikeAwareLoss, self).__init__()
        # Bỏ alpha vì phiên bản đơn giản không dùng Event Weighting phức tạp
        self.gamma = gamma             # Hệ số Focal (Tập trung mẫu khó)
        self.penalty = penalty_weight  # Hệ số phạt khi đoán thiếu

    def forward(self, pred, target):
        diff = target - pred
        mse = diff ** 2
        abs_diff = torch.abs(diff)
        
        # 1. Asymmetric Penalty: Phạt nặng nếu đoán thiếu (Target > Pred)
        # Logic: Nếu diff > 0 (tức là thực tế cao hơn dự báo) -> Nhân thêm penalty
        #        Nếu diff <= 0 (đoán thừa hoặc đúng) -> Giữ nguyên (nhân 1.0)
        asym_weight = torch.where(diff > 0, self.penalty, 1.0)
        
        # 2. Focal Weight: Tập trung vào các lỗi sai lớn
        # Công thức: (1 - exp(-|error|))^gamma
        # Lỗi càng lớn -> Weight càng gần 1. Lỗi nhỏ -> Weight gần 0.
        focal_weight = (1 - torch.exp(-abs_diff)) ** self.gamma
        
        # Tổng hợp
        loss = asym_weight * focal_weight * mse
        
        return loss.mean()