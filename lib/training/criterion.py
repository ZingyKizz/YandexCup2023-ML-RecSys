from torch.nn import BCEWithLogitsLoss
from torch import nn
import torch


class FocalLoss(nn.Module):
    def __init__(self, focus_param=2, balance_param=0.25):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.focus_param = focus_param
        self.balance_param = balance_param

    def forward(self, input, target):
        bce_loss = self.bce_loss(input, target)
        logpt = -bce_loss
        pt = torch.exp(logpt)
        focal_loss = -((1 - pt) ** self.focus_param) * logpt * self.balance_param
        return focal_loss
