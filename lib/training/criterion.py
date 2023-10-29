from torch.nn import BCEWithLogitsLoss
from torch import nn
import torch
from lib.const import NUM_TAGS


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        logits = input.reshape(-1)
        targets = target.reshape(-1)
        p = torch.sigmoid(logits)
        p = torch.where(targets >= 0.5, p, 1 - p)
        logp = -torch.log(torch.clamp(p, 1e-4, 1 - 1e-4))
        loss = logp * ((1 - p) ** self.gamma)
        loss = NUM_TAGS * loss.mean()
        return loss
