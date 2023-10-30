from torch.nn import BCEWithLogitsLoss
from torch import nn
import torch


class FocalLoss(nn.Module):
    def __init__(self, class_weights=None, gamma=2):
        super().__init__()
        self.register_buffer("class_weights", self._class_weights(class_weights))
        self.gamma = gamma

    def forward(self, input, target):
        p = torch.sigmoid(input)
        p = torch.where(target >= 0.5, p, 1 - p)
        logp = -torch.log(torch.clamp(p, 1e-4, 1 - 1e-4))
        loss = logp * ((1 - p) ** self.gamma)
        if self.class_weights is not None:
            loss *= self.class_weights
        loss = loss.size(1) * loss.mean()
        return loss

    @staticmethod
    def _class_weights(class_weights):
        if class_weights is not None:
            w = torch.as_tensor(class_weights, dtype=torch.float)
            w /= torch.sum(w)
            w *= w.size(0)
            return w
