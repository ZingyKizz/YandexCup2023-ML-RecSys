from torch.nn import BCEWithLogitsLoss
from torch import nn
import torch
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, class_weights=None, gamma=2):
        super().__init__()
        self.register_buffer("class_weights", self._class_weights(class_weights))
        self.gamma = gamma

    def forward(self, input, target):
        num_labels = input.size(1)
        p = torch.sigmoid(input)
        p = torch.where(target >= 0.5, p, 1 - p)
        logp = -torch.log(torch.clamp(p, 1e-4, 1 - 1e-4))
        loss = logp * ((1 - p) ** self.gamma)
        if self.class_weights is not None:
            loss = loss * self.class_weights
        loss = num_labels * loss.mean()
        return loss

    @staticmethod
    def _class_weights(class_weights):
        if class_weights is not None:
            w = torch.as_tensor(class_weights, dtype=torch.float)
            w = w / torch.sum(w)
            w = w * w.size(0)
            return w


class ZLPRLoss(nn.Module):
    def forward(self, s, y):
        loss_pos = torch.log(1 + torch.sum(torch.exp(-s) * y, dim=-1))
        loss_neg = torch.log(1 + torch.sum(torch.exp(s) * (1 - y), dim=-1))
        loss = loss_pos + loss_neg
        return loss.mean()


class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        gamma_neg=4.0,
        gamma_pos=1.0,
        clip=0.05,
    ):
        """Asymmetric Loss for Multi-label Classification. https://arxiv.org/abs/2009.14119
        Loss function where negative classes are weighted less than the positive classes.
        Note: the inputs are logits and targets, not sigmoids.
        Usage:
            inputs = torch.randn(5, 3)
            targets = torch.randint(0, 1, (5, 3)) # must be binary
            loss_fn = AsymmetricLoss()
            loss = loss_fn(inputs, targets)
        Args:
            gamma_neg: loss attenuation factor for negative classes
            gamma_pos: loss attenuation factor for positive classes
            clip: shifts the negative class probability and zeros loss if probability > clip
            reduction: how to reduced final loss. Must be one of mean[default], sum, none
        """
        super().__init__()
        if clip < 0.0 or clip > 1.0:
            raise ValueError("Clipping value must be non-negative and less than one")
        if gamma_neg < gamma_pos:
            raise ValueError(
                "Need to ensure that loss for hard positive is penalised less than hard negative"
            )

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def _get_binary_cross_entropy_loss_and_pt_with_logits(self, inputs, targets):
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets.float(), reduction="none"
        )
        pt = torch.exp(-ce_loss)  # probability at y_i=1
        return ce_loss, pt

    def forward(
        self, inputs: torch.FloatTensor, targets: torch.LongTensor
    ) -> torch.FloatTensor:
        ce_loss, pt = self._get_binary_cross_entropy_loss_and_pt_with_logits(
            inputs, targets
        )
        # shift and clamp (therefore zero gradient) high confidence negative cases
        pt_neg = (pt + self.clip).clamp(max=1.0)
        ce_loss_neg = -torch.log(pt_neg)
        loss_neg = (1 - pt_neg) ** self.gamma_neg * ce_loss_neg
        loss_pos = (1 - pt) ** self.gamma_pos * ce_loss
        loss = targets * loss_pos + (1 - targets) * loss_neg

        return loss.mean()
