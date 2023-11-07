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

    def forward(self, inputs, targets):
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


class Hill(nn.Module):
    r""" Hill as described in the paper "Robust Loss Design for Multi-Label Learning with Missing Labels "

    .. math::
        Loss = y \times (1-p_{m})^\gamma\log(p_{m}) + (1-y) \times -(\lambda-p){p}^2

    where : math:`\lambda-p` is the weighting term to down-weight the loss for possibly false negatives,
          : math:`m` is a margin parameter,
          : math:`\gamma` is a commonly used value same as Focal loss.

    .. note::
        Sigmoid will be done in loss.

    Args:
        lambda (float): Specifies the down-weight term. Default: 1.5. (We did not change the value of lambda in our experiment.)
        margin (float): Margin value. Default: 1 . (Margin value is recommended in [0.5,1.0], and different margins have little effect on the result.)
        gamma (float): Commonly used value same as Focal loss. Default: 2

    """

    def __init__(self, lamb: float = 1.5, margin: float = 1.0, gamma: float = 2.0, reduction: str = 'sum') -> None:
        super(Hill, self).__init__()
        self.lamb = lamb
        self.margin = margin
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        call function as forward

        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`

        Returns:
            torch.Tensor: loss
        """

        # Calculating predicted probability
        logits_margin = logits - self.margin
        pred_pos = torch.sigmoid(logits_margin)
        pred_neg = torch.sigmoid(logits)

        # Focal margin for postive loss
        pt = (1 - pred_pos) * targets + (1 - targets)
        focal_weight = pt ** self.gamma

        # Hill loss calculation
        los_pos = targets * torch.log(pred_pos)
        los_neg = (1 - targets) * -(self.lamb - pred_neg) * pred_neg ** 2

        loss = -(los_pos + los_neg)
        loss *= focal_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SPLC(nn.Module):
    r""" SPLC loss as described in the paper "Simple Loss Design for Multi-Label Learning with Missing Labels "

    .. math::
        &L_{SPLC}^+ = loss^+(p)
        &L_{SPLC}^- = \mathbb{I}(p\leq \tau)loss^-(p) + (1-\mathbb{I}(p\leq \tau))loss^+(p)

    where :math:'\tau' is a threshold to identify missing label
          :math:`$\mathbb{I}(\cdot)\in\{0,1\}$` is the indicator function,
          :math: $loss^+(\cdot), loss^-(\cdot)$ refer to loss functions for positives and negatives, respectively.

    .. note::
        SPLC can be combinded with various multi-label loss functions.
        SPLC performs best combined with Focal margin loss in our paper. Code of SPLC with Focal margin loss is released here.
        Since the first epoch can recall few missing labels with high precision, SPLC can be used ater the first epoch.
        Sigmoid will be done in loss.

    Args:
        tau (float): threshold value. Default: 0.6
        change_epoch (int): which epoch to combine SPLC. Default: 1
        margin (float): Margin value. Default: 1
        gamma (float): Hard mining value. Default: 2
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'sum'``

        """

    def __init__(self,
                 tau: float = 0.6,
                 change_epoch: int = 1,
                 margin: float = 1.0,
                 gamma: float = 2.0,
                 reduction: str = 'sum') -> None:
        super(SPLC, self).__init__()
        self.tau = tau
        self.change_epoch = change_epoch
        self.margin = margin
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.LongTensor,
                epoch) -> torch.Tensor:
        """
        call function as forward

        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`
            epoch : The epoch of current training.

        Returns:
            torch.Tensor: loss
        """
        # Subtract margin for positive logits
        logits = torch.where(targets == 1, logits - self.margin, logits)

        # SPLC missing label correction
        if epoch >= self.change_epoch:
            targets = torch.where(
                torch.sigmoid(logits) > self.tau,
                torch.tensor(1).cuda(), targets)

        pred = torch.sigmoid(logits)

        # Focal margin for postive loss
        pt = (1 - pred) * targets + pred * (1 - targets)
        focal_weight = pt ** self.gamma

        los_pos = targets * F.logsigmoid(logits)
        los_neg = (1 - targets) * F.logsigmoid(-logits)

        loss = -(los_pos + los_neg)
        loss *= focal_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
