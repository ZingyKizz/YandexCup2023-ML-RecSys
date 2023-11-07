import torch
from torch import nn
import torch.nn.functional as F


class GeM(nn.Module):
    """
    Code modified from the 2d code in
    https://amaarora.github.io/2020/08/30/gempool.html
    """

    def __init__(self, kernel_size=8, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.kernel_size = kernel_size
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool1d(
            x.clamp(min=eps).pow(p),
            self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1,
        ).pow(1.0 / p)
