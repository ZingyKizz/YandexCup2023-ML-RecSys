import torch
from torch import nn
from scipy.linalg import hadamard
import math


class MeanPooling(nn.Module):
    def forward(self, x, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()
        return torch.sum(x * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.3, residual_connection=True):
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_dim)
        self.residual_connection = residual_connection

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        if self.residual_connection:
            x = x + projected
        x = self.ln(x)
        return x


def ZerO_init_on_matrix(matrix_tensor):
    # Algorithm 1 in the paper.

    m = matrix_tensor.size(0)
    n = matrix_tensor.size(1)

    if m <= n:
        init_matrix = torch.nn.init.eye_(torch.empty(m, n))
    elif m > n:
        clog_m = math.ceil(math.log2(m))
        p = 2 ** (clog_m)
        init_matrix = (
            torch.nn.init.eye_(torch.empty(m, p))
            @ (torch.tensor(hadamard(p)).float() / (2 ** (clog_m / 2)))
            @ torch.nn.init.eye_(torch.empty(p, n))
        )

    return init_matrix


def smart_init_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.data = ZerO_init_on_matrix(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        m.bias.data.zero_()
        m.weight.data.fill_(1.0)
