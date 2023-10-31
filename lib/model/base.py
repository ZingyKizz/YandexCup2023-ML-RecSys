import torch
from torch import nn


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
