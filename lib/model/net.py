from torch import nn
import torch
from lib.const import NUM_TAGS
from transformers.models.bert import BertModel, BertConfig
from transformers.models.deberta_v2 import DebertaV2Model, DebertaV2Config


class MeanPooling(nn.Module):
    def forward(self, x, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()
        return torch.sum(x * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )


class Network(nn.Module):
    def __init__(self, num_classes=NUM_TAGS, input_dim=768, hidden_dim=512):
        super().__init__()
        self.num_classes = num_classes
        self.bn = nn.LayerNorm(hidden_dim)
        self.projector = nn.Linear(input_dim, hidden_dim)
        self.lin = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, embeds, **kwargs):
        x = [self.projector(x) for x in embeds]
        x = [v.mean(0).unsqueeze(0) for v in x]
        x = self.bn(torch.cat(x, dim=0))
        x = self.lin(x)
        outs = self.fc(x)
        return outs


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


class TransNetwork(nn.Module):
    def __init__(
        self,
        num_classes=NUM_TAGS,
        input_dim=768,
        hidden_dim=512,
        n_encoder_layers=2,
        attention_heads=3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.bn = nn.LayerNorm(input_dim)
        self.mp = MeanPooling()
        self.encoder_layer = nn.TransformerEncoderLayer(
            input_dim, attention_heads, input_dim * 4, batch_first=True
        )
        self.encoder_block = nn.TransformerEncoder(self.encoder_layer, n_encoder_layers)
        self.lin = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, embeds, attention_mask=None):
        x = self.encoder_block(
            embeds, src_key_padding_mask=attention_mask.logical_not()
        )
        x = self.mp(x, attention_mask=attention_mask)
        x = self.bn(x)
        x = self.lin(x)
        outs = self.fc(x)
        return outs


class TransNetwork1(nn.Module):
    def __init__(
        self,
        num_classes=NUM_TAGS,
        input_dim=768,
        hidden_dim=512,
        n_encoder_layers=2,
        attention_heads=3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.bn = nn.LayerNorm(input_dim)
        self.mp = MeanPooling()
        self.encoder_layer = nn.TransformerEncoderLayer(
            input_dim, attention_heads, input_dim * 4, batch_first=True
        )
        self.encoder_block = nn.TransformerEncoder(self.encoder_layer, n_encoder_layers)
        self.lin = ProjectionHead(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, embeds, attention_mask=None):
        x = self.encoder_block(
            embeds, src_key_padding_mask=attention_mask.logical_not()
        )
        x = self.mp(x, attention_mask=attention_mask)
        x = self.bn(x)
        x = self.lin(x)
        outs = self.fc(x)
        return outs


class TransNetwork2(nn.Module):
    def __init__(
        self, num_classes=NUM_TAGS, input_dim=768, hidden_dim=512, encoder_cfg=None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.bn = nn.LayerNorm(input_dim)
        self.mp = MeanPooling()
        self.encoder = BertModel(BertConfig(**encoder_cfg))
        self.lin = ProjectionHead(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, embeds, attention_mask=None):
        x = self.encoder(
            inputs_embeds=embeds, attention_mask=attention_mask
        ).last_hidden_state
        x = self.mp(x, attention_mask=attention_mask)
        x = self.bn(x)
        x = self.lin(x)
        outs = self.fc(x)
        return outs


class TransNetwork3(nn.Module):
    def __init__(
        self, num_classes=NUM_TAGS, input_dim=768, hidden_dim=512, encoder_cfg=None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.bn = nn.LayerNorm(input_dim)
        self.mp = MeanPooling()
        self.encoder = DebertaV2Model(DebertaV2Config(**encoder_cfg))
        self.lin = ProjectionHead(input_dim, hidden_dim, dropout=0, residual_connection=False)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, embeds, attention_mask=None):
        x = self.encoder(
            inputs_embeds=embeds, attention_mask=attention_mask
        ).last_hidden_state
        x = self.mp(x, attention_mask=attention_mask)
        x = self.lin(x)
        outs = self.fc(x)
        return outs
