from torch import nn
import torch
from transformers.models.bert import BertModel, BertConfig
from transformers.models.deberta_v2 import DebertaV2Model, DebertaV2Config
from lib.const import NUM_TAGS
from lib.model.base import MeanPooling, ProjectionHead, smart_init_weights
from lib.model.conv_1d import CNN1DModel, LightCNN1DModel


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
        self.lin = ProjectionHead(
            input_dim, hidden_dim, dropout=0, residual_connection=False
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, embeds, attention_mask=None):
        x = self.encoder(
            inputs_embeds=embeds, attention_mask=attention_mask
        ).last_hidden_state
        x = self.mp(x, attention_mask=attention_mask)
        x = self.lin(x)
        outs = self.fc(x)
        return outs


class TransNetwork4(nn.Module):
    def __init__(
        self, num_classes=NUM_TAGS, input_dim=768, hidden_dim=512, encoder_cfg=None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.mp = MeanPooling()
        self.encoder = DebertaV2Model(DebertaV2Config(**encoder_cfg))
        self.lin = ProjectionHead(
            input_dim, hidden_dim, dropout=0.3, residual_connection=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, embeds, attention_mask=None):
        x = self.encoder(
            inputs_embeds=embeds, attention_mask=attention_mask
        ).last_hidden_state
        x = self.mp(x, attention_mask=attention_mask)
        x = self.lin(x)
        outs = self.fc(x)
        return outs


class TransNetwork5(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, num_classes=NUM_TAGS):
        super().__init__()
        self.conv1d = CNN1DModel(input_dim)
        self.mp = MeanPooling()
        self.lin = ProjectionHead(
            input_dim, hidden_dim, dropout=0.3, residual_connection=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, attention_mask):
        x = self.conv1d(x)
        x = self.mp(x, attention_mask=attention_mask)
        x = self.lin(x)
        outs = self.fc(x)
        return outs


class TransNetwork7(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, num_classes=NUM_TAGS):
        super().__init__()
        self.conv1d = LightCNN1DModel(input_dim)
        self.mp = MeanPooling()
        self.lin = ProjectionHead(
            input_dim, hidden_dim, dropout=0.3, residual_connection=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, attention_mask):
        x = self.conv1d(x)
        x = self.mp(x, attention_mask=attention_mask)
        x = self.lin(x)
        outs = self.fc(x)
        return outs


class TransNetwork8(nn.Module):
    def __init__(
        self, num_classes=NUM_TAGS, input_dim=768, hidden_dim=512, encoder_cfg=None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.mp = MeanPooling()
        self.encoder = DebertaV2Model(DebertaV2Config(**encoder_cfg))
        self.lin = ProjectionHead(
            input_dim, hidden_dim, dropout=0.3, residual_connection=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.apply(smart_init_weights)

    def forward(self, embeds, attention_mask=None):
        x = self.encoder(
            inputs_embeds=embeds, attention_mask=attention_mask
        ).last_hidden_state
        x = self.mp(x, attention_mask=attention_mask)
        x = self.lin(x)
        outs = self.fc(x)
        return outs
