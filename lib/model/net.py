from torch import nn
import torch
from transformers.models.bert import BertModel, BertConfig
from transformers.models.deberta_v2 import DebertaV2Model, DebertaV2Config
from transformers.models.convbert import ConvBertModel, ConvBertConfig
from lib.const import NUM_TAGS
from lib.model.base import (
    MeanPooling,
    ProjectionHead,
    smart_init_weights,
    SelfAttention,
)
from lib.model.conv_1d import (
    CNN1DModel,
    LightCNN1DModel,
    VeryLightCNN1DModel,
    GemLightCNN1DModel,
    GemVeryLightCNN1DModel,
    GemVeryLightCNN1DWithDepthMaxPoolModel,
    AdaptiveGemVeryLightCNN1DWithDepthMaxPoolModel,
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

    def forward(self, embeds, *args, **kwargs):
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

    def forward(self, embeds, attention_mask=None, *args, **kwargs):
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

    def forward(self, embeds, attention_mask=None, *args, **kwargs):
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

    def forward(self, embeds, attention_mask=None, *args, **kwargs):
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

    def forward(self, embeds, attention_mask=None, *args, **kwargs):
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

    def forward(self, embeds, attention_mask=None, *args, **kwargs):
        x = self.encoder(
            inputs_embeds=embeds, attention_mask=attention_mask
        ).last_hidden_state
        x = self.mp(x, attention_mask=attention_mask)
        x = self.lin(x)
        outs = self.fc(x)
        return outs


class TransNetwork5(nn.Module):
    def __init__(
        self, input_dim=768, hidden_dim=512, num_classes=NUM_TAGS, cnn_activation="relu"
    ):
        super().__init__()
        self.conv1d = CNN1DModel(input_dim, activation=cnn_activation)
        self.mp = MeanPooling()
        self.lin = ProjectionHead(
            input_dim, hidden_dim, dropout=0.3, residual_connection=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, attention_mask, *args, **kwargs):
        x = self.conv1d(x)
        x = self.mp(x, attention_mask=attention_mask)
        x = self.lin(x)
        outs = self.fc(x)
        return outs


class TransNetwork7(nn.Module):
    def __init__(
        self, input_dim=768, hidden_dim=512, num_classes=NUM_TAGS, cnn_activation="relu"
    ):
        super().__init__()
        self.conv1d = LightCNN1DModel(input_dim, activation=cnn_activation)
        self.mp = MeanPooling()
        self.lin = ProjectionHead(
            input_dim, hidden_dim, dropout=0.3, residual_connection=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, attention_mask, *args, **kwargs):
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
        self.fc.apply(smart_init_weights)

    def forward(self, embeds, attention_mask=None, *args, **kwargs):
        x = self.encoder(
            inputs_embeds=embeds, attention_mask=attention_mask
        ).last_hidden_state
        x = self.mp(x, attention_mask=attention_mask)
        x = self.lin(x)
        outs = self.fc(x)
        return outs


class TransNetwork9(nn.Module):
    def __init__(
        self, input_dim=768, hidden_dim=512, num_classes=NUM_TAGS, cnn_activation="relu"
    ):
        super().__init__()
        self.conv1d = LightCNN1DModel(input_dim, activation=cnn_activation)
        self.mp = MeanPooling()
        self.lin = ProjectionHead(
            input_dim, hidden_dim, dropout=0.3, residual_connection=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.fc.apply(smart_init_weights)

    def forward(self, x, attention_mask, *args, **kwargs):
        x = self.conv1d(x)
        x = self.mp(x, attention_mask=attention_mask)
        x = self.lin(x)
        outs = self.fc(x)
        return outs


class TransNetwork10(nn.Module):
    def __init__(
        self, input_dim=768, hidden_dim=512, num_classes=NUM_TAGS, cnn_activation="relu"
    ):
        super().__init__()
        self.conv1d = VeryLightCNN1DModel(input_dim, activation=cnn_activation)
        self.mp = MeanPooling()
        self.lin = ProjectionHead(
            input_dim, hidden_dim, dropout=0.3, residual_connection=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.fc.apply(smart_init_weights)

    def forward(self, x, attention_mask, *args, **kwargs):
        x = self.conv1d(x)
        x = self.mp(x, attention_mask=attention_mask)
        x = self.lin(x)
        outs = self.fc(x)
        return outs


class TransNetwork11(nn.Module):
    def __init__(
        self, input_dim=768, hidden_dim=512, num_classes=NUM_TAGS, cnn_activation="relu"
    ):
        super().__init__()
        self.ln = nn.LayerNorm(input_dim)
        self.conv1d = LightCNN1DModel(input_dim, activation=cnn_activation)
        self.mp = MeanPooling()
        self.lin = ProjectionHead(
            input_dim, hidden_dim, dropout=0.3, residual_connection=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, attention_mask, *args, **kwargs):
        x = self.ln(x)
        x = self.conv1d(x)
        x = self.mp(x, attention_mask=attention_mask)
        x = self.lin(x)
        outs = self.fc(x)
        return outs


class TransNetwork12(nn.Module):
    def __init__(
        self, input_dim=768, hidden_dim=512, num_classes=NUM_TAGS, cnn_activation="relu"
    ):
        super().__init__()
        self.conv1d = GemLightCNN1DModel(input_dim, activation=cnn_activation)
        self.mp = MeanPooling()
        self.lin = ProjectionHead(
            input_dim, hidden_dim, dropout=0.3, residual_connection=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, attention_mask, *args, **kwargs):
        x = self.conv1d(x)
        x = self.mp(x, attention_mask=attention_mask)
        x = self.lin(x)
        outs = self.fc(x)
        return outs


class TransNetwork13(nn.Module):
    def __init__(
        self, input_dim=768, hidden_dim=512, num_classes=NUM_TAGS, cnn_activation="relu"
    ):
        super().__init__()
        self.conv1d = GemLightCNN1DModel(input_dim, activation=cnn_activation)
        self.lin = ProjectionHead(
            input_dim, hidden_dim, dropout=0.3, residual_connection=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, *args, **kwargs):
        x = self.conv1d(x)
        x = x.mean(dim=1)
        x = self.lin(x)
        outs = self.fc(x)
        return outs


class TransNetwork14(nn.Module):
    def __init__(
        self, input_dim=768, hidden_dim=512, num_classes=NUM_TAGS, cnn_activation="relu"
    ):
        super().__init__()
        self.conv1d = GemLightCNN1DModel(input_dim, activation=cnn_activation)
        self.lin = ProjectionHead(
            input_dim, hidden_dim, dropout=0.3, residual_connection=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, *args, **kwargs):
        x = self.conv1d(x)
        x, _ = x.max(dim=1)
        x = self.lin(x)
        outs = self.fc(x)
        return outs


class TransNetwork15(nn.Module):
    def __init__(
        self, input_dim=768, hidden_dim=512, num_classes=NUM_TAGS, cnn_activation="relu"
    ):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2 * input_dim)
        self.conv1d = GemLightCNN1DModel(2 * input_dim, activation=cnn_activation)
        self.lin = ProjectionHead(
            2 * input_dim, hidden_dim, dropout=0.3, residual_connection=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, *args, **kwargs):
        x = self.linear(x)
        x = self.conv1d(x)
        x, _ = x.max(dim=1)
        x = self.lin(x)
        outs = self.fc(x)
        return outs


class TransNetwork16(nn.Module):
    def __init__(
        self, input_dim=768, hidden_dim=512, num_classes=NUM_TAGS, cnn_activation="relu"
    ):
        super().__init__()
        self.knn_linear = nn.Sequential(
            nn.Linear(72, input_dim // 2),
            nn.SiLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.LayerNorm(input_dim),
        )
        self.mp = MeanPooling()
        self.conv1d = GemLightCNN1DModel(input_dim, activation=cnn_activation)
        # self.ln = nn.LayerNorm(1024)
        self.lin = ProjectionHead(
            768, hidden_dim, dropout=0.3, residual_connection=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, embeds, attention_mask, knn_embeds, length, *args, **kwargs):
        x = self.conv1d(embeds)
        x = self.mp(x, attention_mask=attention_mask)
        y = self.knn_linear(knn_embeds)
        y = y.mean(dim=1)
        z = x + y
        z = self.lin(z)
        outs = self.fc(z)
        return outs


class TransNetwork17(nn.Module):
    def __init__(
        self,
        channels,
        hidden_dim=512,
        num_classes=NUM_TAGS,
        cnn_activation="relu",
        cnn_dropout=0,
    ):
        super().__init__()
        self.conv1d = GemVeryLightCNN1DModel(
            channels, activation=cnn_activation, dropout=cnn_dropout
        )
        self.mp = MeanPooling()
        self.fc = nn.Sequential(
            ProjectionHead(
                channels[-1][1], hidden_dim, dropout=0.3, residual_connection=True
            ),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, attention_mask, *args, **kwargs):
        x = self.conv1d(x)
        x = self.mp(x, attention_mask=attention_mask)
        outs = self.fc(x)
        return outs


class TransNetwork21(nn.Module):
    def __init__(
        self,
        cnn_params,
        gru_params,
        num_classes=NUM_TAGS,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=gru_params["input_size"],
            hidden_size=gru_params["hidden_size"],
            batch_first=True,
        )
        self.conv1d = GemVeryLightCNN1DModel(
            cnn_params["channels"],
            activation=cnn_params.get("activation", "relu"),
            dropout=cnn_params.get("dropout", 0.0),
        )
        self.fc = nn.Linear(gru_params["hidden_size"], num_classes)

    def forward(self, x, *args, **kwargs):
        x = self.conv1d(x)
        x = self.gru(x)[1].squeeze(0)
        outs = self.fc(x)
        return outs


class TransNetwork22(nn.Module):
    def __init__(
        self,
        cnn_params,
        gru_params,
        num_classes=NUM_TAGS,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=gru_params["input_size"],
            hidden_size=gru_params["hidden_size"],
            batch_first=True,
        )
        self.conv1d = GemVeryLightCNN1DModel(
            cnn_params["channels"],
            activation=cnn_params.get("activation", "relu"),
            dropout=cnn_params.get("dropout", 0.0),
        )
        self.fc = nn.Linear(cnn_params["channels"][-1][1], num_classes)

    def forward(self, x, *args, **kwargs):
        x = self.gru(x)[0]
        x = self.conv1d(x)
        x = x.mean(dim=1)
        outs = self.fc(x)
        return outs


class TransNetwork23(nn.Module):
    def __init__(
        self,
        channels,
        hidden_dim=512,
        num_classes=NUM_TAGS,
        cnn_activation="relu",
        cnn_dropout=0,
    ):
        super().__init__()
        self.conv1d = GemVeryLightCNN1DWithDepthMaxPoolModel(
            channels, activation=cnn_activation, dropout=cnn_dropout
        )
        self.mp = MeanPooling()
        self.fc = nn.Sequential(
            ProjectionHead(
                channels[-1][1], hidden_dim, dropout=0.3, residual_connection=True
            ),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, attention_mask, *args, **kwargs):
        x = self.conv1d(x)
        x = self.mp(x, attention_mask=attention_mask)
        outs = self.fc(x)
        return outs


class TransNetwork24(nn.Module):
    def __init__(
        self,
        cnn_params,
        projection_params,
        num_classes=NUM_TAGS,
    ):
        super().__init__()
        channels = cnn_params["channels"]
        hidden_dim = projection_params.get("hidden_dim", 768)
        self.conv1d = GemVeryLightCNN1DWithDepthMaxPoolModel(
            channels,
            activation=cnn_params.get("activation", "relu"),
            dropout=cnn_params.get("dropout", 0.0),
            kernel_size=cnn_params.get("kernel_size", 5),
            gem_kernel_size=cnn_params.get("gem_kernel_size", 3),
        )
        self.mp = MeanPooling()
        self.fc = nn.Sequential(
            ProjectionHead(
                channels[-1][1],
                hidden_dim=hidden_dim,
                dropout=projection_params.get("dropout", 0.3),
                residual_connection=projection_params.get("residual_connection", True),
            ),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, attention_mask, *args, **kwargs):
        x = self.conv1d(x)
        x = self.mp(x, attention_mask=attention_mask)
        outs = self.fc(x)
        return outs


class TransNetwork25(nn.Module):
    def __init__(
        self,
        gru_params,
        num_classes=NUM_TAGS,
    ):
        super().__init__()
        self.bn = nn.BatchNorm1d(768)
        self.gru = nn.GRU(
            input_size=gru_params["input_size"],
            hidden_size=gru_params["hidden_size"],
            batch_first=True,
            num_layers=1,
        )
        self.fc = nn.Linear(gru_params["hidden_size"], num_classes)

    def forward(self, x, *args, **kwargs):
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        x = self.gru(x)[1].squeeze(0)
        outs = self.fc(x)
        return outs


class TransNetwork26(nn.Module):
    def __init__(
        self,
        gru_params,
        num_classes=NUM_TAGS,
    ):
        super().__init__()
        self.bn = nn.BatchNorm1d(768)
        self.gru = nn.GRU(
            input_size=gru_params["input_size"],
            hidden_size=gru_params["hidden_size"],
            batch_first=True,
            num_layers=4,
        )
        self.fc = nn.Linear(gru_params["hidden_size"], num_classes)

    def forward(self, x, *args, **kwargs):
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        x = self.gru(x)[1].mean(dim=0)
        outs = self.fc(x)
        return outs


class TransNetwork27(nn.Module):
    def __init__(
        self,
        gru_params,
        num_classes=NUM_TAGS,
    ):
        super().__init__()
        self.bn = nn.BatchNorm1d(768)
        self.gru = nn.LSTM(
            input_size=gru_params["input_size"],
            hidden_size=gru_params["hidden_size"],
            batch_first=True,
            num_layers=4,
        )
        self.fc = nn.Linear(gru_params["hidden_size"], num_classes)

    def forward(self, x, *args, **kwargs):
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        x = self.gru(x)[1][1].mean(dim=0)
        outs = self.fc(x)
        return outs


class TransNetwork28(nn.Module):
    def __init__(
        self,
        gru_params,
        num_classes=NUM_TAGS,
    ):
        super().__init__()
        self.bn = nn.BatchNorm1d(768)
        self.gru = nn.LSTM(
            input_size=gru_params["input_size"],
            hidden_size=gru_params["hidden_size"],
            batch_first=True,
            num_layers=4,
        )
        self.fc = nn.Linear(gru_params["hidden_size"], num_classes)

    def forward(self, x, *args, **kwargs):
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        x = self.gru(x)[1].mean(dim=0)
        outs = self.fc(x)
        return outs


class TransNetwork29(nn.Module):
    def __init__(
        self, num_classes=NUM_TAGS, input_dim=768, hidden_dim=512, encoder_cfg=None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.mp = MeanPooling()
        self.encoder = ConvBertModel(ConvBertConfig(**encoder_cfg))
        self.lin = ProjectionHead(
            input_dim, hidden_dim, dropout=0.3, residual_connection=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.fc.apply(smart_init_weights)

    def forward(self, embeds, attention_mask=None, *args, **kwargs):
        x = self.encoder(
            inputs_embeds=embeds, attention_mask=attention_mask
        ).last_hidden_state
        x = self.mp(x, attention_mask=attention_mask)
        x = self.lin(x)
        outs = self.fc(x)
        return outs


class TransNetwork30(nn.Module):
    def __init__(self, cnn_params, input_dim=768, hidden_dim=512, num_classes=NUM_TAGS):
        super().__init__()
        self.knn_linear = nn.Sequential(
            nn.Linear(72, input_dim // 2),
            nn.SiLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.LayerNorm(input_dim),
        )
        self.mp = MeanPooling()
        self.conv1d = GemVeryLightCNN1DModel(
            cnn_params["channels"],
            activation=cnn_params["activation"],
            dropout=cnn_params["dropout"],
        )
        self.lin = ProjectionHead(
            768, hidden_dim, dropout=0.3, residual_connection=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, embeds, attention_mask, knn_embeds, length, *args, **kwargs):
        x = self.conv1d(embeds)
        x = self.mp(x, attention_mask=attention_mask)
        y = self.knn_linear(knn_embeds)
        y = y.mean(dim=1)
        z = x + y
        z = self.lin(z)
        outs = self.fc(z)
        return outs


class TransNetwork31(nn.Module):
    def __init__(self, cnn_params, input_dim=768, hidden_dim=512, num_classes=NUM_TAGS):
        super().__init__()
        self.knn_linear = nn.Sequential(
            nn.Linear(72, input_dim),
            nn.SiLU(),
            nn.LayerNorm(input_dim),
        )
        self.mp = MeanPooling()
        self.conv1d = GemVeryLightCNN1DModel(
            cnn_params["channels"],
            activation=cnn_params["activation"],
            dropout=cnn_params["dropout"],
        )
        self.lin = ProjectionHead(
            768, hidden_dim, dropout=0.3, residual_connection=True
        )
        self.ln = nn.LayerNorm(hidden_dim + 1)
        self.fc = nn.Linear(hidden_dim + 1, num_classes)

    def forward(self, embeds, attention_mask, knn_embeds, length, *args, **kwargs):
        x = self.conv1d(embeds)
        x = self.mp(x, attention_mask=attention_mask)
        y = self.knn_linear(knn_embeds)
        y = y.mean(dim=1)
        z = x + y
        z = self.lin(z)
        z = self.ln(torch.cat([z, length], dim=-1))
        outs = self.fc(z)
        return outs


class TransNetwork32(nn.Module):
    def __init__(self, cnn_params, input_dim=768, num_classes=NUM_TAGS):
        super().__init__()
        self.knn_linear = nn.Sequential(
            nn.Linear(72, input_dim),
            nn.SiLU(),
            nn.LayerNorm(input_dim),
        )
        self.conv1d = AdaptiveGemVeryLightCNN1DWithDepthMaxPoolModel(
            cnn_params["channels"],
            activation=cnn_params["activation"],
            dropout=cnn_params["dropout"],
        )
        self.fc = nn.Linear(cnn_params["channels"][-1][1], num_classes)

    def forward(self, embeds, attention_mask, knn_embeds, length, *args, **kwargs):
        x = self.conv1d(embeds)
        y = self.knn_linear(knn_embeds)
        y = y.max(dim=1)[0]
        z = x + y
        outs = self.fc(z)
        return outs
