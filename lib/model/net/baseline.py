from torch import nn
import torch
from lib.const import NUM_TAGS, DEVICE


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

    def forward(self, embeds):
        x = [self.projector(x) for x in embeds]
        x = [v.mean(0).unsqueeze(0) for v in x]
        x = self.bn(torch.cat(x, dim=0))
        x = self.lin(x)
        outs = self.fc(x)
        return outs


class TransNetwork(nn.Module):
    def __init__(self, num_classes=NUM_TAGS, input_dim=768, hidden_dim=512, n_encoder_layers=2, attention_heads=3):
        super().__init__()
        self.num_classes = num_classes
        self.bn = nn.LayerNorm(input_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(input_dim, attention_heads, input_dim * 4, batch_first=True)
        self.encoder_block = nn.TransformerEncoder(self.encoder_layer, n_encoder_layers)
        self.lin = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, embeds):
        x = self.encoder_block(embeds)
        x = x.mean(dim=1)
        x = self.bn(x)
        x = self.lin(x)
        outs = self.fc(x)
        return outs


class TransNetwork1(nn.Module):
    def __init__(self, num_classes=NUM_TAGS, input_dim=768, hidden_dim=512):
        super().__init__()
        self.num_classes = num_classes
        self.bn = nn.LayerNorm(input_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(input_dim, 3, input_dim * 4, batch_first=True)
        self.encoder_block = nn.TransformerEncoder(self.encoder_layer, 2)
        self.pos = nn.Embedding(512, input_dim)
        self.dropout = nn.Dropout(0.2)
        self.lin = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.pos_ind = torch.arange(512).to(DEVICE)

    def forward(self, embeds):
        pos_embeds = self.pos(self.pos_ind)[:embeds.size(1)]
        x = embeds + pos_embeds
        x = self.dropout(x)
        x = self.encoder_block(x)
        x = x.mean(dim=1)
        x = self.bn(x)
        x = self.lin(x)
        outs = self.fc(x)
        return outs
