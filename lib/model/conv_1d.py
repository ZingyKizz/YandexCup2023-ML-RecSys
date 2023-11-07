import torch
from torch import nn
from lib.model.gem import GeM


class Conv1dBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(5,),
        stride=(1,),
        padding=(2,),
        skip_connection=False,
        activation="relu",
        dropout=0.0,
    ):
        super().__init__()
        self.skip_connection = skip_connection
        self.conv_block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode="replicate",
                bias=True,
            ),
            nn.BatchNorm1d(out_channels),
            self._get_activation_module(activation),
            nn.Dropout1d(dropout),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode="replicate",
                bias=True,
            ),
            nn.BatchNorm1d(out_channels),
        )
        self.downsample = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size=(1,), stride=(1,), bias=False
            ),
            nn.BatchNorm1d(out_channels),
        )
        self.act = self._get_activation_module(activation)

    def forward(self, x):
        output = self.conv_block(x)
        if self.skip_connection:
            x = self.downsample(x)
            output = output + x
        output = self.act(output)
        return output

    @staticmethod
    def _get_activation_module(activation):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "leaky_relu":
            return nn.LeakyReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "swish":
            return nn.SiLU()
        elif activation == "elu":
            return nn.ELU()
        else:
            raise ValueError


class CNN1DModel(nn.Module):
    def __init__(self, in_channels, activation="relu"):
        super().__init__()
        self.conv_block1 = Conv1dBlock(
            in_channels=in_channels,
            out_channels=2 * in_channels,
            skip_connection=True,
            activation=activation,
        )
        self.conv_block2 = Conv1dBlock(
            in_channels=2 * in_channels,
            out_channels=4 * in_channels,
            skip_connection=True,
            activation=activation,
        )
        self.conv_block3 = Conv1dBlock(
            in_channels=4 * in_channels,
            out_channels=4 * in_channels,
            skip_connection=True,
            activation=activation,
        )
        self.conv_block4 = Conv1dBlock(
            in_channels=4 * in_channels,
            out_channels=2 * in_channels,
            skip_connection=True,
            activation=activation,
        )
        self.conv_block5 = Conv1dBlock(
            in_channels=2 * in_channels,
            out_channels=in_channels,
            skip_connection=True,
            activation=activation,
        )
        self.pooling = nn.AvgPool1d(kernel_size=(3,), stride=(1,), padding=(1,))

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.conv_block1(x)
        x = self.pooling(x)
        x = self.conv_block2(x)
        x = self.pooling(x)
        x = self.conv_block3(x)
        x = self.pooling(x)
        x = self.conv_block4(x)
        x = self.pooling(x)
        x = self.conv_block5(x)
        x = self.pooling(x)
        x = torch.transpose(x, 1, 2)
        return x


class LightCNN1DModel(nn.Module):
    def __init__(self, in_channels, activation="relu", dropout=0.0):
        super().__init__()
        self.conv_block1 = Conv1dBlock(
            in_channels=in_channels,
            out_channels=2 * in_channels,
            skip_connection=True,
            activation=activation,
            dropout=dropout,
        )
        self.conv_block2 = Conv1dBlock(
            in_channels=2 * in_channels,
            out_channels=2 * in_channels,
            skip_connection=True,
            activation=activation,
            dropout=dropout,
        )
        self.conv_block3 = Conv1dBlock(
            in_channels=2 * in_channels,
            out_channels=in_channels,
            skip_connection=True,
            activation=activation,
            dropout=dropout,
        )
        self.pooling = nn.AvgPool1d(kernel_size=(3,), stride=(1,), padding=(1,))

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.conv_block1(x)
        x = self.pooling(x)
        x = self.conv_block2(x)
        x = self.pooling(x)
        x = self.conv_block3(x)
        x = torch.transpose(x, 1, 2)
        return x


class VeryLightCNN1DModel(nn.Module):
    def __init__(self, in_channels, activation="relu", dropout=0.0):
        super().__init__()
        self.conv_block1 = Conv1dBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            skip_connection=True,
            activation=activation,
            dropout=dropout,
        )
        self.conv_block2 = Conv1dBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            skip_connection=True,
            activation=activation,
            dropout=dropout,
        )
        self.pooling = nn.AvgPool1d(kernel_size=(3,), stride=(1,), padding=(1,))

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.conv_block1(x)
        x = self.pooling(x)
        x = self.conv_block2(x)
        x = self.pooling(x)
        x = torch.transpose(x, 1, 2)
        return x


class GemLightCNN1DModel(nn.Module):
    def __init__(self, in_channels, activation="relu", dropout=0.0):
        super().__init__()
        self.conv_block1 = Conv1dBlock(
            in_channels=in_channels,
            out_channels=2 * in_channels,
            skip_connection=True,
            activation=activation,
            dropout=dropout,
        )
        self.conv_block2 = Conv1dBlock(
            in_channels=2 * in_channels,
            out_channels=2 * in_channels,
            skip_connection=True,
            activation=activation,
            dropout=dropout,
        )
        self.conv_block3 = Conv1dBlock(
            in_channels=2 * in_channels,
            out_channels=in_channels,
            skip_connection=True,
            activation=activation,
            dropout=dropout,
        )
        self.pooling = GeM(kernel_size=3)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.conv_block1(x)
        x = self.pooling(x)
        x = self.conv_block2(x)
        x = self.pooling(x)
        x = self.conv_block3(x)
        x = torch.transpose(x, 1, 2)
        return x


class GemVeryLightCNN1DModel(nn.Module):
    def __init__(self, channels, activation="relu", dropout=0.0):
        super().__init__()
        self.convolutions = nn.ModuleList(
            [
                nn.Sequential(
                    Conv1dBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        skip_connection=True,
                        activation=activation,
                        dropout=dropout,
                    ),
                    GeM(kernel_size=3),
                )
                for i, (in_channels, out_channels) in enumerate(channels)
            ]
        )

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        for conv in self.convolutions:
            x = conv(x)
        x = torch.transpose(x, 1, 2)
        return x


class GemVeryLightCNN1DWithDepthMaxPoolModel(nn.Module):
    def __init__(
        self, channels, activation="relu", dropout=0.0, kernel_size=5, gem_kernel_size=3
    ):
        super().__init__()
        self.convolutions = nn.ModuleList(
            [
                nn.Sequential(
                    Conv1dBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        skip_connection=True,
                        activation=activation,
                        dropout=dropout,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    ),
                    GeM(kernel_size=gem_kernel_size),
                )
                for i, (in_channels, out_channels) in enumerate(channels)
            ]
        )

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        hidden_states = []
        for conv in self.convolutions:
            x = conv(x)
            hidden_states.append(x)
        x_max = torch.max(torch.stack(hidden_states), dim=0)[0]
        outs = torch.transpose(x + x_max, 1, 2)
        return outs
