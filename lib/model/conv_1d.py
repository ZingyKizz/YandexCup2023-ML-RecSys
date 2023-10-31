import torch
from torch import nn


class Conv1dBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(5,),
        stride=(1,),
        padding=(2,),
        skip_connection=False,
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
            nn.ReLU(),
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
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.conv_block(x)
        if self.skip_connection:
            x = self.downsample(x)
            output = output + x
        output = self.relu(output)
        return output


class CNN1DModel(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_block1 = Conv1dBlock(
            in_channels=in_channels, out_channels=2 * in_channels, skip_connection=True
        )
        self.conv_block2 = Conv1dBlock(
            in_channels=2 * in_channels,
            out_channels=4 * in_channels,
            skip_connection=True,
        )
        self.conv_block3 = Conv1dBlock(
            in_channels=4 * in_channels,
            out_channels=4 * in_channels,
            skip_connection=True,
        )
        self.conv_block4 = Conv1dBlock(
            in_channels=4 * in_channels,
            out_channels=2 * in_channels,
            skip_connection=True,
        )
        self.conv_block5 = Conv1dBlock(
            in_channels=2 * in_channels, out_channels=in_channels, skip_connection=True
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
    def __init__(self, in_channels):
        super().__init__()
        self.conv_block1 = Conv1dBlock(
            in_channels=in_channels, out_channels=2 * in_channels, skip_connection=True
        )
        self.conv_block2 = Conv1dBlock(
            in_channels=2 * in_channels,
            out_channels=2 * in_channels,
            skip_connection=True,
        )
        self.conv_block3 = Conv1dBlock(
            in_channels=2 * in_channels, out_channels=in_channels, skip_connection=True
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
    def __init__(self, in_channels):
        super().__init__()
        self.conv_block1 = Conv1dBlock(
            in_channels=in_channels, out_channels=in_channels, skip_connection=True
        )
        self.conv_block2 = Conv1dBlock(
            in_channels=in_channels, out_channels=in_channels, skip_connection=True
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
