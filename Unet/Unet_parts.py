import torch
from torch import nn

# ConvBlock: Two stacked convolutional layers with batch normalization and ReLU activation function
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv_block(x)
        return x


# Encoder: Max pooling followed by a convolutional block
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


# Decoder: Transposed convolution for upsampling followed by a convolutional block
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv_trans = nn.ConvTranspose2d(in_channels, out_channels,
                                             kernel_size=4, stride=2, padding=1)
        self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1 is the output from the previous layer (decoder)
        # x2 is the corresponding feature map from the encoder (skip connection)
        x1 = self.conv_trans(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv_block(x)
        return x