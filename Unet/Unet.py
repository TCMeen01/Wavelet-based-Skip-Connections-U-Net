from torch import nn
from Unet.Unet_parts import ConvBlock, Encoder, Decoder

# U-Net architecture for image segmentation
class Unet(nn.Module):
    # List of channels for each layer in the encoder and decoder
    list_channels = [64, 128, 256, 512, 1024]

    def __init__(self, n_channels, n_classes) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # First convolution block to process the input image
        self.in_conv = ConvBlock(n_channels, self.list_channels[0])

        # Initialize the list of Encoders
        self.enc = nn.ModuleList()
        for i in range(1, len(self.list_channels)):
            self.enc.append(Encoder(self.list_channels[i-1], self.list_channels[i]))

        # Initialize the list of Decoder
        self.dec = nn.ModuleList()
        for i in range(len(self.list_channels)-1, 0, -1):
            self.dec.append(Decoder(self.list_channels[i], self.list_channels[i-1]))

        # Final output convolution to get the desired number of classes
        self.out_conv = nn.Conv2d(self.list_channels[0], n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path
        x0 = self.in_conv(x)
        
        features = [x0]
        for i in range(len(self.enc)):
            features.append(self.enc[i](features[-1]))

        # Decoder path with skip connections
        y = features[-1]  # Start from the bottleneck feature map (1024 channels)
        for i in range(len(self.dec)):
            y = self.dec[i](y, features[len(self.enc)-1-i])  # Using corresponding encoder feature for skip connection

        # Final output layer
        out = self.out_conv(y)

        return out