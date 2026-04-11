from torch import nn
from Wavelet.DWT import DWTransform
from Wavelet.DTCWT import DTCWTransform
from Unet.Unet_parts import ConvBlock, Encoder, Decoder

class WTSC_UNet(nn.Module):
    list_channels = [64, 128, 256, 512, 1024]

    def __init__(self, n_channels, n_classes) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Using Discrete Wavelet Transform for skip connections (Haar wavelet)
        self.wavelet = DWTransform(wave='haar') 
        
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

        # Decoder path with wavelet skip connections
        y = features[-1]  # Start from the bottleneck feature map (1024 channels)

        for i in range(len(self.dec)):
            # Apply wavelet transform to the corresponding encoder feature map for skip connection
            Yl, Yh = self.wavelet(features[len(self.enc)-1-i])  # Yl: low-frequency, Yh: high-frequency
            
            # Resize Yh if necessary to match the spatial dimensions of the decoder input
            if Yh.shape[2:] != y.shape[2:]:
                Yh = nn.functional.interpolate(Yh, size=y.shape[2:], mode='bilinear', align_corners=False)

            y = self.dec[i](y, Yh)  # Using high-frequency component for skip connection

        # Final output layer
        out = self.out_conv(y)

        return out