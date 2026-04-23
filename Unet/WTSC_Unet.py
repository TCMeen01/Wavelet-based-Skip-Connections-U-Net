import torch
from torch import nn
from Wavelet.DTCWT import DTCWTransform
from Unet.Unet_parts import ConvBlock, Encoder, Decoder

class DTCWTSC_UNet(nn.Module):
    list_channels = [64, 128, 256, 512, 1024]

    def __init__(self, n_channels, n_classes, wavelet_level=1) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.wavelet_level = wavelet_level

        # Using DTCWT for skip connections
        self.wavelet = DTCWTransform(level=self.wavelet_level)
        
        # First convolution block to process the input image
        self.in_conv = ConvBlock(n_channels, self.list_channels[0])

        # Initialize the list of Encoders
        self.enc = nn.ModuleList([
            Encoder(self.list_channels[i-1], self.list_channels[i]) 
            for i in range(1, len(self.list_channels))
        ])

        # Initialize the list of Decoder
        self.dec = nn.ModuleList([
            Decoder(self.list_channels[i], self.list_channels[i-1]) 
            for i in range(len(self.list_channels)-1, 0, -1)
        ])

        # Final output convolution to get the desired number of classes
        self.out_conv = nn.Conv2d(self.list_channels[0], n_classes, kernel_size=1)

        # Compress the encoder feature maps to a single channel (grayscale) for wavelet processing
        self.compress = nn.ModuleList([
            nn.Conv2d(self.list_channels[i], 1, kernel_size=1) for i in range(len(self.enc))
        ])

        # Scaler the edge map (learnable)
        self.edge_scales = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in range(len(self.enc))
        ])

        # Wavelet weights for low-frequency components (learnable parameters)
        self.wavelet_weights = nn.ParameterList([
            nn.Parameter(torch.tensor([0.1])) for _ in range(len(self.dec))
        ])

    def forward(self, x):
        # Encoder path
        x0 = self.in_conv(x)
        
        features = [x0]
        for i in range(len(self.enc)):
            features.append(self.enc[i](features[-1]))

        # Decoder path with wavelet skip connections
        y = features[-1]  # Start from the bottleneck feature map (1024 channels)
        
        for i in range(len(self.dec)):
            idx = len(self.enc) - 1 - i  # Corresponding encoder index for skip connection

            # Get the corresponding encoder feature map for skip connection
            encoder_feature = features[idx]

            # Compress the encoder feature map to a single channel (grayscale) for wavelet processing
            compressed_feature = self.compress[idx](encoder_feature)

            # Apply wavelet transform to the corresponding compressed feature map for skip connection
            yl, yh = self.wavelet(compressed_feature)  # yl: low-frequency, yh: high-frequency components

            # Weights the low-frequency component
            yl_weighted = self.wavelet_weights[i] * yl

            # Reconstruct the feature map from wavelet components
            features_weighted = self.wavelet.inverse(yl_weighted, yh)
            
            # Construct Attention Map
            edge_map = torch.abs(features_weighted)
            attention_map = torch.sigmoid(edge_map * self.edge_scales[idx])

            # Apply the attention map to the original encoder feature map (before compression) to get the final weighted skip connection
            features_weighted = encoder_feature + (encoder_feature * attention_map)

            y = self.dec[i](y, features_weighted)  # Pass the weighted skip connection to the decoder

        # Final output layer
        out = self.out_conv(y)

        return out