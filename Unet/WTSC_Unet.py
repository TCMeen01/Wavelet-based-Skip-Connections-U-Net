import torch
from torch import nn
from Wavelet.DWT import DWTransform
from Wavelet.DTCWT import DTCWTransform
from Unet.Unet_parts import ConvBlock, Encoder, Decoder

class DWTSC_UNet(nn.Module):
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

        # Wavelet weights for high-frequency components (learnable parameters)
        self.wavelet_weights = nn.ParameterList([
            nn.Parameter(torch.tensor([0.5, 0.5])) for _ in range(len(self.dec))  # One weight for LL, one for (LH + HL + HH)
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
            # Apply wavelet transform to the corresponding encoder feature map for skip connection
            ll, (lh, hl, hh) = self.wavelet(features[len(self.enc)-1-i])  # ll: low-frequency, lh: low-high-frequency, hl: high-low-frequency, hh: high-frequency

            # Calculate learnable weighted sum of high-feature
            normalized_weights = torch.softmax(self.wavelet_weights[i], dim=0)  # Normalize weights to sum to 1 using soft-max

            ll_weighted = normalized_weights[0] * ll
            lh_weighted = normalized_weights[1] * lh
            hl_weighted = normalized_weights[1] * hl
            hh_weighted = normalized_weights[1] * hh 

            features_weighted = self.wavelet.inverse(ll_weighted, (lh_weighted, hl_weighted, hh_weighted))  # Reconstruct the feature map from wavelet components

            y = self.dec[i](y, features_weighted)  # Pass the weighted skip connection to the decoder

        # Final output layer
        out = self.out_conv(y)

        return out
    

class DTCWTSC_UNet(nn.Module):
    list_channels = [64, 128, 256, 512, 1024]

    def __init__(self, n_channels, n_classes) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Using DTCWT for skip connections (J=1, 6 orientations)
        self.wavelet = DTCWTransform()
        
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

        # Extract Attention Map from the weighted skip connection using a simple 1x1 convolution followed by sigmoid activation
        self.attention = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=1),
                nn.Sigmoid()
            ) for _ in range(len(self.enc))
        ])

        # Wavelet weights for high-frequency components (learnable parameters)
        self.wavelet_weights = nn.ParameterList([
            nn.Parameter(torch.tensor(6.0)) for _ in range(len(self.dec))
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
            yl, yh_real, yh_imag = self.wavelet(compressed_feature)  # yl: low-frequency, yh_real: real part of high-frequency, yh_imag: imaginary part of high-frequency

            # Weights the low-frequency component
            weight_low = torch.sigmoid(self.wavelet_weights[i])  # Learnable weight for low-frequency component
            yl_weighted = weight_low * yl

            # Reconstruct the feature map from wavelet components
            features_weighted = self.wavelet.inverse(yl_weighted, yh_real, yh_imag)  # Reconstruct the feature map from wavelet components

            # Construct Attention Map from the weighted skip connection using a simple 1x1 convolution followed by sigmoid activation
            attention_map = self.attention[idx](features_weighted)

            # Apply the attention map to the original encoder feature map (before compression) to get the final weighted skip connection
            features_weighted = encoder_feature * attention_map

            y = self.dec[i](y, features_weighted)  # Pass the weighted skip connection to the decoder

        # Final output layer
        out = self.out_conv(y)

        return out