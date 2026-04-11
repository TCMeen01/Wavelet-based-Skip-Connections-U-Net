import torch
from torch import nn
from pytorch_wavelets import DTCWTForward, DTCWTInverse

class DTCWTransform(nn.Module):
    def __init__(self, biort='near_sym_b', qshift='qshift_b'):
        """
        Dual-Tree Complex Wavelet Transform (DTCWT) module for U-Net.
        
        Args:
            biort (str): Biorthogonal wavelet type.
            qshift (str): Q-shift wavelet type.
        """
        super().__init__()

        # J=1 for single-level decomposition
        self.dtcwt = DTCWTForward(J=1, biort=biort, qshift=qshift)
        self.idtcwt = DTCWTInverse(biort=biort, qshift=qshift)

        # Freeze wavelet parameters as they represent fixed mathematical operations
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Perform the forward 2D Dual-Tree Complex Wavelet Transform.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            yl (torch.Tensor): Low-frequency component of shape (N, C, H, W)
                (Note: For J=1, DTCWT lowpass has the SAME spatial size as input,
                but captures low-frequency information).
            
            yh_real (tuple of 6 Tensors): Real parts of the 6 directional 
                high-frequency components. Each has shape (N, C, H/2, W/2).
            yh_imag (tuple of 6 Tensors): Imaginary parts of the 6 directional 
                high-frequency components. Each has shape (N, C, H/2, W/2).
        """
        # Perform the forward transform
        yl, yh = self.dtcwt(x)

        # yh is a list of lists. For J=1, we access yh[0].
        # Shape of yh_level1: (N, C, 6, H/2, W/2, 2)
        yh_level1 = yh[0]

        # Extract Real and Imaginary parts: (N, C, 6, H/2, W/2)
        real_parts = yh_level1[..., 0]
        imag_parts = yh_level1[..., 1]

        # Split the 6 orientations into separate tensors
        # d1 to d6 represent the 6 directional subbands (+15, -15, +45, -45, +75, -75 degrees)
        yh_real = (
            real_parts[:, :, 0].contiguous(),
            real_parts[:, :, 1].contiguous(),
            real_parts[:, :, 2].contiguous(),
            real_parts[:, :, 3].contiguous(),
            real_parts[:, :, 4].contiguous(),
            real_parts[:, :, 5].contiguous()
        )

        yh_imag = (
            imag_parts[:, :, 0].contiguous(),
            imag_parts[:, :, 1].contiguous(),
            imag_parts[:, :, 2].contiguous(),
            imag_parts[:, :, 3].contiguous(),
            imag_parts[:, :, 4].contiguous(),
            imag_parts[:, :, 5].contiguous()
        )

        return yl, yh_real, yh_imag

    def inverse(self, yl, yh_real, yh_imag):
        """
        Perform the inverse 2D Dual-Tree Complex Wavelet Transform.

        Args:
            yl (torch.Tensor): Low-frequency component of shape (N, C, H, W)
            yh_real (tuple of 6 Tensors): Real parts of high-frequency components.
            yh_imag (tuple of 6 Tensors): Imaginary parts of high-frequency components.

        Returns:
            x_reconstructed (torch.Tensor): Reconstructed image of shape (N, C, H, W)
        """
        # Stack the 6 directional tensors back together along the orientation dim (dim=2)
        # Output shape for real and imag: (N, C, 6, H/2, W/2)
        real_stacked = torch.stack(yh_real, dim=2)
        imag_stacked = torch.stack(yh_imag, dim=2)

        # Combine real and imaginary parts along the last dimension
        # Output shape: (N, C, 6, H/2, W/2, 2)
        yh_combined = torch.stack([real_stacked, imag_stacked], dim=-1)

        # Perform the inverse transform
        # DTCWTInverse expects a tuple (yl, [yh_level1, yh_level2, ...])
        x_reconstructed = self.idtcwt((yl, [yh_combined]))

        return x_reconstructed