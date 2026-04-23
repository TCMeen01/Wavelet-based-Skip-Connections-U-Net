import torch
from torch import nn
import torch.nn.functional as F
from pytorch_wavelets import DTCWTForward, DTCWTInverse

class DTCWTransform(nn.Module):
    def __init__(self, level=1, biort='near_sym_b', qshift='qshift_b'):
        """
        Dual-Tree Complex Wavelet Transform (DTCWT) module for U-Net.
        
        Args:
            level (int): The level of decomposition.
            biort (str): Biorthogonal wavelet type.
            qshift (str): Q-shift wavelet type.
        """
        super().__init__()

        self.level = level
        self.dtcwt = DTCWTForward(J=level, biort=biort, qshift=qshift)
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
            yl (torch.Tensor): Low-frequency component of shape (N, C, H/2^level, W/2^level)
            yh (list of high-frequency components): yh[k] has shape (N, C, 6, H/2^k, W/2^k, 2) for k in [0, level) where:
                yh[k][..., 0] is the real part
                yh[k][..., 1] is the imaginary part
        """
        # Perform the forward transform
        yl, yh = self.dtcwt(x)

        # Return low-frequency component and high-frequency components
        return yl, yh

    def get_real_imag(self, yh, level=1):
        """
        Utility function to extract real and imaginary parts from the high-frequency components.

        Args:
            yh (list of torch.Tensor): High-frequency components.
            level (int): The level of decomposition to extract from (default is 1 for the first level).

        Returns:
            yh_real (list of 6 Tensors): Real parts of high-frequency components.
            yh_imag (list of 6 Tensors): Imaginary parts of high-frequency components.
        """
        # Extract the specified level's high-frequency components
        yh_level = yh[level - 1]  # Shape: (N, C, 6, H/2^level, W/2^level, 2)

        # Split real and imaginary parts
        real_parts = yh_level[..., 0]
        imag_parts = yh_level[..., 1]

        # Unbind the 6 directional subbands into lists
        yh_real = list(torch.unbind(real_parts, dim=2))
        yh_imag = list(torch.unbind(imag_parts, dim=2))

        # Return as lists for easier handling
        return yh_real, yh_imag

    def inverse(self, yl, yh, list_levels=None):
        """
        Perform the inverse 2D Dual-Tree Complex Wavelet Transform.

        Args:
            yl (torch.Tensor): Low-frequency component of shape (N, C, H, W)
            yh (list of high-frequency components): yh[k] has shape (N, C, 6, H/2^k, W/2^k, 2) for k in [0, level) where:
                yh[k][..., 0] is the real part
                yh[k][..., 1] is the imaginary part
            list_levels (list of int): List of levels to include in the inverse transform. Default is None (include all levels).

        Returns:
            x_reconstructed (torch.Tensor): Reconstructed image of shape (N, C, H, W)
        """
        # If list_levels is None, include all levels
        list_levels = list_levels if list_levels else list(range(1, self.level + 1))

        # Remove levels that are not in list_levels
        list_levels = set(list_levels) # Convert to set for faster lookup
        yh_filtered = [yh[k] if k + 1 in list_levels else None for k in range(self.level)]

        # Perform the inverse transform
        # DTCWTInverse expects a tuple (yl, [yh_level1, yh_level2, ...])
        x_reconstructed = self.idtcwt((yl, yh_filtered))

        return x_reconstructed