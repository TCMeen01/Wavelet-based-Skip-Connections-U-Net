import torch
from torch import nn
from pytorch_wavelets import DTCWTForward, DTCWTInverse

class DTCWTransform(nn.Module):
    def __init__(self, biort='near_dist_b', qshift='qshift_b'):
        """
        biort: Biorthogonal wavelet type (e.g., 'near_dist_b', 'near_sym_a', etc.)
        qshift: Q-shift wavelet type (e.g., 'qshift_b', 'qshift_a', etc.)
        """
        super(DTCWTransform, self).__init__()

        self.xfm = DTCWTForward(J=1, biort=biort, qshift=qshift)
        self.ifm = DTCWTInverse(biort=biort, qshift=qshift)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Perform the forward dual-tree complex wavelet transform
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W)
        Returns:             
            yl (torch.Tensor): Low-frequency component of shape (N, C, H/2, W/2)
            yh_real (torch.Tensor): Real part of high-frequency components of shape (N, 6*C, H/2, W/2)
            yh_imag (torch.Tensor): Imaginary part of high-frequency components of shape (N, 6*C, H/2, W/2)
        """
        # Perform the forward dual-tree complex wavelet transform
        yl, yh = self.xfm(x)

        # yh is a list of high-frequency components for each level, we only have one level (J=1). Shape (N, C, 6, H/2, W/2, 2)
        yh_real = yh[0][..., 0]
        yh_imag = yh[0][..., 1]

        # Concat the real and imaginary parts along the channel dimension
        yh_real = yh_real.view(yh_real.shape[0], -1, yh_real.shape[3], yh_real.shape[4]) # shape (N, 6*C, H/2, W/2)
        yh_imag = yh_imag.view(yh_imag.shape[0], -1, yh_imag.shape[3], yh_imag.shape[4]) # shape (N, 6*C, H/2, W/2)

        return yl, yh_real, yh_imag

    def inverse(self, yl, yh_real, yh_imag):
        """
        Perform the inverse dual-tree complex wavelet transform
        Args:
            yl (torch.Tensor): Low-frequency component of shape (N, C, H/2, W/2)
            yh_real (torch.Tensor): Real part of high-frequency components of shape (N, 6*C, H/2, W/2)
            yh_imag (torch.Tensor): Imaginary part of high-frequency components of shape (N, 6*C, H/2, W/2)
        Returns:
            x_reconstructed (torch.Tensor): Reconstructed image of shape (N, C, H, W)
        """
        N, C_total, H, W = yh_real.shape
        C = yl.shape[1] # Number of channels in the low-frequency component
        
        # Reshape real and imaginary parts back to (N, C, 6, H/2, W/2)
        real = yh_real.reshape(N, C, 6, H//2, W//2)
        imag = yh_imag.reshape(N, C, 6, H//2, W//2)

        # Combine real and imaginary parts into the format expected by the inverse DTCWT
        yh_combined = torch.stack([real, imag], dim=-1)
        
        # DTCWTInverse mong đợi list các level
        x_reconstructed = self.ifm((yl, [yh_combined]))

        return x_reconstructed