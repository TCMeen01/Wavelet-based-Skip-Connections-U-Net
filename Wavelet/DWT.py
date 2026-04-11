import torch
from torch import nn
from pytorch_wavelets import DWTForward, DWTInverse

class DWTransform(nn.Module):
    def __init__(self, wave='haar', mode='zero'):
        '''
        wave: Type of wavelet to use (e.g., 'haar', 'db1', 'sym2', etc.)
        mode: Signal extension mode (e.g., 'zero', 'symmetric', 'periodic', etc.)
        '''
        super().__init__()

        self.dwt = DWTForward(J=1, wave=wave, mode=mode)
        self.idwt = DWTInverse(wave=wave, mode=mode)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        '''
        Perform the forward wavelet transform
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W):
                N: Batch size
                C: Number of channels
                H: Height of the image
                W: Width of the image
        Returns:
            Yl (torch.Tensor): Low-frequency component of shape (N, C, H/2, W/2)
            Yh (torch.Tensor): High-frequency components of shape (N, 3*C, H/2, W/2)
        '''
        # Perform the forward wavelet transform
        Yl, Yh = self.dwt(x)

        # Yl is the low-frequency component, and Yh contains the high-frequency components (LH, HL, HH)
        lh, hl, hh = Yh[0][:, :, 0], Yh[0][:, :, 1], Yh[0][:, :, 2]

        # Concat the high-frequency components along the channel dimension
        Yh = torch.cat([lh, hl, hh], dim=1) # shape (N, 3*C, H/2, W/2)
        
        return Yl, Yh

    def inverse(self, Yl, Yh):
        '''
        Perform the inverse wavelet transform
        Args:
            Yl (torch.Tensor): Low-frequency component of shape (N, C, H/2, W/2)
            Yh (torch.Tensor): High-frequency components of shape (N, 3*C, H/2, W/2)
        Returns:
            x_reconstructed (torch.Tensor): Reconstructed image of shape (N, C, H, W)
        '''
        # Split the high-frequency components back into LH, HL, HH
        C = Yl.shape[1] # Number of channels in the low-frequency component

        lh = Yh[:, :C, :, :]   # shape (N, C, H/2, W/2)
        hl = Yh[:, C:2*C, :, :] # shape (N, C, H/2, W/2)
        hh = Yh[:, 2*C:, :, :] # shape (N, C, H/2, W/2)

        # Stack the high-frequency components into the format expected by the inverse DWT
        Yh = torch.stack([lh, hl, hh], dim=2) # shape (N, C, 3, H/2, W/2)

        # Perform the inverse wavelet transform
        x_reconstructed = self.idwt((Yl, [Yh]))

        return x_reconstructed