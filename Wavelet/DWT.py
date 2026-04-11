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
            ll (torch.Tensor): low-frequency component of shape (N, C, H/2, W/2)
            tuple(lh, hl, hh):
                lh (torch.Tensor): low-high-frequency components of shape (N, C, H/2, W/2)
                hl (torch.Tensor): high-low-frequency components of shape (N, C, H/2, W/2)
                hh (torch.Tensor): high-frequency components of shape (N, C, H/2, W/2)
        '''
        # Perform the forward wavelet transform
        ll, Yh = self.dwt(x)

        # Yl is the low-frequency component, and Yh contains the high-frequency components (LH, HL, HH)
        lh, hl, hh = Yh[0][:, :, 0].contiguous(), Yh[0][:, :, 1].contiguous(), Yh[0][:, :, 2].contiguous()
        
        return ll, (lh, hl, hh)

    def inverse(self, yl, yh):
        '''
        Perform the inverse wavelet transform
        Args:
            yl (torch.Tensor): low-frequency component of shape (N, C, H/2, W/2)
            yh (tuple): A tuple containing the high-frequency components (lh, hl, hh):
                lh (torch.Tensor): low-high-frequency components of shape (N, C, H/2, W/2)
                hl (torch.Tensor): high-low-frequency components of shape (N, C, H/2, W/2)
                hh (torch.Tensor): high-frequency components of shape (N, C, H/2, W/2)
        Returns:
            x_reconstructed (torch.Tensor): Reconstructed image of shape (N, C, H, W)
        '''
        (lh, hl, hh) = yh

        # Stack the high-frequency components into the format expected by the inverse DWT
        Yh = torch.stack([lh, hl, hh], dim=2) # shape (N, C, 3, H/2, W/2)

        # Perform the inverse wavelet transform
        x_reconstructed = self.idwt((yl, [Yh]))

        return x_reconstructed