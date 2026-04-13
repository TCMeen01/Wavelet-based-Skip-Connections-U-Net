import torch
import torch.nn as nn
from monai.metrics import compute_hausdorff_distance

class DiceLoss(nn.Module):
    '''
    Dice Loss for binary segmentation tasks.
    '''
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        '''
        Compute the Dice loss between predicted and target tensors.
        Parameters:
            pred (torch.Tensor): The predicted tensor (output of the model: Raw logits).
            target (torch.Tensor): The target tensor.
        Returns:
            torch.Tensor: The Dice loss.
        '''

        # Apply sigmoid to get probabilities
        pred = torch.sigmoid(pred)

        # Flatten the tensors
        pred = pred.view(-1)
        target = target.view(-1)

        # Intersection
        intersection = (pred * target).sum()

        # Compute Dice Loss
        dice_loss = 1 - (2 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        # Return the average Dice loss for the batch
        return dice_loss
    

class HD95Loss(nn.Module):
    '''
    HD95 Loss for binary segmentation tasks.
    '''
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        '''
        Compute the HD95 loss between predicted and target tensors.
        Parameters:
            pred (torch.Tensor): The predicted tensor (output of the model: Raw logits).
            target (torch.Tensor): The target tensor.
        Returns:
            torch.Tensor: The HD95 loss.
        '''        
        # Convert raw logits from the U-Net into probabilities (0 to 1)
        pred = torch.sigmoid(pred)

        # Calculate HD95 using MONAI
        # percentile=95 computes the 95th percentile
        # include_background=True assuming the passed channel directly contains the target mask
        hd95_tensor = compute_hausdorff_distance(
            y_pred=pred, 
            y=target, 
            include_background=True, 
            percentile=95
        )

        # Return the average HD95 score for the batch as a loss (higher HD95 means worse performance)
        return hd95_tensor.mean()