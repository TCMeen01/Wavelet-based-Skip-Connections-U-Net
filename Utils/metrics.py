import torch
import torch.nn as nn

def dice_score(pred, target, smooth=1e-6):
    '''
    Compute the Dice score between predicted and target tensors.
    Parameters:
        pred (torch.Tensor): The predicted tensor (output of the model: Raw logits).
        target (torch.Tensor): The target tensor.
        smooth (float): A small value to avoid division by zero.
    Returns:
        float: The Dice score.
    '''
    # Apply sigmoid to get probabilities (from raw logits)
    pred = torch.sigmoid(pred)
    
    # Binarize predictions (threshold at 0.5)
    pred = (pred > 0.5).float()

    # Flatten the tensors
    pred = pred.view(-1)
    target = target.view(-1)

    # Intersection
    intersection = (pred * target).sum()

    # Compute Dice Score
    dice = (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return dice.item()


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

        return dice_loss