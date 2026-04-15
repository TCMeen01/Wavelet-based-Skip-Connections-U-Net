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
    
class BCEDiceLoss(nn.Module):
    """
    Combine BCE and Dice Loss for training binary segmentation models.
    """
    def __init__(self, weight_bce=0.5, weight_dice=0.5, smooth=1e-6):
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        
        # BCE loss (with raw logit)
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Dice loss
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw output from the model (before sigmoid).
            targets (torch.Tensor): Ground truth binary mask (0 and 1).
        Returns:
            torch.Tensor: The combined BCE and Dice loss.
        """
        # 1. Calc BCE loss
        bce = self.bce_loss(logits, targets.float())
        
        # 2. Calc Dice loss
        dice = self.dice_loss(logits, targets)
        
        # 3. Combine Loss
        combined_loss = (self.weight_bce * bce) + (self.weight_dice * dice)
        
        return combined_loss
    
class BoundaryLoss(nn.Module):
    '''
    Boundary Loss for binary segmentation tasks, which penalizes predictions based on their distance from the true boundary.
    This loss is used for fine-tuning the model to better capture the boundaries of the segmented objects.
    '''
    def __init__(self):
        super().__init__()

    def forward(self, logits, dist_maps):
        """
        Args:
            logits (torch.Tensor): Raw output from the model (before sigmoid), shape (B, 1, H, W)
            dist_maps (torch.Tensor): Precomputed distance maps for the ground truth masks, shape (B, 1, H, W).
        Returns:
            torch.Tensor: The scalar loss value.
        """
        # Convert logits to probabilities using sigmoid
        probs = torch.sigmoid(logits)
        
        # Multiply element-wise between probabilities and distance maps to get the boundary loss
        loss = probs * dist_maps
        
        # Return the mean of the entire batch
        return loss.mean()