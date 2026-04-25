import torch
import torch.nn as nn
from monai.metrics import compute_hausdorff_distance

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

def iou_score(pred, target, smooth=1e-6):
    '''
    Compute the Intersection over Union (IoU) score between predicted and target tensors.
    Parameters:
        pred (torch.Tensor): The predicted tensor (output of the model: Raw logits).
        target (torch.Tensor): The target tensor.
        smooth (float): A small value to avoid division by zero.
    Returns:
        float: The IoU score.
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

    # Union
    union = pred.sum() + target.sum() - intersection

    # Compute IoU Score
    iou = (intersection + smooth) / (union + smooth)

    return iou.item()

def hd95_score(pred, target):
    """
    Compute the 95th percentile Hausdorff Distance (HD95) between the predicted mask 
    and the ground truth target using MONAI.
    
    Args:
        pred (torch.Tensor): Raw logits output from the model. Shape: (Batch, Channel, H, W)
        target (torch.Tensor): Ground truth binary mask. Shape: (Batch, Channel, H, W)

    Returns:
        float: The calculated average HD95 score for the batch.
    """
    # Convert raw logits from the U-Net into probabilities (0 to 1)
    pred = torch.sigmoid(pred)
    # Binarize the predictions and target using a 0.5 threshold
    pred = (pred > 0.5).float()

    # Move to CPU
    pred = pred.cpu()
    target = target.cpu()
    
    # Calculate HD95 using MONAI
    # percentile=95 computes the 95th percentile
    # include_background=True assuming the passed channel directly contains the target mask
    hd95_tensor = compute_hausdorff_distance(
        y_pred=pred, 
        y=target, 
        include_background=True, 
        percentile=95
    )

    # Filter out NaN or Inf values 
    valid_hd95 = hd95_tensor[~torch.isnan(hd95_tensor) & ~torch.isinf(hd95_tensor)]

    # If the entire batch is invalid/empty, return a default value 999 for infty
    if len(valid_hd95) == 0:
        return 999.0

    # Return the mean HD95 of the valid batch as a float
    return valid_hd95.mean().item()