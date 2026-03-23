import torch

def iou_score(pred, target, smooth=1e-6):
    '''
    Compute the Intersection over Union (IoU) score between predicted and target tensors.
    Parameters:
        pred (torch.Tensor): The predicted tensor (output of the model).
        target (torch.Tensor): The target tensor.
        smooth (float): A small value to avoid division by zero.
    Returns:
        float: The IoU score.
    '''
    # Flatten the tensors
    pred = pred.view(-1)
    target = target.view(-1)

    # Intersection
    intersection = (pred * target).sum()

    # Union
    union = pred.sum() + target.sum() - intersection

    # Compute IoU
    iou = (intersection + smooth) / (union + smooth)

    return iou.item()

def dice_score(pred, target, smooth=1e-6):
    '''
    Compute the Dice score between predicted and target tensors.
    Parameters:
        pred (torch.Tensor): The predicted tensor (output of the model).
        target (torch.Tensor): The target tensor.
        smooth (float): A small value to avoid division by zero.
    Returns:
        float: The Dice score.
    '''
    # Flatten the tensors
    pred = pred.view(-1)
    target = target.view(-1)

    # Intersection
    intersection = (pred * target).sum()

    # Compute Dice Score
    dice = (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return dice.item()