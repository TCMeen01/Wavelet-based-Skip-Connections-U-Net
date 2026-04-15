import matplotlib.pyplot as plt
import random

from Utils.metrics import dice_score
import numpy as np
from scipy.ndimage import distance_transform_edt

def compute_distance_map(mask_np):
    """
    Compute the distance map for a given binary mask.
    Args:
        mask_np (numpy.ndarray): A 2D binary numpy array where foreground pixels are 1 and background pixels are 0.
    Returns:
        numpy.ndarray: A 2D array of the same shape as mask_np containing the distance map values.
    """
    posmask = mask_np.astype(bool)
    
    if posmask.any(): # If there are any foreground pixels
        negmask = ~posmask
        res = distance_transform_edt(negmask) * negmask - (distance_transform_edt(posmask) - 1) * posmask
    else: # If there are no foreground pixels
        res = np.zeros_like(mask_np)
        
    return res.astype(np.float32)

def random_visualize(loader, title="Dataset Sample", n_samples=3):
    '''
    Fetch a batch from the dataloader and visualize 'n_samples' random image-mask pairs.
    Assumes the image has been normalized with mean=0.5 and std=0.5.

    Args:
        loader (torch.utils.data.DataLoader): The DataLoader to sample from.
        title (str): Title prefix for the plot.
        n_samples (int): Number of random samples to visualize. Default is 3.
    '''
    # Fetch the first batch of data
    images, masks = next(iter(loader))
    
    # Ensure n_samples does not exceed the actual batch size
    batch_size = images.shape[0]
    n_samples = min(n_samples, batch_size)
    
    # Randomly select indices from the batch
    random_indices = random.sample(range(batch_size), n_samples)
    
    # Create a large figure: rows = n_samples, columns = 2 (Image and Mask)
    plt.figure(figsize=(10, 4 * n_samples))
    
    for i, idx in enumerate(random_indices):
        # Process the Image
        img = images[idx].permute(1, 2, 0).cpu().numpy()
        img = img * 0.5 + 0.5  # Denormalize
        img = img.clip(0, 1)   # Ensure valid range for matplotlib
        
        # Process the Mask
        mask = masks[idx][0].cpu().numpy()
        
        # Plotting
        # Position the Image on the subplot grid (Left column: 1, 3, 5...)
        plt.subplot(n_samples, 2, 2 * i + 1)
        plt.imshow(img)
        plt.title(f"{title} - Image (Batch Idx: {idx})")
        plt.axis("off")
        
        # Position the Mask on the subplot grid (Right column: 2, 4, 6...)
        plt.subplot(n_samples, 2, 2 * i + 2)
        plt.imshow(mask, cmap="gray")
        plt.title(f"{title} - Mask (Batch Idx: {idx})")
        plt.axis("off")
        
    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    '''
    Plot the training and validation loss and dice score over epochs.

    Args:
        history (dict): A dictionary containing lists of 'train_loss', 'train_dice', 'val_loss', and 'val_dice'.
    Returns:
        None: Displays the plots of training history.
    '''
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    plt.plot(epochs, history['val_loss'], label='Val Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    # Plot Dice Score
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_dice'], label='Train Dice Score', marker='o')
    plt.plot(epochs, history['val_dice'], label='Val Dice Score', marker='o')
    plt.title('Training and Validation Dice Score')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()