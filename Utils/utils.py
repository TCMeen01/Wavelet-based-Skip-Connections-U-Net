import torch
from tqdm import tqdm
import torch.optim as optim
import time
import copy

import matplotlib.pyplot as plt
import random

from Utils.metrics import dice_score

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

def train_model(model, train_loader, val_loader, epochs, learning_rate, criterion, device):
    """
    Train a PyTorch model for Binary Image Segmentation.

    Args:
        model (nn.Module): The neural network model (e.g., U-Net).
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for the optimizer.
        criterion (nn.Module): Loss function (e.g., BCEWithLogitsLoss, DiceLoss).
        device (torch.device): Device to run the training on ('cuda' or 'cpu').

    Returns:
        tuple: (best_model_state, history)
            - best_model_state (dict): The state dictionary of the best model.
            - history (dict): Lists of train/val loss and dice scores for plotting.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Dictionary to keep track of the best model based on validation performance
    best_model = {
        'Epoch': -1,
        'Train Loss': float('inf'),
        'Train Dice': 0.0,
        'Val Loss': float('inf'),
        'Val Dice': 0.0,
        'Model_State': None
    }
    
    # Dictionary to save the history of metrics for line plotting later
    history = {
        'train_loss': [],
        'train_dice': [],
        'val_loss': [],
        'val_dice': []
    }
    
    print(f'Training using device: {device} ...\n' + '-'*60)
    
    for epoch in range(epochs):
        # Record the start time of the current epoch
        start_time = time.time()
        
        # TRAINING PHASE
        model.train()
        running_train_loss = 0.0
        running_train_dice = 0.0

        # Wrap train_loader with tqdm for a visual progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{epochs:03d} [Train]", leave=False)
        
        for images, masks in train_pbar:
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass
            outputs = model(images) # raw logits
            loss = criterion(outputs, masks)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()      
            optimizer.step()     
            
            # Accumulate metrics
            running_train_loss += loss.item()
            running_train_dice += dice_score(outputs.detach(), masks)
                        
        # Calculate average metrics for the training epoch
        epoch_train_loss = running_train_loss / len(train_loader)
        epoch_train_dice = running_train_dice / len(train_loader)
        
        # VALIDATION PHASE
        model.eval()
        running_val_loss = 0.0
        running_val_dice = 0.0
        
        # Wrap val_loader with tqdm for a visual progress bar
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1:03d}/{epochs:03d} [Val]  ", leave=False)
        
        with torch.no_grad(): 
            for images, masks in val_pbar:
                images, masks = images.to(device), masks.to(device)
                
                # Forward pass only (no gradients needed)
                outputs = model(images) # raw logits
                loss = criterion(outputs, masks)
                
                # Accumulate metrics
                running_val_loss += loss.item()
                running_val_dice += dice_score(outputs, masks)
                
        # Calculate average metrics for the validation epoch
        epoch_val_loss = running_val_loss / len(val_loader)
        epoch_val_dice = running_val_dice / len(val_loader)

        # Calculate the total elapsed time for the epoch
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        # Append current epoch metrics to the history dictionary
        history['train_loss'].append(epoch_train_loss)
        history['train_dice'].append(epoch_train_dice)
        history['val_loss'].append(epoch_val_loss)
        history['val_dice'].append(epoch_val_dice)

        # Save the model if it achieves the best Validation Dice Score so far
        if epoch_val_dice > best_model['Val Dice']:
            best_model = {
                'Epoch': epoch + 1,
                'Train Loss': epoch_train_loss,
                'Train Dice': epoch_train_dice,
                'Val Loss': epoch_val_loss,
                'Val Dice': epoch_val_dice,
                'Model_State': copy.deepcopy(model.state_dict())
            }
        
        # Print a summarized log of the epoch, including the elapsed time
        print(f"Epoch [{epoch+1:03d}/{epochs:03d}] | Time: {int(epoch_mins)}m {int(epoch_secs)}s | "
              f"Train Loss: {epoch_train_loss:.4f} - Train Dice: {epoch_train_dice:.4f} || "
              f"Val Loss: {epoch_val_loss:.4f} - Val Dice: {epoch_val_dice:.4f}")

    # Print final training summary
    print('-'*70)
    print(f"Training Completed. Best Model found at Epoch {best_model['Epoch']}:")
    print(f"Val Dice: {best_model['Val Dice']:.4f} | Val Loss: {best_model['Val Loss']:.4f}")
    
    # Return both the best model's state dictionary and the training history
    return best_model['Model_State'], history