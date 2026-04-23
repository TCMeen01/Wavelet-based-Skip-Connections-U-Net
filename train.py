import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import time
import copy
import tqdm
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from Unet.Unet import Unet
from Unet.WTSC_Unet import DTCWTSC_UNet
from DataHandle.DataLoader import *
from DataHandle.Dataset import *
from Utils.objectives import DiceLoss
from Utils.utils import *

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNet or DWTSC-UNet")

    # Required args
    parser.add_argument("--model", type=str, required=True, help="Model architecture to train ('Unet' or 'DTCWTSC_UNet')")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset ('ISIC' or 'Kvasir')")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset directory")

    # Optional args with defaults
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--img_size", type=int, default=256, help="Size to resize images and masks (assumes square size img_size x img_size)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of CPU subprocesses for data loading")
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--device", type=str, default="cpu", choices=['cpu', 'gpu'], help="Device to train on")
    parser.add_argument("--criterion", type=str, default="BCEWithLogitsLoss", choices=['BCEWithLogitsLoss', 'DiceLoss'], help="Loss function to use for training")
    parser.add_argument("--model_save_path", type=str, help="Path to save the trained model. If not provided, the model will not be saved.")
    parser.add_argument("--Wavelet_Level", type=int, default=1, help="The level of wavelet decomposition to use for the DTCWTSC-UNet model (default is 1, which means only the first level of wavelet decomposition will be used)")

    args = parser.parse_args()

    # Get dataloaders
    if args.dataset_name.lower() == 'isic':
        train_loader, val_loader, test_loader = get_isic_dataloaders(args.dataset_path, args.batch_size, 
                                                                     args.img_size, args.num_workers)
    elif args.dataset_name.lower() == 'kvasir':
        train_loader, val_loader, test_loader = get_kvasir_dataloaders(args.dataset_path, args.batch_size, 
                                                                       args.img_size, args.num_workers)
    else:
        raise ValueError("Unsupported dataset name. Please choose 'ISIC' or 'Kvasir'.")
    
    # Initialize model
    if args.model.lower() == 'unet':
        model = Unet(n_channels=3, n_classes=1)
    elif args.model.lower() == 'dtcwtsc_unet':
        model = DTCWTSC_UNet(n_channels=3, n_classes=1, wavelet_level=args.Wavelet_Level)
    else:
        raise ValueError("Unsupported model architecture. Please choose 'Unet' or 'DTCWTSC_UNet'.")

    # Set device
    device = torch.device("cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu")

    # Train the model
    if args.criterion == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss()
    elif args.criterion == "DiceLoss":
        criterion = DiceLoss()
    else:
        raise ValueError("Unsupported criterion. Please choose 'BCEWithLogitsLoss' or 'DiceLoss'.")
    
    best_state, history = train_model(model, train_loader, val_loader, 
                                      epochs=args.n_epochs, learning_rate=args.lr, 
                                      criterion=criterion, device=device)
    
    # Save the best model if a save path is provided
    if args.model_save_path:
        torch.save(best_state, args.model_save_path)
        print(f"Best model saved to {args.model_save_path}")

    # Plot training history
    plot_training_history(history)