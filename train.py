import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

import argparse

import torch
import torch.nn as nn

# Unet and WTSC_Unet models
from Unet.Unet import Unet
from Unet.WTSC_Unet import WTSC_UNet

# Dataset and Dataloader
from DataHandle.DataLoader import *
from DataHandle.Dataset import *

# Metrics
from Utils.metrics import DiceLoss

# Utils
from Utils.utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNet or WTSC-UNet")

    # Required args
    parser.add_argument("--model", type=str, required=True, choices=['Unet', 'WTSC_UNet'], help="Model architecture to train")
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
    if args.model == 'Unet':
        model = Unet(in_channels=3, out_channels=1)
    elif args.model == 'WTSC_UNet':
        model = WTSC_UNet(in_channels=3, out_channels=1)
    else:
        raise ValueError("Unsupported model architecture. Please choose 'Unet' or 'WTSC_UNet'.")

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