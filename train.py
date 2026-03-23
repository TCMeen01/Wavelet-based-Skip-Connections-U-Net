import torch
import os
from tqdm import tqdm
from torch import nn

# Unet and WTSC_Unet models
from Unet.Unet import Unet
from Unet.WTSC_Unet import WTSC_UNet

# Dataset and Dataloader
from Utils.DataLoader import dataloader

# Metrics
from Utils.metrics import iou_score, dice_score

def train(model, 
          dataloader, 
          criterion, 
          optimizer, 
          device,
          num_epochs=25,
          save_checkpoints=True,
          checkpoint_dir='checkpoints',
          checkpoint_interval=5):

    # Create checkpoint directory if needed
    if save_checkpoints and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Created checkpoint directory: {checkpoint_dir}")

    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_iou = 0.0
        running_dice = 0.0
        
        # Use tqdm for progress bar
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            # --- Preprocessing mask logic (Dataset Specific) ---
            # Oxford Pet masks: 1 (Fore), 2 (Back), 3 (Trimap).
            # Convert to: 0, 1, 2
            if masks.max() <= 1.0: 
                 masks = (masks * 255).long()
                 masks = masks - 1
                 masks = torch.clamp(masks, min=0)
                 
            # Squeeze channel dim for CrossEntropyLoss: (B, 1, H, W) -> (B, H, W)
            if masks.dim() == 4 and (masks.shape[1] == 1):
                masks = masks.squeeze(1)

            optimizer.zero_grad()
            
            outputs = model(images) # Output: (B, n_classes, H, W)
            
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # --- Metrics Calculation ---
            # Tính IoU/Dice cho lớp quan trọng nhất (ví dụ class 0: Pet)
            # Hoặc trung bình tất cả các lớp. Ở đây demo tính cho class 0.
            with torch.no_grad():
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                # Binary metrics for Class 0 (Pet)
                pred_pet = (preds == 0).float()
                target_pet = (masks == 0).float()
                
                batch_iou = iou_score(pred_pet, target_pet)
                batch_dice = dice_score(pred_pet, target_pet)
                
                running_iou += batch_iou
                running_dice += batch_dice
            
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}", 'iou_pet': f"{batch_iou:.4f}"})
        
        epoch_loss = running_loss / len(dataloader)
        epoch_iou = running_iou / len(dataloader)
        epoch_dice = running_dice / len(dataloader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} | IoU (Pet): {epoch_iou:.4f} | Dice (Pet): {epoch_dice:.4f}")
        
        # Save Checkpoint
        if save_checkpoints and (epoch + 1) % checkpoint_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            
            # Saving model parameters (state_dict) and optimizer state
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'iou': epoch_iou,
                'dice': epoch_dice
            }, ckpt_path)
            print(f"Saved checkpoint (params included): {ckpt_path}")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model parameters for Oxford Pet (3 classes: Pet, BG, Border)
    n_channels = 3
    n_classes = 3
    
    # Initialize model
    model = WTSC_UNet(n_channels, n_classes) 
    # model = Unet(n_channels, n_classes)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Train
    train(model, 
          dataloader, 
          criterion, 
          optimizer, 
          device, 
          num_epochs=50, 
          save_checkpoints=True, 
          checkpoint_interval=10)