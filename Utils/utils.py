import matplotlib.pyplot as plt
import random
import os
from PIL import Image
from torchvision import transforms

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

def load_random_rgb_image_tensor(dir_path, device):
    """
    Randomly select and load an image from a directory as an RGB PyTorch tensor.

    Args:
        dir_path (str): The directory containing the input images.
        device (torch.device): The device (CPU or CUDA) to load the tensor onto.

    Returns:
        img_tensor_rgb (torch.Tensor): The RGB image tensor of shape (1, 3, 256, 256).
        img_path (str): The absolute path to the selected image file.
    """
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
    all_files = [f for f in os.listdir(dir_path) if f.lower().endswith(valid_exts)]
    
    if not all_files:
        raise FileNotFoundError(f"No valid images found in directory: {dir_path}")
        
    random_file = random.choice(all_files)
    # random_file = all_files[20] # For debugging: use a specific file instead of random selection
    img_path = os.path.join(dir_path, random_file)
    
    # Force loading as RGB image
    img = Image.open(img_path).convert('RGB') 
    
    # Resize to a power of 2 (e.g., 256x256)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    # Add batch dimension and move to specified device
    img_tensor_rgb = transform(img).unsqueeze(0).to(device) # Shape: (1, 3, 256, 256)
    
    return img_tensor_rgb, img_path

def convert_rgb_to_gray_tensor(rgb_tensor):
    """
    Convert an RGB tensor (N, 3, H, W) to a Grayscale tensor (N, 1, H, W)
    using the standard luminosity method.
    """
    # Standard weighting for human eye sensitivity: R: 0.299, G: 0.587, B: 0.114
    gray_tensor = 0.299 * rgb_tensor[:, 0:1, :, :] + \
                  0.587 * rgb_tensor[:, 1:2, :, :] + \
                  0.114 * rgb_tensor[:, 2:3, :, :]
    return gray_tensor