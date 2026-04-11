import os
import random
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from Wavelet.DWT import DWTransform
from Wavelet.DTCWT import DTCWTransform

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

def random_visualize_dwt(dir_path, device):
    """
    Randomly select an RGB image, convert to Grayscale, and visualize its 
    Discrete Wavelet Transform (DWT) outputs (LL, LH, HL, HH subbands).
    """
    img_rgb, img_path = load_random_rgb_image_tensor(dir_path, device)
    
    # Convert RGB to Grayscale for Wavelet Processing
    img_gray = convert_rgb_to_gray_tensor(img_rgb)
    
    # Initialize DWT model using 'db2' filter
    dwt = DWTransform(wave='db2').to(device)
    
    # Perform forward DWT on the grayscale image
    with torch.no_grad():
        ll, (lh, hl, hh) = dwt(img_gray)
        
    # Convert tensors to numpy arrays
    # .squeeze() removes batch and channel dims, except for RGB which keeps channels
    img_rgb_np = img_rgb.squeeze().permute(1, 2, 0).cpu().numpy() # (H, W, 3) for plt.imshow
    img_rgb_np = (img_rgb_np * 0.5) + 0.5 # Denormalize for visualization

    ll_np = ll.squeeze().cpu().numpy()
    lh_np = lh.squeeze().cpu().numpy()
    hl_np = hl.squeeze().cpu().numpy()
    hh_np = hh.squeeze().cpu().numpy()
    
    # Setup matplotlib figure (1 row, 5 columns to include Original RGB)
    fig, axes = plt.subplots(1, 5, figsize=(12, 4))
    fig.suptitle(f"DWT (db2) - {os.path.basename(img_path)}", fontsize=14)
    
    titles = ['Original RGB', 'Low-Frequency (LL)', 'Horizontal (LH)', 'Vertical (HL)', 'Diagonal (HH)']
    
    # 1. Plot Original RGB
    axes[0].imshow(img_rgb_np)
    axes[0].set_title(titles[0])
    axes[0].axis('off')
    
    # 2. Plot Wavelet Features
    images_wavelet = [ll_np, lh_np, hl_np, hh_np]
    for i in range(4):
        axes[i+1].imshow(images_wavelet[i], cmap='gray')
        axes[i+1].set_title(titles[i+1])
        axes[i+1].axis('off')
        
    plt.tight_layout()
    plt.show()

def random_visualize_dtcwt(dir_path, device):
    """
    Randomly select an RGB image, convert to Grayscale, and visualize its 
    Dual-Tree Complex Wavelet Transform (DTCWT) magnitude outputs.
    """
    img_rgb, img_path = load_random_rgb_image_tensor(dir_path, device)
    
    # Convert RGB to Grayscale for Wavelet Processing
    img_gray = convert_rgb_to_gray_tensor(img_rgb)
    
    # Initialize DTCWT model
    dtcwt = DTCWTransform(biort='near_sym_b').to(device)
    
    # Perform forward DTCWT on the grayscale image
    with torch.no_grad():
        yl, yh_real, yh_imag = dtcwt(img_gray)
        
    # Convert to numpy arrays
    img_rgb_np = img_rgb.squeeze().permute(1, 2, 0).cpu().numpy()
    img_rgb_np = (img_rgb_np * 0.5) + 0.5 # Denormalize for visualization
    yl_np = yl.squeeze().cpu().numpy()
    
    # Setup a grid layout: 
    # Top row: Original RGB (2 cells) + Low-Frequency Gray (2 cells)
    # Bottom row: 6 high-frequency directional subbands
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(f"DTCWT (Magnitude) - {os.path.basename(img_path)}", fontsize=16)
    
    # 1. Plot Original RGB Image
    ax_rgb = fig.add_subplot(2, 6, (1, 2)) 
    ax_rgb.imshow(img_rgb_np)
    ax_rgb.set_title('Original RGB')
    ax_rgb.axis('off')
    
    # 2. Plot Low-Frequency Grayscale Image
    ax_yl = fig.add_subplot(2, 6, (3, 4)) 
    ax_yl.imshow(yl_np, cmap='gray')
    ax_yl.set_title('Low-Frequency Gray (yl)')
    ax_yl.axis('off')
    
    # 3. Plot 6 Directions of High-Frequency (Magnitude)
    angles = ['+15°', '-15°', '+45°', '-45°', '+75°', '-75°']
    
    for i in range(6):
        # Extract individual directional band and calculate magnitude
        real_band = yh_real[i].squeeze().cpu().numpy()
        imag_band = yh_imag[i].squeeze().cpu().numpy()
        magnitude = (real_band**2 + imag_band**2)**0.5
        
        ax = fig.add_subplot(2, 6, i + 7) # Second row
        ax.imshow(magnitude, cmap='gray')
        ax.set_title(f'High-Freq {angles[i]}')
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

# ==================== MAIN ====================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_dir = 'Datasets\Kvasir-SEG\images'
    
    print("Visualizing DWT...")
    random_visualize_dwt(dataset_dir, device)
    
    print("Visualizing DTCWT...")
    random_visualize_dtcwt(dataset_dir, device)