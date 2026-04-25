import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from Wavelet.DTCWT import DTCWTransform

from Utils.utils import load_random_rgb_image_tensor, convert_rgb_to_gray_tensor

def random_visualize_dtcwt(dir_path, low_pass_weight, device):
    """
    Randomly select an RGB image, convert to Grayscale, and visualize its 
    Dual-Tree Complex Wavelet Transform (DTCWT) magnitude outputs.
    """
    img_rgb, img_path = load_random_rgb_image_tensor(dir_path, device)
    
    # Convert RGB to Grayscale for Wavelet Processing
    img_gray = convert_rgb_to_gray_tensor(img_rgb)
    
    # Initialize DTCWT model
    level = 4
    dtcwt = DTCWTransform(level=level, biort='near_sym_b').to(device)
    
    # Perform forward DTCWT on the grayscale image
    with torch.no_grad():
        yl, yh = dtcwt(img_gray)
        yh_real, yh_imag = dtcwt.get_real_imag(yh, level=1)
        
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
    
    # 3. Plot Reconstructed Image from DTCWT to verify correctness
    reconstructed = dtcwt.inverse(low_pass_weight * yl, yh).squeeze().cpu().numpy()
    reconstructed = np.abs(reconstructed)
    ax_recon = fig.add_subplot(2, 6, (5, 6))
    ax_recon.imshow(reconstructed, cmap='gray')
    ax_recon.set_title('Reconstructed from DTCWT')
    ax_recon.axis('off')

    angles = ['+15°', '-15°', '+45°', '-45°', '+75°', '-75°']
    
    for i in range(6):
        # Extract individual directional band and calculate magnitude
        real_band = yh_real[i].squeeze().cpu().numpy()
        imag_band = yh_imag[i].squeeze().cpu().numpy()
        magnitude = np.sqrt(real_band**2 + imag_band**2)
        ax = fig.add_subplot(2, 6, i + 7) # Second row
        ax.imshow(magnitude, cmap='gray')
        ax.set_title(f'High-Freq {angles[i]}')
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

def random_compare_dtcwt(dir_path, level_1, level_2, low_pass_weight, device):
    '''
    Randomly select an RGB image, convert to Grayscale, and compare the DTCWT outputs at two different levels of decomposition.

    '''
    img_rgb, img_path = load_random_rgb_image_tensor(dir_path, device)
    
    # Convert RGB to Grayscale for Wavelet Processing
    img_gray = convert_rgb_to_gray_tensor(img_rgb)
    
    # Initialize DTCWT models
    dtcwt_1 = DTCWTransform(level=level_1, biort='near_sym_b').to(device)
    dtcwt_2 = DTCWTransform(level=level_2, biort='near_sym_b').to(device)
    
    # Perform forward DTCWT on the grayscale image
    with torch.no_grad():
        yl_1, yh_1 = dtcwt_1(img_gray)
        yl_2, yh_2 = dtcwt_2(img_gray)

    # Denoising the low-frequency components
    reconstructed_1 = dtcwt_1.inverse(low_pass_weight * yl_1, yh_1).squeeze().cpu().numpy()
    reconstructed_2 = dtcwt_2.inverse(low_pass_weight * yl_2, yh_2).squeeze().cpu().numpy()

    # Visualize the original image and the two reconstructions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # fig.suptitle(f"DTCWT Comparison - {os.path.basename(img_path)}", fontsize=16)
    fig.suptitle(f"DTCWT Comparison", fontsize=16)
    axes[0].imshow(img_rgb.squeeze().permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
    axes[0].set_title('Original RGB')
    axes[0].axis('off')

    axes[1].imshow(np.abs(reconstructed_1), cmap='gray')
    axes[1].set_title(f'Reconstructed (Level {level_1})')
    axes[1].axis('off')

    axes[2].imshow(np.abs(reconstructed_2), cmap='gray')
    axes[2].set_title(f'Reconstructed (Level {level_2})')
    axes[2].axis('off')

    plt.tight_layout()
    # plt.savefig(f"Datasets/DTCWT_Comparison.png")
    plt.show()

# ==================== MAIN ====================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_dir = 'Datasets\Kvasir-SEG\images'
    # dataset_dir = 'Datasets\ISIC-2018'

    # print("Visualizing DTCWT...")
    # random_visualize_dtcwt(dataset_dir, low_pass_weight=0.1, device=device)

    print("Comparing DTCWT at different levels...")
    random_compare_dtcwt(dataset_dir, level_1=1, level_2=4, 
                         low_pass_weight=0.0, device=device)