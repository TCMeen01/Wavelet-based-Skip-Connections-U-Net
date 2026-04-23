from PIL import Image
import matplotlib.pyplot as plt
import random
import os
import argparse

import torch
from DataHandle.DataLoader import get_transforms
from Utils.metrics import dice_score, iou_score, hd95_score
from Unet.WTSC_Unet import DTCWTSC_UNet
from Unet.Unet import Unet

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="monai")

def random_inference(model, input_dir, mask_dir, img_size, device):
    """
    Perform inference using the given model and input data.

    Args:
        model: The trained model to use for inference.
        input_dir: Path to the input image.
        mask_dir: Path to the ground truth mask.
        img_size: The size to which the images should be resized.
        device: The device to run the inference on (e.g., 'cpu' or 'cuda').
    Returns:
        None: Displays the input image, ground truth mask, and predicted mask. Also prints the Dice score and HD95 score.
    """
    # Get all image file
    all_input = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Random choose image
    selected_input = random.sample(all_input, 1)

    # Load the input image and mask
    input_image = Image.open(os.path.join(input_dir, selected_input[0])).convert('RGB')
    mask_image = Image.open(os.path.join(mask_dir, selected_input[0])).convert('L')

    # Get the transformations for inference
    img_transform, mask_transform = get_transforms(img_size=img_size)

    # Apply transformations to the input image and mask
    input_tensor = img_transform(input_image).unsqueeze(0).to(device)
    mask_tensor = mask_transform(mask_image).unsqueeze(0).to(device)

    # Binary thresholding for the mask
    mask_tensor = (mask_tensor > 0.5).float()

    # Set the model to evaluation mode and perform inference
    model.to(device)
    model.eval()

    # Run inference
    with torch.inference_mode():
        # Get the model's output (Raw logits)
        output = model(input_tensor)

        # Compute Dice score and HD95 score
        dice_val = dice_score(output, mask_tensor)
        iou_val = iou_score(output, mask_tensor)
        hd95 = hd95_score(output, mask_tensor)

        # Apply sigmoid activation and threshold to get binary predictions
        predicted_mask = (torch.sigmoid(output) > 0.5).float()

        # Convert output to CPU and numpy for visualization
        predicted = predicted_mask.squeeze().cpu().numpy()

    # Display the input image, ground truth mask, and predicted mask
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

    # Denormalize input
    input_tensor[:] = input_tensor * 0.5 + 0.5
    
    # Setup image and titles
    images = [input_tensor.squeeze().cpu().permute(1, 2, 0).numpy(),
              mask_tensor.squeeze().cpu().numpy(), 
              predicted]
    titles = ['Input Image\n', 'Ground Truth Mask\n', f'Predicted Mask\nDice Score: {dice_val:.4f}\nIoU: {iou_val:.4f}\nHD95: {hd95:.4f}']

    # Draw image
    for i in range(3):
        axes[i].imshow(images[i], cmap='gray' if i > 0 else None)
        axes[i].set_title(titles[i], fontsize=12, pad=10)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

# ==================== MAIN ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for UNet or DTCWTSC-UNet")

    # Required args
    parser.add_argument("--model_type", type=str, required=True, 
                        help="Model architecture to use for inference ('Unet' or 'DTCWTSC_UNet')")
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                        help="Path to the .pth file (e.g., Models/unet_baseline_model_Kvasir.pth)")

    # Data paths
    parser.add_argument("--image_dir", type=str, required=True, 
                        help="Path to the directory containing input images (e.g., Datasets/Kvasir-SEG/images)")
    parser.add_argument("--mask_dir", type=str, required=True, 
                        help="Path to the directory containing mask images (e.g., Datasets/Kvasir-SEG/masks)")

    # Inference parameters
    parser.add_argument("--device", type=str, default="gpu", choices=['cpu', 'gpu'], 
                        help="Device to run inference on")
    parser.add_argument("--img_size", type=int, default=256, 
                        help="Size to which the input images should be resized (must match the training size)")
    parser.add_argument("--Wavelet_Level", type=int, default=1,
                        help="The level of wavelet decomposition to use for the DTCWTSC-UNet model (default is 1, which means only the first level of wavelet decomposition will be used)")
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load model
    print('Loading model...')
    if args.model_type.lower() == 'unet':
        model = Unet(n_channels=3, n_classes=1)
    elif args.model_type.lower() == 'dtcwtsc_unet':
        model = DTCWTSC_UNet(n_channels=3, n_classes=1, wavelet_level=args.Wavelet_Level)
    model.to(device)

    # Load the model weights from the specified checkpoint
    state_dict = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print('Model loaded successfully.')
    print('-'*50)

    # Perform inference
    print(f'Performing inference on device {device}')
    random_inference(model, 
              input_dir=args.image_dir, 
              mask_dir=args.mask_dir,
              img_size=args.img_size,
              device=device)
    print('Inference completed.')