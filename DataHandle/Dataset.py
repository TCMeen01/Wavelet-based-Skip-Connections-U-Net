import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class MedicalSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, 
                 image_transform=None, mask_transform=None, aug_transform=None):
        '''
        Initializes the MedicalSegmentationDataset.
        Parameters:
            image_paths (list): List of file paths to the input images.
            mask_paths (list): List of file paths to the corresponding masks.
            image_transform (callable, optional): A function/transform to apply to the input images.
            mask_transform (callable, optional): A function/transform to apply to the masks.
            aug_transform (albumentations.Compose, optional): An albumentations Compose object containing the augmentations.
        '''
        super().__init__()

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.aug_transform = aug_transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        '''
        Retrieves the image and mask at the specified index, applies transformations and augmentations (if provided), and returns them as tensors.
        Parameters:
            idx (int): The index of the item to retrieve.
        Returns:
            tuple: A tuple containing the transformed image and mask tensors.
        '''
        # Get the file paths for the image and mask
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load the image and mask using PIL
        image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB
        mask = Image.open(mask_path).convert('L')      # Ensure mask is in grayscale

        # Apply augmentations if provided (using albumentations)
        if self.aug_transform:
            augmented = self.aug_transform(image=np.array(image), mask=np.array(mask))
            image = augmented['image']
            mask = augmented['mask']

            if mask.ndim == 2:
                mask = mask.unsqueeze(0)  # Add channel dimension for grayscale mask

            mask = (mask > 127).float()  # Binarize the mask (assuming binary segmentation)
        else:
            # Apply transformations if provided
            if self.image_transform:
                image = self.image_transform(image)
            if self.mask_transform:
                mask = self.mask_transform(mask)

            mask = (mask > 0.5).float()  # Binarize the mask (assuming binary segmentation)
        
        # Return the image and mask
        return image, mask