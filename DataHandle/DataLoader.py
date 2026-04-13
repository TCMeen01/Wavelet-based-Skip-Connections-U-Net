import os
import torch
from torch.utils.data import DataLoader
from DataHandle.Dataset import MedicalSegmentationDataset
from DataHandle.Transforms import get_transforms, get_augmentations_transform

from sklearn.model_selection import train_test_split

def get_isic_dataloaders(root_dir, batch_size=8, img_size=256, num_workers=4, augmentation=False):
    '''
    Create and return DataLoaders for the pre-split ISIC 2018 dataset (Train/Val/Test).

    Args:
        root_dir (str): Path to the root 'skin-dataset' directory.
        batch_size (int): Number of samples per batch. Defaults to 8.
        img_size (int): Size to resize the images and masks (assumes square size img_size x img_size). Defaults to 256.
        num_workers (int): Number of CPU subprocesses for data loading. Defaults to 4.
        augmentation (bool): Whether to apply data augmentation. Defaults to False.

    Returns:
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        test_loader (DataLoader): DataLoader for the testing set.
    '''
    # Define paths to subdirectories
    train_img_dir = os.path.join(root_dir, 'train', 'images', 'ISIC2018_Task1-2_Training_Input')
    train_mask_dir = os.path.join(root_dir, 'train', 'masks', 'ISIC2018_Task1_Training_GroundTruth')
    
    val_img_dir = os.path.join(root_dir, 'validation', 'images', 'ISIC2018_Task1-2_Validation_Input')
    val_mask_dir = os.path.join(root_dir, 'validation', 'masks', 'ISIC2018_Task1_Validation_GroundTruth')

    test_img_dir = os.path.join(root_dir, 'test', 'images', 'ISIC2018_Task1-2_Test_Input')
    test_mask_dir = os.path.join(root_dir, 'test', 'masks', 'ISIC2018_Task1_Test_GroundTruth')
    
    # Helper function: Get and sort file paths
    def get_sorted_paths(img_dir, mask_dir):
        '''Retrieve and sort absolute paths for images and masks to ensure correct alignment'''
        # Assert directories exist
        assert os.path.exists(img_dir) and os.path.exists(mask_dir)

        # Filter valid image extensions
        valid_extensions = ('.jpg', '.png', '.jpeg')
        images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(valid_extensions)])
        masks = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(valid_extensions)])
        
        # Create absolute paths
        image_paths = [os.path.join(img_dir, img) for img in images]
        mask_paths = [os.path.join(mask_dir, msk) for msk in masks]
        
        # Ensure a image_paths and mask_paths have the same length
        assert len(image_paths) == len(mask_paths)
        
        return image_paths, mask_paths

    # Retrieve file path lists for each split
    train_images, train_masks = get_sorted_paths(train_img_dir, train_mask_dir)
    val_images, val_masks = get_sorted_paths(val_img_dir, val_mask_dir)
    test_images, test_masks = get_sorted_paths(test_img_dir, test_mask_dir)

    # Get transforms
    image_transform, mask_transform = get_transforms(img_size=img_size)
    aug_transform = get_augmentations_transform(img_size=img_size) if augmentation else None
    
    # Initialize Datasets
    train_dataset = MedicalSegmentationDataset(train_images, train_masks, image_transform, mask_transform, aug_transform)
    val_dataset = MedicalSegmentationDataset(val_images, val_masks, image_transform, mask_transform)
    test_dataset = MedicalSegmentationDataset(test_images, test_masks, image_transform, mask_transform)
    
    # Initialize DataLoaders with optimized configurations
    pin_mem = True if torch.cuda.is_available() else False  # Use pin_memory only if CUDA is available
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_mem,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_mem, 
        prefetch_factor=2
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_mem, 
        prefetch_factor=2
    )

    print("DONE LOADING ISIC-2018 DATASET")
    
    # Return dataloaders
    return train_loader, val_loader, test_loader

def get_kvasir_dataloaders(root_dir, batch_size=8, img_size=256,
                          num_workers=4, val_size=0.1, test_size=0.1, augmentation=False):
    '''
    Create and return train, validation, and test dataloaders from the given `root_dir`.

    Args:
        root_dir (str): Path to the root directory containing the data. It is assumed to contain 2 subfolders: `images` and `masks`.
        batch_size (int): Batch size for the dataloaders (number of samples loaded per batch). Defaults to 8.
        img_size (int): Size to resize the images and masks (assumes square size img_size x img_size). Defaults to 256.
        val_size (float): Proportion of the dataset to include in the validation split. Defaults to 0.1 (10%).
        test_size (float,): Proportion of the dataset to include in the test split. Defaults to 0.1 (10%).
        augmentation (bool): Whether to apply data augmentation. Defaults to False.
        num_workers (int): Number of CPU subprocesses to use for data loading. Defaults to 4.

    Returns:
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
    '''

    # Get image and mask directories
    image_dir, mask_dir = os.path.join(root_dir, 'images'), os.path.join(root_dir, 'masks')

    # Get list of image and mask paths. Sort to ensure matching order
    valid_extensions = ('.jpg', '.png', '.jpeg')
    image_paths = sorted([f for f in os.listdir(image_dir) if f.endswith(valid_extensions)])
    mask_paths = sorted([f for f in os.listdir(mask_dir) if f.endswith(valid_extensions)])

    # Assert that image_paths and mask_paths have the same size
    assert len(image_paths) == len(mask_paths)

    # Get full path
    image_paths = [os.path.join(image_dir, f) for f in image_paths]
    mask_paths = [os.path.join(mask_dir, f) for f in mask_paths]
    
    # Train - Val - Test Split
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        image_paths, mask_paths, test_size=val_size+test_size, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, y_val_test, test_size=test_size/(val_size+test_size), random_state=42
    )

    # Get transforms
    image_transform, mask_transform = get_transforms(img_size=img_size)
    aug_transform = get_augmentations_transform(img_size=img_size) if augmentation else None

    # Init datasets
    train_dataset = MedicalSegmentationDataset(X_train, y_train, image_transform, mask_transform, aug_transform)
    val_dataset = MedicalSegmentationDataset(X_val, y_val, image_transform, mask_transform)
    test_dataset = MedicalSegmentationDataset(X_test, y_test, image_transform, mask_transform)

    # Init dataloaders, add pin_memory and prefetch_factor for optimization purpose
    pin_mem = True if torch.cuda.is_available() else False  # Use pin_memory only if CUDA is available
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,             
        num_workers=num_workers, 
        pin_memory=pin_mem,          
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=pin_mem, 
        prefetch_factor=2
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=pin_mem, 
        prefetch_factor=2
    )

    print("DONE LOADING KVASIR DATASET")
    
    # Return dataloaders
    return train_loader, val_loader, test_loader