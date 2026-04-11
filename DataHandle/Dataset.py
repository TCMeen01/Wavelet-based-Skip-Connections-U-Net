from torch.utils.data import Dataset
from PIL import Image

class MedicalSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, image_transform=None, mask_transform=None):
        '''
        Initializes the MedicalSegmentationDataset.
        Parameters:
            image_paths (list): List of file paths to the input images.
            mask_paths (list): List of file paths to the corresponding masks.
            image_transform (callable, optional): A function/transform to apply to the input images.
            mask_transform (callable, optional): A function/transform to apply to the masks.
        '''
        super().__init__()

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        '''
        Retrieves the image and mask at the specified index, applies transformations, and returns them as tensors.
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

        # Apply transformations if provided
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        # Binarize the mask (assuming binary segmentation)
        mask = (mask > 0).float()  # Convert to binary mask (0 and 1)

        return image, mask