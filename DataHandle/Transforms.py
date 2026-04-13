from torchvision.transforms import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(img_size=256):
    '''
    Get image and mask transformations for data preprocessing.
    Parameters:
        img_size (int): The size to which images and masks will be resized.
    Returns:
        tuple: A tuple containing the image and mask transformations.
    '''
    img_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    return img_transform, mask_transform

def get_augmentations_transform(img_size=256):
    '''
    Get image and mask augmentations transform for data augmentation during training.
    Parameters:
        img_size (int): The size to which images and masks will be resized.
    Returns:
        aug_transform (albumentations.Compose): An albumentations Compose object containing the augmentations.
    '''
    aug_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(translate_percent=(-0.1, 0.1), scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
        A.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ToTensorV2()
    ])

    return aug_transform