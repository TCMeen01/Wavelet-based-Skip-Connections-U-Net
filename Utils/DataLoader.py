import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader

my_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = OxfordIIITPet(root='../datasets', split='trainval', target_types='segmentation', download=True,
                        transform=my_transform, target_transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]))
dataloader = DataLoader(
    dataset, 
    batch_size=16, 
    shuffle=True, 
    num_workers=2
)

# Test the dataloader
if __name__ == "__main__":
    for images, masks in dataloader:
        # Plot the first image and its corresponding mask
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow((images[0].permute(1, 2, 0) * 0.5 + 0.5).numpy())  # Denormalize and convert to HWC format
        axes[0].set_title('Image')
        axes[0].axis('off')
        axes[1].imshow(masks[0][0], cmap='gray')  # Show the mask (assuming it's a single-channel image)
        axes[1].set_title('Mask')
        axes[1].axis('off')
        plt.show()
        break  # Remove this break to visualize more batches

