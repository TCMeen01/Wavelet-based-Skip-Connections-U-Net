import matplotlib.pyplot as plt

def plot_img_and_mask(image, mask, title1='Image', title2='Mask'):
    '''
    Plot an image and its corresponding mask side by side.
    Parameters:
        image: The input image (H, W, C) format.
        mask: The corresponding mask (H, W) format.
        title1: Title for the image subplot.
        title2: Title for the mask subplot.
    Returns:
        Displays the image and mask in a side-by-side plot.
    '''
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(title1)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title(title2)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

