from PIL import Image
import numpy as np
import os
import torch
from u2net import *
def segment_cloth(image_path, checkpoint_path='model/cloth_segm.pth', use_cuda=False, save_output=None):
    """
    Segments cloth from the given image.

    Args:
        image_path (str): Path to the input image.
        checkpoint_path (str): Path to the model checkpoint.
        use_cuda (bool): Whether to use CUDA (GPU) for processing.
        save_output (str or None): If specified, saves the output to the given path. Otherwise, returns the output image.

    Returns:
        PIL.Image or None: The segmented cloth image if save_output is None. Otherwise, saves the image and returns None.
    """
    device = 'cuda:0' if torch.cuda.is_available() and use_cuda else 'cpu'

    # Load the segmentation model
    model = load_seg_model(checkpoint_path, device=device)

    # Define the color palette for visualization
    palette = get_palette(4)

    # Load and convert the input image
    img = Image.open(image_path).convert('RGB')

    # Generate the segmentation mask

    cloth_seg = generate_mask(img, net=model, palette=palette,image_path = image_path ,  device=device)

    if save_output:
        # Save the segmentation result to the specified path
        cloth_seg.save(save_output)
        return None
    else:
        # Return the segmentation result directly
        return cloth_seg

#segmented_image = segment_cloth(image_path="AISIA_BOUTIQUE_DATASET\\cardigans\\img_4433485.jpg", use_cuda=True)
def apply_segmentation_mask(original_image_path, mask_image_path):
    """
    Apply a segmentation mask to an original image and save the segmented image.
    The save path is dynamically generated based on the mask_image_path.

    Parameters:
    - original_image_path: Path to the original image.
    - mask_image_path: Path to the segmentation mask.
    """
    # Load the original image
    original_image = Image.open(original_image_path).convert('RGB')

    # Load the segmentation mask and convert to grayscale
    mask_image = Image.open(mask_image_path).convert('L')

    # Convert images to numpy arrays
    original_array = np.array(original_image)
    mask_array = np.array(mask_image)

    # Ensure the mask is boolean for multiplication
    mask_array = mask_array.astype(bool)

    # Prepare an empty array with the same shape as the original image
    segmented_array = np.zeros_like(original_array)

    # Apply the mask
    for i in range(3):  # Assuming RGB
        segmented_array[:, :, i] = original_array[:, :, i] * mask_array

    # Convert back to PIL image
    segmented_image = Image.fromarray(segmented_array)

    # Dynamically generate the save_path
    save_path = mask_image_path.replace('mask_image', 'segmented')

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the segmented image
    segmented_image.save(save_path)
    print(f"Segmented image saved to {save_path}")
