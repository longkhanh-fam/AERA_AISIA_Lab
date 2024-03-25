import os
from PIL import Image
from segment import *
def process_dataset(dataset_path, base_output_dir="/kaggle/working"):

    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.jpg'):
                input_image_path = os.path.join(root, file)
                                
                original_image_path = input_image_path
                
                 # Base output directory
                base_output_dir = '..\\output'

                # Extract category and image ID from the image_path
                _, category, image_file_name = input_image_path.rsplit('/', 2)
                #new_category_name = 'women_sweaters'  # Set the new category for saving outputs
                new_image_name = image_file_name.replace('.jpg', '.png')  # Change file extension

                # Construct the output paths
                mask_image_path = os.path.join(base_output_dir, category, new_image_name)
                
                #Segmantating
                segment_cloth(image_path=original_image_path, use_cuda=True)
                #Apply segmentation
                apply_segmentation_mask(original_image_path, mask_image_path)
                print(f"Processed and saved: {input_image_path}")

# Example usage:
dataset_base_path = 'segmentation\AISIA_BOUTIQUE_DATASET'
process_dataset(dataset_base_path)
