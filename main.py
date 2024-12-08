from rembg import remove
from rembg.bg import (
    alpha_matting_cutout,
    apply_background_color,
    naive_cutout
)
from PIL import Image
import os

# Create output directory if it doesn't exist
output_dir = 'output_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def get_output_path(filename):
    """
    Get the full output path for a given filename.
    
    Args:
        filename (str): The name of the file to save.
    
    Returns:
        str: The full path where the file will be saved.
    """
    return os.path.join(output_dir, filename)

# Load the input image
input_path = 'group.jpg'  # Specify the input image file path
input_image = Image.open(input_path)

# 1. Basic removal (default method)
# Remove the background using the default method and save the output
basic_output = remove(input_image)
basic_output.save(get_output_path('basic_remove_output.png'))

# 2. Alpha matting method
# Remove the background using the alpha matting method for improved edge details
alpha_matting_output = alpha_matting_cutout(
    input_image,
    mask=remove(input_image, only_mask=True),  # Generate a mask for alpha matting
    foreground_threshold=240,  # Threshold for foreground intensity
    background_threshold=10,   # Threshold for background intensity
    erode_structure_size=10    # Size of the erode structure for refining edges
)
alpha_matting_output.save(get_output_path('alpha_matting_output.png'))

# 3. With colored background (light blue)
# Apply a light blue background to the image with the background removed
mask_output = remove(input_image, only_mask=True)  # Generate the mask
cutout = naive_cutout(input_image, mask_output)    # Apply the mask to the image
colored_bg_output = apply_background_color(cutout, (173, 216, 230, 255))  # RGBA for light blue
colored_bg_output.save(get_output_path('colored_background_output.png'))

# 4. Only mask output
# Save the mask only (binary representation of the background)
mask_only_output = remove(input_image, only_mask=True)
mask_only_output.save(get_output_path('mask_only_output.png'))

print(f"All outputs have been generated in the '{output_dir}' directory!")
