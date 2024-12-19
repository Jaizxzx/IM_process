import numpy as np
import cv2
import rembg
from PIL import Image
import io
from rembg.bg import (
    alpha_matting_cutout,
    apply_background_color,
    naive_cutout
)

class ImageProcessor:
    """
    Comprehensive image processing class for background removal and image transformations
    """
    
    BG_REMOVAL_METHODS = {
        "basic": 0,
        "alpha_matting": 1,
        "colored_background": 2,
        "mask_only": 3
    }
    
    @staticmethod
    def remove_background(
        input_image: np.ndarray, 
        method: str | int = "basic",
        foreground_threshold: int = 240,
        background_threshold: int = 10,
        erode_size: int = 10,
        bg_color: tuple = (173, 216, 230, 255)  # Light blue by default
    ) -> np.ndarray:
        """
        Remove background from the input image using various methods
        
        Args:
            input_image (np.ndarray): Input image array
            method (str | int): Background removal method or its ID
                - "basic" or 0: Default rembg removal
                - "alpha_matting" or 1: Alpha matting method
                - "colored_background" or 2: Colored background
                - "mask_only" or 3: Only output the mask
            foreground_threshold (int): Threshold for foreground in alpha matting
            background_threshold (int): Threshold for background in alpha matting
            erode_size (int): Size of erode structure for alpha matting
            bg_color (tuple): RGBA color tuple for colored background
        
        Returns:
            np.ndarray: Processed image with modified background
        """
        try:
            # Convert numpy array to PIL Image if needed
            if not isinstance(input_image, Image.Image):
                input_image = Image.fromarray(input_image)
            
            # Convert method to string if it's an integer
            if isinstance(method, int):
                method = {v: k for k, v in ImageProcessor.BG_REMOVAL_METHODS.items()}.get(method, "basic")
            
            # Convert method to lowercase
            method = method.lower()
            
            if method == "basic":
                output = rembg.remove(input_image)
            
            elif method == "alpha_matting":
                mask = rembg.remove(input_image, only_mask=True)
                output = alpha_matting_cutout(
                    input_image,
                    mask=mask,
                    foreground_threshold=foreground_threshold,
                    background_threshold=background_threshold,
                    erode_structure_size=erode_size
                )
            
            elif method == "colored_background":
                mask = rembg.remove(input_image, only_mask=True)
                cutout = naive_cutout(input_image, mask)
                output = apply_background_color(cutout, bg_color)
            
            elif method == "mask_only":
                output = rembg.remove(input_image, only_mask=True)
            
            else:
                print(f"Unknown method '{method}', falling back to basic removal")
                output = rembg.remove(input_image)
            
            # Convert back to numpy array
            return np.array(output)
        
        except Exception as e:
            print(f"Background removal error: {e}")
            return np.array(input_image)
    
    @staticmethod
    def apply_image_filter(
        image: np.ndarray, 
        filter_type: str = "none"
    ) -> np.ndarray:
        """
        Apply various image filters
        
        Args:
            image (np.ndarray): Input image
            filter_type (str): Type of filter to apply
        
        Returns:
            np.ndarray: Processed image
        """
        # Ensure image is in the right format
        if image.shape[-1] == 4:  # If RGBA, convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        filters = {
            "grayscale": lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
            "sepia": lambda img: ImageProcessor._apply_sepia(img),
            "blur": lambda img: cv2.GaussianBlur(img, (5, 5), 0),
            "sharpen": lambda img: ImageProcessor._sharpen_image(img),
            "edge_detect": lambda img: cv2.Canny(img, 100, 200),
            "none": lambda img: img
        }
        
        # Apply selected filter
        return filters.get(filter_type.lower(), filters["none"])(image)
    
    @staticmethod
    def _apply_sepia(image: np.ndarray) -> np.ndarray:
        """Apply sepia tone to image"""
        sepia_matrix = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        sepia_image = cv2.transform(image, sepia_matrix)
        return np.clip(sepia_image, 0, 255).astype(np.uint8)
    
    @staticmethod
    def _sharpen_image(image: np.ndarray) -> np.ndarray:
        """Sharpen image using kernel"""
        kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])
        return cv2.filter2D(image, -1, kernel)
    
    @staticmethod
    def convert_to_format(
        image: np.ndarray, 
        format: str = "png"
    ) -> bytes:
        """Convert image to specific format"""
        pil_image = Image.fromarray(image)
        byte_arr = io.BytesIO()
        pil_image.save(byte_arr, format=format.upper())
        return byte_arr.getvalue()

def process_image(
    input_image: np.ndarray, 
    remove_bg: bool = True,
    bg_method: str | int = "basic",
    filter_type: str = "none",
    **bg_params
) -> np.ndarray:
    """
    Main image processing function
    
    Args:
        input_image (np.ndarray): Input image
        remove_bg (bool): Whether to remove background
        bg_method (str | int): Background removal method or its ID
        filter_type (str): Filter to apply
        **bg_params: Additional parameters for background removal
            - foreground_threshold (int): For alpha matting
            - background_threshold (int): For alpha matting
            - erode_size (int): For alpha matting
            - bg_color (tuple): For colored background
    
    Returns:
        np.ndarray: Processed image
    """
    if remove_bg:
        input_image = ImageProcessor.remove_background(
            input_image,
            method=bg_method,
            **bg_params
        )
    
    processed_image = ImageProcessor.apply_image_filter(
        input_image, 
        filter_type
    )
    
    return processed_image