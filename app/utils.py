import os
import uuid
import logging
from typing import Optional, Union
import numpy as np
import cv2

class ImageUtils:
    """
    Utility class for common image processing helper functions
    """
    
    @staticmethod
    def generate_unique_filename(prefix: str = "image", extension: str = ".png") -> str:
        """
        Generate a unique filename to prevent overwriting
        
        Args:
            prefix (str): Prefix for the filename
            extension (str): File extension
        
        Returns:
            str: Unique filename
        """
        unique_id = str(uuid.uuid4())[:8]
        return f"{prefix}_{unique_id}{extension}"
    
    @staticmethod
    def ensure_directory(directory: str) -> None:
        """
        Create directory if it doesn't exist
        
        Args:
            directory (str): Path to directory
        """
        os.makedirs(directory, exist_ok=True)
    
    @staticmethod
    def validate_image(image: Union[np.ndarray, None]) -> bool:
        """
        Validate if the input is a valid image
        
        Args:
            image: Input image (numpy array)
        
        Returns:
            bool: Whether image is valid
        """
        if image is None:
            return False
        
        try:
            # Check if it's a numpy array with correct dimensions
            if not isinstance(image, np.ndarray):
                return False
            
            # Check image has at least 2 dimensions (height, width)
            if len(image.shape) < 2 or len(image.shape) > 3:
                return False
            
            return True
        except Exception as e:
            logging.error(f"Image validation error: {e}")
            return False
    
    @staticmethod
    def resize_image(
        image: np.ndarray, 
        max_size: Optional[int] = 1024, 
        maintain_aspect_ratio: bool = True
    ) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio
        
        Args:
            image (np.ndarray): Input image
            max_size (int): Maximum dimension size
            maintain_aspect_ratio (bool): Keep original aspect ratio
        
        Returns:
            np.ndarray: Resized image
        """
        if not ImageUtils.validate_image(image):
            raise ValueError("Invalid image")
        
        # Get current dimensions
        height, width = image.shape[:2]
        
        # Calculate resize dimensions
        if maintain_aspect_ratio:
            # Calculate scaling factor
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
        else:
            # Force resize to exact dimensions
            new_width = new_height = max_size
        
        # Resize image
        resized_image = cv2.resize(
            image, 
            (new_width, new_height), 
            interpolation=cv2.INTER_AREA
        )
        
        return resized_image
    
    @staticmethod
    def log_error(message: str) -> None:
        """
        Centralized error logging
        
        Args:
            message (str): Error message to log
        """
        logging.error(message)
        # Optionally could add more complex logging like 
        # sending to monitoring system, etc.

def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure logging for the application
    
    Args:
        log_level (str): Logging level
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            # Uncomment to add file logging
            # logging.FileHandler('app.log')
        ]
    )

# Configure logging when module is imported
setup_logging()