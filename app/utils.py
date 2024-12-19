import os
import uuid
import logging
from datetime import datetime
from typing import Optional, Union, Tuple
import hashlib
import mimetypes
from pathlib import Path
import numpy as np
import cv2
import PIL
from PIL import Image

# Constants for image constraints
MAX_IMAGE_DIMENSION = 3840  # 4K resolution width
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

class ImageUtils:
    """
    Utility class for image processing helper functions
    """
    
    @staticmethod
    def generate_unique_filename(prefix: str = "image", extension: str = ".png") -> str:
        """
        Generate a unique filename using UUID to prevent overwriting
        
        Args:
            prefix (str): Prefix for the filename
            extension (str): File extension
        
        Returns:
            str: Unique filename
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        return f"{prefix}_{timestamp}_{unique_id}{extension}"

    @staticmethod
    def ensure_directory(directory: str) -> None:
        """
        Create directory if it doesn't exist
        
        Args:
            directory (str): Path to directory
        """
        os.makedirs(directory, exist_ok=True)

    @staticmethod
    def setup_directories() -> Tuple[str, str]:
        """
        Create necessary directories for storing uploaded and processed images.
        
        Returns:
            Tuple[str, str]: Paths to upload and output directories
        """
        base_dir = Path("storage")
        upload_dir = base_dir / "uploads"
        output_dir = base_dir / "processed"
        
        for dir_path in [upload_dir, output_dir]:
            ImageUtils.ensure_directory(str(dir_path))
            
        return str(upload_dir), str(output_dir)

    @staticmethod
    def validate_image_file(file_path: str) -> Tuple[bool, str]:
        """
        Validate if the uploaded file is a valid image file.
        
        Args:
            file_path (str): Path to the uploaded file
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        try:
            if not os.path.exists(file_path):
                return False, "File does not exist"
                
            if os.path.getsize(file_path) > MAX_FILE_SIZE_BYTES:
                return False, f"File size exceeds {MAX_FILE_SIZE_MB}MB limit"
                
            mime_type = mimetypes.guess_type(file_path)[0]
            if not mime_type or not mime_type.startswith('image/'):
                return False, "Invalid file format. Please upload an image file"
                
            with Image.open(file_path) as img:
                img.verify()
                width, height = img.size
                if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
                    return False, f"Image dimensions exceed {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION} pixels"
                
            return True, ""
            
        except Exception as e:
            ImageUtils.log_error(f"Image validation failed: {str(e)}")
            return False, f"Invalid image file: {str(e)}"

    @staticmethod
    def validate_image_array(image: Union[np.ndarray, None]) -> bool:
        """
        Validate if the input is a valid image array
        
        Args:
            image: Input image (numpy array)
        
        Returns:
            bool: Whether image is valid
        """
        if image is None:
            return False
        
        try:
            if not isinstance(image, np.ndarray):
                return False
            
            if len(image.shape) < 2 or len(image.shape) > 3:
                return False
            
            return True
        except Exception as e:
            ImageUtils.log_error(f"Image array validation error: {e}")
            return False

    @staticmethod
    def resize_image(
        image: Union[np.ndarray, PIL.Image.Image], 
        max_size: int = MAX_IMAGE_DIMENSION,
        maintain_aspect_ratio: bool = True
    ) -> Union[np.ndarray, PIL.Image.Image]:
        """
        Resize image while maintaining aspect ratio
        
        Args:
            image: Input image (numpy array or PIL Image)
            max_size: Maximum dimension size (default: 3840 for 4K)
            maintain_aspect_ratio: Keep original aspect ratio
        
        Returns:
            Resized image in same format as input
        """
        is_pil = isinstance(image, PIL.Image.Image)
        
        if is_pil:
            width, height = image.size
            if width > max_size or height > max_size:
                if maintain_aspect_ratio:
                    ratio = min(max_size / width, max_size / height)
                    new_size = (int(width * ratio), int(height * ratio))
                else:
                    new_size = (max_size, max_size)
                return image.resize(new_size, PIL.Image.LANCZOS)
            return image
        else:
            if not ImageUtils.validate_image_array(image):
                raise ValueError("Invalid image array")
            
            height, width = image.shape[:2]
            if maintain_aspect_ratio:
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
            else:
                new_width = new_height = max_size
            
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def get_image_info(image_path: str) -> dict:
        """
        Get image metadata and basic information.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            dict: Image information including size, format, and mode
        """
        try:
            with Image.open(image_path) as img:
                info = {
                    'format': img.format,
                    'mode': img.mode,
                    'size': img.size,
                    'dimensions': f"{img.size[0]}x{img.size[1]} pixels",
                    'resolution': 'Full HD' if max(img.size) <= 1920 else '4K' if max(img.size) <= 3840 else 'Higher than 4K',
                    'file_size_mb': os.path.getsize(image_path) / (1024 * 1024),
                    'created': datetime.fromtimestamp(os.path.getctime(image_path)),
                    'modified': datetime.fromtimestamp(os.path.getmtime(image_path))
                }
            return info
        except Exception as e:
            ImageUtils.log_error(f"Error getting image info: {str(e)}")
            return {}

    @staticmethod
    def cleanup_old_files(directory: str, max_age_hours: int = 24) -> None:
        """
        Remove files older than specified hours from directory.
        
        Args:
            directory (str): Directory path
            max_age_hours (int): Maximum age of files in hours
        """
        try:
            current_time = datetime.now().timestamp()
            for file_path in Path(directory).glob('*'):
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_hours * 3600:
                        file_path.unlink()
                        logging.info(f"Removed old file: {file_path}")
        except Exception as e:
            ImageUtils.log_error(f"Error during cleanup: {str(e)}")

    @staticmethod
    def ensure_rgb(image: PIL.Image.Image) -> PIL.Image.Image:
        """
        Convert image to RGB mode if it's not already.
        
        Args:
            image (PIL.Image.Image): Input image
            
        Returns:
            PIL.Image.Image: RGB image
        """
        if image.mode != 'RGB':
            return image.convert('RGB')
        return image

    @staticmethod
    def array_to_pil(array: np.ndarray) -> PIL.Image.Image:
        """
        Convert numpy array to PIL Image.
        
        Args:
            array (np.ndarray): Input array
            
        Returns:
            PIL.Image.Image: PIL Image
        """
        return Image.fromarray((array * 255).astype(np.uint8))

    @staticmethod
    def pil_to_array(image: PIL.Image.Image) -> np.ndarray:
        """
        Convert PIL Image to numpy array.
        
        Args:
            image (PIL.Image.Image): Input image
            
        Returns:
            np.ndarray: Numpy array
        """
        return np.array(image) / 255.0

    @staticmethod
    def log_error(message: str) -> None:
        """
        Centralized error logging
        
        Args:
            message (str): Error message to log
        """
        logging.error(message)

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
        ]
    )

# Configure logging when module is imported
setup_logging()