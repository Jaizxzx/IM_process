import gradio as gr
import cv2
import numpy as np
import requests
from app.image_processing import ImageProcessor
from app.utils import ImageUtils
import logging
import os
from tempfile import NamedTemporaryFile
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def remove_background_bria(image: np.ndarray) -> np.ndarray:
    """Remove background using Bria API and preserve transparency."""
    logger.info("Starting Bria background removal process...")

    # Validate API token early
    api_token = os.getenv('BRIA_API_TOKEN')
    if not api_token:
        logger.error("BRIA_API_TOKEN not found in environment variables")
        return image
    logger.info("Found BRIA_API_TOKEN in environment variables")

    # Create temporary file
    temp_file = None
    processed_temp_file = None
    try:
        # Log input image details
        logger.info(f"Input image shape: {image.shape}")
        logger.info(f"Input image dtype: {image.dtype}")

        # Save numpy array as temporary image file
        temp_file = NamedTemporaryFile(suffix='.png', delete=False)
        logger.info(f"Creating temporary file: {temp_file.name}")

        # Convert RGB to BGR for OpenCV
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        logger.info("Converted image from RGB to BGR")

        # Save image
        success = cv2.imwrite(temp_file.name, bgr_image)
        if not success:
            logger.error("Failed to save temporary image file")
            return image
        logger.info("Successfully saved temporary image file")

        # Verify file exists and get size
        file_size = os.path.getsize(temp_file.name)
        logger.info(f"Temporary file size: {file_size} bytes")

        # Prepare API request
        url = "https://engine.prod.bria-api.com/v1/background/remove"
        headers = {'api_token': api_token}
        files = [('file', ('image.png', open(temp_file.name, 'rb'), 'image/png'))]

        logger.info("Sending request to Bria API...")
        response = requests.post(url, headers=headers, files=files)

        # Log response details
        logger.info(f"API Response Status Code: {response.status_code}")
        logger.info(f"API Response Headers: {response.headers}")

        if response.status_code != 200:
            logger.error(f"Bria API error: {response.status_code}")
            logger.error(f"Response content: {response.text}")
            return image

        # Parse API response
        try:
            response_data = response.json()
            logger.info(f"API Response JSON: {response_data}")

            result_url = response_data.get('result_url')
            if not result_url:
                logger.error("No result_url in API response")
                return image

            logger.info(f"Downloading result from: {result_url}")
            
            # Download and process the image
            processed_temp_file = NamedTemporaryFile(suffix='.png', delete=False)
            with requests.get(result_url, stream=True) as r:
                r.raise_for_status()
                with open(processed_temp_file.name, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            # Read the processed image with transparency
            processed_img = cv2.imread(processed_temp_file.name, cv2.IMREAD_UNCHANGED)
            
            if processed_img is None:
                logger.error("Failed to load processed image")
                return image

            logger.info(f"Processed image shape: {processed_img.shape}")

            # Ensure the image has an alpha channel
            if len(processed_img.shape) == 3 and processed_img.shape[2] == 3:
                processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2BGRA)
                processed_img[:, :, 3] = 255  # Set full opacity

            # Convert from BGRA to RGBA for Gradio
            if processed_img.shape[2] == 4:
                processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGRA2RGBA)
                logger.info("Converted BGRA to RGBA")
            else:
                processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                logger.info("Converted BGR to RGB")

            return processed_img

        except Exception as e:
            logger.error(f"Error processing API response: {str(e)}")
            logger.exception("Full traceback:")
            return image

    except Exception as e:
        logger.error(f"Error in Bria background removal: {str(e)}")
        logger.exception("Full traceback:")
        return image
    
    finally:
        # Clean up temporary files
        try:
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            if processed_temp_file and os.path.exists(processed_temp_file.name):
                os.unlink(processed_temp_file.name)
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {str(e)}")


def cleanup_on_shutdown():
    """Cleanup function to be called on application shutdown"""
    logger.info("Cleaning up temporary files...")
    try:
        upload_dir, output_dir = ImageUtils.setup_directories()
        ImageUtils.cleanup_old_files(upload_dir)
        ImageUtils.cleanup_old_files(output_dir)
        logger.info("Temporary files cleaned up successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

def process_image_pipeline(
    input_image: np.ndarray,
    bg_method: str,
    filter_type: str,
    bg_color_r: int,
    bg_color_g: int,
    bg_color_b: int,
    brightness: float,
    contrast: float,
    saturation: float,
    rotation: int,
    flip_horizontal: bool,
    flip_vertical: bool
) -> np.ndarray:
    """Process image with selected parameters"""
    try:
        # Validate input image
        if not ImageUtils.validate_image_array(input_image):
            logger.error("Invalid input image")
            return input_image
            
        # Resize image if needed
        input_image = ImageUtils.resize_image(input_image)
            
        # Create background color tuple
        bg_color = (bg_color_r, bg_color_g, bg_color_b, 255)
        
        try:
            # For advanced (Bria) method, apply background removal first
            if bg_method == "advanced":
                # Apply adjustments to input image
                adjusted_image = ImageProcessor.adjust_image(
                    input_image,
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    rotation=rotation,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical
                )
                
                # Remove background using Bria
                processed = remove_background_bria(adjusted_image)
                
                # Apply filter if needed
                if filter_type != "none":
                    processed = ImageProcessor.apply_image_filter(
                        processed,
                        filter_type=filter_type
                    )
                
                # Ensure we have the correct shape for RGBA
                if len(processed.shape) == 3 and processed.shape[2] == 4:
                    return processed
                else:
                    logger.error(f"Unexpected processed image shape: {processed.shape}")
                    return input_image
                    
            else:
                # For other methods, follow the original pipeline
                # Apply adjustments first
                input_image = ImageProcessor.adjust_image(
                    input_image,
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    rotation=rotation,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical
                )
                
                # Then remove background
                processed = ImageProcessor.remove_background(
                    input_image,
                    method=bg_method,
                    bg_color=bg_color
                )
                
                # Finally apply filter
                processed = ImageProcessor.apply_image_filter(
                    processed,
                    filter_type=filter_type
                )
                
                return processed
            
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            logger.exception("Full traceback:")
            return input_image
            
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        logger.exception("Full traceback:")
        return input_image

def create_gradio_interface():
    """Create and configure Gradio interface with support for RGBA images."""
    # Ensure storage directories exist
    upload_dir, output_dir = ImageUtils.setup_directories()
    
    # Define input components
    input_image = gr.Image(
        label="Upload Image",
        type="numpy",
        sources="upload"
    )
    
    # Add advanced method to background removal choices
    bg_methods = list(ImageProcessor.BG_REMOVAL_METHODS.keys())
    bg_methods.append("advanced")
    
    bg_method = gr.Dropdown(
        choices=bg_methods,
        value="basic",
        label="Background Removal Method",
        info="Choose method for removing image background. 'Advanced' uses Bria's RMBG 2.0 model."
    )
    
    # Define available filters from the image processor's filter dictionary
    filter_type = gr.Dropdown(
        choices=["none", "grayscale", "sepia", "blur", "sharpen", 
                "edge_detect", "emboss", "sketch", "watercolor", "invert"],
        value="none",
        label="Filter Type",
        info="Select a filter to apply to the image"
    )
    
    with gr.Row():
        bg_color_r = gr.Slider(
            minimum=0,
            maximum=255,
            value=173,
            step=1,
            label="Background Red",
            info="Red component of background color"
        )
        bg_color_g = gr.Slider(
            minimum=0,
            maximum=255,
            value=216,
            step=1,
            label="Background Green",
            info="Green component of background color"
        )
        bg_color_b = gr.Slider(
            minimum=0,
            maximum=255,
            value=230,
            step=1,
            label="Background Blue",
            info="Blue component of background color"
        )
    
    with gr.Row():
        brightness = gr.Slider(
            minimum=0.0,
            maximum=2.0,
            value=1.0,
            step=0.1,
            label="Brightness"
        )
        contrast = gr.Slider(
            minimum=0.0,
            maximum=2.0,
            value=1.0,
            step=0.1,
            label="Contrast"
        )
        saturation = gr.Slider(
            minimum=0.0,
            maximum=2.0,
            value=1.0,
            step=0.1,
            label="Saturation"
        )
    
    with gr.Row():
        rotation = gr.Slider(
            minimum=-180,
            maximum=180,
            value=0,
            step=90,
            label="Rotation"
        )
        flip_horizontal = gr.Checkbox(
            label="Flip Horizontal",
            value=False
        )
        flip_vertical = gr.Checkbox(
            label="Flip Vertical",
            value=False
        )

    # Define output with PNG format
    output_image = gr.Image(
        label="Processed Image",
        type="numpy",
        format="png",
        show_download_button=True,  # Enable download button
        height=500  # Set a reasonable display height
    )
    
    # Create interface
    iface = gr.Interface(
        fn=process_image_pipeline,
        inputs=[
            input_image,
            bg_method,
            filter_type,
            bg_color_r,
            bg_color_g,
            bg_color_b,
            brightness,
            contrast,
            saturation,
            rotation,
            flip_horizontal,
            flip_vertical
        ],
        outputs=output_image,
        title="Advanced Image Processing Tool",
        description="""
        Upload an image to remove its background and apply professional filters.
        
        Features:
        - Multiple background removal methods including Bria's RMBG 2.0 model
        - Various image filters
        - Custom background color selection
        - Automatic image validation and resizing
        - Support for high-resolution images
        - Download processed images
        
        Note: The 'Advanced' background removal method requires a valid Bria API token.
        """,
        cache_examples=True,
        theme=gr.themes.Soft()
    )
    
    return iface

if __name__ == "__main__":
    try:
        # Clean up old files before starting
        upload_dir, output_dir = ImageUtils.setup_directories()
        ImageUtils.cleanup_old_files(upload_dir)
        ImageUtils.cleanup_old_files(output_dir)
        
        # Create and launch the interface
        iface = create_gradio_interface()
        import atexit
        atexit.register(cleanup_on_shutdown)
        iface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False
        )
    except Exception as e:
        logger.error(f"Application startup error: {e}")