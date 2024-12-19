import gradio as gr
import cv2
import numpy as np
from app.image_processing import ImageProcessor
from app.utils import ImageUtils
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_image_pipeline(
    input_image: np.ndarray,
    bg_method: str,
    filter_type: str,
    bg_color_r: int,
    bg_color_g: int,
    bg_color_b: int
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
            # Process image using the ImageProcessor class
            # First remove background
            processed = ImageProcessor.remove_background(
                input_image,
                method=bg_method,
                bg_color=bg_color
            )
            
            # Then apply filter
            processed = ImageProcessor.apply_image_filter(
                processed,
                filter_type=filter_type
            )
            
            return processed
            
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            return input_image
            
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        return input_image

def create_gradio_interface():
    """Create and configure Gradio interface"""
    
    # Ensure storage directories exist
    upload_dir, output_dir = ImageUtils.setup_directories()
    
    # Define input components
    input_image = gr.Image(
        label="Upload Image",
        type="numpy",
        sources="upload"
    )
    
    # Get available background removal methods from ImageProcessor
    bg_method = gr.Dropdown(
        choices=list(ImageProcessor.BG_REMOVAL_METHODS.keys()),
        value="basic",
        label="Background Removal Method",
        info="Choose method for removing image background"
    )
    
    # Define available filters from the image processor's filter dictionary
    filter_type = gr.Dropdown(
        choices=["none", "grayscale", "sepia", "blur", "sharpen", "edge_detect"],
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
    
    # Define output with PNG format
    output_image = gr.Image(
        label="Processed Image",
        type="numpy",
        format="png"  # Added this line to force PNG format
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
            bg_color_b
        ],
        outputs=output_image,
        title="Advanced Image Processing Tool",
        description="""
        Upload an image to remove its background and apply professional filters.
        
        Features:
        - Multiple background removal methods
        - Various image filters
        - Custom background color selection
        - Automatic image validation and resizing
        - Support for high-resolution images
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
        iface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False
        )
    except Exception as e:
        logger.error(f"Application startup error: {e}")
