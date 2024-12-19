# IM_process

## Overview
This application provides a web-based interface for advanced image processing operations, including background removal, filter application, and image transformation.

[![Try on Hugging Face](https://img.shields.io/badge/Hugging%20Face-Try-orange?logo=huggingface)](https://huggingface.co/spaces/Jaizxzx/im_process)


## Features
- Background removal with multiple methods
- Image filtering and effects
- Custom background color selection
- Automatic image validation and resizing
- Support for high-resolution images up to 4K
- File cleanup and management
- Containerized deployment

## Technical Architecture

### Core Components

1. **ImageProcessor (`image_processing.py`)**
   - Handles core image processing operations
   - Implements background removal with multiple methods:
     - Basic removal
     - Alpha matting
     - Colored background
     - Mask only
   - Provides image filtering capabilities:
     - Grayscale
     - Sepia
     - Blur
     - Sharpen
     - Edge detection

2. **ImageUtils (`utils.py`)**
   - Provides utility functions for image handling
   - Manages file operations and directory structure
   - Handles image validation and constraints
   - Implements image resizing and format conversion
   - Manages cleanup of old files

3. **Web Interface (`app.py`)**
   - Creates Gradio-based web interface
   - Implements the image processing pipeline
   - Manages user input and output
   - Handles error logging and reporting

### System Requirements

- Python 3.11 or higher
- Docker (for containerized deployment)
- Minimum 4GB RAM recommended
- GPU optional but recommended for better performance

### Dependencies

```plaintext
numpy>=1.21.0
opencv-python>=4.5.0
Pillow>=9.0.0
gradio>=3.50.0
rembg>=2.0.0
onnxruntime>=1.15.0
```

## Installation and Deployment

### Local Development Setup

1. Clone the repository:
```bash
git clone [https://github.com/Jaizxzx/im_app.git](https://github.com/Jaizxzx/IM_process.git)
cd image-processing-app
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

### Docker Deployment

#### Option 1: Build from Source
1. Build the Docker image:
```bash
docker build -t image-processor .
```

2. Run the container:
```bash
docker run -p 7860:7860 image-processor
```

#### Option 2: Use Pre-built Image
1. Pull the pre-built image:
```bash
docker pull jaizxzx/im_app:1.0
```

2. Run the container:
```bash
docker run -p 7860:7868 jaizxzx/im_app:1.0
```

3. Access the application:
```
http://localhost:7868
```

## Usage Guide

### Web Interface

1. Access the application:
   - If using default port: `http://localhost:7860`
   - If using pre-built image: `http://localhost:7868`
2. Upload an image using the file upload component
3. Select background removal method:
   - Basic: Standard background removal
   - Alpha matting: Advanced edge detection
   - Colored background: Replace background with custom color
   - Mask only: Generate alpha mask
4. Choose image filter (optional):
   - None: No filter
   - Grayscale: Convert to black and white
   - Sepia: Apply vintage effect
   - Blur: Gaussian blur
   - Sharpen: Enhance edges
   - Edge detect: Highlight edges
5. Adjust background color using RGB sliders (if using colored background method)
6. Process image and download result

### Configuration

The application uses several constants that can be modified in the code:

- `MAX_IMAGE_DIMENSION`: Maximum image dimension (default: 3840 for 4K)
- `MAX_FILE_SIZE_MB`: Maximum file size in MB (default: 20)
- Storage directories:
  - `storage/uploads`: Temporary storage for uploaded files
  - `storage/processed`: Storage for processed images

## API Reference

### ImageProcessor Class

#### `remove_background()`
```python
def remove_background(
    input_image: np.ndarray, 
    method: str | int = "basic",
    foreground_threshold: int = 240,
    background_threshold: int = 10,
    erode_size: int = 10,
    bg_color: tuple = (173, 216, 230, 255)
) -> np.ndarray
```

#### `apply_image_filter()`
```python
def apply_image_filter(
    image: np.ndarray, 
    filter_type: str = "none"
) -> np.ndarray
```

### ImageUtils Class

#### `resize_image()`
```python
def resize_image(
    image: Union[np.ndarray, PIL.Image.Image], 
    max_size: int = MAX_IMAGE_DIMENSION,
    maintain_aspect_ratio: bool = True
) -> Union[np.ndarray, PIL.Image.Image]
```

#### `validate_image_file()`
```python
def validate_image_file(file_path: str) -> Tuple[bool, str]
```

## Error Handling

The application implements comprehensive error handling:
- Input validation for images and parameters
- Logging of processing errors
- Fallback mechanisms for failed operations
- Automatic cleanup of temporary files

## Security Considerations

- Input validation prevents processing of malicious files
- File size limits prevent resource exhaustion
- Temporary file cleanup prevents storage overflow
- Docker containerization provides isolation

## Performance Optimization

- Automatic image resizing for large inputs
- Efficient memory management
- Optional GPU acceleration
- Cleanup of old files
