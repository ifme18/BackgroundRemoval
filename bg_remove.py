import streamlit as st
from rembg import remove
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
from io import BytesIO
import os
import traceback
import time

st.set_page_config(layout="wide", page_title="Image Background Remover")

st.write("## Remove background from your image")
st.write(
    ":dog: Try uploading an image to watch the background magically removed. Full quality images can be downloaded from the sidebar. This code is open source and available [here](https://github.com/tyler-simons/BackgroundRemoval) on GitHub. Special thanks to the [rembg library](https://github.com/danielgatis/rembg) :grin:"
)
st.sidebar.write("## Upload and download :gear:")

# Increased file size limit
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Max dimensions for processing
MAX_IMAGE_SIZE = 2000  # pixels

# Download the fixed image
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

# Resize image while maintaining aspect ratio
def resize_image(image, max_size):
    width, height = image.size
    if width <= max_size and height <= max_size:
        return image
    
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    
    return image.resize((new_width, new_height), Image.LANCZOS)

# Preprocess image to enhance contrast, especially for hair
def preprocess_image(image):
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Enhance contrast significantly to make hair stand out
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)  # Increased from 1.5 to 2.0 for better hair detection
    
    # Optional: Enhance brightness to further separate hair from background
    brightness = ImageEnhance.Brightness(image)
    image = brightness.enhance(1.2)
    
    return image

# Refine alpha channel to preserve hair details
def refine_hair_details(image):
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Ensure RGBA
    if img_array.shape[2] == 3:
        img_array = np.pad(img_array, ((0,0),(0,0),(0,1)), mode='constant', constant_values=255)
    
    # Extract channels
    r, g, b, a = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2], img_array[:,:,3]
    
    # Create a mask for content (lower threshold to preserve semi-transparent hair)
    content_mask = a > 10  # Lowered to 10 to catch fine hair strands
    
    # Detect greyish edge pixels (background remnants)
    intensity = (r.astype(float) + g.astype(float) + b.astype(float)) / 3
    grey_mask = (a > 0) & (a <= 50) & (abs(r - g) < 20) & (abs(g - b) < 20) & (abs(b - r) < 20)
    
    # For edge pixels and greyish areas, set to fully transparent
    edge_mask = (a <= 50) | grey_mask
    
    # Preserve semi-transparent hair by not fully removing pixels with low alpha
    # Only set to transparent if they are grey or very low alpha
    r[edge_mask] = 0
    g[edge_mask] = 0
    b[edge_mask] = 0
    a[edge_mask] = 0
    
    # Enhance alpha channel for hair: boost semi-transparent areas
    semi_transparent = (a > 10) & (a < 100)
    a[semi_transparent] = np.minimum(a[semi_transparent] * 1.5, 255).astype(np.uint8)
    
    # Optional: Smooth transition at content boundaries
    if np.any(content_mask):
        from scipy.ndimage import binary_dilation
        boundary = binary_dilation(content_mask) & ~content_mask
        a[boundary] = a[boundary] * 0.8  # Slightly less aggressive fade
    
    # Reconstruct the image
    cleaned_array = np.stack([r, g, b, a], axis=-1)
    
    # Convert back to PIL Image
    refined_image = Image.fromarray(cleaned_array.astype('uint8'), 'RGBA')
    
    return refined_image

@st.cache_data
def process_image(image_bytes):
    """Process image with caching to avoid redundant processing"""
    try:
        image = Image.open(BytesIO(image_bytes))
        # Preprocess to enhance contrast
        preprocessed = preprocess_image(image)
        # Resize large images to prevent memory issues
        resized = resize_image(preprocessed, MAX_IMAGE_SIZE)
        
        # Background removal with parameters tuned for hair
        fixed = remove(
            resized, 
            alpha_matting=True,
            alpha_matting_foreground_threshold=200,  # Lowered to 200 for better hair detection
            alpha_matting_erode_size=5,  # Reduced to 5 to preserve hair details
            alpha_matting_base_threshold=10,  # Lowered to catch fine details
            alpha_matting_method='natural',
            bgcolor=(0, 0, 0, 0)  # Force transparent background
        )
        
        # Refine hair details and clean up edges
        refined = refine_hair_details(fixed)
        
        return image, refined
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None

def fix_image(upload):
    try:
        start_time = time.time()
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        status_text.text("Loading image...")
        progress_bar.progress(10)
        
        # Read image bytes
        if isinstance(upload, str):
            # Default image path
            if not os.path.exists(upload):
                st.error(f"Default image not found at path: {upload}")
                return
            with open(upload, "rb") as f:
                image_bytes = f.read()
        else:
            # Uploaded file
            image_bytes = upload.getvalue()
        
        status_text.text("Processing image...")
        progress_bar.progress(30)
        
        # Process image (using cache if available)
        image, fixed = process_image(image_bytes)
        if image is None or fixed is None:
            return
        
        progress_bar.progress(80)
        status_text.text("Displaying results...")
        
        # Display images
        col1.write("Original Image :camera:")
        col1.image(image)
        
        col2.write("Fixed Image :wrench:")
        col2.image(fixed)
        
        # Prepare download button
        st.sidebar.markdown("\n")
        st.sidebar.download_button(
            "Download fixed image", 
            convert_image(fixed), 
            "fixed.png", 
            "image/png"
        )
        
        progress_bar.progress(100)
        processing_time = time.time() - start_time
        status_text.text(f"Completed in {processing_time:.2f} seconds")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.sidebar.error("Failed to process image")
        # Log the full error for debugging
        print(f"Error in fix_image: {traceback.format_exc()}")

# UI Layout
col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Information about limitations
with st.sidebar.expander("ℹ️ Image Guidelines"):
    st.write("""
    - Maximum file size: 10MB
    - Large images will be automatically resized
    - Supported formats: PNG, JPG, JPEG
    - Processing time depends on image size
    """)

# Process the image
if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error(f"The uploaded file is too large. Please upload an image smaller than {MAX_FILE_SIZE/1024/1024:.1f}MB.")
    else:
        fix_image(upload=my_upload)
else:
    # Try default images in order of preference
    default_images = ["./zebra.jpg", "./wallaby.png"]
    for img_path in default_images:
        if os.path.exists(img_path):
            fix_image(img_path)
            break
    else:
        st.info("Please upload an image to get started!")