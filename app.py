import numpy as np
import cv2
import streamlit as st
from PIL import Image, ImageEnhance
from streamlit_cropper import st_cropper
from colorization import Colorizer
from streamlit_image_comparison import image_comparison

# Initialize Colorizer
model_dir = r"models"
colorizer = Colorizer(model_dir)

def apply_filter(image, filter_type):
    if filter_type == "Grayscale":
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif filter_type == "Sepia":
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        return cv2.transform(image, kernel)
    elif filter_type == "Negative":
        return cv2.bitwise_not(image)
    elif filter_type == "Sketch":
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        inv = cv2.bitwise_not(gray)
        return cv2.divide(gray, 255 - inv, scale=256)
    elif filter_type == "Vintage":
        return cv2.applyColorMap(image, cv2.COLORMAP_OCEAN)
    elif filter_type == "Cool":
        return cv2.applyColorMap(image, cv2.COLORMAP_HOT )
    elif filter_type == "Warm":
        return cv2.applyColorMap(image, cv2.COLORMAP_WINTER)
    elif filter_type == "Summer":
        return cv2.applyColorMap(image, cv2.COLORMAP_SUMMER)
    elif filter_type == "Pencil":
        gray, sketch = cv2.pencilSketch(image, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
        return sketch
    elif filter_type == "Stylization":
        return cv2.stylization(image, sigma_s=60, sigma_r=0.45)
    return image

def blend_filters(image, filter_1, filter_2, blend_ratio):
    filtered_1 = apply_filter(image, filter_1)
    filtered_2 = apply_filter(image, filter_2)
    # Ensure both filtered images have the same size
    if filtered_1.shape != filtered_2.shape:
        filtered_2 = cv2.resize(filtered_2, (filtered_1.shape[1], filtered_1.shape[0]))
    
    # Ensure both images have the same number of channels
    if len(filtered_1.shape) == 2:  # If grayscale, convert to RGB
        filtered_1 = cv2.cvtColor(filtered_1, cv2.COLOR_GRAY2RGB)
    if len(filtered_2.shape) == 2:  # If grayscale, convert to RGB
        filtered_2 = cv2.cvtColor(filtered_2, cv2.COLOR_GRAY2RGB)

    # Blend the images with the given blend ratio
    blended_image = cv2.addWeighted(filtered_1, blend_ratio, filtered_2, 1 - blend_ratio, 0)
    
    return blended_image
    #return cv2.addWeighted(filtered_1, blend_ratio, filtered_2, 1 - blend_ratio, 0)

def auto_enhance(image):
    pil_img = Image.fromarray(image)
    enhancer = ImageEnhance.Contrast(pil_img)
    image = enhancer.enhance(1.5)  # Increase contrast
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.2)  # Increase brightness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.3)  # Increase sharpness
    return np.array(image)

def load_image(file):
    image = Image.open(file)
    return np.array(image)

def display_image(image, caption="Image", use_column_width=True):
    st.image(image, caption=caption, use_column_width=use_column_width)

def main():
    st.title("🎨 Colorify: Image Colorization")
    st.write("Upload your black & white images to convert them to color! 🌈")

    # Initialize session state variables
    if "step" not in st.session_state:
        st.session_state.step = 0
    if "original_image" not in st.session_state:
        st.session_state.original_image = None
    if "cropped_image" not in st.session_state:
        st.session_state.cropped_image = None
    if "colorized_image" not in st.session_state:
        st.session_state.colorized_image = None
    if "filtered_image" not in st.session_state:
        st.session_state.filtered_image = None

    # Function to handle the "Back" button
    def go_back():
        if st.session_state.step > 0:
            st.session_state.step -= 1

    # Upload image file
    if st.session_state.step == 0:
        file = st.sidebar.file_uploader("Please upload a black & white image 🖼️", type=["jpg", "png", "jpeg"])
        if file is not None:
            st.session_state.original_image = load_image(file)
            st.session_state.step = 1

    # Display and crop image
    if st.session_state.step == 1:
        if st.session_state.original_image is not None:
            st.subheader("Original Image")
            display_image(st.session_state.original_image, caption="Original Image")

            st.subheader("🖼️ Crop Image")
            cropped_img = st_cropper(Image.fromarray(st.session_state.original_image), aspect_ratio=None)

             # Create a container for buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Confirm Crop"):
                    st.session_state.cropped_image = np.array(cropped_img)
                    st.session_state.step = 2

            with col2:
                if st.button("No Crop"):
                    st.session_state.cropped_image = st.session_state.original_image  # Use original image
                    st.session_state.step = 2


    # Colorize the image
    if st.session_state.step == 2:
        st.subheader("🌈 Colorized Image")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎨 Colorize Image"):
                try:
                    colorized_img = colorizer.colorize(st.session_state.cropped_image)
                    st.session_state.colorized_image = colorized_img
                    st.session_state.filtered_image = colorized_img  # Ensure it is set
                    display_image(colorized_img, caption="Colorized Image")
                    st.session_state.step = 3
                except Exception as e:
                    st.error(f"❌ An error occurred during colorization: {e}")
        
        with col2:
            # Back button
            if st.button("🔙 Back", key="back_step_2"):
                go_back()

    # Apply filters
    if st.session_state.step == 3:
        st.subheader("✨ Apply Filters")
        if st.button("Auto-Enhance"):
            enhanced_img = auto_enhance(st.session_state.colorized_image)
            st.session_state.filtered_image = enhanced_img
            display_image(enhanced_img, caption="Auto-Enhanced Image")

        filters = ["None", "Grayscale", "Sepia", "Negative", "Sketch", "Vintage", "Cool", "Warm", "Summer", "Pencil", "Stylization"]
        filter_1 = st.selectbox("Choose first filter:", filters)
        filter_2 = st.selectbox("Choose second filter:", filters)
        blend_ratio = st.slider("Blending Ratio (0 - 1):", min_value=0.0, max_value=1.0, value=0.5)

        # Apply blended filters
        filtered_img = st.session_state.colorized_image
        if filter_1 != "None" or filter_2 != "None":
            try:
                filtered_img = blend_filters(st.session_state.colorized_image, filter_1, filter_2, blend_ratio)
                st.session_state.filtered_image = filtered_img
            except Exception as e:
                st.error(f"Error applying filter: {e}")
                filtered_img = st.session_state.colorized_image

        if filtered_img is not None:
            display_image(filtered_img, caption="Filtered Image")

        col1, col2 = st.columns(2)
        with col1:
            # Proceed to comparison
            if st.button("Next: Compare Before and After"):
                st.session_state.step = 4
        with col2:
            # Back button
            if st.button("🔙 Back", key="back_step_3"):
                go_back()

    # Before and after comparison
    if st.session_state.step == 4:
        st.subheader("🆚 Before and After Comparison")
        if st.session_state.original_image is not None and st.session_state.filtered_image is not None:
            image_comparison(
                img1=Image.fromarray(st.session_state.original_image),
                img2=Image.fromarray(st.session_state.filtered_image),
                label1="Original",
                label2="Colorized",
                width=700
            )

            if st.button("Finalize and Download"):

                def convert_bgr_to_rgb(image):
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                if st.session_state.filtered_image is not None:
                    # Convert the colorized image to RGB before encoding
                    rgb_image = convert_bgr_to_rgb(st.session_state.filtered_image)

                    # Encode the image in PNG format
                    is_success, buffer = cv2.imencode(".png", rgb_image)

                    col1, col2 = st.columns(2)
                    with col1:
    
                        # If successful, allow the download
                        if is_success:
                            st.download_button("⬇️ Download Final Image",
                                   data=buffer.tobytes(),
                                   file_name='final_image.png',
                                   mime='image/png')
                            
                    with col2:
                        # Back button
                        if st.button("🔙 Back", key="back_step_4s"):
                            go_back()

if __name__ == "__main__":
    main()
