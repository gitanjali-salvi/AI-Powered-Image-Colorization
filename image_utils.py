import numpy as np
import streamlit as st
import cv2
from PIL import Image
from streamlit_image_comparison import image_comparison

def colorize_image(img):
    # Sample placeholder for colorization logic
    colorized_img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return colorized_img

def blend_filters(img, filter_1, filter_2, blend_ratio):
    # Apply first filter
    if filter_1 == "Sepia":
        img_1 = cv2.transform(img, np.array([[0.393, 0.769, 0.189],
                                             [0.349, 0.686, 0.168],
                                             [0.272, 0.534, 0.131]]))
    elif filter_1 == "GrayScale":
        img_1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_1 = cv2.cvtColor(img_1, cv2.COLOR_GRAY2BGR)
    else:
        img_1 = img

    # Apply second filter
    if filter_2 == "Vintage":
        img_2 = cv2.applyColorMap(img, cv2.COLORMAP_PINK)
    else:
        img_2 = img

    # Blend Images
    blended_image = cv2.addWeighted(img_1, blend_ratio, img_2, 1 - blend_ratio, 0)
    return blended_image

def compare_images_slider(img1, img2):
    # Convert to PIL for compatibility
    img1_pil = Image.fromarray(img1)
    img2_pil = Image.fromarray(img2)
    
    # Before/After slider comparison
    image_comparison(
        img1_pil,
        img2_pil,
        label1="Original",
        label2="Colorized",
        width=700
    )

def display_image(image, caption="", use_column_width=True):
    st.image(image, caption=caption, use_column_width=use_column_width)
