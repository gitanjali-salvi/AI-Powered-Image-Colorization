import cv2
import numpy as np

def apply_filter(image, filter_type):
    if filter_type == "Grayscale":
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Other filters remain unchanged
