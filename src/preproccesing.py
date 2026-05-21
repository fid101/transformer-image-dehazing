# src/preprocessing.py

import cv2
import numpy as np


def rgb_to_yuv(image):
    """
    Convert RGB image to YUV.

    Input:
        NumPy image array
        shape -> [H, W, 3]

    Output:
        YUV image
    """

    # OpenCV internally uses BGR by default.
    # Since PIL loads RGB images,
    # we use RGB2YUV conversion explicitly.
    yuv_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

    return yuv_image

def yuv_to_rgb(image):
    """
    Convert YUV image back to RGB.

    Used only for visualization.
    """

    rgb_image = cv2.cvtColor(image, cv2.COLOR_YUV2RGB)

    return rgb_image