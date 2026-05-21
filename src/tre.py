# src/tre.py

import cv2
import numpy as np


def texture_recovery(image):
    """
    TRE Module
    Texture Recovery and Enhancement

    Input:
        YUV image as NumPy array
        shape -> [H, W, 3]

    Output:
        Texture-enhanced image
    """

    # Apply Gaussian Blur
    #
    # Kernel size:
    # (5,5) means 5x5 smoothing filter
    #
    # Sigma=0 lets OpenCV choose automatically
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Extract texture/high-frequency details
    #
    # Original - Blurred
    #
    # High frequencies remain after subtraction
    texture = cv2.subtract(image, blurred)

    # Enhance texture
    #
    # addWeighted performs:
    #
    # output = alpha*image + beta*texture
    #
    # This sharpens details
    enhanced = cv2.addWeighted(
        image,     # original image
        1.0,       # weight for original
        texture,   # extracted texture
        1.5,       # texture enhancement strength
        0          # brightness offset
    )

    return enhanced