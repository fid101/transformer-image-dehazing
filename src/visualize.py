# src/visualize.py

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from preproccesing import rgb_to_yuv, yuv_to_rgb
from tre import texture_recovery

import torchvision.transforms as transforms


# ---------------------------------------------------
# Load Image
# ---------------------------------------------------

# Replace with your own image path
image_path = "datasets/test/hazy/001.png"

# PIL image
image = Image.open(image_path).convert("RGB")

# Convert PIL -> NumPy
rgb_image = np.array(image)


# ---------------------------------------------------
# RGB -> YUV
# ---------------------------------------------------

yuv_image = rgb_to_yuv(rgb_image)


# ---------------------------------------------------
# TRE Enhancement
# ---------------------------------------------------

tre_image = texture_recovery(yuv_image)


# ---------------------------------------------------
# Convert Back To RGB For Visualization
# ---------------------------------------------------

yuv_rgb = yuv_to_rgb(yuv_image)
tre_rgb = yuv_to_rgb(tre_image)


# ---------------------------------------------------
# Tensor Conversion
# ---------------------------------------------------

to_tensor = transforms.ToTensor()

tensor_image = to_tensor(tre_image)

print("Tensor Shape:", tensor_image.shape)
print("Tensor Min:", tensor_image.min())
print("Tensor Max:", tensor_image.max())


# ---------------------------------------------------
# Visualization
# ---------------------------------------------------

# ---------------------------------------------------
# Visualization
# ---------------------------------------------------

plt.figure(figsize=(20, 10))


# Original RGB
plt.subplot(2, 3, 1)
plt.imshow(rgb_image)
plt.title("Original RGB")
plt.axis("off")


# Raw YUV visualization
# Colors appear strange because matplotlib assumes RGB
plt.subplot(2, 3, 2)
plt.imshow(yuv_image)
plt.title("Raw YUV")
plt.axis("off")


# YUV converted back to RGB
plt.subplot(2, 3, 3)
plt.imshow(yuv_rgb)
plt.title("YUV -> RGB")
plt.axis("off")


# TRE enhanced image
plt.subplot(2, 3, 4)
plt.imshow(tre_rgb)
plt.title("TRE Enhanced")
plt.axis("off")


# Y channel (luminance)
plt.subplot(2, 3, 5)
plt.imshow(yuv_image[:, :, 0], cmap="gray")
plt.title("Y Channel (Luminance)")
plt.axis("off")


# Texture-enhanced Y channel
plt.subplot(2, 3, 6)
plt.imshow(tre_image[:, :, 0], cmap="gray")
plt.title("TRE Y Channel")
plt.axis("off")


plt.tight_layout()
plt.show()