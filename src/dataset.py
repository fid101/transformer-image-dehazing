# src/dataset.py

# PyTorch dataset base class
from torch.utils.data import Dataset

# PIL for image loading
from PIL import Image

# File handling
import os

# NumPy for OpenCV compatibility
import numpy as np

# Torchvision transforms
from torchvision import transforms

# RGB -> YUV conversion
from preproccesing import rgb_to_yuv

# TRE module
from tre import texture_recovery


class DehazingDataset(Dataset):

    """
    Custom Dataset for HITFormer Dehazing

    Pipeline:
        RGB
        ↓
        Resize
        ↓
        RGB -> YUV
        ↓
        TRE enhancement
        ↓
        Tensor conversion
    """

    def __init__(self, hazy_dir, clean_dir, image_size=256):

        """
        Parameters:
            hazy_dir  -> path to hazy images
            clean_dir -> path to clean images
            image_size -> resize dimension
        """

        # Store dataset paths
        self.hazy_dir = hazy_dir
        self.clean_dir = clean_dir

        # Read all image filenames
        # Assumes matching filenames exist in both folders
        self.image_names = os.listdir(hazy_dir)

        # Resize transform
        #
        # Neural networks require
        # consistent image dimensions
        self.resize = transforms.Resize((image_size, image_size))

        # Convert image -> tensor
        #
        # Changes:
        # H x W x C  ->  C x H x W
        #
        # Also normalizes:
        # 0-255 -> 0-1
        self.to_tensor = transforms.ToTensor()

    def __len__(self):

        """
        Returns total dataset size.
        """

        return len(self.image_names)

    def __getitem__(self, idx):

        """
        Fetch one training sample.
        """

        # Get filename
        image_name = self.image_names[idx]

        # Construct full image paths
        hazy_path = os.path.join(self.hazy_dir, image_name)
        clean_path = os.path.join(self.clean_dir, image_name)

        # Load images using PIL
        #
        # convert("RGB") ensures:
        # 3-channel RGB format
        hazy_image = Image.open(hazy_path).convert("RGB")
        clean_image = Image.open(clean_path).convert("RGB")

        # Resize images
        #
        # Resize works on PIL images
        hazy_image = self.resize(hazy_image)
        clean_image = self.resize(clean_image)

        # Convert PIL -> NumPy
        #
        # OpenCV operations require NumPy arrays
        hazy_image = np.array(hazy_image)
        clean_image = np.array(clean_image)

        # RGB -> YUV conversion
        #
        # Following HITFormer preprocessing pipeline
        hazy_image = rgb_to_yuv(hazy_image)
        clean_image = rgb_to_yuv(clean_image)

        # Apply TRE enhancement
        #
        # Enhances texture/high-frequency details
        hazy_image = texture_recovery(hazy_image)
        clean_image = texture_recovery(clean_image)

        # Convert NumPy -> Tensor
        #
        # Neural networks operate on tensors
        hazy_image = self.to_tensor(hazy_image)
        clean_image = self.to_tensor(clean_image)

        # Return paired sample
        return hazy_image, clean_image