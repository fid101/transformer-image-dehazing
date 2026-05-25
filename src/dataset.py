# src/dataset.py

import os

from PIL import Image

from torch.utils.data import Dataset

import torchvision.transforms as transforms


class DehazingDataset(Dataset):

    """
    RESIDE ITS Dataset Loader

    Maps:
        1_2_0.98.png
    ->
        1.png
    """

    def __init__(

        self,

        hazy_dir,

        clean_dir,

        image_size=256

    ):

        self.hazy_dir = hazy_dir

        self.clean_dir = clean_dir

        # List all hazy images
        self.hazy_images = sorted(

            os.listdir(hazy_dir)
        )

        # ---------------------------------------------------
        # Image Transform
        # ---------------------------------------------------

        self.transform = transforms.Compose([

            transforms.Resize(

                (image_size, image_size)
            ),

            transforms.ToTensor()
        ])

    def __len__(self):

        return len(self.hazy_images)

    def __getitem__(self, idx):

        # ---------------------------------------------------
        # Hazy Image Name
        # ---------------------------------------------------

        hazy_name = self.hazy_images[idx]

        """
        Example:

        1_2_0.97842.png
        """

        # ---------------------------------------------------
        # Extract Clean Image ID
        # ---------------------------------------------------

        clean_id = hazy_name.split("_")[0]

        """
        "1"
        """

        clean_name = clean_id + ".png"

        """
        1.png
        """

        # ---------------------------------------------------
        # Full Paths
        # ---------------------------------------------------

        hazy_path = os.path.join(

            self.hazy_dir,

            hazy_name
        )

        clean_path = os.path.join(

            self.clean_dir,

            clean_name
        )

        # ---------------------------------------------------
        # Load Images
        # ---------------------------------------------------

        hazy_image = Image.open(

            hazy_path

        ).convert("RGB")

        clean_image = Image.open(

            clean_path

        ).convert("RGB")

        # ---------------------------------------------------
        # Apply Transforms
        # ---------------------------------------------------

        hazy_image = self.transform(

            hazy_image
        )

        clean_image = self.transform(

            clean_image
        )

        return hazy_image, clean_image