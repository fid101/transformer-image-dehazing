# src/dataset.py

import os
import torch

from PIL import Image
from torch.utils.data import Dataset

import torchvision.transforms as transforms


class DehazingDataset(Dataset):

    def __init__(
        self,
        hazy_dir,
        clean_dir,
        image_size=256,
        train=True
    ):

        self.hazy_dir = hazy_dir
        self.clean_dir = clean_dir

        self.hazy_images = sorted(
            os.listdir(hazy_dir)
        )

        # -----------------------------------------
        # TRAINING TRANSFORMS
        # -----------------------------------------

        if train:

            self.transform = transforms.Compose([

                transforms.RandomCrop(
                    image_size
                ),

                transforms.RandomHorizontalFlip(
                    p=0.5
                ),

                transforms.RandomVerticalFlip(
                    p=0.5
                ),

                transforms.RandomRotation(
                    degrees=90
                ),

                transforms.ToTensor()

            ])

        # -----------------------------------------
        # EVALUATION TRANSFORMS
        # -----------------------------------------

        else:

            self.transform = transforms.Compose([

                transforms.Resize(
                    (512,512)
                ),

                transforms.ToTensor()

            ])

    def __len__(self):

        return len(self.hazy_images)

    def __getitem__(self, idx):

        hazy_name = self.hazy_images[idx]

        # -----------------------------------------
        # RESIDE / SOTS Mapping
        #
        # 1400_1.png -> 1400.png
        # 1_2_0.98.png -> 1.png
        # -----------------------------------------

        clean_id = hazy_name.split("_")[0]

        clean_name = clean_id + ".png"

        hazy_path = os.path.join(
            self.hazy_dir,
            hazy_name
        )

        clean_path = os.path.join(
            self.clean_dir,
            clean_name
        )

        hazy_image = Image.open(
            hazy_path
        ).convert("RGB")

        clean_image = Image.open(
            clean_path
        ).convert("RGB")

        # -----------------------------------------
        # SAME RANDOM AUGMENTATION
        # -----------------------------------------

        if isinstance(
            self.transform.transforms[0],
            transforms.RandomCrop
        ):

            seed = torch.randint(
                0,
                999999,
                (1,)
            ).item()

            torch.manual_seed(seed)

            hazy_image = self.transform(
                hazy_image
            )

            torch.manual_seed(seed)

            clean_image = self.transform(
                clean_image
            )

        else:

            hazy_image = self.transform(
                hazy_image
            )

            clean_image = self.transform(
                clean_image
            )

        return hazy_image, clean_image