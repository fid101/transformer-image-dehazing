# src/evaluate.py

import os

import torch
import numpy as np

from torch.utils.data import DataLoader

from model import HITFormer
from dataset import DehazingDataset

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

import torchvision.utils as vutils


# ---------------------------------------------------
# Device
# ---------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using Device:", device)


# ---------------------------------------------------
# Create Results Folder
# ---------------------------------------------------

os.makedirs(
    "results",
    exist_ok=True
)


# ---------------------------------------------------
# Load Model
# ---------------------------------------------------

model = HITFormer().to(device)

model.load_state_dict(

    torch.load(

        "hitformer_slca_final.pth",

        map_location=device
    )
)

model.eval()

print("Model Loaded Successfully!")


# ---------------------------------------------------
# Dataset
# ---------------------------------------------------

test_dataset = DehazingDataset(

    hazy_dir="datasets_small/test/hazy",

    clean_dir="datasets_small/test/clear",

    image_size=512,

    train=False
)

test_loader = DataLoader(

    test_dataset,

    batch_size=1,

    shuffle=False
)


# ---------------------------------------------------
# Evaluation Variables
# ---------------------------------------------------

total_psnr = 0.0

total_ssim = 0.0

num_images = 0


# ---------------------------------------------------
# Evaluation Loop
# ---------------------------------------------------

with torch.no_grad():

    for hazy, clean in test_loader:

        hazy = hazy.to(device)

        clean = clean.to(device)

        # -------------------------------------------
        # Forward Pass
        # -------------------------------------------

        output = model(hazy)

        output = torch.clamp(

            output,

            0,

            1
        )

        # -------------------------------------------
        # Save First 20 Comparisons
        # -------------------------------------------

        if num_images < 20:

            vutils.save_image(

                hazy.cpu(),

                f"results/{num_images:03d}_hazy.png"
            )

            vutils.save_image(

                output.cpu(),

                f"results/{num_images:03d}_output.png"
            )

            vutils.save_image(

                clean.cpu(),

                f"results/{num_images:03d}_gt.png"
            )

        # -------------------------------------------
        # Tensor -> NumPy
        # -------------------------------------------

        output_np = output[0].cpu().permute(

            1,
            2,
            0

        ).numpy()

        clean_np = clean[0].cpu().permute(

            1,
            2,
            0

        ).numpy()

        # -------------------------------------------
        # PSNR
        # -------------------------------------------

        psnr = peak_signal_noise_ratio(

            clean_np,

            output_np,

            data_range=1.0
        )

        # -------------------------------------------
        # SSIM
        # -------------------------------------------

        ssim = structural_similarity(

            clean_np,

            output_np,

            channel_axis=2,

            data_range=1.0
        )

        total_psnr += psnr

        total_ssim += ssim

        num_images += 1

        print(

            f"Image {num_images} "

            f"| PSNR: {psnr:.2f} "

            f"| SSIM: {ssim:.4f}"
        )


# ---------------------------------------------------
# Final Metrics
# ---------------------------------------------------

avg_psnr = total_psnr / num_images

avg_ssim = total_ssim / num_images


print("\n===================================")

print(

    f"Average PSNR: {avg_psnr:.2f} dB"
)

print(

    f"Average SSIM: {avg_ssim:.4f}"
)

print("===================================\n")