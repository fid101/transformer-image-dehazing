# src/train.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import DehazingDataset

from model import HITFormer


# ---------------------------------------------------
# Device
# ---------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using Device:", device)


# ---------------------------------------------------
# Dataset
# ---------------------------------------------------

train_dataset = DehazingDataset(

    hazy_dir="datasets_small/train/hazy",

    clean_dir="datasets_small/train/clear",

    image_size=256
)

train_loader = DataLoader(

    train_dataset,

    batch_size=1,

    shuffle=True,

    num_workers=4,

    pin_memory=True
)


# ---------------------------------------------------
# Model
# ---------------------------------------------------

model = HITFormer().to(device)

print("Model Loaded Successfully!")


# ---------------------------------------------------
# PSNR Loss
# ---------------------------------------------------

class PSNRLoss(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, pred, target):

        mse = torch.mean(

            (pred - target) ** 2
        )

        psnr = 10 * torch.log10(

            1.0 / (mse + 1e-8)
        )

        return -psnr


# ---------------------------------------------------
# SSIM Loss
# ---------------------------------------------------

class SSIMLoss(nn.Module):

    """
    Lightweight SSIM Approximation
    """

    def __init__(self):

        super().__init__()

    def forward(self, pred, target):

        mu_x = pred.mean()

        mu_y = target.mean()

        sigma_x = pred.var()

        sigma_y = target.var()

        sigma_xy = ((pred - mu_x) * (target - mu_y)).mean()

        C1 = 0.01 ** 2

        C2 = 0.03 ** 2

        ssim = (

            (2 * mu_x * mu_y + C1)

            *

            (2 * sigma_xy + C2)

        ) / (

            (mu_x ** 2 + mu_y ** 2 + C1)

            *

            (sigma_x + sigma_y + C2)
        )

        return 1 - ssim


# ---------------------------------------------------
# Loss Functions
# ---------------------------------------------------

l1_loss = nn.L1Loss()

psnr_loss = PSNRLoss()

ssim_loss = SSIMLoss()


# ---------------------------------------------------
# Optimizer
# ---------------------------------------------------

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=1e-4
)


# ---------------------------------------------------
# Scheduler
# ---------------------------------------------------

epochs = 10

scheduler = CosineAnnealingLR(

    optimizer,

    T_max=epochs,

    eta_min=1e-6
)


# ---------------------------------------------------
# Training Loop
# ---------------------------------------------------

for epoch in range(epochs):

    model.train()

    epoch_loss = 0.0

    print(f"\nStarting Epoch {epoch+1}/{epochs}\n")

    for batch_idx, (hazy, clean) in enumerate(train_loader):

        # ---------------------------------------------------
        # Move To GPU
        # ---------------------------------------------------

        hazy = hazy.to(device)

        clean = clean.to(device)

        # ---------------------------------------------------
        # Forward Pass
        # ---------------------------------------------------

        output = model(hazy)

        # ---------------------------------------------------
        # Combined Loss
        # ---------------------------------------------------

        loss_l1 = l1_loss(

            output,

            clean
        )

        loss_psnr = psnr_loss(

            output,

            clean
        )

        loss_ssim = ssim_loss(

            output,

            clean
        )

        # ---------------------------------------------------
        # Final Combined Loss
        # ---------------------------------------------------

        loss = (

            1.0 * loss_l1

            +

            0.1 * loss_psnr

            +

            0.5 * loss_ssim
        )

        # ---------------------------------------------------
        # Backpropagation
        # ---------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        # ---------------------------------------------------
        # Gradient Clipping
        # ---------------------------------------------------

        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            max_norm=1.0
        )

        optimizer.step()

        # ---------------------------------------------------
        # Logging
        # ---------------------------------------------------

        epoch_loss += loss.item()

        if batch_idx % 100 == 0:

            current_lr = optimizer.param_groups[0]['lr']

            print(

                f"Epoch [{epoch+1}/{epochs}] "

                f"Batch [{batch_idx}/{len(train_loader)}] "

                f"Loss: {loss.item():.4f} "

                f"LR: {current_lr:.7f}"
            )

    # ---------------------------------------------------
    # Scheduler Step
    # ---------------------------------------------------

    scheduler.step()

    # ---------------------------------------------------
    # Epoch Summary
    # ---------------------------------------------------

    avg_loss = epoch_loss / len(train_loader)

    print(

        f"\nEpoch {epoch+1} Completed "

        f"| Average Loss: {avg_loss:.4f}\n"
    )

    # ---------------------------------------------------
    # Save Checkpoint
    # ---------------------------------------------------

    torch.save(

        {

            'epoch': epoch + 1,

            'model_state_dict': model.state_dict(),

            'optimizer_state_dict': optimizer.state_dict(),

        },

        f"checkpoint_lslca_epoch_{epoch+1}.pth"
    )

    print(

        f"Checkpoint Saved: checkpoint_lslca_epoch_{epoch+1}.pth"
    )


# ---------------------------------------------------
# Final Save
# ---------------------------------------------------


torch.save(

    model.state_dict(),

    "hitformer_slca_final.pth"
)

print("\nTraining Completed Successfully!")

print("Final Model Saved: hitformer_lslca_final.pth")