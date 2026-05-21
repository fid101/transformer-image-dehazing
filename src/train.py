# src/train.py

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

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

    hazy_dir="datasets/train/hazy",

    clean_dir="datasets/train/clean",

    image_size=256
)

train_loader = DataLoader(

    train_dataset,

    batch_size=2,

    shuffle=True
)


# ---------------------------------------------------
# Model
# ---------------------------------------------------

model = HITFormer().to(device)


# ---------------------------------------------------
# PSNR Loss
# ---------------------------------------------------

class PSNRLoss(nn.Module):

    """
    Negative PSNR Loss

    Higher PSNR = better image

    So we minimize negative PSNR.
    """

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


criterion = PSNRLoss()


# ---------------------------------------------------
# Optimizer
# ---------------------------------------------------

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=1e-4
)


# ---------------------------------------------------
# Training Settings
# ---------------------------------------------------

epochs = 10


# ---------------------------------------------------
# Training Loop
# ---------------------------------------------------

for epoch in range(epochs):

    model.train()

    epoch_loss = 0.0

    for batch_idx, (hazy, clean) in enumerate(train_loader):

        # Move to GPU
        hazy = hazy.to(device)

        clean = clean.to(device)

        # ---------------------------------------------------
        # Forward Pass
        # ---------------------------------------------------

        output = model(hazy)

        # ---------------------------------------------------
        # Loss
        # ---------------------------------------------------

        loss = criterion(output, clean)

        # ---------------------------------------------------
        # Backpropagation
        # ---------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # Accumulate loss
        epoch_loss += loss.item()

        # ---------------------------------------------------
        # Batch Logging
        # ---------------------------------------------------

        if batch_idx % 10 == 0:

            print(

                f"Epoch [{epoch+1}/{epochs}] "
                f"Batch [{batch_idx}] "
                f"Loss: {loss.item():.4f}"
            )

    # ---------------------------------------------------
    # Epoch Summary
    # ---------------------------------------------------

    avg_loss = epoch_loss / len(train_loader)

    print(

        f"\nEpoch {epoch+1} Completed "
        f"| Average Loss: {avg_loss:.4f}\n"
    )


# ---------------------------------------------------
# Save Model
# ---------------------------------------------------

torch.save(

    model.state_dict(),

    "hitformer_baseline.pth"
)

print("Model Saved Successfully!")