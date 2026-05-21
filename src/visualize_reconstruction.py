# src/visualize_reconstruction.py

import torch
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader

from dataset import DehazingDataset

from model import (

    PatchEmbedding,

    window_partition,

    window_reverse,

    shift_window,

    WindowAttention,

    ReconstructionHead
)


# ---------------------------------------------------
# Device
# ---------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using Device:", device)


# ---------------------------------------------------
# Dataset
# ---------------------------------------------------

dataset = DehazingDataset(

    hazy_dir="datasets/test/hazy",

    clean_dir="datasets/test/clean",

    image_size=256
)

loader = DataLoader(

    dataset,

    batch_size=1,

    shuffle=True
)


# ---------------------------------------------------
# Models
# ---------------------------------------------------

patch_embed = PatchEmbedding().to(device)

attention_block = WindowAttention().to(device)

decoder = ReconstructionHead().to(device)


# ---------------------------------------------------
# Get One Image
# ---------------------------------------------------

for hazy, clean in loader:

    hazy = hazy.to(device)

    clean = clean.to(device)

    break


# ---------------------------------------------------
# Forward Pipeline
# ---------------------------------------------------

# Patch embedding
features = patch_embed(hazy)

# Window attention
windows = window_partition(features, window_size=8)

attended = attention_block(windows)

# Reverse windows
restored = window_reverse(

    attended,

    window_size=8,

    H=64,

    W=64,

    B=1
)

# Shifted windows
shifted = shift_window(restored, shift_size=4)

shifted_windows = window_partition(

    shifted,

    window_size=8
)

shifted_attention = attention_block(

    shifted_windows
)

# Reverse shifted windows
final_features = window_reverse(

    shifted_attention,

    window_size=8,

    H=64,

    W=64,

    B=1
)

# Reconstruction
output = decoder(final_features)


# ---------------------------------------------------
# Move To CPU
# ---------------------------------------------------

hazy = hazy.detach().cpu()

output = output.detach().cpu()


# ---------------------------------------------------
# Convert Tensor -> NumPy
# ---------------------------------------------------

hazy_image = hazy[0].permute(1, 2, 0).numpy()

output_image = output[0].permute(1, 2, 0).numpy()


# ---------------------------------------------------
# Normalize Output For Visualization
# ---------------------------------------------------

output_image = (output_image - output_image.min()) / (

    output_image.max() - output_image.min() + 1e-8
)


# ---------------------------------------------------
# Visualization
# ---------------------------------------------------

plt.figure(figsize=(10, 5))


# Input Hazy Image
plt.subplot(1, 2, 1)

plt.imshow(hazy_image)

plt.title("Input Hazy Image")

plt.axis("off")


# Reconstructed Output
plt.subplot(1, 2, 2)

plt.imshow(output_image)

plt.title("Untrained Reconstruction")

plt.axis("off")


plt.tight_layout()

plt.show()