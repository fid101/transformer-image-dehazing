# src/visualize_features.py

import torch
import matplotlib.pyplot as plt

from dataset import DehazingDataset
from model import (
    PatchEmbedding,
    window_partition,
    WindowAttention
)

from torch.utils.data import DataLoader


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

patch_embed = PatchEmbedding(

    in_channels=3,

    embed_dim=96,

    patch_size=4

).to(device)

attention_block = WindowAttention(

    embed_dim=96,

    num_heads=4

).to(device)


# ---------------------------------------------------
# Get One Image
# ---------------------------------------------------

for hazy, clean in loader:

    hazy = hazy.to(device)

    break


# ---------------------------------------------------
# Patch Embedding
# ---------------------------------------------------

features = patch_embed(hazy)

print("Patch Feature Shape:", features.shape)

"""
Expected:
[1, 96, 64, 64]
"""


# ---------------------------------------------------
# Window Attention
# ---------------------------------------------------

windows = window_partition(

    features,

    window_size=8
)

attended = attention_block(windows)

print("Attention Window Shape:", attended.shape)


# ---------------------------------------------------
# Convert Attention Windows Back To Feature Grid
# ---------------------------------------------------

"""
For visualization simplicity,
we visualize original features first.

Attention windows are harder to reconstruct fully right now.
"""


# ---------------------------------------------------
# Move To CPU
# ---------------------------------------------------

features = features.detach().cpu()


# ---------------------------------------------------
# Visualization
# ---------------------------------------------------

plt.figure(figsize=(12, 12))

"""
Visualize first 16 feature channels
"""

for i in range(16):

    plt.subplot(4, 4, i + 1)

    feature_map = features[0, i]

    plt.imshow(feature_map, cmap="gray")

    plt.title(f"Channel {i}")

    plt.axis("off")


plt.tight_layout()

plt.show()