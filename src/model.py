# src/model.py

import torch
import torch.nn as nn


# ---------------------------------------------------
# Patch Embedding
# ---------------------------------------------------

class PatchEmbedding(nn.Module):

    """
    Input:
        [B, 3, H, W]

    Output:
        [B, 96, H/4, W/4]
    """

    def __init__(

        self,

        in_channels=3,

        embed_dim=96,

        patch_size=4

    ):

        super().__init__()

        # Patch extraction + feature projection
        self.proj = nn.Conv2d(

            in_channels=in_channels,

            out_channels=embed_dim,

            kernel_size=patch_size,

            stride=patch_size
        )

    def forward(self, x):

        x = self.proj(x)

        return x


# ---------------------------------------------------
# Window Partition
# ---------------------------------------------------

def window_partition(x, window_size=8):

    """
    Input:
        [B, C, H, W]

    Output:
        [num_windows, window_size, window_size, C]
    """

    B, C, H, W = x.shape

    x = x.view(

        B,

        C,

        H // window_size,
        window_size,

        W // window_size,
        window_size
    )

    x = x.permute(

        0,

        2,
        4,

        3,
        5,

        1
    )

    windows = x.reshape(

        -1,

        window_size,
        window_size,

        C
    )

    return windows


# ---------------------------------------------------
# Window Reverse
# ---------------------------------------------------

def window_reverse(

    windows,

    window_size,

    H,

    W,

    B

):

    """
    Reverse window partition.

    Input:
        [num_windows, win_h, win_w, C]

    Output:
        [B, C, H, W]
    """

    C = windows.shape[-1]

    # Restore window grid
    x = windows.view(

        B,

        H // window_size,
        W // window_size,

        window_size,
        window_size,

        C
    )

    # Rearrange dimensions
    x = x.permute(

        0,

        5,

        1,
        3,

        2,
        4
    )

    # Merge spatial windows
    x = x.reshape(

        B,

        C,

        H,

        W
    )

    return x


# ---------------------------------------------------
# Shift Window
# ---------------------------------------------------

def shift_window(x, shift_size=4):

    """
    Spatial cyclic shift.

    Input:
        [B, C, H, W]
    """

    shifted_x = torch.roll(

        x,

        shifts=(-shift_size, -shift_size),

        dims=(2, 3)
    )

    return shifted_x


# ---------------------------------------------------
# Window Attention
# ---------------------------------------------------

class WindowAttention(nn.Module):

    """
    Swin-style local attention.

    Input:
        [num_windows, 8, 8, 96]
    """

    def __init__(

        self,

        embed_dim=96,

        num_heads=4

    ):

        super().__init__()

        self.attention = nn.MultiheadAttention(

            embed_dim=embed_dim,

            num_heads=num_heads,

            batch_first=True
        )

    def forward(self, x):

        # Extract dimensions
        B_windows, H, W, C = x.shape

        # Flatten windows into token sequences
        x = x.reshape(

            B_windows,

            H * W,

            C
        )

        """
        [128, 64, 96]
        """

        # Self attention
        attended, attention_weights = self.attention(

            x,  # Query
            x,  # Key
            x   # Value
        )

        # Restore spatial layout
        attended = attended.reshape(

            B_windows,

            H,
            W,

            C
        )

        return attended


# ---------------------------------------------------
# Reconstruction Head
# ---------------------------------------------------

class ReconstructionHead(nn.Module):

    """
    Transformer features -> image reconstruction

    Input:
        [B, 96, 64, 64]

    Output:
        [B, 3, 256, 256]
    """

    def __init__(self):

        super().__init__()

        # 64x64 -> 128x128
        self.up1 = nn.ConvTranspose2d(

            in_channels=96,

            out_channels=48,

            kernel_size=2,

            stride=2
        )

        # 128x128 -> 256x256
        self.up2 = nn.ConvTranspose2d(

            in_channels=48,

            out_channels=24,

            kernel_size=2,

            stride=2
        )

        # Final image reconstruction
        self.final = nn.Conv2d(

            in_channels=24,

            out_channels=3,

            kernel_size=3,

            padding=1
        )

        self.relu = nn.ReLU()

    def forward(self, x):

        # First upsample
        x = self.up1(x)

        x = self.relu(x)

        """
        [B, 48, 128, 128]
        """

        # Second upsample
        x = self.up2(x)

        x = self.relu(x)

        """
        [B, 24, 256, 256]
        """

        # Final reconstruction
        x = self.final(x)

        """
        [B, 3, 256, 256]
        """

        return x
    # ---------------------------------------------------
# Full HITFormer Model
# ---------------------------------------------------

class HITFormer(nn.Module):

    """
    Full Transformer Dehazing Network
    """

    def __init__(self):

        super().__init__()

        # ---------------------------------------------------
        # Patch Embedding
        # ---------------------------------------------------

        self.patch_embed = PatchEmbedding(

            in_channels=3,

            embed_dim=96,

            patch_size=4
        )

        # ---------------------------------------------------
        # Attention Block
        # ---------------------------------------------------

        self.attention = WindowAttention(

            embed_dim=96,

            num_heads=4
        )

        # ---------------------------------------------------
        # Reconstruction Decoder
        # ---------------------------------------------------

        self.decoder = ReconstructionHead()

    def forward(self, x):

        """
        Input:
            [B, 3, 256, 256]
        """

        B = x.shape[0]

        # ---------------------------------------------------
        # Patch Embedding
        # ---------------------------------------------------

        features = self.patch_embed(x)

        """
        [B, 96, 64, 64]
        """

        # ---------------------------------------------------
        # Window Attention
        # ---------------------------------------------------

        windows = window_partition(

            features,

            window_size=8
        )

        attended = self.attention(windows)

        # ---------------------------------------------------
        # Reverse Windows
        # ---------------------------------------------------

        restored = window_reverse(

            attended,

            window_size=8,

            H=64,

            W=64,

            B=B
        )

        """
        [B, 96, 64, 64]
        """

        # ---------------------------------------------------
        # Shifted Window Attention
        # ---------------------------------------------------

        shifted = shift_window(

            restored,

            shift_size=4
        )

        shifted_windows = window_partition(

            shifted,

            window_size=8
        )

        shifted_attention = self.attention(

            shifted_windows
        )

        # ---------------------------------------------------
        # Reverse Shifted Windows
        # ---------------------------------------------------

        final_features = window_reverse(

            shifted_attention,

            window_size=8,

            H=64,

            W=64,

            B=B
        )

        """
        [B, 96, 64, 64]
        """

        # ---------------------------------------------------
        # Reconstruction
        # ---------------------------------------------------

        output = self.decoder(final_features)

        """
        [B, 3, 256, 256]
        """

        return output