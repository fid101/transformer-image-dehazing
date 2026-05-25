# src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------
# Patch Embedding
# ---------------------------------------------------

class PatchEmbedding(nn.Module):

    def __init__(

        self,

        in_channels=3,

        embed_dim=96,

        patch_size=4

    ):

        super().__init__()

        self.proj = nn.Conv2d(

            in_channels,

            embed_dim,

            kernel_size=patch_size,

            stride=patch_size
        )

    def forward(self, x):

        return self.proj(x)


# ---------------------------------------------------
# Window Partition
# ---------------------------------------------------

def window_partition(x, window_size=8):

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

    C = windows.shape[-1]

    x = windows.view(

        B,

        H // window_size,
        W // window_size,

        window_size,
        window_size,

        C
    )

    x = x.permute(

        0,

        5,

        1,
        3,

        2,
        4
    )

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

        B_windows, H, W, C = x.shape

        x = x.reshape(

            B_windows,

            H * W,

            C
        )

        attended, _ = self.attention(

            x,
            x,
            x
        )

        attended = attended.reshape(

            B_windows,

            H,
            W,

            C
        )

        return attended


# ---------------------------------------------------
# Feed Forward MLP
# ---------------------------------------------------

class MLP(nn.Module):

    def __init__(

        self,

        dim,

        hidden_dim

    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(

                dim,

                hidden_dim
            ),

            nn.GELU(),

            nn.Linear(

                hidden_dim,

                dim
            )
        )

    def forward(self, x):

        return self.net(x)


# ---------------------------------------------------
# Swin Transformer Block
# ---------------------------------------------------

class SwinTransformerBlock(nn.Module):

    def __init__(

        self,

        embed_dim=96,

        num_heads=4,

        window_size=8,

        shift_size=0

    ):

        super().__init__()

        self.window_size = window_size

        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(embed_dim)

        self.attention = WindowAttention(

            embed_dim=embed_dim,

            num_heads=num_heads
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = MLP(

            dim=embed_dim,

            hidden_dim=embed_dim * 4
        )

    def forward(self, x):

        B, C, H, W = x.shape

        residual = x

        # Shift
        if self.shift_size > 0:

            x = shift_window(

                x,

                self.shift_size
            )

        # LayerNorm
        x_ln = x.permute(

            0,
            2,
            3,
            1
        )

        x_ln = self.norm1(x_ln)

        x = x_ln.permute(

            0,
            3,
            1,
            2
        )

        # Window Attention
        windows = window_partition(

            x,

            self.window_size
        )

        attended = self.attention(windows)

        x = window_reverse(

            attended,

            self.window_size,

            H,

            W,

            B
        )

        # Reverse Shift
        if self.shift_size > 0:

            x = torch.roll(

                x,

                shifts=(

                    self.shift_size,

                    self.shift_size

                ),

                dims=(2, 3)
            )

        # Residual Attention
        x = x + residual

        # MLP Refinement
        residual2 = x

        x_mlp = x.permute(

            0,
            2,
            3,
            1
        )

        x_mlp = self.norm2(x_mlp)

        x_mlp = self.mlp(x_mlp)

        x_mlp = x_mlp.permute(

            0,
            3,
            1,
            2
        )

        x = residual2 + x_mlp

        return x


# ---------------------------------------------------
# Residual Block
# ---------------------------------------------------

class ResidualBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.conv1 = nn.Conv2d(

            channels,

            channels,

            kernel_size=3,

            padding=1
        )

        self.conv2 = nn.Conv2d(

            channels,

            channels,

            kernel_size=3,

            padding=1
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        residual = x

        x = self.conv1(x)

        x = self.relu(x)

        x = self.conv2(x)

        x = x + residual

        return x


# ---------------------------------------------------
# AHIP
# ---------------------------------------------------

class AHIP_Block(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.global_branch = nn.Sequential(

            nn.Conv2d(

                channels,

                channels // 4,

                kernel_size=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(

                channels // 4,

                channels,

                kernel_size=1
            )
        )

        self.local_branch = nn.Sequential(

            nn.Conv2d(

                channels,

                channels,

                kernel_size=3,

                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(

                channels,

                channels,

                kernel_size=3,

                padding=1
            )
        )

        self.fusion = nn.Sequential(

            nn.Conv2d(

                channels,

                channels,

                kernel_size=1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        residual = x

        global_feat = self.global_pool(x)

        global_feat = self.global_branch(

            global_feat
        )

        local_feat = self.local_branch(x)

        haze_map = global_feat + local_feat

        haze_weights = self.fusion(

            haze_map
        )

        enhanced = x * haze_weights

        output = enhanced + residual

        return output


# ---------------------------------------------------
# SLCA
# ---------------------------------------------------

class SLCA_Block(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.channel_attention = nn.Sequential(

            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(

                channels,

                channels // 8,

                kernel_size=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(

                channels // 8,

                channels,

                kernel_size=1
            ),

            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(

            nn.Conv2d(

                channels,

                1,

                kernel_size=7,

                padding=3
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        channel_weights = self.channel_attention(x)

        x = x * channel_weights

        spatial_weights = self.spatial_attention(x)

        x = x * spatial_weights

        return x


# ---------------------------------------------------
# Reconstruction Decoder
# ---------------------------------------------------

class ReconstructionHead(nn.Module):

    def __init__(self):

        super().__init__()

        # Stage 1
        self.up1 = nn.ConvTranspose2d(

            96,

            96,

            kernel_size=2,

            stride=2
        )

        self.refine1 = nn.Sequential(

            nn.Conv2d(

                96,

                96,

                kernel_size=3,

                padding=1
            ),

            nn.ReLU(inplace=True),

            ResidualBlock(96),

            ResidualBlock(96)
        )

        self.slca1 = SLCA_Block(96)

        # Stage 2
        self.up2 = nn.ConvTranspose2d(

            96,

            64,

            kernel_size=2,

            stride=2
        )

        self.refine2 = nn.Sequential(

            nn.Conv2d(

                64,

                64,

                kernel_size=3,

                padding=1
            ),

            nn.ReLU(inplace=True),

            ResidualBlock(64),

            ResidualBlock(64)
        )

        self.slca2 = SLCA_Block(64)

        # Final RGB
        self.final = nn.Sequential(

            nn.Conv2d(

                64,

                32,

                kernel_size=3,

                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(

                32,

                3,

                kernel_size=3,

                padding=1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.up1(x)

        x = self.refine1(x)

        x = self.slca1(x)

        x = self.up2(x)

        x = self.refine2(x)

        x = self.slca2(x)

        output = self.final(x)

        return output


# ---------------------------------------------------
# HITFormer
# ---------------------------------------------------

class HITFormer(nn.Module):

    def __init__(self):

        super().__init__()

        self.patch_embed = PatchEmbedding(

            in_channels=3,

            embed_dim=96,

            patch_size=4
        )

        # Hierarchical Swin Blocks
        self.swin_block1 = SwinTransformerBlock(

            embed_dim=96,

            num_heads=4,

            window_size=8,

            shift_size=0
        )

        self.swin_block2 = SwinTransformerBlock(

            embed_dim=96,

            num_heads=4,

            window_size=8,

            shift_size=4
        )

        self.swin_block3 = SwinTransformerBlock(

            embed_dim=96,

            num_heads=4,

            window_size=8,

            shift_size=0
        )

        # AHIP
        self.ahip = AHIP_Block(96)

        # Decoder
        self.decoder = ReconstructionHead()

    def forward(self, x):

        # ---------------------------------------------------
        # Patch Embedding
        # ---------------------------------------------------

        features = self.patch_embed(x)

        # Save shallow features
        skip_features = features

        # ---------------------------------------------------
        # Hierarchical Swin
        # ---------------------------------------------------

        features = self.swin_block1(features)

        features = self.swin_block2(features)

        features = self.swin_block3(features)

        # ---------------------------------------------------
        # AHIP
        # ---------------------------------------------------

        features = self.ahip(features)

        # ---------------------------------------------------
        # Skip Fusion
        # ---------------------------------------------------

        features = features + skip_features

        # ---------------------------------------------------
        # Reconstruction
        # ---------------------------------------------------

        output = self.decoder(features)

        return output