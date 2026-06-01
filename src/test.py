# src/test.py

import torch
import matplotlib.pyplot as plt

from PIL import Image

import torchvision.transforms as transforms

from model import HITFormer


# ---------------------------------------------------
# Device
# ---------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using Device:", device)


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
# Image Transform
# ---------------------------------------------------

transform = transforms.Compose([

    transforms.Resize((256, 256)),

    transforms.ToTensor()
])


# ---------------------------------------------------
# Test Image
# ---------------------------------------------------

hazy_path = "datasets/train/hazy/1_1_0.90179.png"

clean_path = "datasets/train/clear/1.png"


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
# Transform
# ---------------------------------------------------

input_tensor = transform(

    hazy_image

).unsqueeze(0).to(device)

clean_tensor = transform(

    clean_image
)


# ---------------------------------------------------
# Inference
# ---------------------------------------------------

with torch.no_grad():

    output = model(input_tensor)


# ---------------------------------------------------
# Tensor -> Image
# ---------------------------------------------------

hazy_np = input_tensor[0].cpu().permute(

    1, 2, 0

).numpy()

output_np = output[0].cpu().permute(

    1, 2, 0

).numpy()

clean_np = clean_tensor.permute(

    1, 2, 0

).numpy()


# ---------------------------------------------------
# Clip Output
# ---------------------------------------------------

output_np = output_np.clip(0, 1)


# ---------------------------------------------------
# Visualization
# ---------------------------------------------------

plt.figure(figsize=(15, 5))


# Hazy
plt.subplot(1, 3, 1)

plt.imshow(hazy_np)

plt.title("Hazy Input")

plt.axis("off")


# Predicted
plt.subplot(1, 3, 2)

plt.imshow(output_np)

plt.title("Model Output")

plt.axis("off")


# Ground Truth
plt.subplot(1, 3, 3)

plt.imshow(clean_np)

plt.title("Ground Truth")

plt.axis("off")


plt.tight_layout()

plt.show()