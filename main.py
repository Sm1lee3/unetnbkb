import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import OxfordPetsDataset
from unet import UNet
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# === 1. Підготовка трансформацій і датасету ===
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def mask_transform(mask):
    mask = mask.resize((256, 256))  # resize через PIL
    mask = np.array(mask)
    mask = (mask >= 2).astype(np.float32)  # контур+об'єкт = 1, фон = 0
    return torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)

target_transform = mask_transform

dataset = OxfordPetsDataset(
    root="data/oxford-iiit-pet",
    split="train",
    transform=transform,
    target_transform=target_transform
)
dataset = torch.utils.data.Subset(dataset, range(10))
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# === 2. Підготовка моделі та оптимізатора ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
criterion = nn.BCEWithLogitsLoss()  # Для бінарної сегментації з logit
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# === 3. Тренування ===
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    print(f"Epoch {epoch+1}/{num_epochs}")

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        masks = (masks > 0).float()

        outputs = model(images)
        
        # Якщо розміри не збігаються, інтерполюємо вихід до потрібного розміру
        if outputs.size(2) != masks.size(2) or outputs.size(3) != masks.size(3):
            outputs = F.interpolate(outputs, size=(256, 256), mode="bilinear", align_corners=False)

        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# === 4. Візуалізація передбачень ===
model.eval()
with torch.no_grad():
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)

        # Якщо розміри не збігаються, інтерполюємо вихід до потрібного розміру
        if outputs.size(2) != masks.size(2) or outputs.size(3) != masks.size(3):
            outputs = F.interpolate(outputs, size=(256, 256), mode="bilinear", align_corners=False)

        # Перетворюємо логіти в ймовірності через sigmoid
        preds = torch.sigmoid(outputs).cpu().numpy()

        # Видаляємо зайвий вимір для правильного виведення
        preds = preds.squeeze()  # Розмір стане (256, 256)

        print(preds.min(), preds.max())

        # Візуалізація
        plt.subplot(1, 3, 1)
        plt.imshow(images[0].cpu().permute(1, 2, 0))
        plt.title("Image")
        print("Mask min:", masks.min().item(), "Mask max:", masks.max().item())  # Додаємо цей рядок
        plt.subplot(1, 3, 2)
        plt.imshow(masks[0][0].cpu().numpy(), cmap="gray")  # Переконайся, що це маска з 0 та 1
        plt.title("Ground Truth")

        plt.subplot(1, 3, 3)
        plt.imshow(preds > 0.7, cmap="gray")  # Бінаризація на основі порогу 0.5
        plt.title("Prediction")

        plt.show()
        break
