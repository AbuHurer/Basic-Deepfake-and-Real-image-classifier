import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from collections import Counter
from pathlib import Path
from PIL import Image

# 🔰 SECTION 2: Upload & Prepare Dataset


# 📁 Replace this path with your dataset location
train_dir = r"C:\Users\saeem\Desktop\Deepfake\Data\train\train"
# 🔰 SECTION 3: Check Class Balance
class_counts = {}
for cls in os.listdir(train_dir):
    cls_path = os.path.join(train_dir, cls)
    class_counts[cls] = len(os.listdir(cls_path))

print("Class distribution:", class_counts)

# 🔰 SECTION 4: Define Transforms and Dataset
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = ImageFolder(train_dir, transform=train_transform)
val_dataset = ImageFolder(train_dir, transform=val_transform)

print("Class to index:", train_dataset.class_to_idx)  # {'fake': 0, 'real': 1}


train_dataset = ImageFolder(train_dir, transform=train_transform)
val_dataset = ImageFolder(train_dir, transform=val_transform)

targets = [label for _, label in train_dataset]
class_sample_counts = np.bincount(targets)
print("Class sample counts:", class_sample_counts)

# Inverse frequency weights
class_weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
sample_weights = [class_weights[label] for label in targets]

sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 🔰 SECTION 5: Define Model, Loss, Optimizer
from torchvision.models import resnet18, ResNet18_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# Weighted cross-entropy loss
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# 🔰 SECTION 6: Training Loop
EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct = 0.0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    acc = correct / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Train Accuracy: {acc:.4f}")

model.eval()
correct = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        correct += (outputs.argmax(1) == labels).sum().item()

val_acc = correct / len(val_loader.dataset)
print(f"✅ Validation Accuracy: {val_acc:.4f}")

# 🔰 SECTION 7: Save Trained Model
model_path = r"C:\Users\saeem\Desktop\Deepfake\deepfake_model_augmented.pth"
torch.save(model.state_dict(), model_path)
print("✅ Model saved at:", model_path)
