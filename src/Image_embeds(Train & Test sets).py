#TRAIN IMAGE EMBEDDINGS:


import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class ImageDataset(Dataset):
    def __init__(self, df, image_folder, transform):
        self.df = df
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]["image_name"]
        img_path = os.path.join(self.image_folder, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception as e:
            # fallback: black image if something fails
            image = torch.zeros(3, 224, 224)

        return image

image_folder = "/kaggle/working/train_images"

dataset = ImageDataset(train_df, image_folder, image_transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)


all_features = []

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Extracting image embeddings (batched)"):
        batch = batch.to(device)
        features = resnet(batch)                      # shape: [B, 2048, 1, 1]
        features = features.view(features.size(0), -1)  # shape: [B, 2048]
        all_features.append(features.cpu().numpy())

# Combine all batches
image_embeddings = np.concatenate(all_features, axis=0)
print("✅ Image embeddings shape:", image_embeddings.shape)
np.save("/kaggle/working/image_embeddings.npy", image_embeddings)
print("✅ Saved image embeddings successfully!")


#TEST IMAGE EMBEDDINGS:

from torchvision import transforms

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class ImageDataset(Dataset):
    def __init__(self, df, image_folder, transform):
        self.df = df
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]["image_name"]
        img_path = os.path.join(self.image_folder, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception as e:
            # fallback: black image if something fails
            image = torch.zeros(3, 224, 224)

        return image


resnet = models.resnet50(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])  # remove final layer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = resnet.to(device)
resnet.eval()

print("✅ Using device:", device)


image_folder = "/kaggle/working/test_images"

dataset = ImageDataset(train_df, image_folder, image_transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

all_features = []

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Extracting image embeddings (batched)"):
        batch = batch.to(device)
        features = resnet(batch)                      # shape: [B, 2048, 1, 1]
        features = features.view(features.size(0), -1)  # shape: [B, 2048]
        all_features.append(features.cpu().numpy())

# Combine all batches
image_embeddings = np.concatenate(all_features, axis=0)
print("✅ Image embeddings shape:", image_embeddings.shape)


np.save("/kaggle/working/image_test_embeddings.npy", image_embeddings)
print("✅ Saved image embeddings successfully!")


