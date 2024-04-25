from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random as rnd
import pandas as pd
import numpy as np
import pickle
import torch
import cv2
import os


class CustomDataset(Dataset):
    def __init__(
        self,
        csv_file,
        root_dir,
        transform=None,
        vae_model=None,
        have_mask=False,
        mask_transform=None
    ):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.have_mask = have_mask
        self.vae_model = vae_model
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, f"img_{self.data_frame.iloc[idx, 0]}.png")
        image = Image.open(img_name).convert('RGB').resize((224, 224))
        target = self.data_frame.iloc[idx, 1]

        if self.have_mask:
            self.vae_model.eval()
            with torch.no_grad():
                # Применяем модель vae_model для получения маски
                mask = self.vae_model(self.mask_transform(image).unsqueeze(0))
                # Преобразуем маску обратно в изображение PIL
                mask = transforms.ToPILImage()(mask.squeeze(0))
                image = Image.fromarray(cv2.bitwise_and(np.array(image), np.array(image), mask=np.array(mask)))
        
        if self.transform:
            image = self.transform(image)

        return image, target
