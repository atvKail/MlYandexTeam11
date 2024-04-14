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
        save_file=None,
        transform=None,
        mask_dir=None,
        have_mask=False,
    ):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.save_file = save_file
        self.transform = transform
        self.have_mask = have_mask
        self.mask_dir = mask_dir

        if not save_file:
            return

        if os.path.exists(self.save_file):
            with open(self.save_file, "rb") as f:
                self.data = pickle.load(f)
        else:
            self.data = []
            for idx in range(len(self.data_frame)):
                img_name = os.path.join(
                    self.root_dir, f"img_{self.data_frame.iloc[idx, 0]}.png"
                )
                image = Image.open(img_name)
                target = self.data_frame.iloc[idx, 1]

                if self.have_mask:
                    mask_name = os.path.join(
                        self.mask_dir, f"img_{self.data_frame.iloc[idx, 0]}.png"
                    )
                    mask = Image.open(mask_name)
                    image = Image.fromarray(
                        cv2.bitwise_and(
                            np.array(image), np.array(image), mask=np.array(mask)
                        )
                    )

                if self.transform:
                    image = self.transform(image)

                self.data.append((image, target))

            # Сохраняем данные в файл
            with open(self.save_file, "wb") as f:
                pickle.dump(self.data, f)

    def __len__(self):
        if self.save_file:
            return len(self.data)
        return len(self.data_frame)

    def __getitem__(self, idx):
        if self.save_file:
            return self.data[idx]

        img_name = os.path.join(self.root_dir, f"img_{self.data_frame.iloc[idx, 0]}.png")
        image = Image.open(img_name)
        target = self.data_frame.iloc[idx, 1]

        if self.have_mask:
            mask_name = os.path.join(self.mask_dir, f"img_{self.data_frame.iloc[idx, 0]}.png")
            mask = Image.open(mask_name)
            image = Image.fromarray(cv2.bitwise_and(np.array(image), np.array(image), mask=np.array(mask)))

        if self.transform:
            image = self.transform(image)

        return image, target

    def get_balance_data(self, augmentation_transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=45)
    ])):
        class_count = {}
        balance_data = []

        for i in range(len(self)):
            if self[i][1] not in class_count.keys():
                class_count[self[i][1]] = [1, [i]]
            else:
                class_count[self[i][1]] = [class_count[self[i][1]][0] + 1, class_count[self[i][1]][1] + [i]]
                balance_data.append(self[i])

        mx_count = max(class_count.items(), key=lambda x:x[1][0])

        for item in class_count.items():
            for _ in range(mx_count[1][0] - item[1][0]):
                x = rnd.choice(item[1][1])
                image = self[x][0]
                balance_data.append((augmentation_transform(image), self[x][1]))
        return balance_data


class TestDataClassifer:
    def __init__(self, model, image_folder, transform=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model.eval()
        self.image_folder = image_folder
        self.transform = transform

    def load_images_and_predict(self):
        predictions = []
        image_paths = sorted([os.path.join(self.image_folder, file) for file in os.listdir(self.image_folder) if file.endswith(('png', 'jpg', 'jpeg'))])

        for image_path in image_paths:
            image = Image.open(image_path).convert('RGB')

            image = self.transform(image)
            image = image.unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(image)
                _, predicted = torch.max(output, 1)
                predictions.append(predicted.item())

        return predictions

    def save_predictions_to_csv(self, predictions):
        data = {"ID": list(range(len(predictions))), "target_feature": predictions}
        df = pd.DataFrame(data)
        df.to_csv("predictions.csv", index=False)