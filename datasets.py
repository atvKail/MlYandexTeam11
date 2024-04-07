from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
import pickle
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
        return len(self.data)

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



# Пример использования
# # Путь к файлам
# csv_file = os.getcwd() + '\\data\\train_answers.csv'
# root_dir = os.getcwd() + '\\data\\train_images'

# # Преобразования изображений
# transform = transforms.Compose([
#     transforms.ToTensor()
# ])

# # Создание датасета
# dataset = CustomDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
