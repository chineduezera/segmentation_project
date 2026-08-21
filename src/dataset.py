# pyrefly: ignore [missing-import]
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import torch
from utils import kaggle_rle_to_mask

class SteelDataset(Dataset):
    def __init__(self, df:pd.DataFrame, img_dir, transforms = None,):
        self.df = df
        self.img_dir = img_dir
        self.img_ids = df["ImageId"].unique().tolist()
        self.transforms = transforms

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        # Get image id and image path
        img_id = self.img_ids[idx]
        img_pth = f"{self.img_dir}/{img_id}"

        # Load image using image path and change to rgb
        image = cv2.imread(img_pth)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = np.zeros((256, 1600, 4), dtype=np.float32)
        rows = self.df[self.df["ImageId"] == img_id]

        for _, row in rows.iterrows():
            class_value = str(row["ClassId"]).strip()
            if class_value.lower() == "clean":
                continue

            class_id = int(row["ClassId"]) - 1
            rle = row['EncodedPixels']
            mask[:, :, class_id] = kaggle_rle_to_mask(rle, 256, 1600)

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask = torch.tensor(mask, dtype=torch.float32).permute(2, 0, 1)

        return image, mask


