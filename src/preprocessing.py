# Perform dataloading
# StratifiedKFold
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from .dataset import SteelDataset
import pandas as pd
import csv
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
data_pth = Path(os.getenv("DATA_PATH", "data"))
train_csv_pth = f"{data_pth}/train.csv"
train_img_pth = Path(f"{data_pth}/train_images")

train_csv = pd.read_csv(train_csv_pth)
train_csv = train_csv.copy()


imageid_count = train_csv["ImageId"].value_counts()
duplicates_combined = imageid_count.index.tolist()
train_file_name = [
    file_path.name for file_path in train_img_pth.iterdir() if file_path.is_file()
]

#Create Function
# Grouped defect profiles
grouped_clean_label = []
for item in duplicates_combined:
    duplicate_statuses = train_csv[["ImageId", "ClassId"]].loc[
        train_csv["ImageId"] == item
    ]
    grouped_clean_label.append((item, "&".join(map(str, duplicate_statuses.ClassId.tolist()))))

#Create function
# Matched train.csv with train_images and assigned "Clean" profile to train_images without defects
for obj in train_file_name:
    if obj not in duplicates_combined:
        grouped_clean_label.append((obj, "Clean"))

#Create Function
headers = ["ImageId", "Class"]
file_pth = Path(os.getenv("CURRENT_FILE_PATH", ""))
with open("train_folds.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    writer.writerows(grouped_clean_label)

train_fold = pd.read_csv("train_folds.csv")
train_fold["Fold"] = -1

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(train_fold, train_fold["Class"])):
    train_fold.loc[val_idx, "Fold"] = fold

train_fold.to_csv("train_folds.csv", index=False)
