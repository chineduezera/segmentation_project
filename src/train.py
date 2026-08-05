from utils import get_validation_subset_for_fold
import pandas as pd
import os
from dotenv import load_dotenv
from .dataset import SteelDataset
from torch.utils.data import DataLoader

load_dotenv()
data_path = os.getenv("DATA_PATH", "data")
train_fold = pd.read_csv("train_fold.csv")
train_csv = pd.read_csv(f"{data_path}/train.csv")

train_df, val_df = get_validation_subset_for_fold(1, train_fold, train_csv)

train_dataset = SteelDataset(df = train_df, img_dir= f"{data_path}/train_images")
val_dataset = SteelDataset(df = val_df, img_dir=f"{data_path}/train_images")

train_dataloader = DataLoader(train_dataset, batch_size= 128, shuffle= True)
val_dataloader = DataLoader(val_dataset, batch_size= 128, shuffle = False)
 