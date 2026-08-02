# Perform dataloading
# StratifiedKFold
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from .dataset import SteelDataset
import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
data_path = Path(os.getenv("DATA_PATH", "data"))
csv_pth = f"{data_path}/train.csv"
trn_img_pth = f"{data_path}/train_images"

data = pd.read_csv(csv_pth)
train_csv = data.copy()

imageids = train_csv["ImageId"].value_counts()


