from utils import get_validation_subset_for_fold
import pandas as pd
import os
from dotenv import load_dotenv
from .dataset import SteelDataset
from torch.utils.data import DataLoader
import torch
from .model import Unet
from torch.nn import Sigmoid
from tqdm import trange
from .metrics import BCEDiceLoss
from torch.optim import AdamW

def main():
    load_dotenv()
    data_path = os.getenv("DATA_PATH", "data")
    train_fold = pd.read_csv("train_fold.csv")
    train_csv = pd.read_csv(f"{data_path}/train.csv")

    train_df, val_df = get_validation_subset_for_fold(1, train_fold, train_csv)

    train_dataset = SteelDataset(df = train_df, img_dir= f"{data_path}/train_images")
    val_dataset = SteelDataset(df = val_df, img_dir=f"{data_path}/train_images")

    BATCH_SIZE = 32
    LR = 1e-4
    EPOCH = 100
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(train_dataset, batch_size= BATCH_SIZE, shuffle= True)
    val_dataloader = DataLoader(val_dataset, batch_size= BATCH_SIZE, shuffle = False)

    model = Unet(in_channels= 3, out_channels= 4).to(DEVICE)
    criterion = BCEDiceLoss()
    optim = AdamW(params= model.parameters(), lr= LR, )

    for epoch in trange(EPOCH):
        model.train()
        train_loss = []

        for image, mask in trange(train_dataloader):
            img = image.to(device= DEVICE)
            msk = mask.to(device= DEVICE)

            y_pred = model(img)
            optim.zero_grad()
            loss= criterion(y_pred, msk)
            
            loss.backward()
            optim.step()

        model.eval()
        with torch.no_grad():
            pass




if __name__ == "__main__":
    main()