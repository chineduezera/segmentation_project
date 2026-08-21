from utils import (
    get_validation_subset_for_fold,
    calculate_slice_bboxes,
    dice_coefficient,
)
import pandas as pd
import os
from dotenv import load_dotenv
from dataset import SteelDataset
from torch.utils.data import DataLoader
import torch
from model import Unet
from tqdm import trange
from metrics import BCEDiceLoss
from torch.optim import AdamW
import albumentations as A
import numpy as np
from tqdm import tqdm


def main():
    load_dotenv()
    data_path = os.getenv("DATA_PATH", "data")
    current_path = os.getenv("CURRENT_PATH", ".")
    train_fold = pd.read_csv(f"{current_path}/train_folds.csv")
    train_csv = pd.read_csv(f"{data_path}/train.csv")

    train_df, val_df = get_validation_subset_for_fold(1, train_fold, train_csv)
    print(f"Train size: {len(train_df["ImageId"])}, Val size: {len(val_df["ImageId"])}")
    train_transform = A.Compose(
        [
            A.RandomCrop(height=256, width=256, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ]
    )

    train_dataset = SteelDataset(
        df=train_df, img_dir=f"{data_path}/train_images", transforms=train_transform
    )
    val_dataset = SteelDataset(df=val_df, img_dir=f"{data_path}/train_images")

    BATCH_SIZE = 4
    LR = 1e-3
    EPOCH = 20
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = Unet(in_channels=3, out_channels=4).to(DEVICE)
    criterion = BCEDiceLoss()
    optim = AdamW(
        params=model.parameters(),
        lr=LR,
    )

    slice_h = 256
    slice_w = 400
    overlap_ratio_h = 0
    overlap_ratio_w = 100 / slice_w  # 100px overlap b/w each slice
    train_loss = []
    train_dice = []
    val_dice = []
    val_loss = []

    for epoch in trange(EPOCH):
        model.train()
        total_train_loss = 0
        total_train_dice = 0
        for batch in tqdm(train_dataloader):
            images, masks = batch
            image, mask = images.to(DEVICE), masks.to(DEVICE)

            optim.zero_grad()
            ypred = model(image)
            loss = criterion(ypred, mask)
            loss.backward()
            optim.step()

            dice_train = dice_coefficient(
                ypred.detach().cpu().numpy(), mask.detach().cpu().numpy()
            )
            total_train_loss += loss.item()
            total_train_dice += dice_train.mean().item()

        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            total_val_dice = 0
            for batch in tqdm(val_dataloader):
                image_batch, mask_batch = batch
                images, masks = image_batch.to(DEVICE), mask_batch.to(DEVICE)
                B, C, H, W = images.shape

                for img in range(B): # This loop is to slice each image in the batch, segment it and merge it back to a full image
                    image = images[img]
                    mask = masks[img]
                    slices = calculate_slice_bboxes(
                        image_height=H,
                        image_width=W,
                        slice_height=slice_h,
                        slice_width=slice_w,
                        overlap_height_ratio=overlap_ratio_h,
                        overlap_width_ratio=overlap_ratio_w,
                    )

                    final_mask = torch.zeros((4, H, W), dtype=torch.float32, device=DEVICE)
                    for x1, y1, x2, y2 in slices:
                        # crop slice from original image
                        window = image[:, y1:y2, x1:x2]

                        # get segmentation mask for slice
                        y_pred = model(window.unsqueeze(0)).squeeze(0)

                        # merge slice result to final mask
                        final_mask[:, y1:y2, x1:x2] = y_pred

                    loss = criterion(final_mask.unsqueeze(0), mask.unsqueeze(0))
                    total_val_loss += loss.item()

                    dice_val = dice_coefficient(
                        final_mask.unsqueeze(0).cpu().numpy(),
                        mask.unsqueeze(0).cpu().numpy(),
                    )
                    total_val_dice += dice_val.mean().item()

        num_train_batches = len(train_dataloader)
        num_val_images = len(val_dataset)

        avg_train_loss = total_train_loss / num_train_batches
        avg_train_dice = total_train_dice / num_train_batches
        avg_val_loss = total_val_loss / num_val_images
        avg_val_dice = total_val_dice / num_val_images

        train_loss.append(avg_train_loss)
        train_dice.append(avg_train_dice)
        val_loss.append(avg_val_loss)
        val_dice.append(avg_val_dice)

        # print(
        #     f"epoch {epoch+1}\n"
        #     f"      train loss: {train_loss[-1]:.4f}, val loss: {val_loss[-1]:.4f}\n"
        #     f"      train dice: {train_dice[-1]:.4f}, val dice: {val_dice[-1]:.4f}"
        # )


if __name__ == "__main__":
    main()
