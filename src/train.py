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
from torch.nn import Sigmoid
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

    train_transform = A.Compose(
        [
            A.RandomCrop(height=256, width=800, p=1.0),  # or smaller patches
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),  # steel images have no fixed "up" — safe here
            A.RandomRotate90(p=0.3),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2
                    ),
                    A.RandomGamma(),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.GaussNoise(),
                    A.GaussianBlur(blur_limit=3),
                ],
                p=0.3,
            ),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.4
            ),
            A.CoarseDropout(
                max_holes=4, max_height=20, max_width=20, p=0.3
            ),  # cutout, forces robustness
            A.Normalize(),
        ]
    )

    train_dataset = SteelDataset(
        df=train_df, img_dir=f"{data_path}/train_images", transforms=train_transform
    )
    val_dataset = SteelDataset(df=val_df, img_dir=f"{data_path}/train_images")

    BATCH_SIZE = 32
    LR = 1e-4
    EPOCH = 100
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
            total_train_dice += dice_train

        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            total_val_dice = 0
            for batch in tqdm(val_dataloader):
                image_batch, mask_batch = batch
                images, masks = image_batch.to(DEVICE), mask_batch.to(DEVICE)
                B, C, H, W = images.shape

                for b in range(B):
                    image = images[b]
                    mask = masks[b]
                    slices = calculate_slice_bboxes(
                        image.shape[0],
                        image.shape[1],
                        slice_h,
                        slice_w,
                        overlap_ratio_h,
                        overlap_ratio_w,
                    )

                    final_mask = np.zeros(
                        (image.shape[0], image.shape[1], 4), dtype=np.uint8
                    )
                    for x1, y1, x2, y2 in slices:
                        # crop slice from original image
                        window = image[y1:y2, x1:x2]

                        # get segmentation mask for slice
                        y_pred = model(window.unsqueeze(0))

                        # merge slice result to final mask
                        final_mask[y1:y2, x1:x2] = (
                            y_pred.squeeze(0).detach().cpu().numpy()
                        )

                final_mask = torch.as_tensor(final_mask, dtype=torch.float32).to(DEVICE)
                loss = criterion(final_mask, mask)  # Note: mask (singular), not masks
                total_val_loss += loss.item()
                dice_val = dice_coefficient(
                    final_mask.detach().cpu().numpy(), mask.detach().cpu().numpy()
                )
                total_val_dice += dice_val

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

        print(
    f"epoch {epoch+1}\n"
    f"      train loss: {train_loss[-1]:.4f}, val loss: {val_loss[-1]:.4f}\n"
    f"      train dice: {train_dice[-1]:.4f}, val dice: {val_dice[-1]:.4f}"
)


if __name__ == "__main__":
    main()
