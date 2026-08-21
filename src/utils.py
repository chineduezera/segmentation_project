import pandas as pd
import torch
import numpy as np

def get_validation_subset_for_fold(
    val_fold_no: int,
    train_fold_df: pd.DataFrame,
    train_csv_df: pd.DataFrame,
):
    """
    Splits a dataset into training and validation subsets based on a specified fold number.

    This function combines the clean data from the training fold with the training CSV data, merges them on 'ImageId', and then splits the resulting DataFrame into:
    - `train_df`: All other data points (i.e., data points where 'Fold' does not equal the specified fold number).
    - `val_df`: Data points where the 'Fold' column equals the specified fold number.

    Parameters:
    - val_fold_no (int): The fold number to use as the validation set.
    - train_fold_df (pd.DataFrame): DataFrame containing training fold data with 'ImageId' and 'ClassId' columns.
    - train_csv_df (pd.DataFrame): DataFrame containing training CSV data.

    Returns:
    - tuple: A tuple of two pandas DataFrames (train_df, val_df).
    """

    train_fold = train_fold_df.copy()
    train_csv = train_csv_df.copy()

    clean = train_fold[["ImageId", "ClassId"]].loc[train_fold["ClassId"] == "Clean"]
    concat_train_csv = pd.concat([train_csv, clean], ignore_index=True)
    concat_train_csv = concat_train_csv.merge(train_fold[["ImageId", "Fold"]], how="inner", on="ImageId")
    train_df = concat_train_csv.loc[concat_train_csv["Fold"] != val_fold_no]
    val_df = concat_train_csv.loc[concat_train_csv["Fold"] == val_fold_no]
    return train_df, val_df


def dice_coefficient(pred, target, threshold=0.5, eps=1e-7):
    pred = torch.from_numpy(pred)
    target = torch.from_numpy(target)
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice  # shape: (batch, num_classes)


def calculate_slice_bboxes(
    image_height: int,
    image_width: int,
    slice_height: int = 512,
    slice_width: int = 512,
    overlap_height_ratio: float = 0,
    overlap_width_ratio: float = 0,
):
    slice_bboxes = []
    y_max = y_min = 0
    y_overlap = int(overlap_height_ratio * slice_height)
    x_overlap = int(overlap_width_ratio * slice_width)
    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + slice_height
        while x_max < image_width:
            x_max = x_min + slice_width
            if y_max > image_height or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height, y_max)
                xmin = max(0, xmax - slice_width)
                ymin = max(0, ymax - slice_height)
                slice_bboxes.append([xmin, ymin, xmax, ymax])
            else:
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes


def kaggle_rle_to_mask(rle, height=256, width=1600):
    if rle is None or (isinstance(rle, float) and np.isnan(rle)):
        return np.zeros((height, width), dtype=np.float32)

    rle = str(rle).strip()
    if rle == "":
        return np.zeros((height, width), dtype=np.float32)

    vals = np.array([int(v) for v in rle.split()], dtype=np.int32)
    if len(vals) % 2 != 0:
        raise ValueError(f"Malformed RLE for mask: {rle}")

    mask = np.zeros(height * width, dtype=np.float32)
    starts = vals[0::2]
    lengths = vals[1::2]

    for start, length in zip(starts, lengths):
        if length == 0:
            continue
        end = start + length
        if 0 <= start < len(mask) and end <= len(mask):
            mask[start:end] = 1.0

    return mask.reshape((height, width))
