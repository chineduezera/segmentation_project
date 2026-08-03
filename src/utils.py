import pandas as pd
from pathlib import Path
def split_df_on_fold(val_fold_no: int, fold_df: pd.DataFrame, train_csv_df: pd.DataFrame, train_img_pth: Path):
    #Get all folds based on val_fold_no
    val_idx = fold_df.loc[fold_df["Folds"] == val_fold_no]
    image_id = val_idx.ImageId.tolist() #Image Id in list

    for id in image_id:
        if id in train_csv_df["ClassId"]:
            val = train_csv_df.loc[id, :]
             
