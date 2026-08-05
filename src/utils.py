import pandas as pd

def get_validation_subset_for_fold(
    val_fold_no: int,
    train_fold_df: pd.DataFrame,
    train_csv_df: pd.DataFrame,
):
    """
    Splits a dataset into training and validation subsets based on a specified fold number.

    This function combines the clean data from the training fold with the training CSV data, merges them on 'ImageId', and then splits the resulting DataFrame into:
    - `train_df`: Data points where the 'Fold' column equals the specified fold number.
    - `val_df`: All other data points (i.e., data points where 'Fold' does not equal the specified fold number).

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
    concat_train_csv = concat_train_csv.merge(train_fold, how="inner", on="ImageId")
    train_df = concat_train_csv.loc[concat_train_csv["Fold"] == val_fold_no]
    val_df = concat_train_csv.loc[concat_train_csv["Fold"] != val_fold_no]
    return train_df, val_df
