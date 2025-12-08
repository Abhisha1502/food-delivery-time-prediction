import os
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(
    filename: str = "Food_Delivery_Time_Prediction.csv",
    data_dir: str = "data/raw"
) -> pd.DataFrame:
    """
    Load the dataset from the raw data directory.
    """
    filepath = os.path.join(data_dir, filename)
    df = pd.read_csv(filepath)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic missing value handling:
    - Numeric columns: fill with median
    - Categorical columns: fill with mode
    """
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    for col in numeric_cols:
        median_value = df[col].median()
        df[col].fillna(median_value, inplace=True)

    for col in cat_cols:
        mode_value = df[col].mode().iloc[0]
        df[col].fillna(mode_value, inplace=True)

    return df


def encode_categorical_features(
    df: pd.DataFrame,
    categorical_cols: List[str]
) -> Tuple[pd.DataFrame, OneHotEncoder]:
    """
    One-hot encode specified categorical columns and return transformed DataFrame.
    """
    df = df.copy()

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded_array = encoder.fit_transform(df[categorical_cols])

    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=df.index,
    )

    df = df.drop(columns=categorical_cols)
    df = pd.concat([df, encoded_df], axis=1)

    return df, encoder


def scale_features(
    X: pd.DataFrame,
    scaler: StandardScaler | None = None
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Scale features using StandardScaler. If scaler is not provided, fit a new one.
    Returns scaled array and the scaler object.
    """
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, scaler


def save_processed_data(
    df: pd.DataFrame,
    filename: str = "food_delivery_processed.csv",
    data_dir: str = "data/processed"
) -> None:
    """
    Save processed DataFrame to the processed data directory.
    """
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, filename)
    df.to_csv(filepath, index=False)


def load_and_preprocess(
    categorical_cols: List[str],
    target_col: str
) -> Tuple[pd.DataFrame, pd.Series, OneHotEncoder]:
    """
    High-level helper:
    1. Load raw data
    2. Handle missing values
    3. Encode categorical variables
    Returns:
      X (features), y (target), encoder
    """
    df = load_data()
    df = handle_missing_values(df)

    # Separate target
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Encode categoricals
    X, encoder = encode_categorical_features(X, categorical_cols)

    return X, y, encoder
