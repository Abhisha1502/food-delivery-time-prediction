import numpy as np
import pandas as pd


def create_delivery_status(
    df: pd.DataFrame,
    time_col: str = "Delivery_Time",
    threshold_minutes: int = 30,
    new_col: str = "Delivery_Status"
) -> pd.DataFrame:
    """
    Create a binary classification target:
    - 1 (Delayed) if delivery time > threshold
    - 0 (Fast) otherwise
    """
    df = df.copy()
    df[new_col] = (df[time_col] > threshold_minutes).astype(int)
    return df


def extract_time_features(
    df: pd.DataFrame,
    time_col: str = "Order_Time"
) -> pd.DataFrame:
    """
    Example: create hour-of-day and rush hour from an order time column.
    Assumes time_col is parseable by pandas.to_datetime.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    df["Order_Hour"] = df[time_col].dt.hour
    df["Order_DayOfWeek"] = df[time_col].dt.dayofweek

    # Simple rush hour feature (customize as needed)
    df["Is_Rush_Hour"] = df["Order_Hour"].isin([11, 12, 13, 19, 20, 21]).astype(int)

    return df


def haversine_distance(
    lat1, lon1, lat2, lon2
) -> np.ndarray:
    """
    Compute Haversine distance between two points on Earth.
    All inputs are in degrees. Output in kilometers.
    """
    R = 6371  # Earth radius (km)
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c
    return distance


def add_distance_feature(
    df: pd.DataFrame,
    rest_lat_col: str = "Restaurant_Latitude",
    rest_lon_col: str = "Restaurant_Longitude",
    cust_lat_col: str = "Customer_Latitude",
    cust_lon_col: str = "Customer_Longitude",
    new_col: str = "Distance_km"
) -> pd.DataFrame:
    """
    Add a distance feature based on restaurant and customer coordinates.
    """
    df = df.copy()
    df[new_col] = haversine_distance(
        df[rest_lat_col],
        df[rest_lon_col],
        df[cust_lat_col],
        df[cust_lon_col],
    )
    return df
