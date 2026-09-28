import numpy as np
import pandas as pd

TRAFFIC_ORDER = {"Low": 0, "Medium": 1, "High": 2, "Jam": 3}
ONE_HOT_COLS = ["Weatherconditions", "Type_of_order", "Type_of_vehicle", "City"]
DROP_COLS = ["ID", "Delivery_person_ID", "Order_Date", "Time_Orderd", "Time_Order_picked",
             "Restaurant_latitude", "Restaurant_longitude",
             "Delivery_location_latitude", "Delivery_location_longitude"]


def haversine_km(lat1, lon1, lat2, lon2):
    """Distance in km between two GPS points (works on whole columns at once)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def add_features(df):
    """Create distance_km and prep_time_min."""
    df = df.copy()
    df["distance_km"] = haversine_km(df["Restaurant_latitude"], df["Restaurant_longitude"],
                                     df["Delivery_location_latitude"], df["Delivery_location_longitude"])
    ordered = pd.to_datetime(df["Time_Orderd"], format="%H:%M:%S")
    picked = pd.to_datetime(df["Time_Order_picked"], format="%H:%M:%S")
    minutes = (picked - ordered).dt.total_seconds() / 60
    df["prep_time_min"] = np.where(minutes < 0, minutes + 1440, minutes)  # order passed midnight
    return df


def encode_features(df):
    """Turn text columns into numbers and keep only the model columns."""
    df = df.copy()
    df["Road_traffic_density"] = df["Road_traffic_density"].map(TRAFFIC_ORDER)
    df["Festival"] = df["Festival"].map({"No": 0, "Yes": 1})
    df = pd.get_dummies(df, columns=ONE_HOT_COLS, drop_first=True)
    return df.drop(columns=DROP_COLS).astype(float)