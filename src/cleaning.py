import numpy as np
import pandas as pd

COORD_COLS = ["Restaurant_latitude", "Restaurant_longitude",
              "Delivery_location_latitude", "Delivery_location_longitude"]


def load_raw(path):
    """Read the raw CSV file."""
    return pd.read_csv(path)


def clean_data(df):
    """Clean the raw delivery data and return a new DataFrame."""
    df = df.copy()

    # 1. remove spaces around every text value
    text_cols = df.select_dtypes(include=["object", "str"]).columns
    for col in text_cols:
        df[col] = df[col].str.strip()

    # 2. turn hidden "NaN" texts (even "conditions NaN") into real missing values
    for col in text_cols:
        mask = df[col].str.contains("NaN", na=False)
        df.loc[mask, col] = np.nan

    # 3. remove text prefixes and fix types
    df["Weatherconditions"] = df["Weatherconditions"].str.replace("conditions ", "", regex=False)
    df["Time_taken(min)"] = df["Time_taken(min)"].str.replace("(min) ", "", regex=False).astype(float)
    for col in ["Delivery_person_Age", "Delivery_person_Ratings", "multiple_deliveries"]:
        df[col] = df[col].astype(float)

    # 4. missing values: median for numbers, mode for categories
    for col in ["Delivery_person_Age", "Delivery_person_Ratings"]:
        df[col] = df[col].fillna(df[col].median())
    for col in ["multiple_deliveries", "Weatherconditions", "Road_traffic_density",
                "Festival", "City"]:
        df[col] = df[col].fillna(df[col].mode()[0])
    df = df.dropna(subset=["Time_Orderd"])  # a clock time can't be guessed

    # 5. coordinates: fix wrong signs, drop impossible values near 0
    df[COORD_COLS] = df[COORD_COLS].abs()
    df = df[(df[COORD_COLS] >= 1).all(axis=1)]

    # 6. duplicates
    df = df.drop_duplicates()
    return df.reset_index(drop=True)