from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL = joblib.load(ROOT / "models" / "final_model.joblib")
COLUMNS = joblib.load(ROOT / "models" / "feature_columns.joblib")

TRAFFIC = {"Low": 0, "Medium": 1, "High": 2, "Jam": 3}


def predict_delivery_time(age, rating, traffic, vehicle_condition,
                          multiple_deliveries, festival, distance_km,
                          prep_time_min, weather, order_type,
                          vehicle_type, city):
    if traffic not in TRAFFIC:
        raise ValueError(f"traffic must be one of {list(TRAFFIC)}")
    if not 0 <= distance_km <= 100:
        raise ValueError("distance_km must be between 0 and 100")
    if not 1 <= rating <= 5:
        raise ValueError("rating must be between 1 and 5")

    # start with all columns at 0, then fill in the ones we know
    row = dict.fromkeys(COLUMNS, 0.0)
    row["Delivery_person_Age"] = age
    row["Delivery_person_Ratings"] = rating
    row["Road_traffic_density"] = TRAFFIC[traffic]
    row["Vehicle_condition"] = vehicle_condition
    row["multiple_deliveries"] = multiple_deliveries
    row["Festival"] = 1.0 if festival else 0.0
    row["distance_km"] = distance_km
    row["prep_time_min"] = prep_time_min

    # one-hot columns: the dropped category has no column, so all zeros = that category
    for name in (f"Weatherconditions_{weather}", f"Type_of_order_{order_type}",
                 f"Type_of_vehicle_{vehicle_type}", f"City_{city}"):
        if name in row:
            row[name] = 1.0

    X = pd.DataFrame([row])[COLUMNS]
    return float(MODEL.predict(X)[0])