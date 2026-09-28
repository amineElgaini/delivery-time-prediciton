import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.cleaning import load_raw, clean_data
from src.features import add_features, encode_features

ROOT = Path(__file__).resolve().parent.parent
TARGET = "Time_taken(min)"
NUMERIC_FEATURES = ["Delivery_person_Age", "Delivery_person_Ratings", "distance_km", "prep_time_min"]
BEST_PARAMS = dict(n_estimators=300, max_depth=15, min_samples_leaf=5, max_features=0.5)


def evaluate(model, X, y):
    """Return MAE, RMSE, R2 and adjusted R2."""
    pred = model.predict(X)
    r2 = r2_score(y, pred)
    n, p = X.shape
    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "R2": r2,
        "Adj_R2": 1 - (1 - r2) * (n - 1) / (n - p - 1),
    }


def build_pipeline():
    preprocess = ColumnTransformer(
        [("scale", StandardScaler(), NUMERIC_FEATURES)],
        remainder="passthrough",
    )
    model = RandomForestRegressor(**BEST_PARAMS, random_state=42, n_jobs=-1)
    return Pipeline([("prep", preprocess), ("model", model)])


def main():
    models_dir = ROOT / "models"
    processed_dir = ROOT / "data" / "processed"
    models_dir.mkdir(exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. data
    raw = load_raw(ROOT / "data" / "raw" / "Food_delivery.csv")
    df = encode_features(add_features(clean_data(raw)))
    df.to_csv(processed_dir / "clean_data.csv", index=False)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. train
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    # 3. evaluate
    metrics = {"train": evaluate(pipe, X_train, y_train), "test": evaluate(pipe, X_test, y_test)}
    for label, m in metrics.items():
        print(f"{label:5s} | MAE {m['MAE']:.2f} | RMSE {m['RMSE']:.2f} | "
              f"R2 {m['R2']:.3f} | Adj R2 {m['Adj_R2']:.3f}")

    # 4. save everything
    joblib.dump(pipe, models_dir / "final_model.joblib")
    joblib.dump(pipe.named_steps["prep"].named_transformers_["scale"], models_dir / "scaler.joblib")
    joblib.dump(list(X.columns), models_dir / "feature_columns.joblib")
    with open(models_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame({"actual": y_test.values, "predicted": pipe.predict(X_test)}) \
        .to_csv(models_dir / "test_predictions.csv", index=False)

    names = NUMERIC_FEATURES + [c for c in X.columns if c not in NUMERIC_FEATURES]
    pd.DataFrame({"feature": names, "importance": pipe.named_steps["model"].feature_importances_}) \
        .sort_values("importance", ascending=False) \
        .to_csv(models_dir / "feature_importance.csv", index=False)
    print("Saved model and files in", models_dir)


if __name__ == "__main__":
    main()