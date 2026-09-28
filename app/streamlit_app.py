import sys
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.predict import predict_delivery_time

st.set_page_config(page_title="Delivery ETA", layout="wide")
st.title("Delivery time prediction")


@st.cache_data
def load_assets():
    data = pd.read_csv(ROOT / "data" / "processed" / "clean_data.csv")
    preds = pd.read_csv(ROOT / "models" / "test_predictions.csv")
    imp = pd.read_csv(ROOT / "models" / "feature_importance.csv")
    with open(ROOT / "models" / "metrics.json") as f:
        metrics = json.load(f)
    return data, preds, imp, metrics


data, preds, imp, metrics = load_assets()
tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Data", "Model metrics", "Accuracy"])

# ---------- Tab 1: prediction ----------
with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        distance = st.slider("Distance (km)", 0.5, 25.0, 8.0, 0.5)
        traffic = st.selectbox("Traffic", ["Low", "Medium", "High", "Jam"])
        weather = st.selectbox("Weather", ["Cloudy", "Fog", "Sandstorms", "Stormy", "Sunny", "Windy"])
        city = st.selectbox("City type", ["Metropolitian", "Urban", "Semi-Urban"])
    with c2:
        age = st.slider("Rider age", 20, 39, 30)
        rating = st.slider("Rider rating", 2.5, 5.0, 4.7, 0.1)
        multiple = st.selectbox("Deliveries in the same trip", [0, 1, 2, 3])
        condition = st.selectbox("Vehicle condition (0 = worst, 2 = best)", [0, 1, 2])
    with c3:
        vehicle = st.selectbox("Vehicle type", ["motorcycle", "scooter", "electric_scooter"])
        order = st.selectbox("Order type", ["Buffet", "Drinks", "Meal", "Snack"])
        prep = st.selectbox("Preparation time (min)", [5, 10, 15], index=1)
        festival = st.checkbox("Festival day")

    if st.button("Predict", type="primary"):
        try:
            minutes = predict_delivery_time(age, rating, traffic, condition, multiple,
                                            festival, distance, prep, weather, order,
                                            vehicle, city)
            mae = metrics["test"]["MAE"]
            st.metric("Estimated delivery time", f"{minutes:.0f} min")
            st.info(f"Likely range: {minutes - mae:.0f} to {minutes + mae:.0f} minutes "
                    f"(the model is off by {mae:.1f} minutes on average)")
        except ValueError as e:
            st.error(str(e))

# ---------- Tab 2: data ----------
with tab2:
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots()
        sns.histplot(data["Time_taken(min)"], bins=30, kde=True, ax=ax)
        ax.set_title("Distribution of delivery time")
        st.pyplot(fig)
    with b:
        fig, ax = plt.subplots()
        sns.boxplot(data=data, x="Road_traffic_density", y="Time_taken(min)", ax=ax)
        ax.set_xticklabels(["Low", "Medium", "High", "Jam"])
        ax.set_title("Delivery time by traffic")
        st.pyplot(fig)

    cols = ["Delivery_person_Age", "Delivery_person_Ratings", "Road_traffic_density",
            "multiple_deliveries", "Vehicle_condition", "distance_km", "Time_taken(min)"]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(data[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation matrix")
    st.pyplot(fig)

# ---------- Tab 3: metrics ----------
with tab3:
    t = metrics["test"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE (min)", f"{t['MAE']:.2f}")
    m2.metric("RMSE (min)", f"{t['RMSE']:.2f}")
    m3.metric("R²", f"{t['R2']:.3f}")
    m4.metric("Adjusted R²", f"{t['Adj_R2']:.3f}")
    st.subheader("Train vs test")
    st.table(pd.DataFrame(metrics).T.round(3))
    st.subheader("Most important features")
    st.bar_chart(imp.head(10).set_index("feature")["importance"])

# ---------- Tab 4: accuracy ----------
with tab4:
    errors = preds["predicted"] - preds["actual"]
    st.metric("Predictions within ±5 minutes", f"{(errors.abs() <= 5).mean() * 100:.0f}%")
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots()
        ax.scatter(preds["actual"], preds["predicted"], alpha=0.2)
        lim = [preds["actual"].min(), preds["actual"].max()]
        ax.plot(lim, lim, "r--")
        ax.set_xlabel("Actual time (min)")
        ax.set_ylabel("Predicted time (min)")
        ax.set_title("Actual vs predicted")
        st.pyplot(fig)
    with b:
        fig, ax = plt.subplots()
        sns.histplot(errors, bins=40, kde=True, ax=ax)
        ax.set_xlabel("Error (predicted - actual, min)")
        ax.set_title("Prediction errors")
        st.pyplot(fig)