"""NYC Taxi Trip Duration - CRISP-DM Streamlit Dashboard.

Run from the VS Code terminal with:
    pip install streamlit pandas numpy scikit-learn plotly
    streamlit run app.py

The application uses 1,500 reproducible synthetic records modeled after the
feature structure of the Kaggle NYC Taxi Trip Duration challenge.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
N_RECORDS = 1_500
MODEL_FEATURES = [
    "passenger_count",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "haversine_distance_km",
    "manhattan_distance_km",
    "pickup_hour",
    "day_of_week",
]

st.set_page_config(
    page_title="NYC Taxi Trip Duration",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)


# CRISP-DM Phase 1-2: Business Understanding and Data Understanding/Generation
@st.cache_data
def generate_synthetic_data(n_records: int = N_RECORDS) -> pd.DataFrame:
    """Generate realistic-looking NYC taxi trip records."""
    rng = np.random.default_rng(RANDOM_STATE)

    # NYC bounds centered around Manhattan, with a small surrounding area.
    pickup_lat = rng.uniform(40.700, 40.850, n_records)
    pickup_lon = rng.uniform(-74.020, -73.930, n_records)

    # Generate destinations as local coordinate offsets and clip to NYC bounds.
    lat_offset = rng.normal(0.0, 0.045, n_records)
    lon_offset = rng.normal(0.0, 0.050, n_records)
    dropoff_lat = np.clip(pickup_lat + lat_offset, 40.630, 40.900)
    dropoff_lon = np.clip(pickup_lon + lon_offset, -74.120, -73.850)

    start_date = datetime(2016, 1, 1)
    minutes = rng.integers(0, 180 * 24 * 60, n_records)
    pickup_datetime = pd.to_datetime(
        [start_date + timedelta(minutes=int(m)) for m in minutes]
    )

    passenger_count = rng.choice(
        [1, 2, 3, 4, 5, 6],
        size=n_records,
        p=[0.62, 0.20, 0.08, 0.05, 0.03, 0.02],
    )

    generated = pd.DataFrame(
        {
            "pickup_datetime": pickup_datetime,
            "passenger_count": passenger_count,
            "pickup_longitude": pickup_lon,
            "pickup_latitude": pickup_lat,
            "dropoff_longitude": dropoff_lon,
            "dropoff_latitude": dropoff_lat,
        }
    )
    generated["pickup_hour"] = generated["pickup_datetime"].dt.hour
    generated["day_of_week"] = generated["pickup_datetime"].dt.dayofweek

    generated["haversine_distance_km"] = haversine_distance(
        generated["pickup_latitude"],
        generated["pickup_longitude"],
        generated["dropoff_latitude"],
        generated["dropoff_longitude"],
    )
    generated["manhattan_distance_km"] = manhattan_distance(
        generated["pickup_latitude"],
        generated["pickup_longitude"],
        generated["dropoff_latitude"],
        generated["dropoff_longitude"],
    )

    # Construct duration from distance, traffic patterns, passenger count, and noise.
    rush_hour = generated["pickup_hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(float)
    weekend = generated["day_of_week"].isin([5, 6]).astype(float)
    distance = generated["manhattan_distance_km"]
    speed_kmh = 24 - (rush_hour * 7) + (weekend * 3)
    base_minutes = (distance / speed_kmh.clip(lower=10)) * 60
    traffic_delay = rush_hour * rng.uniform(3, 12, n_records)
    passenger_delay = np.maximum(generated["passenger_count"] - 1, 0) * 0.7
    noise = rng.normal(0, 3.5, n_records)

    generated["trip_duration"] = np.maximum(
        90, base_minutes + traffic_delay + passenger_delay + noise + 2.5
    ).round().astype(int)

    return generated


# CRISP-DM Phase 3: Data Preparation and Feature Engineering
def haversine_distance(
    lat1: pd.Series | np.ndarray,
    lon1: pd.Series | np.ndarray,
    lat2: pd.Series | np.ndarray,
    lon2: pd.Series | np.ndarray,
) -> np.ndarray:
    """Return great-circle distance in kilometers."""
    earth_radius_km = 6371.0
    lat1_rad, lat2_rad = np.radians(lat1), np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    )
    return earth_radius_km * 2 * np.arcsin(np.sqrt(a))


def manhattan_distance(
    lat1: pd.Series | np.ndarray,
    lon1: pd.Series | np.ndarray,
    lat2: pd.Series | np.ndarray,
    lon2: pd.Series | np.ndarray,
) -> np.ndarray:
    """Approximate Manhattan distance in kilometers using latitude/longitude legs."""
    vertical = haversine_distance(lat1, lon1, lat2, lon1)
    horizontal = haversine_distance(lat2, lon1, lat2, lon2)
    return vertical + horizontal


def prepare_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create model matrix X and log-transformed target y."""
    prepared = data.copy()
    prepared["pickup_datetime"] = pd.to_datetime(prepared["pickup_datetime"])
    prepared["pickup_hour"] = prepared["pickup_datetime"].dt.hour
    prepared["day_of_week"] = prepared["pickup_datetime"].dt.dayofweek
    prepared["haversine_distance_km"] = haversine_distance(
        prepared["pickup_latitude"],
        prepared["pickup_longitude"],
        prepared["dropoff_latitude"],
        prepared["dropoff_longitude"],
    )
    prepared["manhattan_distance_km"] = manhattan_distance(
        prepared["pickup_latitude"],
        prepared["pickup_longitude"],
        prepared["dropoff_latitude"],
        prepared["dropoff_longitude"],
    )
    x = prepared[MODEL_FEATURES]
    y = np.log1p(prepared["trip_duration"])
    return x, y


# CRISP-DM Phase 4-5: Modeling and Evaluation
@st.cache_resource
def train_model(data: pd.DataFrame) -> dict:
    """Train the Random Forest and return evaluation artifacts."""
    x, y = prepare_features(data)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=RANDOM_STATE
    )

    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=16,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    predicted_log = model.predict(x_test)
    actual = np.expm1(y_test)
    predicted = np.maximum(0, np.expm1(predicted_log))
    residuals = actual - predicted

    metrics = {
        "R² Score": r2_score(actual, predicted),
        "RMSE (minutes)": np.sqrt(mean_squared_error(actual, predicted)),
        "MAE (minutes)": mean_absolute_error(actual, predicted),
    }
    evaluation = pd.DataFrame(
        {
            "Actual Duration": actual,
            "Predicted Duration": predicted,
            "Residual": residuals,
        }
    ).reset_index(drop=True)
    importance = pd.DataFrame(
        {"Feature": MODEL_FEATURES, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=True)

    return {
        "model": model,
        "metrics": metrics,
        "evaluation": evaluation,
        "importance": importance,
    }


def make_prediction_row(values: dict) -> pd.DataFrame:
    """Convert dashboard controls into the same engineered feature schema."""
    row = pd.DataFrame([values])
    row["haversine_distance_km"] = haversine_distance(
        row["pickup_latitude"],
        row["pickup_longitude"],
        row["dropoff_latitude"],
        row["dropoff_longitude"],
    )
    row["manhattan_distance_km"] = manhattan_distance(
        row["pickup_latitude"],
        row["pickup_longitude"],
        row["dropoff_latitude"],
        row["dropoff_longitude"],
    )
    return row[MODEL_FEATURES]


# CRISP-DM Phase 6: Deployment - Streamlit user interface
def main() -> None:
    st.title("🚕 NYC Taxi Trip Duration Predictor")
    st.caption(
        "CRISP-DM project | Synthetic version of the Kaggle NYC Taxi Trip Duration challenge"
    )

    data = generate_synthetic_data()
    results = train_model(data)

    with st.sidebar:
        st.header("Trip Inputs")
        st.write("Adjust the trip variables to generate a live prediction.")
        passenger_count = st.slider("Passenger count", 1, 6, 1)
        pickup_hour = st.slider("Pickup hour", 0, 23, 12)
        day_of_week = st.slider("Day of week (0=Mon, 6=Sun)", 0, 6, 2)
        pickup_latitude = st.slider(
            "Pickup latitude", 40.63, 40.90, 40.7580, 0.0001
        )
        pickup_longitude = st.slider(
            "Pickup longitude", -74.12, -73.85, -73.9855, 0.0001
        )
        dropoff_latitude = st.slider(
            "Drop-off latitude", 40.63, 40.90, 40.7484, 0.0001
        )
        dropoff_longitude = st.slider(
            "Drop-off longitude", -74.12, -73.85, -73.9857, 0.0001
        )

        input_values = {
            "passenger_count": passenger_count,
            "pickup_longitude": pickup_longitude,
            "pickup_latitude": pickup_latitude,
            "dropoff_longitude": dropoff_longitude,
            "dropoff_latitude": dropoff_latitude,
            "pickup_hour": pickup_hour,
            "day_of_week": day_of_week,
        }
        prediction_row = make_prediction_row(input_values)
        prediction = float(np.expm1(results["model"].predict(prediction_row)[0]))

    metric_cols = st.columns(4)
    metric_cols[0].metric("Predicted duration", f"{prediction:.1f} min")
    metric_cols[1].metric("R² Score", f"{results['metrics']['R² Score']:.3f}")
    metric_cols[2].metric(
        "RMSE", f"{results['metrics']['RMSE (minutes)']:.2f} min"
    )
    metric_cols[3].metric(
        "MAE", f"{results['metrics']['MAE (minutes)']:.2f} min"
    )

    overview_tab, evaluation_tab, prediction_tab, data_tab = st.tabs(
        ["Project Overview", "Evaluation", "Live Prediction", "Generated Data"]
    )

    with overview_tab:
        st.subheader("CRISP-DM workflow")
        overview = pd.DataFrame(
            {
                "Phase": [
                    "Business Understanding",
                    "Data Understanding",
                    "Data Preparation",
                    "Modeling",
                    "Evaluation",
                    "Deployment",
                ],
                "Implementation": [
                    "Estimate taxi trip duration for operational planning.",
                    "Generate and inspect 1,500 NYC-like taxi records.",
                    "Engineer Haversine/Manhattan distances and time features.",
                    "Train a Random Forest Regressor on log1p(duration).",
                    "Compare R², RMSE, MAE, feature importance, and residuals.",
                    "Provide a Streamlit dashboard with live prediction and map.",
                ],
            }
        )
        st.table(overview)
        st.info(
            "This educational app uses synthetic data so it runs without downloading "
            "the large Kaggle file. The modeling pipeline can be adapted to the real "
            "train.csv by replacing the generation function."
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Trip duration distribution")
            fig = px.histogram(
                data,
                x="trip_duration",
                nbins=35,
                labels={"trip_duration": "Trip duration (minutes)"},
                color_discrete_sequence=["#2E86AB"],
            )
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Distance relationship")
            fig = px.scatter(
                data,
                x="manhattan_distance_km",
                y="trip_duration",
                color="pickup_hour",
                labels={
                    "manhattan_distance_km": "Manhattan distance (km)",
                    "trip_duration": "Trip duration (minutes)",
                },
                color_continuous_scale="Turbo",
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    with evaluation_tab:
        st.subheader("Feature importance")
        importance_fig = px.bar(
            results["importance"],
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues",
        )
        importance_fig.update_layout(height=430, showlegend=False)
        st.plotly_chart(importance_fig, use_container_width=True)

        st.subheader("Residual analysis")
        left, right = st.columns(2)
        with left:
            residual_fig = px.scatter(
                results["evaluation"],
                x="Predicted Duration",
                y="Residual",
                labels={"Residual": "Actual - Predicted (minutes)"},
                color_discrete_sequence=["#F18F01"],
            )
            residual_fig.add_hline(y=0, line_dash="dash", line_color="black")
            residual_fig.update_layout(height=400)
            st.plotly_chart(residual_fig, use_container_width=True)
        with right:
            distribution_fig = px.histogram(
                results["evaluation"],
                x="Residual",
                nbins=30,
                labels={"Residual": "Residual (minutes)"},
                color_discrete_sequence=["#C73E1D"],
            )
            distribution_fig.add_vline(x=0, line_dash="dash", line_color="black")
            distribution_fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(distribution_fig, use_container_width=True)

        st.subheader("Evaluation metrics")
        metric_table = pd.DataFrame(
            [{"Metric": key, "Value": value} for key, value in results["metrics"].items()]
        )
        st.dataframe(metric_table, hide_index=True, use_container_width=True)

    with prediction_tab:
        st.subheader("Live trip prediction")
        st.success(f"Estimated trip duration: {prediction:.1f} minutes")
        distance_col1, distance_col2 = st.columns(2)
        distance_col1.metric(
            "Haversine distance",
            f"{prediction_row['haversine_distance_km'].iloc[0]:.2f} km",
        )
        distance_col2.metric(
            "Manhattan distance",
            f"{prediction_row['manhattan_distance_km'].iloc[0]:.2f} km",
        )

        st.subheader("Pickup and drop-off map")
        map_data = pd.DataFrame(
            {
                "latitude": [pickup_latitude, dropoff_latitude],
                "longitude": [pickup_longitude, dropoff_longitude],
                "location": ["Pickup", "Drop-off"],
            }
        )
        map_fig = px.scatter_map(
            map_data,
            lat="latitude",
            lon="longitude",
            hover_name="location",
            color="location",
            zoom=10,
            center={"lat": pickup_latitude, "lon": pickup_longitude},
            height=500,
            map_style="open-street-map",
            color_discrete_map={"Pickup": "#1f77b4", "Drop-off": "#d62728"},
        )
        map_fig.add_trace(
            go.Scattermap(
                lat=[pickup_latitude, dropoff_latitude],
                lon=[pickup_longitude, dropoff_longitude],
                mode="lines",
                line={"width": 3, "color": "#555555"},
                name="Trip route",
                hoverinfo="skip",
            )
        )
        st.plotly_chart(map_fig, use_container_width=True)

    with data_tab:
        st.subheader("Synthetic dataset preview")
        st.write(f"Rows: {len(data):,} | Columns: {len(data.columns)}")
        st.dataframe(data.head(25), use_container_width=True, hide_index=True)
        st.download_button(
            "Download generated CSV",
            data=data.to_csv(index=False).encode("utf-8"),
            file_name="synthetic_nyc_taxi_trip_duration.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
