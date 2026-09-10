"""Project 3 - Credit Card Fraud & Anomaly Detection.

Run:
    pip install streamlit pandas numpy scikit-learn plotly
    streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
N_TRANSACTIONS = 1_000
MODEL_FEATURES = ["Amount", "Time_Hour", "Distance_From_Home"]

st.set_page_config(
    page_title="Credit Card Fraud & Anomaly Detection",
    page_icon="💳",
    layout="wide",
)


# CRISP-DM Phase 1-2: Business Understanding and Data Understanding
@st.cache_data
def generate_transaction_data(n_records: int = N_TRANSACTIONS) -> pd.DataFrame:
    """Generate 1,000 reproducible synthetic credit card transactions."""
    rng = np.random.default_rng(RANDOM_STATE)
    anomaly_count = int(round(n_records * 0.05))
    anomaly_indices = rng.choice(n_records, size=anomaly_count, replace=False)
    anomaly_mask = np.zeros(n_records, dtype=bool)
    anomaly_mask[anomaly_indices] = True

    normal_count = n_records - anomaly_count
    amount = np.empty(n_records)
    time_hour = np.empty(n_records)
    distance = np.empty(n_records)

    normal_indices = np.flatnonzero(~anomaly_mask)
    amount[normal_indices] = np.clip(
        rng.lognormal(mean=3.35, sigma=0.65, size=normal_count),
        1,
        450,
    )
    time_hour[normal_indices] = rng.normal(13.0, 4.5, normal_count) % 24
    distance[normal_indices] = np.clip(
        rng.gamma(shape=2.0, scale=5.0, size=normal_count),
        0.2,
        55,
    )

    amount[anomaly_indices] = np.clip(
        rng.lognormal(mean=5.15, sigma=0.55, size=anomaly_count),
        250,
        2_500,
    )
    time_hour[anomaly_indices] = rng.choice(
        [0.5, 1.5, 2.5, 3.5, 22.5, 23.5],
        size=anomaly_count,
    )
    distance[anomaly_indices] = np.clip(
        rng.gamma(shape=3.5, scale=18.0, size=anomaly_count),
        35,
        250,
    )

    return pd.DataFrame(
        {
            "TransactionID": np.arange(1, n_records + 1),
            "Amount": np.round(amount, 2),
            "Time_Hour": np.round(time_hour, 2),
            "Distance_From_Home": np.round(distance, 2),
            "Is_Anomaly": anomaly_mask.astype(int),
        }
    )


# CRISP-DM Phase 3-5: Data Preparation, Modeling, and Evaluation
@st.cache_resource
def train_isolation_forest(
    data: pd.DataFrame,
    contamination_rate: float,
    n_estimators: int,
) -> dict:
    """Scale model features, fit IsolationForest, and return model outputs."""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[MODEL_FEATURES])

    isolation_forest = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination_rate,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    predicted_labels = isolation_forest.fit_predict(scaled_features)
    decision_scores = isolation_forest.decision_function(scaled_features)

    scored_data = data.copy()
    scored_data["Predicted_Label"] = predicted_labels
    scored_data["Decision_Score"] = np.round(decision_scores, 5)
    scored_data["Predicted_Status"] = np.where(
        predicted_labels == -1,
        "Anomaly",
        "Normal",
    )

    # PCA is included to provide a compact, optional 2D model-space view.
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_coordinates = pca.fit_transform(scaled_features)
    scored_data["PCA_1"] = pca_coordinates[:, 0]
    scored_data["PCA_2"] = pca_coordinates[:, 1]

    return {
        "model": isolation_forest,
        "scaler": scaler,
        "scaled_features": scaled_features,
        "scored_data": scored_data,
        "pca": pca,
    }


def build_overview_table() -> pd.DataFrame:
    """Describe how each CRISP-DM phase is applied to this project."""
    return pd.DataFrame(
        {
            "CRISP-DM Phase": [
                "Business Understanding",
                "Data Understanding",
                "Data Preparation",
                "Modeling",
                "Evaluation",
                "Deployment",
            ],
            "Application to Credit Card Anomaly Detection": [
                "Identify potentially fraudulent transactions early to reduce financial loss and protect customers.",
                "Explore transaction amount, time of day, distance from home, and approximately 5% synthetic ground-truth anomalies.",
                "Scale Amount, Time_Hour, and Distance_From_Home so each feature contributes appropriately to anomaly detection.",
                "Train IsolationForest to isolate unusual transactions using the selected contamination rate and estimators.",
                "Compare detected anomaly counts with the synthetic ground truth and inspect decision scores and visual patterns.",
                "Deploy an interactive Streamlit dashboard that scores new transactions in real time.",
            ],
        }
    )


def status_from_prediction(prediction: int) -> tuple[str, str]:
    """Return the required display text and an emoji for an IsolationForest label."""
    if prediction == -1:
        return "SUSPICIOUS / ANOMALY DETECTED", "🚨"
    return "NORMAL", "✅"


# CRISP-DM Phase 6: Deployment
def main() -> None:
    st.title("💳 Project 3 - Credit Card Fraud & Anomaly Detection")
    st.caption(
        "CRISP-DM project | IsolationForest anomaly detection with synthetic transactions"
    )

    data = generate_transaction_data()

    with st.sidebar:
        st.header("Model Parameters")
        contamination_rate = st.slider(
            "Contamination Rate",
            min_value=0.01,
            max_value=0.15,
            value=0.05,
            step=0.01,
            format="%.2f",
        )
        n_estimators = st.slider(
            "Number of Estimators",
            min_value=50,
            max_value=200,
            value=100,
            step=10,
        )
        st.markdown(
            "The model uses Amount, Time_Hour, and Distance_From_Home after scaling."
        )

    results = train_isolation_forest(data, contamination_rate, n_estimators)
    scored_data = results["scored_data"]
    detected_anomalies = int((scored_data["Predicted_Label"] == -1).sum())
    anomaly_percentage = detected_anomalies / len(scored_data) * 100

    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Total Transactions", f"{len(scored_data):,}")
    metric_2.metric("Detected Anomalies Count", f"{detected_anomalies:,}")
    metric_3.metric("Anomaly Percentage", f"{anomaly_percentage:.2f}%")

    overview_tab, data_tab, visualizer_tab, evaluator_tab = st.tabs(
        [
            "Project Overview",
            "Data Understanding",
            "Anomaly Visualizer",
            "Live Transaction Evaluator",
        ]
    )

    with overview_tab:
        st.subheader("CRISP-DM Framework")
        st.dataframe(
            build_overview_table(),
            use_container_width=True,
            hide_index=True,
        )
        st.info(
            "IsolationForest returns -1 for an anomaly and 1 for a normal transaction. "
            "The Contamination Rate controls the expected proportion of anomalies."
        )

        comparison = pd.crosstab(
            scored_data["Is_Anomaly"].map({0: "Normal", 1: "Anomaly"}),
            scored_data["Predicted_Status"],
            rownames=["Ground Truth"],
            colnames=["Model Prediction"],
        )
        st.subheader("Ground Truth vs. Model Prediction")
        st.dataframe(comparison, use_container_width=True)

    with data_tab:
        st.subheader("Transaction Dataset Preview")
        st.dataframe(data.head(10), use_container_width=True, hide_index=True)

        st.subheader("Transaction Amount Distribution")
        amount_histogram = px.histogram(
            scored_data,
            x="Amount",
            color="Predicted_Status",
            nbins=35,
            barmode="overlay",
            opacity=0.78,
            labels={
                "Amount": "Transaction Amount ($)",
                "Predicted_Status": "Model Status",
            },
            color_discrete_map={"Normal": "#2E8B57", "Anomaly": "#D62728"},
            title="Transaction Amounts: Normal vs. Detected Anomalies",
        )
        amount_histogram.update_layout(height=500)
        st.plotly_chart(amount_histogram, use_container_width=True)

    with visualizer_tab:
        st.subheader("Three-Dimensional Anomaly View")
        scatter_3d = px.scatter_3d(
            scored_data,
            x="Amount",
            y="Distance_From_Home",
            z="Time_Hour",
            color="Predicted_Status",
            hover_data=[
                "TransactionID",
                "Is_Anomaly",
                "Decision_Score",
            ],
            labels={
                "Amount": "Amount ($)",
                "Distance_From_Home": "Distance From Home (miles)",
                "Time_Hour": "Time of Day (hour)",
                "Predicted_Status": "Model Status",
            },
            color_discrete_map={"Normal": "#2E8B57", "Anomaly": "#D62728"},
            title="IsolationForest Results in Transaction Feature Space",
        )
        scatter_3d.update_traces(marker={"size": 4})
        scatter_3d.update_layout(height=680)
        st.plotly_chart(scatter_3d, use_container_width=True)

        st.subheader("PCA Model-Space View")
        pca_fig = px.scatter(
            scored_data,
            x="PCA_1",
            y="PCA_2",
            color="Predicted_Status",
            hover_data=["TransactionID", "Decision_Score"],
            color_discrete_map={"Normal": "#2E8B57", "Anomaly": "#D62728"},
            title="Two Principal Components of Scaled Transaction Features",
        )
        pca_fig.update_layout(height=450)
        st.plotly_chart(pca_fig, use_container_width=True)

    with evaluator_tab:
        st.subheader("Evaluate a New Transaction")
        st.write(
            "Adjust the transaction details below. The same scaler and IsolationForest "
            "model will classify the transaction."
        )

        input_col_1, input_col_2, input_col_3 = st.columns(3)
        with input_col_1:
            input_amount = st.slider(
                "Amount ($)",
                min_value=1.0,
                max_value=2_500.0,
                value=75.0,
                step=1.0,
            )
        with input_col_2:
            input_time = st.slider(
                "Time of Day (Hour)",
                min_value=0.0,
                max_value=24.0,
                value=13.0,
                step=0.5,
            )
        with input_col_3:
            input_distance = st.slider(
                "Distance from Home (miles)",
                min_value=0.0,
                max_value=250.0,
                value=5.0,
                step=0.5,
            )

        new_transaction = pd.DataFrame(
            [[input_amount, input_time, input_distance]],
            columns=MODEL_FEATURES,
        )
        new_transaction_scaled = results["scaler"].transform(new_transaction)
        prediction = int(results["model"].predict(new_transaction_scaled)[0])
        score = float(results["model"].decision_function(new_transaction_scaled)[0])
        status, emoji = status_from_prediction(prediction)

        if prediction == -1:
            st.error(f"{emoji} {status}")
        else:
            st.success(f"{emoji} {status}")

        result_col_1, result_col_2 = st.columns(2)
        result_col_1.metric("IsolationForest Label", prediction)
        result_col_2.metric("Decision Score", f"{score:.5f}")

        st.dataframe(new_transaction, use_container_width=True, hide_index=True)
        st.caption(
            "A lower decision score indicates a more unusual transaction. "
            "Scores below the learned threshold are classified as anomalies."
        )

        st.download_button(
            "Download Scored Transactions",
            data=scored_data.to_csv(index=False).encode("utf-8"),
            file_name="scored_credit_card_transactions.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()