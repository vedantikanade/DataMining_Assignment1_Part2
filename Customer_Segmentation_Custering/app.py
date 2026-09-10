"""Project 2 - Customer Segmentation & Clustering.

Run:
    pip install streamlit pandas numpy scikit-learn plotly
    streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
N_CUSTOMERS = 400
FEATURES = ["Age", "Annual_Income_k", "Spending_Score"]

st.set_page_config(
    page_title="Project 2 - Customer Segmentation",
    page_icon="👥",
    layout="wide",
)


# CRISP-DM Phase 1-2: Business Understanding and Data Understanding
@st.cache_data
def generate_customer_data(n_records: int = N_CUSTOMERS) -> pd.DataFrame:
    """Generate a reproducible synthetic customer dataset."""
    rng = np.random.default_rng(RANDOM_STATE)
    data = pd.DataFrame(
        {
            "CustomerID": np.arange(1, n_records + 1),
            "Age": rng.integers(18, 71, n_records),
            "Annual_Income_k": rng.uniform(15, 140, n_records).round(1),
            "Spending_Score": rng.integers(1, 101, n_records),
        }
    )
    return data


# CRISP-DM Phase 3: Data Preparation
@st.cache_resource
def fit_clustering_model(data: pd.DataFrame, n_clusters: int) -> dict:
    """Standardize customer features, fit KMeans, and calculate silhouette score."""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[FEATURES])

    model = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init=10,
    )
    cluster_labels = model.fit_predict(scaled_features)

    clustered_data = data.copy()
    clustered_data["Cluster"] = cluster_labels
    score = silhouette_score(scaled_features, cluster_labels)

    return {
        "model": model,
        "scaler": scaler,
        "scaled_features": scaled_features,
        "clustered_data": clustered_data,
        "silhouette_score": score,
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
            "Application to Customer Segmentation": [
                "Group customers into useful personas to support targeted marketing and business decisions.",
                "Explore 400 synthetic customers using age, annual income, and spending score.",
                "Check feature ranges and standardize the three clustering variables with StandardScaler.",
                "Use KMeans to partition customers into the selected number of clusters.",
                "Use the Silhouette Score and visual cluster separation to assess clustering quality.",
                "Deploy an interactive Streamlit dashboard with charts and a live customer persona predictor.",
            ],
        }
    )


# CRISP-DM Phase 6: Deployment
def main() -> None:
    st.title("👥 Project 2 - Customer Segmentation & Clustering")
    st.caption(
        "CRISP-DM project | KMeans clustering with synthetic customer data"
    )

    data = generate_customer_data()

    with st.sidebar:
        st.header("Model Parameters")
        n_clusters = st.slider(
            "Number of Clusters (K)",
            min_value=2,
            max_value=8,
            value=4,
            step=1,
        )
        st.markdown(
            "The three clustering features are standardized before KMeans is fitted."
        )

    results = fit_clustering_model(data, n_clusters)
    clustered_data = results["clustered_data"]
    silhouette = results["silhouette_score"]

    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Total Customers", f"{len(data):,}")
    metric_2.metric("Selected K Clusters", n_clusters)
    metric_3.metric("Silhouette Score", f"{silhouette:.3f}")

    overview_tab, data_tab, analysis_tab, predictor_tab = st.tabs(
        [
            "Project Overview",
            "Data Understanding",
            "Cluster Analysis",
            "Live Predictor",
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
            "A higher Silhouette Score generally indicates better-defined clusters. "
            "The score should be interpreted together with the visualizations and "
            "the business meaning of each customer segment."
        )

    with data_tab:
        st.subheader("Customer Dataset Preview")
        st.dataframe(data.head(10), use_container_width=True, hide_index=True)

        st.subheader("Spending Score Distribution by Cluster")
        histogram = px.histogram(
            clustered_data,
            x="Spending_Score",
            color="Cluster",
            nbins=20,
            barmode="overlay",
            opacity=0.78,
            labels={
                "Spending_Score": "Spending Score",
                "Cluster": "Cluster",
            },
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )
        histogram.update_layout(
            height=480,
            legend_title_text="Cluster",
            bargap=0.08,
        )
        st.plotly_chart(histogram, use_container_width=True)

    with analysis_tab:
        st.subheader("Three-Dimensional Cluster Visualization")
        scatter_3d = px.scatter_3d(
            clustered_data,
            x="Annual_Income_k",
            y="Spending_Score",
            z="Age",
            color="Cluster",
            hover_data=["CustomerID"],
            labels={
                "Annual_Income_k": "Annual Income ($k)",
                "Spending_Score": "Spending Score",
                "Age": "Age",
                "Cluster": "Cluster",
            },
            color_discrete_sequence=px.colors.qualitative.Vivid,
            title="Customer Segments by Income, Spending, and Age",
        )
        scatter_3d.update_traces(marker={"size": 5})
        scatter_3d.update_layout(height=650, legend_title_text="Cluster")
        st.plotly_chart(scatter_3d, use_container_width=True)

        st.subheader("Cluster Summary")
        summary = (
            clustered_data.groupby("Cluster")[FEATURES]
            .agg(["count", "mean"])
            .round(2)
        )
        st.dataframe(summary, use_container_width=True)

    with predictor_tab:
        st.subheader("Predict a New Customer's Cluster")
        st.write(
            "Use the controls below to standardize a new customer's attributes "
            "and assign the nearest KMeans cluster."
        )

        input_col_1, input_col_2, input_col_3 = st.columns(3)
        with input_col_1:
            age = st.slider("Age", 18, 70, 35)
        with input_col_2:
            annual_income = st.slider(
                "Annual Income ($k)",
                15.0,
                140.0,
                60.0,
                step=0.5,
            )
        with input_col_3:
            spending_score = st.slider("Spending Score", 1, 100, 50)

        new_customer = pd.DataFrame(
            [[age, annual_income, spending_score]],
            columns=FEATURES,
        )
        new_customer_scaled = results["scaler"].transform(new_customer)
        predicted_cluster = int(
            results["model"].predict(new_customer_scaled)[0]
        )

        st.success(
            f"Predicted customer persona: Cluster {predicted_cluster}"
        )

        st.subheader("New Customer Details")
        st.dataframe(new_customer, use_container_width=True, hide_index=True)

        centroid = results["model"].cluster_centers_[predicted_cluster]
        centroid_original = results["scaler"].inverse_transform([centroid])[0]
        persona_profile = pd.DataFrame(
            {
                "Feature": FEATURES,
                "New Customer": [age, annual_income, spending_score],
                "Cluster Average": np.round(centroid_original, 2),
            }
        )
        st.write("Comparison with the predicted cluster's average profile:")
        st.dataframe(persona_profile, use_container_width=True, hide_index=True)

        st.caption(
            "Cluster numbers are machine-generated labels. Their numeric order "
            "does not represent a ranking from low to high value."
        )


if __name__ == "__main__":
    main()