"""Project 5 - Spotify Music Audio Features & Hit Predictor Skills Lab.

Run:
    pip install streamlit pandas numpy scikit-learn plotly
    streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
N_TRACKS = 500
GENRES = ["Pop", "Rock", "Hip-Hop", "Indie", "Electronic"]
AUDIO_FEATURES = ["Danceability", "Energy", "Valence"]

st.set_page_config(
    page_title="Spotify Hit Predictor Skills Lab",
    page_icon="🎵",
    layout="wide",
)


# CRISP-DM Phase 1-2: Business Understanding and Data Understanding
@st.cache_data
def generate_spotify_data(n_tracks: int = N_TRACKS) -> pd.DataFrame:
    """Generate a reproducible Kaggle-style Spotify track dataset."""
    rng = np.random.default_rng(RANDOM_STATE)
    genre_values = rng.choice(GENRES, size=n_tracks, replace=True)

    genre_energy_shift = {
        "Pop": 0.08,
        "Rock": 0.18,
        "Hip-Hop": 0.02,
        "Indie": -0.08,
        "Electronic": 0.15,
    }
    genre_dance_shift = {
        "Pop": 0.10,
        "Rock": -0.10,
        "Hip-Hop": 0.15,
        "Indie": -0.04,
        "Electronic": 0.12,
    }

    energy = np.array(
        [
            np.clip(rng.beta(5, 3) + genre_energy_shift[genre], 0, 1)
            for genre in genre_values
        ]
    )
    danceability = np.array(
        [
            np.clip(rng.beta(5, 3) + genre_dance_shift[genre], 0, 1)
            for genre in genre_values
        ]
    )
    valence = np.clip(rng.beta(4, 4, n_tracks), 0, 1)
    popularity = np.clip(
        100 * (0.45 * energy + 0.35 * danceability + 0.20 * valence)
        + rng.normal(0, 13, n_tracks),
        0,
        100,
    )
    loudness = np.clip(
        -18 + 10 * energy + rng.normal(0, 1.8, n_tracks),
        -30,
        -2,
    )

    hit_score = (
        0.045 * (popularity - 50)
        + 2.2 * (energy - 0.5)
        + 1.5 * (danceability - 0.5)
        + 0.9 * (valence - 0.5)
        + rng.normal(0, 0.7, n_tracks)
    )
    hit_probability = 1 / (1 + np.exp(-hit_score))
    is_hit = rng.binomial(1, hit_probability)
    # Guarantee both target classes in the synthetic sample.
    is_hit[0] = 0
    is_hit[1] = 1

    artists = [
        "Nova Pulse",
        "The Echoes",
        "Maya Lane",
        "Urban Static",
        "Solar Avenue",
        "Velvet Circuit",
        "Golden Hour",
        "Midnight Bloom",
    ]
    return pd.DataFrame(
        {
            "Track_Name": [f"Track {i:03d}" for i in range(1, n_tracks + 1)],
            "Artist": rng.choice(artists, size=n_tracks),
            "Genre": genre_values,
            "Popularity": np.round(popularity, 1),
            "Danceability": np.round(danceability, 3),
            "Energy": np.round(energy, 3),
            "Valence": np.round(valence, 3),
            "Loudness": np.round(loudness, 2),
            "Is_Hit": is_hit.astype(int),
        }
    )


# CRISP-DM Phase 3-5: Data Preparation, Modeling, and Evaluation
@st.cache_resource
def train_hit_classifier(data: pd.DataFrame, n_estimators: int) -> dict:
    """Scale audio features and train a Random Forest hit classifier."""
    x = data[AUDIO_FEATURES]
    y = data["Is_Hit"]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    return {
        "model": model,
        "scaler": scaler,
        "accuracy": accuracy,
    }


def build_overview_table() -> pd.DataFrame:
    """Describe the six CRISP-DM phases for Spotify analytics."""
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
            "Application to Spotify Kaggle Music Analytics": [
                "Understand which audio characteristics are associated with popular hit tracks.",
                "Explore genre, popularity, danceability, energy, valence, and loudness distributions.",
                "Clean numeric features and standardize audio attributes for machine learning.",
                "Train a Random Forest Classifier to predict whether a custom track is a hit.",
                "Evaluate the classifier using held-out test accuracy and interactive analytics.",
                "Deploy the learning experience as a Streamlit dashboard with a live hit predictor.",
            ],
        }
    )


def main() -> None:
    st.title("🎵 Project 5 - Spotify Music Audio Features & Hit Predictor Skills Lab")
    st.caption(
        "CRISP-DM project | Explore audio features and predict potential global hits"
    )

    data = generate_spotify_data()

    with st.sidebar:
        st.header("Filters and Model Controls")
        selected_genres = st.multiselect(
            "Genre selection",
            options=GENRES,
            default=GENRES,
        )
        minimum_popularity = st.slider(
            "Minimum Popularity",
            min_value=0,
            max_value=100,
            value=0,
            step=1,
        )
        n_estimators = st.slider(
            "Random Forest Number of Estimators",
            min_value=50,
            max_value=200,
            value=100,
            step=10,
        )

    if not selected_genres:
        filtered_data = data.iloc[0:0].copy()
    else:
        filtered_data = data[
            data["Genre"].isin(selected_genres)
            & (data["Popularity"] >= minimum_popularity)
        ].copy()

    if filtered_data.empty:
        st.warning(
            "No tracks match the current filters. The charts below use the full dataset "
            "until at least one track is selected."
        )
        display_data = data.copy()
    else:
        display_data = filtered_data

    model_results = train_hit_classifier(data, n_estimators)
    accuracy_percent = model_results["accuracy"] * 100

    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Total Songs in Dataset", f"{len(display_data):,}")
    metric_2.metric(
        "Average Popularity",
        f"{display_data['Popularity'].mean():.1f}",
    )
    metric_3.metric("Model Accuracy Score", f"{accuracy_percent:.2f}%")

    overview_tab, eda_tab, correlation_tab, predictor_tab = st.tabs(
        [
            "Project Overview",
            "Data Understanding & EDA",
            "Correlation & Feature Analytics",
            "Live Hit Predictor",
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
            "Use the sidebar to filter the exploratory views. The Random Forest model "
            "is trained on all 500 synthetic tracks so its accuracy remains comparable "
            "when the dashboard filters change."
        )

    with eda_tab:
        st.subheader("Dataset Preview")
        st.dataframe(
            display_data.head(10),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Audio Feature Distributions by Genre")
        feature_to_plot = st.selectbox(
            "Choose an audio feature",
            options=AUDIO_FEATURES,
        )
        distribution_fig = px.histogram(
            display_data,
            x=feature_to_plot,
            color="Genre",
            nbins=25,
            barmode="overlay",
            opacity=0.75,
            marginal="box",
            title=f"{feature_to_plot} Distribution by Genre",
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )
        distribution_fig.update_layout(height=520)
        st.plotly_chart(
            distribution_fig,
            use_container_width=True,
            key="spotify_chart_feature_distribution",
        )

        box_fig = px.box(
            display_data,
            x="Genre",
            y=feature_to_plot,
            color="Genre",
            points="all",
            title=f"{feature_to_plot} Box Plot by Genre",
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )
        box_fig.update_layout(height=520, showlegend=False)
        st.plotly_chart(
            box_fig,
            use_container_width=True,
            key="spotify_chart_feature_box",
        )

    with correlation_tab:
        st.subheader("Feature Correlation Heatmap")
        correlation_columns = [
            "Popularity",
            "Danceability",
            "Energy",
            "Valence",
            "Loudness",
        ]
        correlation_matrix = display_data[correlation_columns].corr()
        heatmap_fig = px.imshow(
            correlation_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
            title="Correlation Between Popularity and Audio Features",
        )
        heatmap_fig.update_layout(height=560)
        st.plotly_chart(
            heatmap_fig,
            use_container_width=True,
            key="spotify_chart_correlation_heatmap",
        )

        scatter_col_1, scatter_col_2 = st.columns(2)
        with scatter_col_1:
            energy_loudness_fig = px.scatter(
                display_data,
                x="Energy",
                y="Loudness",
                color="Genre",
                size="Popularity",
                hover_data=["Track_Name", "Artist", "Is_Hit"],
                title="Energy vs. Loudness",
                color_discrete_sequence=px.colors.qualitative.Vivid,
            )
            energy_loudness_fig.update_layout(height=500)
            st.plotly_chart(
                energy_loudness_fig,
                use_container_width=True,
                key="spotify_chart_energy_loudness",
            )

        with scatter_col_2:
            energy_valence_fig = px.scatter(
                display_data,
                x="Energy",
                y="Valence",
                color="Genre",
                size="Popularity",
                hover_data=["Track_Name", "Artist", "Is_Hit"],
                title="Energy vs. Valence",
                color_discrete_sequence=px.colors.qualitative.Vivid,
            )
            energy_valence_fig.update_layout(height=500)
            st.plotly_chart(
                energy_valence_fig,
                use_container_width=True,
                key="spotify_chart_energy_valence",
            )

    with predictor_tab:
        st.subheader("Predict Hit Potential for a Custom Track")
        st.write(
            "Adjust the audio features below. The Random Forest Classifier will "
            "standardize the values and predict the track category."
        )

        predictor_col_1, predictor_col_2, predictor_col_3 = st.columns(3)
        with predictor_col_1:
            custom_danceability = st.slider(
                "Danceability",
                min_value=0.0,
                max_value=1.0,
                value=0.70,
                step=0.01,
            )
        with predictor_col_2:
            custom_energy = st.slider(
                "Energy",
                min_value=0.0,
                max_value=1.0,
                value=0.70,
                step=0.01,
            )
        with predictor_col_3:
            custom_valence = st.slider(
                "Valence",
                min_value=0.0,
                max_value=1.0,
                value=0.60,
                step=0.01,
            )

        custom_track = pd.DataFrame(
            [[custom_danceability, custom_energy, custom_valence]],
            columns=AUDIO_FEATURES,
        )
        custom_scaled = model_results["scaler"].transform(custom_track)
        prediction = int(model_results["model"].predict(custom_scaled)[0])
        probabilities = model_results["model"].predict_proba(custom_scaled)[0]
        hit_probability = float(
            probabilities[list(model_results["model"].classes_).index(1)]
        )

        if prediction == 1:
            st.success(
                f"🌍 GLOBAL HIT 🚀 — Estimated hit probability: "
                f"{hit_probability:.1%}"
            )
        else:
            st.info(
                f"🎵 NICHE TRACK 🎵 — Estimated hit probability: "
                f"{hit_probability:.1%}"
            )

        st.dataframe(
            custom_track,
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "This educational prediction reflects patterns in the synthetic dataset "
            "and should not be interpreted as a guarantee of real-world success."
        )


if __name__ == "__main__":
    main()
