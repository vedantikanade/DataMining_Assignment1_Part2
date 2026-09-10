# Prompt - Project 5: Data Science Skills Mastery Lab (Spotify / Netflix Analytics)

Act as a Senior Data Scientist and Lead Streamlit Developer.

Write a complete, single-file Streamlit application (`app.py`) for a data science course project following the CRISP-DM framework.

Project Details:
- Title: Project 5 - Spotify Music Audio Features & Hit Predictor Skills Lab
- Directory: 05_data_science_skills_lab
- Tech Stack: Python, Streamlit, Plotly Express, Scikit-learn (RandomForestClassifier, StandardScaler), Pandas, NumPy.

Requirements:
1. Data Generation / Loading:
   - Generate a synthetic Kaggle-style Spotify dataset of 500 tracks with columns: 'Track_Name', 'Artist', 'Genre' (Pop, Rock, Hip-Hop, Indie, Electronic), 'Popularity' (0-100), 'Danceability' (0-1.0), 'Energy' (0-1.0), 'Valence' (0-1.0), and 'Is_Hit' (Binary 0 or 1). Cache data generation.

2. Sidebar & Filtering Controls:
   - Sidebar filters for Genre selection, minimum Popularity slider, and model hyperparameters (Number of Estimators for Random Forest).

3. Top-Level Metrics Display:
   - 3 columns showing: Total Songs in Dataset, Average Popularity, and Model Accuracy Score (%).

4. Multi-Tab Layout:
   - Tab 1 ("Project Overview"): Display a clean DataFrame detailing the 6 CRISP-DM phases applied to Spotify Kaggle Music Analytics.
   - Tab 2 ("Data Understanding & EDA"): Display `df.head(10)` and interactive Plotly distribution/box plots for Danceability, Energy, and Valence grouped by Genre.
   - Tab 3 ("Correlation & Feature Analytics"): Display an interactive Plotly heatmap showing feature correlations (Energy vs Loudness/Valence) and scatter plots.
   - Tab 4 ("Live Hit Predictor"): Interactive sliders for Danceability, Energy, and Valence. Train a Random Forest Classifier on audio features and display a live prediction of whether the custom track configuration will be a "GLOBAL HIT 🚀" or "NICHE TRACK 🎵".

CRITICAL FIX REQUIREMENT:
- Ensure EVERY call to `st.plotly_chart()` includes a unique `key` parameter (e.g., `key="spotify_chart_1"`, `key="spotify_chart_2"`) to avoid StreamlitDuplicateElementId errors.

Ensure the code is complete, bug-free, fully self-contained, and ready to run with `streamlit run app.py`.