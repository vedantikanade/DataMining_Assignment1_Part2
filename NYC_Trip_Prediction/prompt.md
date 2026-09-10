# Project 1: NYC Taxi Trip Duration Predictor

## Baseline Prompt
Now on to another project - You will do end2end a data science project including data, training, deployment, crisp-dm framework, and an awesome front end... Kaggle NYC taxi challenge.

## Expanded AI Prompt Used
Act as an expert Data Scientist. Generate a complete, end-to-end Python script using Streamlit (`app.py`) for the Kaggle NYC Taxi Trip Duration challenge following the CRISP-DM framework.
Requirements:
1. Data Understanding & Generation: Generate 1,500 synthetic records with realistic NYC taxi features (pickup_datetime, passenger_count, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, trip_duration).
2. Data Preparation & Feature Engineering: Calculate Haversine and Manhattan distance in kilometers, extract time features (pickup_hour, day_of_week), and log-transform the target variable trip_duration (log1p).
3. Modeling & Evaluation: Train a Random Forest Regressor and display evaluation metrics (R² Score, RMSE, MAE) alongside Plotly feature importance and residual plots.
4. Deployment (Streamlit UI): Build an interactive dashboard with input sliders for trip variables, a live duration prediction card, and an interactive Plotly map displaying pickup and dropoff points.