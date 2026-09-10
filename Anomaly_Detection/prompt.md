Act as a Senior Data Scientist and Lead Streamlit Developer.

Write a complete, single-file Streamlit application (`app.py`) for a data science course project following the CRISP-DM framework.

Project Details:
- Title: Project 3 - Credit Card Fraud & Anomaly Detection
- Directory: Anomaly_Detection
- Tech Stack: Python, Streamlit, Plotly Express, Scikit-learn (IsolationForest, StandardScaler, PCA).

Requirements:
1. Data Generation:
   - Create a synthetic dataset of 1,000 credit card transaction records with columns: 'TransactionID', 'Amount', 'Time_Hour' (0-24), 'Distance_From_Home' (miles), and a generated 'Is_Anomaly' ground truth (around 5% anomalies). Cache data generation.

2. Sidebar & Model Parameters:
   - Sidebar controls for Contamination Rate slider (0.01 to 0.15, default = 0.05) and Number of Estimators (50 to 200, default = 100).
   - Fit an IsolationForest model on features ['Amount', 'Time_Hour', 'Distance_From_Home'].
   - Output predicted anomaly labels (-1 for anomaly, 1 for normal) and decision scores.

3. Top-Level Metrics Display:
   - 3 columns showing: Total Transactions, Detected Anomalies Count, Anomaly Percentage (%).

4. Multi-Tab Layout:
   - Tab 1 ("Project Overview"): Display a clean DataFrame detailing the 6 CRISP-DM phases applied to credit card anomaly detection.
   - Tab 2 ("Data Understanding"): Display `df.head(10)` and a Plotly histogram comparing transaction amounts between normal and anomalous transactions.
   - Tab 3 ("Anomaly Visualizer"): Display a 3D Scatter plot (`px.scatter_3d`) plotting Amount vs Distance_From_Home vs Time_Hour, colored by Anomaly status.
   - Tab 4 ("Live Transaction Evaluator"): Provide interactive sliders for Amount ($), Time of Day (Hour), and Distance from Home (miles). Scale the input, run `isolation_forest.predict()`, and display whether the transaction is "NORMAL" (green) or "SUSPICIOUS / ANOMALY DETECTED" (red).

Ensure the code is complete, bug-free, fully self-contained, and ready to run with `streamlit run app.py`.