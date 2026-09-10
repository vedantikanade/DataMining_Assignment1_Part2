Act as a Senior Data Scientist and Lead Streamlit Developer. 

Write a complete, single-file Streamlit application (`app.py`) for a data science course project following the CRISP-DM framework.

Project Details:
- Title: Project 2 - Customer Segmentation & Clustering
- Directory: 03_customer_segmentation_clustering
- Tech Stack: Python, Streamlit, Plotly Express, Scikit-learn (KMeans, StandardScaler, silhouette_score).

Requirements:
1. Data Generation:
   - Create a synthetic dataset of 400 records with columns: 'CustomerID', 'Age' (18-70), 'Annual_Income_k' (15-140), and 'Spending_Score' (1-100). Cache the data generation.

2. Sidebar & Model Parameters:
   - Sidebar slider for selecting Number of Clusters (K) between 2 and 8 (default = 4).
   - Standardize features ('Age', 'Annual_Income_k', 'Spending_Score') using StandardScaler and fit KMeans.
   - Calculate and store cluster assignments and the overall Silhouette Score.

3. Top-Level Metrics Display:
   - 3 columns showing: Total Customers, Selected K Clusters, and Silhouette Score.

4. Multi-Tab Layout:
   - Tab 1 ("Project Overview"): Display a clean Markdown/DataFrame table detailing the 6 CRISP-DM phases (Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, Deployment) applied specifically to this customer clustering problem.
   - Tab 2 ("Data Understanding"): Display `df.head(10)` and an interactive Plotly histogram of Spending Score colored by cluster.
   - Tab 3 ("Cluster Analysis"): Display an interactive Plotly 3D scatter plot (`px.scatter_3d`) plotting Annual Income vs Spending Score vs Age, colored by Cluster.
   - Tab 4 ("Live Predictor"): Provide interactive sliders for Age, Annual Income, and Spending Score. Standardize the user input, predict the assigned cluster persona using the fitted KMeans model, and display a highlighted success message with the cluster assignment.

Ensure the code is complete, bug-free, fully self-contained, and ready to run with `streamlit run app.py`.