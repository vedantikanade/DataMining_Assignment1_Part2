# Prompt - Project 4: Data Science Visual Mastery (08_datascience_visual_mastery)

Act as a Senior Data Scientist and Lead Streamlit Developer.

Write a complete, single-file Streamlit application (`app.py`) for an educational data science project following the CRISP-DM framework.

Project Details:
- Title: Project 4 - Data Science Visual Mastery & Interactive Intuition Lab
- Assignment Spec: Educational visual simulation with math & live widget controls (adjusting learning rate live).
- Tech Stack: Python, Streamlit, Plotly Express, Plotly Graph Objects, NumPy, Scikit-learn.

Requirements:
1. Live Simulation Engine:
   - Build a interactive Gradient Descent & Linear Regression simulator.
   - Generate synthetic linear data with controllable Gaussian noise and sample size (N).

2. Sidebar & Live Interactive Controls:
   - Sliders for Learning Rate ($\alpha$: 0.001 to 0.5), Epochs/Iterations (10 to 200), Noise Level, and Sample Size ($N$).

3. Top-Level Metrics Display:
   - 3 columns showing: Final Loss (MSE), Learning Rate ($\alpha$), and Total Iterations.

4. Multi-Tab Layout:
   - Tab 1 ("Project Overview"): Clean table detailing CRISP-DM applied to interactive visual learning systems.
   - Tab 2 ("Mathematical Rigor"): Mathematical equations using LaTeX ($J(\theta) = \frac{1}{2m} \sum (h_\theta(x) - y)^2$, parameter update rule $\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$) paired with intuition notes.
   - Tab 3 ("Live Learning Rate Simulation"): Interactive Plotly visualization showing:
     1. The cost function curve / loss trajectory over epochs.
     2. The regression line fitting to data points step-by-step or at final iteration.
   - Tab 4 ("Interactive Experiment Lab"): Provide controls to test underfitting vs. overfitting or exploding gradients when learning rate is set too high.

Ensure the code is complete, bug-free, fully self-contained, and ready to run with `streamlit run app.py`.