"""Project 4 - Data Science Visual Mastery & Interactive Intuition Lab.

Run:
    pip install streamlit pandas numpy scikit-learn plotly
    streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


RANDOM_STATE = 42

st.set_page_config(
    page_title="Data Science Visual Mastery",
    page_icon="📈",
    layout="wide",
)


# CRISP-DM Phase 1-2: Business Understanding and Data Understanding
@st.cache_data
def generate_linear_data(
    sample_size: int,
    noise_level: float,
) -> pd.DataFrame:
    """Generate reproducible synthetic data for linear regression."""
    rng = np.random.default_rng(RANDOM_STATE)
    x = np.linspace(-5, 5, sample_size)
    true_intercept = 4.0
    true_slope = 2.5
    y = true_intercept + true_slope * x + rng.normal(0, noise_level, sample_size)
    return pd.DataFrame({"Feature_X": x, "Target_Y": y})


# CRISP-DM Phase 3-5: Data Preparation, Modeling, and Evaluation
def gradient_descent(
    data: pd.DataFrame,
    learning_rate: float,
    iterations: int,
) -> dict:
    """Fit y = theta_0 + theta_1*x with batch gradient descent."""
    x = data["Feature_X"].to_numpy(dtype=float)
    y = data["Target_Y"].to_numpy(dtype=float)
    sample_count = len(x)

    theta_0 = 0.0
    theta_1 = 0.0
    losses: list[float] = []
    parameter_history: list[tuple[float, float]] = []

    for _ in range(iterations):
        predictions = theta_0 + theta_1 * x
        errors = predictions - y

        # J(theta) = 1/(2m) * sum(error^2)
        cost = float(np.mean(errors**2) / 2)
        losses.append(cost)
        parameter_history.append((theta_0, theta_1))

        if not np.isfinite(cost):
            break

        gradient_0 = float(np.mean(errors))
        gradient_1 = float(np.mean(errors * x))
        theta_0 -= learning_rate * gradient_0
        theta_1 -= learning_rate * gradient_1

    final_predictions = theta_0 + theta_1 * x
    with np.errstate(over="ignore", invalid="ignore"):
        final_mse_value = float(np.mean((y - final_predictions) ** 2))
    final_mse = final_mse_value if np.isfinite(final_mse_value) else float("inf")
    return {
        "theta_0": theta_0,
        "theta_1": theta_1,
        "losses": losses,
        "parameter_history": parameter_history,
        "predictions": final_predictions,
        "final_mse": final_mse,
        "iterations_completed": len(losses),
    }


def line_for_iteration(
    data: pd.DataFrame,
    result: dict,
    iteration_number: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return regression-line coordinates for a chosen training iteration."""
    x_line = np.linspace(
        data["Feature_X"].min() - 0.25,
        data["Feature_X"].max() + 0.25,
        100,
    )
    history = result["parameter_history"]
    if not history:
        intercept, slope = 0.0, 0.0
    else:
        index = min(max(iteration_number - 1, 0), len(history) - 1)
        intercept, slope = history[index]
    return x_line, intercept + slope * x_line


def loss_figure(result: dict, title: str) -> go.Figure:
    """Create a Plotly loss trajectory chart."""
    losses = np.asarray(result["losses"], dtype=float)
    losses = np.where(np.isfinite(losses), losses, np.nan)
    loss_data = pd.DataFrame(
        {"Iteration": np.arange(1, len(losses) + 1), "Cost (J)": losses}
    )
    fig = px.line(
        loss_data,
        x="Iteration",
        y="Cost (J)",
        markers=True,
        title=title,
        labels={"Cost (J)": "Cost / Loss"},
    )
    fig.update_layout(height=430)
    return fig


def regression_figure(
    data: pd.DataFrame,
    result: dict,
    shown_iteration: int,
) -> go.Figure:
    """Create a data-and-regression-line chart for a chosen iteration."""
    x_line, y_line = line_for_iteration(data, result, shown_iteration)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["Feature_X"],
            y=data["Target_Y"],
            mode="markers",
            name="Synthetic data",
            marker={"size": 8, "color": "#2E86AB"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name=f"Regression line at iteration {shown_iteration}",
            line={"width": 4, "color": "#D1495B"},
        )
    )
    fig.update_layout(
        title="Regression Line During Learning",
        xaxis_title="Feature X",
        yaxis_title="Target Y",
        height=430,
    )
    return fig


def build_overview_table() -> pd.DataFrame:
    """Describe the six CRISP-DM phases for this educational simulator."""
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
            "Application to Visual Mastery Lab": [
                "Help learners understand how a model minimizes prediction error through iterative optimization.",
                "Generate synthetic linear data and expose the effects of sample size and Gaussian noise.",
                "Represent features and targets numerically and initialize model parameters.",
                "Use linear regression with batch Gradient Descent to update intercept and slope.",
                "Inspect MSE/cost trajectories and regression-line behavior across iterations.",
                "Deliver an interactive Streamlit learning environment with live controls and diagnostic experiments.",
            ],
        }
    )


def diagnose_experiment(result: dict, requested_iterations: int) -> tuple[str, str]:
    """Classify common learning behaviors for the experiment lab."""
    losses = np.asarray(result["losses"], dtype=float)
    if len(losses) == 0 or not np.all(np.isfinite(losses)):
        return "Exploding gradients", "The loss became non-finite."
    initial_loss = losses[0]
    final_loss = losses[-1]
    if final_loss > max(initial_loss * 10, 1_000_000):
        return "Exploding gradients", "The learning rate is too high for stable updates."
    if requested_iterations <= 25 or final_loss > initial_loss * 0.35:
        return "Underfitting / incomplete convergence", "The model needs more iterations or a better learning rate."
    return "Stable learning", "The loss is decreasing toward a useful fit."


# CRISP-DM Phase 6: Deployment
def main() -> None:
    st.title("📈 Project 4 - Data Science Visual Mastery & Interactive Intuition Lab")
    st.caption(
        "CRISP-DM project | Learn linear regression and Gradient Descent visually"
    )

    with st.sidebar:
        st.header("Live Simulation Controls")
        learning_rate = st.slider(
            "Learning Rate (α)",
            min_value=0.001,
            max_value=0.5,
            value=0.05,
            step=0.001,
            format="%.3f",
        )
        iterations = st.slider(
            "Epochs / Iterations",
            min_value=10,
            max_value=200,
            value=100,
            step=10,
        )
        noise_level = st.slider(
            "Gaussian Noise Level",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.5,
        )
        sample_size = st.slider(
            "Sample Size (N)",
            min_value=20,
            max_value=300,
            value=100,
            step=10,
        )

    data = generate_linear_data(sample_size, noise_level)
    result = gradient_descent(data, learning_rate, iterations)
    final_loss = result["final_mse"]

    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Final Loss (MSE)", f"{final_loss:.4f}")
    metric_2.metric("Learning Rate (α)", f"{learning_rate:.3f}")
    metric_3.metric("Total Iterations", f"{result['iterations_completed']}")

    overview_tab, math_tab, simulation_tab, lab_tab = st.tabs(
        [
            "Project Overview",
            "Mathematical Rigor",
            "Live Learning Rate Simulation",
            "Interactive Experiment Lab",
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
            "Change one sidebar control at a time and observe how it changes the "
            "loss curve, fitted line, and final MSE."
        )

    with math_tab:
        st.subheader("The Objective Function")
        st.latex(r"J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2")
        st.write(
            "The cost function measures the average squared difference between "
            "the model predictions and the observed targets. The factor 1/2 "
            "simplifies the derivative."
        )

        st.subheader("The Parameter Update Rule")
        st.latex(
            r"\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)"
        )
        st.write(
            "At every iteration, each parameter moves opposite to the gradient. "
            "The learning rate α controls the step size: very small values learn "
            "slowly, while very large values can overshoot the minimum."
        )

        st.subheader("This App's Linear Model")
        st.latex(r"h_\theta(x) = \theta_0 + \theta_1 x")
        st.write(
            "Here, θ₀ is the intercept and θ₁ is the slope. Gradient Descent "
            "updates both parameters using the same batch of synthetic data."
        )

    with simulation_tab:
        st.subheader("Watch the Model Learn")
        max_iteration = max(1, result["iterations_completed"])
        shown_iteration = st.slider(
            "Show regression line after iteration",
            min_value=1,
            max_value=max_iteration,
            value=max_iteration,
            step=1,
        )

        chart_col_1, chart_col_2 = st.columns(2)
        with chart_col_1:
            st.plotly_chart(
                loss_figure(result, "Cost Function / Loss Trajectory"),
                use_container_width=True,
                key="chart_loss_trajectory",
            )
        with chart_col_2:
            st.plotly_chart(
                regression_figure(data, result, shown_iteration),
                use_container_width=True,
                key="chart_regression_line",
            )

        parameters = result["parameter_history"][
            min(shown_iteration - 1, len(result["parameter_history"]) - 1)
        ]
        st.write(
            f"At iteration {shown_iteration}: intercept θ₀ = {parameters[0]:.4f}, "
            f"slope θ₁ = {parameters[1]:.4f}"
        )

    with lab_tab:
        st.subheader("Experiment Lab")
        st.write(
            "Use a separate experiment to see how iteration count and learning "
            "rate affect optimization stability."
        )

        experiment_type = st.selectbox(
            "Choose an experiment",
            [
                "Stable learning",
                "Underfitting / too few iterations",
                "Exploding gradients",
            ],
        )
        lab_col_1, lab_col_2 = st.columns(2)
        with lab_col_1:
            lab_learning_rate = st.slider(
                "Experiment learning rate",
                min_value=0.001,
                max_value=0.5,
                value=0.05 if experiment_type == "Stable learning" else (
                    0.01 if experiment_type == "Underfitting / too few iterations" else 0.5
                ),
                step=0.001,
                format="%.3f",
            )
        with lab_col_2:
            lab_iterations = st.slider(
                "Experiment iterations",
                min_value=10,
                max_value=200,
                value=100 if experiment_type != "Underfitting / too few iterations" else 10,
                step=10,
            )

        lab_result = gradient_descent(
            data,
            lab_learning_rate,
            lab_iterations,
        )
        diagnosis, explanation = diagnose_experiment(lab_result, lab_iterations)

        if diagnosis == "Exploding gradients":
            st.error(f"⚠️ {diagnosis}: {explanation}")
        elif diagnosis == "Underfitting / incomplete convergence":
            st.warning(f"ℹ️ {diagnosis}: {explanation}")
        else:
            st.success(f"✅ {diagnosis}: {explanation}")

        lab_metric_1, lab_metric_2 = st.columns(2)
        lab_metric_1.metric("Experiment Final MSE", f"{lab_result['final_mse']:.4f}")
        lab_metric_2.metric(
            "Completed Iterations",
            f"{lab_result['iterations_completed']}",
        )

        lab_chart_1, lab_chart_2 = st.columns(2)
        with lab_chart_1:
            st.plotly_chart(
                loss_figure(lab_result, "Experiment Loss Trajectory"),
                use_container_width=True,
                key="chart_experiment_loss",
            )
        with lab_chart_2:
            lab_line_iteration = max(1, lab_result["iterations_completed"])
            st.plotly_chart(
                regression_figure(data, lab_result, lab_line_iteration),
                use_container_width=True,
                key="chart_experiment_regression",
            )


if __name__ == "__main__":
    main()
