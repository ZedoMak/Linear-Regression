import streamlit as st
import numpy as np

# Dataset: [Square Footage (in 1000s), Number of Bedrooms]
X_train = np.array([
    [1.0, 2],
    [1.5, 3],
    [2.0, 3],
    [2.5, 4],
    [3.0, 5]
])

# Target: Price (in $1,000s)
y_train = np.array([250, 310, 400, 440, 520])


def zscore_normalize_features(X):
    """Normalizes features to have mean=0 and std=1"""
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


# Normalize our training data
X_features_norm, data_mean, data_std = zscore_normalize_features(X_train)

def compute_cost(X, y, w, b):
    """Computes the Mean Squared Error cost function"""
    n = X.shape[0]
    predictions = np.dot(X, w) + b
    cost = (1 / (2 * n)) * np.sum((predictions - y) ** 2)
    return cost


def train_model(X, y, alpha=0.1, epochs=1000):
    """Trains the model weights and bias using Multiple Gradient Descent"""
    n, num_features = X.shape
    w = np.zeros(num_features)  # Initialize weights to zero
    b = 0.0  # Initialize bias to zero

    for _ in range(epochs):
        # 1. Calculate predictions (Dot product of features and weights matrix)
        predictions = np.dot(X, w) + b
        error = predictions - y

        # 2. Compute gradients (Vectorized partial derivatives)
        dj_dw = (1 / n) * np.dot(X.T, error)
        dj_db = (1 / n) * np.sum(error)

        # 3. Update parameters simultaneously
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

    return w, b


# Train the model right when the app loads
w_trained, b_trained = train_model(X_features_norm, y_train, alpha=0.1, epochs=1000)


st.set_page_config(page_title="Multi-Feature House Predictor", page_icon="🏡", layout="centered")

st.title("🏡 Multi-Feature House Price Predictor")
st.markdown("""
This web application runs a **Multiple Linear Regression** model trained entirely from scratch using **raw NumPy**. 
It dynamically optimizes parameters using **Vectorized Gradient Descent** and uses **Z-score Normalization** to keep feature scales balanced.
""")

st.divider()

st.subheader("🛠️ Step 1: Input House Attributes")
col1, col2 = st.columns(2)

with col1:
    sqft_input = st.number_input(
        "Square Footage (in 1,000s sq ft)",
        min_value=0.5,
        max_value=10.0,
        value=1.8,
        step=0.1,
        help="Example: 1.8 means 1,800 square feet."
    )

with col2:
    bedroom_input = st.slider(
        "Number of Bedrooms",
        min_value=1,
        max_value=6,
        value=3,
        help="Select the exact number of bedrooms."
    )


# Put raw user inputs into an array
raw_user_input = np.array([sqft_input, bedroom_input])

# CRITICAL STEP: Normalize user input using the exact same mean/std dev from the training set
normalized_user_input = (raw_user_input - data_mean) / data_std

# Calculate final prediction using the learned weights
predicted_price_thousands = np.dot(normalized_user_input, w_trained) + b_trained
final_price = predicted_price_thousands * 1000

st.divider()

st.subheader("🎯 Step 2: Model Prediction")
st.metric(label="Estimated Market Value", value=f"${final_price:,.2f}")

# Display the underlying mathematics parameters
with st.expander("🔍 Inspect Trained Parameters & Math"):
    st.markdown("### Feature Normalization Metrics")
    st.write(f"**Calculated Dataset Mean ($\mu$):** Size = {data_mean[0]:.2f}, Bedrooms = {data_mean[1]:.2f}")
    st.write(
        f"**Calculated Dataset Standard Deviation ($\sigma$):** Size = {data_std[0]:.2f}, Bedrooms = {data_std[1]:.2f}")

    st.markdown("### Optimized Values (After 1,000 Epochs)")
    st.write(f"**Optimal Weights ($w$):** Size Weight = {w_trained[0]:.4f}, Bedroom Weight = {w_trained[1]:.4f}")
    st.write(f"**Optimal Bias ($b$):** {b_trained:.4f}")

    st.markdown("### Mathematical Equation Implemented Behind the Scenes:")
    st.latex(
        r"Price\_Predicted = w_1 \cdot \left(\frac{Size - \mu_1}{\sigma_1}\right) + w_2 \cdot \left(\frac{Bedrooms - \mu_2}{\sigma_2}\right) + b")