import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# ─────────────────────────────────────────────
#  DATA & MODEL
# ─────────────────────────────────────────────
X_train = np.array([
    [1.0, 2],
    [1.5, 3],
    [2.0, 3],
    [2.5, 4],
    [3.0, 5]
])
y_train = np.array([250, 310, 400, 440, 520])
feature_names = ["Square Footage (k)", "Bedrooms"]

def zscore_normalize(X):
    mu    = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    return (X - mu) / sigma, mu, sigma

X_norm, data_mean, data_std = zscore_normalize(X_train)

def compute_cost(X, y, w, b):
    n = X.shape[0]
    return (1 / (2 * n)) * np.sum((np.dot(X, w) + b - y) ** 2)

def train_model(X, y, alpha=0.1, epochs=1000):
    n, nf = X.shape
    w, b  = np.zeros(nf), 0.0
    history = []
    for i in range(epochs):
        pred  = np.dot(X, w) + b
        err   = pred - y
        dj_dw = (1 / n) * np.dot(X.T, err)
        dj_db = (1 / n) * np.sum(err)
        w    -= alpha * dj_dw
        b    -= alpha * dj_db
        if i % 10 == 0:
            history.append({"epoch": i, "cost": compute_cost(X, y, w, b),
                             "w0": w[0], "w1": w[1], "b": b})
    return w, b, pd.DataFrame(history)

w_trained, b_trained, history_df = train_model(X_norm, y_train, alpha=0.1, epochs=1000)

# ─────────────────────────────────────────────
#  PAGE CONFIG & THEME
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* ── dark canvas ── */
.stApp {
    background: #0a0e17;
    color: #e8eaf0;
}

/* ── hero banner ── */
.hero {
    background: linear-gradient(135deg, #0d1b2a 0%, #0a0e17 60%, #0d1b2a 100%);
    border: 1px solid #1e2d40;
    border-radius: 20px;
    padding: 48px 56px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(99,179,237,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #63b3ed;
    margin-bottom: 12px;
}
.hero-title {
    font-size: 48px;
    font-weight: 700;
    line-height: 1.1;
    background: linear-gradient(135deg, #e8eaf0 30%, #63b3ed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 16px;
}
.hero-sub {
    font-size: 15px;
    color: #8892a0;
    max-width: 600px;
    line-height: 1.7;
}

/* ── pill badges ── */
.pill {
    display: inline-block;
    background: rgba(99,179,237,0.1);
    border: 1px solid rgba(99,179,237,0.25);
    color: #63b3ed;
    border-radius: 100px;
    padding: 4px 14px;
    font-size: 12px;
    font-family: 'Space Mono', monospace;
    margin-right: 8px;
    margin-top: 16px;
}

/* ── section cards ── */
.card {
    background: #0d1b2a;
    border: 1px solid #1e2d40;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
}
.card-label {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #63b3ed;
    margin-bottom: 6px;
}
.card-title {
    font-size: 18px;
    font-weight: 600;
    color: #e8eaf0;
    margin-bottom: 20px;
}

/* ── prediction result ── */
.result-box {
    background: linear-gradient(135deg, #0d1b2a, #0f2234);
    border: 1px solid #2a4a6b;
    border-radius: 20px;
    padding: 40px 48px;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 24px;
}
.result-box::after {
    content: '';
    position: absolute;
    bottom: -80px; left: 50%;
    transform: translateX(-50%);
    width: 400px; height: 160px;
    background: radial-gradient(ellipse, rgba(99,179,237,0.06) 0%, transparent 70%);
}
.result-label {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #63b3ed;
    margin-bottom: 8px;
}
.result-price {
    font-size: 64px;
    font-weight: 700;
    background: linear-gradient(135deg, #63b3ed 0%, #90cdf4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    font-family: 'Space Mono', monospace;
}
.result-sub {
    font-size: 13px;
    color: #4a5568;
    margin-top: 8px;
    font-family: 'Space Mono', monospace;
}

/* ── stat chips ── */
.stat-row {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    justify-content: center;
    margin-top: 24px;
}
.stat-chip {
    background: rgba(30,45,64,0.8);
    border: 1px solid #1e2d40;
    border-radius: 12px;
    padding: 12px 20px;
    text-align: center;
    min-width: 110px;
}
.stat-chip-val {
    font-size: 20px;
    font-weight: 700;
    color: #63b3ed;
    font-family: 'Space Mono', monospace;
}
.stat-chip-lbl {
    font-size: 10px;
    color: #4a5568;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 4px;
}

/* ── streamlit widget overrides ── */
div[data-testid="stNumberInput"] input,
div[data-testid="stSlider"] {
    accent-color: #63b3ed;
}
div[data-testid="stMetric"] {
    background: #0d1b2a;
    border: 1px solid #1e2d40;
    border-radius: 12px;
    padding: 16px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    letter-spacing: 1px;
    color: #4a5568;
}
.stTabs [aria-selected="true"] {
    color: #63b3ed !important;
    border-bottom-color: #63b3ed !important;
}
div[data-testid="stExpander"] {
    background: #0d1b2a;
    border: 1px solid #1e2d40;
    border-radius: 12px;
}

/* ── plotly chart background ── */
.js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}

/* hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PLOTLY THEME DEFAULTS
# ─────────────────────────────────────────────
PLOT_BG    = "rgba(0,0,0,0)"
PAPER_BG   = "rgba(0,0,0,0)"
GRID_COLOR = "#1e2d40"
TEXT_COLOR = "#8892a0"
ACCENT     = "#63b3ed"
ACCENT2    = "#90cdf4"
ACCENT3    = "#4299e1"

def base_layout(**kwargs):
    return dict(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="Space Grotesk, sans-serif", color=TEXT_COLOR, size=12),
        margin=dict(t=40, b=40, l=40, r=20),
        **kwargs
    )

# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">Machine Learning · NumPy · From Scratch</div>
  <div class="hero-title">House Price<br>Predictor</div>
  <div class="hero-sub">
    A <strong style="color:#e8eaf0">Multiple Linear Regression</strong> model built entirely in NumPy —
    no scikit-learn, no shortcuts. Vectorized gradient descent converges in real-time
    to estimate property values from raw features.
  </div>
  <div>
    <span class="pill">Z-score Normalization</span>
    <span class="pill">Gradient Descent</span>
    <span class="pill">MSE Cost Function</span>
    <span class="pill">Vectorized NumPy</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  INPUTS
# ─────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-label">Step 01</div><div class="card-title">Configure the Property</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")
with col1:
    sqft_input = st.number_input(
        "Square Footage (thousands)",
        min_value=0.5, max_value=10.0, value=1.8, step=0.1,
        help="E.g. 1.8 = 1,800 sq ft"
    )
with col2:
    bedroom_input = st.slider(
        "Bedrooms", min_value=1, max_value=6, value=3
    )

st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PREDICTION
# ─────────────────────────────────────────────
raw_input       = np.array([sqft_input, bedroom_input])
norm_input      = (raw_input - data_mean) / data_std
predicted_k     = np.dot(norm_input, w_trained) + b_trained
final_price     = predicted_k * 1000

# Confidence band: simple ±RMSE from training
train_preds   = (np.dot(X_norm, w_trained) + b_trained) * 1000
rmse          = np.sqrt(np.mean((train_preds - y_train * 1000) ** 2))
price_low     = max(0, final_price - rmse)
price_high    = final_price + rmse

# ─────────────────────────────────────────────
#  RESULT BOX
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="result-box">
  <div class="result-label">Estimated Market Value</div>
  <div class="result-price">${final_price:,.0f}</div>
  <div class="result-sub">±${rmse:,.0f} model uncertainty (training RMSE)</div>
  <div class="stat-row">
    <div class="stat-chip">
      <div class="stat-chip-val">{sqft_input:.1f}k</div>
      <div class="stat-chip-lbl">Sq Ft</div>
    </div>
    <div class="stat-chip">
      <div class="stat-chip-val">{bedroom_input}</div>
      <div class="stat-chip-lbl">Bedrooms</div>
    </div>
    <div class="stat-chip">
      <div class="stat-chip-val">${price_low/1000:.0f}k</div>
      <div class="stat-chip-lbl">Low Est.</div>
    </div>
    <div class="stat-chip">
      <div class="stat-chip-val">${price_high/1000:.0f}k</div>
      <div class="stat-chip-lbl">High Est.</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  VISUALIZATIONS  (tabs)
# ─────────────────────────────────────────────
st.markdown('<div class="card-label" style="margin-bottom:12px;font-family:Space Mono,monospace;font-size:10px;letter-spacing:2.5px;text-transform:uppercase;color:#63b3ed;">Step 02 — Explore the Model</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📉  Training Curve",
    "🔵  Feature Space",
    "🧊  Cost Surface",
    "📊  Predictions vs Actual",
])

# ── TAB 1: Training Curve ──
with tab1:
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Cost (MSE)", "Weight — Sq Ft", "Weight — Bedrooms"],
        horizontal_spacing=0.08
    )

    line_kw = dict(mode="lines", line=dict(width=2))
    fig.add_trace(go.Scatter(x=history_df.epoch, y=history_df.cost,
                             line=dict(color=ACCENT,  width=2), name="Cost", **{k:v for k,v in line_kw.items() if k!='line'}), row=1, col=1)
    fig.add_trace(go.Scatter(x=history_df.epoch, y=history_df.w0,
                             line=dict(color=ACCENT2, width=2), name="w₀", **{k:v for k,v in line_kw.items() if k!='line'}), row=1, col=2)
    fig.add_trace(go.Scatter(x=history_df.epoch, y=history_df.w1,
                             line=dict(color=ACCENT3, width=2), name="w₁", **{k:v for k,v in line_kw.items() if k!='line'}), row=1, col=3)

    for col_idx in range(1, 4):
        fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False,
                         title_text="Epoch", row=1, col=col_idx)
        fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False, row=1, col=col_idx)

    fig.update_layout(**base_layout(height=340, showlegend=False,
                                    title=dict(text="Gradient descent convergence over 1,000 epochs",
                                               font=dict(size=13, color=TEXT_COLOR))))
    fig.update_annotations(font_color=TEXT_COLOR)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"Final cost: **{history_df.cost.iloc[-1]:.4f}**  ·  "
               f"w₀ = **{w_trained[0]:.4f}**  ·  "
               f"w₁ = **{w_trained[1]:.4f}**  ·  "
               f"b = **{b_trained:.4f}**")

# ── TAB 2: Feature Space ──
with tab2:
    c1, c2 = st.columns(2, gap="large")

    with c1:
        # Sqft vs Price with prediction line
        sqft_range = np.linspace(0.5, 4.0, 120)
        line_prices = []
        for s in sqft_range:
            ni = (np.array([s, bedroom_input]) - data_mean) / data_std
            line_prices.append((np.dot(ni, w_trained) + b_trained) * 1000)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=sqft_range, y=line_prices,
            mode="lines", line=dict(color=ACCENT, width=2, dash="dot"),
            name=f"Model (beds={bedroom_input})"
        ))
        fig2.add_trace(go.Scatter(
            x=X_train[:, 0], y=y_train * 1000,
            mode="markers", marker=dict(color=ACCENT2, size=10, symbol="circle",
                                        line=dict(color="#0a0e17", width=2)),
            name="Training data"
        ))
        # User input point
        fig2.add_trace(go.Scatter(
            x=[sqft_input], y=[final_price],
            mode="markers", marker=dict(color="#f6ad55", size=14, symbol="star",
                                        line=dict(color="#0a0e17", width=2)),
            name="Your estimate"
        ))
        fig2.update_layout(**base_layout(
            height=320,
            xaxis=dict(title="Square Footage (k)", showgrid=True, gridcolor=GRID_COLOR, zeroline=False),
            yaxis=dict(title="Price ($)", showgrid=True, gridcolor=GRID_COLOR, zeroline=False),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
            title=dict(text="Price vs. Square Footage", font=dict(size=13, color=TEXT_COLOR))
        ))
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        # Bedrooms vs Price
        bed_vals  = np.arange(1, 7)
        bed_prices = []
        for b_ in bed_vals:
            ni = (np.array([sqft_input, b_]) - data_mean) / data_std
            bed_prices.append((np.dot(ni, w_trained) + b_trained) * 1000)

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=bed_vals, y=bed_prices,
            marker=dict(
                color=bed_prices,
                colorscale=[[0, "#1a3a5c"], [0.5, ACCENT3], [1, ACCENT]],
                line=dict(width=0)
            ),
            text=[f"${p/1000:.0f}k" for p in bed_prices],
            textposition="outside",
            textfont=dict(color=TEXT_COLOR, size=11),
            name="Predicted price"
        ))
        # Highlight selected
        fig3.add_vline(x=bedroom_input, line_color="#f6ad55", line_dash="dot", line_width=2)

        fig3.update_layout(**base_layout(
            height=320,
            xaxis=dict(title="Bedrooms", showgrid=False, zeroline=False, dtick=1),
            yaxis=dict(title="Price ($)", showgrid=True, gridcolor=GRID_COLOR, zeroline=False),
            showlegend=False,
            title=dict(text=f"Price vs. Bedrooms (sqft={sqft_input:.1f}k)", font=dict(size=13, color=TEXT_COLOR))
        ))
        st.plotly_chart(fig3, use_container_width=True)

# ── TAB 3: Cost Surface ──
with tab3:
    # 2D cost surface over w0, w1 (bias fixed at trained b)
    w0_range = np.linspace(w_trained[0] - 120, w_trained[0] + 120, 60)
    w1_range = np.linspace(w_trained[1] - 120, w_trained[1] + 120, 60)
    W0, W1   = np.meshgrid(w0_range, w1_range)
    Z        = np.array([[compute_cost(X_norm, y_train,
                                       np.array([w0, w1]), b_trained)
                          for w0 in w0_range] for w1 in w1_range])

    fig4 = go.Figure()
    fig4.add_trace(go.Surface(
        x=W0, y=W1, z=Z,
        colorscale=[[0, "#0d2a45"], [0.4, ACCENT3], [0.7, ACCENT], [1, "#f6ad55"]],
        opacity=0.85,
        showscale=True,
        colorbar=dict(thickness=10, len=0.6, tickfont=dict(color=TEXT_COLOR)),
        contours=dict(
            z=dict(show=True, usecolormap=True, project_z=True,
                   highlightcolor="#f6ad55", width=1)
        ),
        name="Cost Surface"
    ))
    # Mark the minimum
    fig4.add_trace(go.Scatter3d(
        x=[w_trained[0]], y=[w_trained[1]],
        z=[compute_cost(X_norm, y_train, w_trained, b_trained)],
        mode="markers",
        marker=dict(size=8, color="#f6ad55", symbol="diamond"),
        name="Optimal w"
    ))

    fig4.update_layout(**base_layout(
        height=480,
        scene=dict(
            xaxis=dict(title="w₀ (Sq Ft)", gridcolor=GRID_COLOR,
                       backgroundcolor=PAPER_BG, color=TEXT_COLOR),
            yaxis=dict(title="w₁ (Bedrooms)", gridcolor=GRID_COLOR,
                       backgroundcolor=PAPER_BG, color=TEXT_COLOR),
            zaxis=dict(title="Cost (MSE)", gridcolor=GRID_COLOR,
                       backgroundcolor=PAPER_BG, color=TEXT_COLOR),
            bgcolor=PAPER_BG,
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.9))
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        title=dict(text="Cost surface — the bowl gradient descent descends",
                   font=dict(size=13, color=TEXT_COLOR))
    ))
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("🟡 Diamond = learned optimum. The bowl shape confirms MSE is convex — gradient descent always finds the global minimum.")

# ── TAB 4: Predictions vs Actual ──
with tab4:
    train_preds_full = (np.dot(X_norm, w_trained) + b_trained) * 1000
    actual_full      = y_train * 1000
    residuals        = actual_full - train_preds_full

    c3, c4 = st.columns(2, gap="large")

    with c3:
        # Predicted vs actual scatter
        fig5 = go.Figure()
        perfect = np.linspace(230000, 540000, 50)
        fig5.add_trace(go.Scatter(
            x=perfect, y=perfect,
            mode="lines", line=dict(color=GRID_COLOR, width=1, dash="dot"),
            name="Perfect fit"
        ))
        fig5.add_trace(go.Scatter(
            x=actual_full, y=train_preds_full,
            mode="markers+text",
            marker=dict(color=ACCENT, size=14, line=dict(color="#0a0e17", width=2)),
            text=[f"#{i+1}" for i in range(len(actual_full))],
            textposition="top center",
            textfont=dict(size=10, color=TEXT_COLOR),
            name="Training samples"
        ))
        fig5.update_layout(**base_layout(
            height=320,
            xaxis=dict(title="Actual Price ($)", showgrid=True, gridcolor=GRID_COLOR, zeroline=False),
            yaxis=dict(title="Predicted Price ($)", showgrid=True, gridcolor=GRID_COLOR, zeroline=False),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            title=dict(text="Predicted vs. Actual", font=dict(size=13, color=TEXT_COLOR))
        ))
        st.plotly_chart(fig5, use_container_width=True)

    with c4:
        # Residuals bar
        colors = [ACCENT if r >= 0 else "#fc8181" for r in residuals]
        fig6 = go.Figure()
        fig6.add_trace(go.Bar(
            x=[f"House {i+1}" for i in range(len(residuals))],
            y=residuals,
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{r:+.0f}" for r in residuals],
            textposition="outside",
            textfont=dict(color=TEXT_COLOR, size=11),
        ))
        fig6.add_hline(y=0, line_color=GRID_COLOR, line_width=1)
        fig6.update_layout(**base_layout(
            height=320,
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(title="Residual ($)", showgrid=True, gridcolor=GRID_COLOR, zeroline=False),
            showlegend=False,
            title=dict(text="Residuals (Actual − Predicted)", font=dict(size=13, color=TEXT_COLOR))
        ))
        st.plotly_chart(fig6, use_container_width=True)

    # Metrics row
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((actual_full - np.mean(actual_full)) ** 2)
    r2     = 1 - ss_res / ss_tot
    mae    = np.mean(np.abs(residuals))

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("R² Score",   f"{r2:.4f}")
    with m2: st.metric("RMSE",       f"${rmse:,.0f}")
    with m3: st.metric("MAE",        f"${mae:,.0f}")
    with m4: st.metric("Final Cost", f"{history_df.cost.iloc[-1]:.4f}")

# ─────────────────────────────────────────────
#  MATH EXPANDER
# ─────────────────────────────────────────────
with st.expander("🔬  Inspect Trained Parameters & Mathematics"):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Feature Normalization (Z-score)**")
        st.latex(r"x_{\text{norm}} = \frac{x - \mu}{\sigma}")
        st.write(f"μ: sqft = {data_mean[0]:.2f}, beds = {data_mean[1]:.2f}")
        st.write(f"σ: sqft = {data_std[0]:.2f}, beds = {data_std[1]:.2f}")
    with col_b:
        st.markdown("**Prediction Equation**")
        st.latex(
            r"\hat{y} = w_1 \cdot \frac{x_1-\mu_1}{\sigma_1} + w_2 \cdot \frac{x_2-\mu_2}{\sigma_2} + b"
        )
        st.write(f"w₀ = {w_trained[0]:.4f}  ·  w₁ = {w_trained[1]:.4f}  ·  b = {b_trained:.4f}")

    st.markdown("**Gradient Descent Update Rules**")
    st.latex(r"w_j := w_j - \alpha \frac{\partial J}{\partial w_j} \qquad b := b - \alpha \frac{\partial J}{\partial b}")
    st.latex(r"\frac{\partial J}{\partial w_j} = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}^{(i)} - y^{(i)}) x_j^{(i)}")

st.markdown("<br><div style='text-align:center;font-family:Space Mono,monospace;font-size:10px;color:#2a3a50;letter-spacing:2px;'>BUILT FROM SCRATCH · NUMPY ONLY · NO SCIKIT-LEARN</div>", unsafe_allow_html=True)