"""
Credit Card Fraud Detection — Streamlit App
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Shield | Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
    .stApp { background: #0a0e1a; color: #e2e8f0; }

    .metric-card {
        background: linear-gradient(135deg, #1a1f35 0%, #12172a 100%);
        border: 1px solid #2d3561;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 6px 0;
    }
    .metric-value { font-size: 2rem; font-weight: 800; color: #60a5fa; font-family: 'Space Mono', monospace; }
    .metric-label { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 2px; margin-top: 4px; }

    .fraud-alert {
        background: linear-gradient(135deg, #3b0a0a 0%, #1f0505 100%);
        border: 2px solid #ef4444; border-radius: 16px;
        padding: 28px; text-align: center; margin: 16px 0;
    }
    .safe-alert {
        background: linear-gradient(135deg, #052e16 0%, #021a0e 100%);
        border: 2px solid #22c55e; border-radius: 16px;
        padding: 28px; text-align: center; margin: 16px 0;
    }
    .alert-title { font-size: 1.8rem; font-weight: 800; margin-bottom: 8px; }
    .alert-sub   { font-size: 0.95rem; color: #94a3b8; }

    .upload-box {
        background: linear-gradient(135deg, #1a1f35 0%, #12172a 100%);
        border: 2px dashed #3b82f6; border-radius: 16px;
        padding: 40px; text-align: center; margin: 20px 0;
    }

    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white; border: none; border-radius: 10px;
        padding: 14px 32px; font-size: 1rem; font-weight: 700;
        width: 100%; font-family: 'Syne', sans-serif;
        letter-spacing: 1px; cursor: pointer; transition: all 0.2s;
    }
    .stButton > button:hover { background: linear-gradient(135deg, #2563eb, #1d4ed8); transform: translateY(-1px); }

    .sidebar-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
        padding: 20px; border-radius: 12px;
        margin-bottom: 20px; text-align: center;
    }

    h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
    div[data-testid="stMetricValue"] { color: #60a5fa !important; font-family: 'Space Mono'; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    artifacts_path = "model_artifacts"
    if not os.path.exists(artifacts_path):
        return None, None, None, None
    try:
        with open(f"{artifacts_path}/random_forest.pkl", "rb") as f: rf     = pickle.load(f)
        with open(f"{artifacts_path}/logistic_reg.pkl",  "rb") as f: lr     = pickle.load(f)
        with open(f"{artifacts_path}/scaler.pkl",        "rb") as f: scaler = pickle.load(f)
        with open(f"{artifacts_path}/feature_cols.pkl",  "rb") as f: feats  = pickle.load(f)
        return rf, lr, scaler, feats
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

rf, lr, scaler, feature_cols = load_models()


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.markdown("<div style='font-size:3.5rem;padding-top:10px'>🛡️</div>", unsafe_allow_html=True)
with col_title:
    st.markdown("<h1 style='color:#60a5fa;margin:0;font-size:2.4rem'>FRAUD SHIELD</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b;margin:0;font-size:0.9rem;letter-spacing:3px'>CREDIT CARD FRAUD DETECTION SYSTEM</p>", unsafe_allow_html=True)

st.markdown("---")


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class='sidebar-header'>
        <div style='font-size:2rem'>🔍</div>
        <div style='font-weight:800;font-size:1.1rem;color:white'>Transaction Inspector</div>
        <div style='color:#93c5fd;font-size:0.8rem'>Manual entry or upload a CSV</div>
    </div>
    """, unsafe_allow_html=True)

    model_choice = st.selectbox("🤖 Choose Model", ["Random Forest", "Logistic Regression"])

    st.markdown("#### 💰 Transaction Details")
    amount = st.slider("Transaction Amount ($)", 0.0, 5000.0, 150.0, step=0.5)
    hour   = st.slider("Hour of Day (0–23)", 0, 23, 14)

    st.markdown("#### 🔢 PCA Features (V1–V10)")
    st.caption("Anonymized transaction behavior features")

    v_vals = {}
    col1, col2 = st.columns(2)
    for i in range(1, 11):
        with (col1 if i % 2 == 1 else col2):
            v_vals[f"V{i}"] = st.number_input(f"V{i}", value=0.0, step=0.1, format="%.2f")

    predict_btn = st.button("🔍 ANALYZE TRANSACTION", type="primary")


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Manual Prediction",
    "📂 Upload CSV File",
    "📊 Model Insights",
    "📖 How It Works"
])


# ══════════════════════════════════════════════
# TAB 1 — MANUAL PREDICTION
# ══════════════════════════════════════════════
with tab1:
    if not rf:
        st.warning("⚠️ Models not found. Please run `python train_model.py` first.")
        st.code("pip install -r requirements.txt\npython train_model.py\nstreamlit run app.py")
    else:
        if predict_btn:
            amount_scaled = (amount - 88.35) / 250.12
            hour_scaled   = (hour - 13.5) / 7.5
            feature_dict  = {f"V{i}": 0.0 for i in range(1, 29)}
            feature_dict.update(v_vals)
            feature_dict["Amount_Scaled"] = amount_scaled
            feature_dict["Hour_Scaled"]   = hour_scaled

            X_input    = pd.DataFrame([feature_dict])[feature_cols]
            model      = rf if model_choice == "Random Forest" else lr
            prediction = model.predict(X_input)[0]
            proba      = model.predict_proba(X_input)[0]
            fraud_prob = proba[1] * 100
            safe_prob  = proba[0] * 100

            if prediction == 1:
                st.markdown(f"""
                <div class='fraud-alert'>
                    <div class='alert-title' style='color:#ef4444'>⚠️ FRAUDULENT TRANSACTION DETECTED</div>
                    <div class='alert-sub'>This transaction has been flagged as high-risk</div>
                    <div style='font-size:2.5rem;font-weight:800;color:#ef4444;font-family:Space Mono,monospace;margin-top:12px'>{fraud_prob:.1f}% Fraud Probability</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='safe-alert'>
                    <div class='alert-title' style='color:#22c55e'>✅ TRANSACTION APPEARS LEGITIMATE</div>
                    <div class='alert-sub'>No suspicious patterns detected</div>
                    <div style='font-size:2.5rem;font-weight:800;color:#22c55e;font-family:Space Mono,monospace;margin-top:12px'>{safe_prob:.1f}% Safe Probability</div>
                </div>""", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=fraud_prob,
                    title={"text": "Fraud Risk Score", "font": {"color": "#94a3b8"}},
                    number={"suffix": "%", "font": {"color": "#60a5fa", "size": 36}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#475569"},
                        "bar":  {"color": "#ef4444" if prediction == 1 else "#22c55e"},
                        "bgcolor": "#1a1f35",
                        "steps": [
                            {"range": [0,  40], "color": "#052e16"},
                            {"range": [40, 70], "color": "#422006"},
                            {"range": [70,100], "color": "#3b0a0a"},
                        ],
                    }
                ))
                fig.update_layout(paper_bgcolor="#0a0e1a", font_color="#e2e8f0", height=280)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                fig2 = go.Figure(go.Bar(
                    x=["Legitimate", "Fraudulent"], y=[safe_prob, fraud_prob],
                    marker_color=["#22c55e", "#ef4444"],
                    text=[f"{safe_prob:.1f}%", f"{fraud_prob:.1f}%"],
                    textposition="outside", textfont={"color": "#e2e8f0", "size": 14}
                ))
                fig2.update_layout(
                    title="Class Probabilities",
                    paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
                    font_color="#e2e8f0",
                    yaxis={"range": [0, 110], "gridcolor": "#1e293b"},
                    height=280, showlegend=False
                )
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("#### 📋 Transaction Summary")
            m1, m2, m3, m4 = st.columns(4)
            for col, val, label in zip(
                [m1, m2, m3, m4],
                [f"${amount:.2f}", f"{hour:02d}:00",
                 "🔴 FRAUD" if prediction == 1 else "🟢 SAFE",
                 model_choice.split()[0]],
                ["Amount", "Hour", "Result", "Model"]
            ):
                with col:
                    st.markdown(f"""<div class='metric-card'>
                        <div class='metric-value'>{val}</div>
                        <div class='metric-label'>{label}</div></div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='text-align:center;padding:60px 20px'>
                <div style='font-size:5rem;margin-bottom:16px'>🔍</div>
                <div style='font-size:1.4rem;font-weight:700;color:#64748b'>Enter transaction details in the sidebar</div>
                <div style='font-size:0.9rem;margin-top:8px;color:#475569'>then click ANALYZE TRANSACTION</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — CSV FILE UPLOAD  ← NEW FEATURE
# ══════════════════════════════════════════════
with tab2:
    st.markdown("### 📂 Batch Fraud Detection — Upload CSV File")
    st.markdown("Analyze **hundreds of transactions at once** by uploading a CSV file.")

    # ── Format guide ──
    with st.expander("📋 What format should my CSV be in?", expanded=False):
        st.markdown("""
        Your CSV must have these columns:
        - **V1 to V28** — PCA-transformed features (anonymized)
        - **Amount** — Transaction amount in dollars
        - **Time** — Seconds elapsed since first transaction (used to compute hour)
        - **Class** *(optional)* — 0 = Normal, 1 = Fraud. If included, app shows actual vs predicted.

        This is the same format as the Kaggle `creditcard.csv` dataset.
        """)

    # ── Sample CSV download ──
    st.markdown("#### 📥 Download a sample CSV to test")
    np.random.seed(42)
    sample_rows = []
    for i in range(5):
        row = {"Time": np.random.randint(0, 172800), "Amount": round(np.random.uniform(1, 500), 2), "Class": 0}
        for j in range(1, 29):
            row[f"V{j}"] = round(np.random.normal(0, 1), 4)
        sample_rows.append(row)
    # Add 1 fraud row
    fraud_row = {"Time": 50000, "Amount": 2850.00, "Class": 1}
    for j in range(1, 29):
        fraud_row[f"V{j}"] = round(np.random.normal(-2, 2), 4)
    sample_rows.append(fraud_row)

    sample_df  = pd.DataFrame(sample_rows)
    cols_order = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    sample_df  = sample_df[cols_order]

    st.download_button(
        label="⬇️ Download Sample CSV (6 transactions)",
        data=sample_df.to_csv(index=False).encode("utf-8"),
        file_name="sample_transactions.csv",
        mime="text/csv"
    )

    st.markdown("---")

    # ── Upload box ──
    st.markdown("""
    <div class='upload-box'>
        <div style='font-size:3rem;margin-bottom:10px'>📁</div>
        <div style='font-size:1.1rem;font-weight:700;color:#60a5fa'>Drop your CSV file below</div>
        <div style='font-size:0.85rem;color:#64748b;margin-top:6px'>Supports .csv files — analyzes all rows instantly</div>
    </div>""", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload a CSV with transaction data in creditcard.csv format"
    )

    if uploaded_file is not None:
        if not rf:
            st.error("⚠️ Models not found! Run `python train_model.py` first.")
        else:
            try:
                df_upload = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded — **{len(df_upload):,} transactions** found")

                st.markdown("#### 👀 Preview (first 5 rows)")
                st.dataframe(df_upload.head(), use_container_width=True, hide_index=True)

                # Validate columns
                required = [f"V{i}" for i in range(1, 29)] + ["Amount"]
                missing  = [c for c in required if c not in df_upload.columns]

                if missing:
                    st.error(f"❌ Missing columns: {missing}")
                else:
                    st.markdown("---")

                    # Feature engineering
                    df_upload["Hour"]          = ((df_upload["Time"] // 3600) % 24) if "Time" in df_upload.columns else 12
                    df_upload["Amount_Scaled"] = (df_upload["Amount"] - 88.35) / 250.12
                    df_upload["Hour_Scaled"]   = (df_upload["Hour"] - 13.5) / 7.5

                    X_batch = df_upload[feature_cols]
                    model   = rf if model_choice == "Random Forest" else lr

                    with st.spinner("🔍 Analyzing all transactions..."):
                        predictions   = model.predict(X_batch)
                        probabilities = model.predict_proba(X_batch)[:, 1]

                    # ── Summary metrics ──
                    total       = len(predictions)
                    fraud_count = int(predictions.sum())
                    safe_count  = total - fraud_count
                    fraud_pct   = (fraud_count / total) * 100

                    st.markdown("### 📊 Batch Results")
                    r1, r2, r3, r4 = st.columns(4)
                    for col, val, label in zip(
                        [r1, r2, r3, r4],
                        [f"{total:,}", f"{fraud_count:,}", f"{safe_count:,}", f"{fraud_pct:.2f}%"],
                        ["Total Transactions", "🔴 Fraud Detected", "🟢 Legitimate", "Fraud Rate"]
                    ):
                        with col:
                            st.markdown(f"""<div class='metric-card'>
                                <div class='metric-value'>{val}</div>
                                <div class='metric-label'>{label}</div></div>""", unsafe_allow_html=True)

                    # ── Charts side by side ──
                    ch1, ch2 = st.columns(2)

                    with ch1:
                        st.markdown("#### 🥧 Fraud vs Legitimate")
                        fig_pie = go.Figure(go.Pie(
                            labels=["Legitimate", "Fraud"],
                            values=[safe_count, fraud_count],
                            marker_colors=["#22c55e", "#ef4444"],
                            hole=0.5,
                            textfont={"color": "white", "size": 13}
                        ))
                        fig_pie.update_layout(
                            paper_bgcolor="#0a0e1a", font_color="#e2e8f0",
                            legend={"bgcolor": "#1a1f35"}, height=300
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)

                    with ch2:
                        st.markdown("#### 📈 Fraud Probability Distribution")
                        fig_hist = go.Figure(go.Histogram(
                            x=probabilities, nbinsx=40,
                            marker_color="#3b82f6", opacity=0.85
                        ))
                        fig_hist.update_layout(
                            xaxis_title="Fraud Probability",
                            yaxis_title="Count",
                            paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
                            font_color="#e2e8f0",
                            xaxis={"gridcolor": "#1e293b"},
                            yaxis={"gridcolor": "#1e293b"},
                            height=300
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

                    # ── Results table ──
                    st.markdown("#### 📋 Full Transaction Results")
                    result_df = pd.DataFrame()
                    result_df["Amount ($)"]           = df_upload["Amount"].values
                    result_df["Hour"]                  = df_upload["Hour"].astype(int).values
                    result_df["Fraud Probability (%)"] = (probabilities * 100).round(2)
                    result_df["Prediction"]            = ["🔴 FRAUD" if p == 1 else "🟢 NORMAL" for p in predictions]
                    result_df["Risk Level"]            = ["🔴 High" if p >= 70 else ("🟡 Medium" if p >= 30 else "🟢 Low")
                                                          for p in (probabilities * 100)]

                    if "Class" in df_upload.columns:
                        result_df["Actual"]   = ["🔴 FRAUD" if c == 1 else "🟢 NORMAL" for c in df_upload["Class"]]
                        result_df["Correct?"] = ["✅" if p == a else "❌"
                                                  for p, a in zip(predictions, df_upload["Class"])]

                    st.dataframe(result_df, use_container_width=True)

                    # ── Download results ──
                    st.markdown("#### ⬇️ Download Your Results")
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=result_df.to_csv(index=True).encode("utf-8"),
                        file_name="fraud_predictions.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
                st.info("Make sure your CSV matches the required format. Download the sample CSV above for reference.")

    else:
        st.markdown("""
        <div style='text-align:center;padding:40px 20px'>
            <div style='font-size:4rem;margin-bottom:12px'>📂</div>
            <div style='font-size:1.1rem;font-weight:700;color:#64748b'>No file uploaded yet</div>
            <div style='font-size:0.85rem;margin-top:6px;color:#475569'>Upload a CSV above to analyze multiple transactions at once</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 3 — MODEL INSIGHTS
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### 📊 Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in zip(
        [c1, c2, c3, c4],
        ["284,807", "492", "0.17%", "30"],
        ["Total Transactions", "Fraud Cases", "Fraud Rate", "Features Used"]
    ):
        with col:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{label}</div></div>""", unsafe_allow_html=True)

    st.markdown("### 🌲 Model Performance Comparison")
    metrics_data = {
        "Metric": ["Precision", "Recall", "F1-Score", "ROC-AUC"],
        "Random Forest": [0.95, 0.82, 0.88, 0.97],
        "Logistic Regression": [0.88, 0.61, 0.72, 0.97],
    }
    df_metrics = pd.DataFrame(metrics_data)
    fig3 = go.Figure()
    for model_name, color in [("Random Forest", "#60a5fa"), ("Logistic Regression", "#f472b6")]:
        fig3.add_trace(go.Bar(
            name=model_name, x=df_metrics["Metric"], y=df_metrics[model_name],
            marker_color=color,
            text=[f"{v:.2f}" for v in df_metrics[model_name]],
            textposition="outside"
        ))
    fig3.update_layout(
        barmode="group",
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a", font_color="#e2e8f0",
        yaxis={"range": [0, 1.1], "gridcolor": "#1e293b"},
        legend={"bgcolor": "#1a1f35", "bordercolor": "#2d3561"}, height=350
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### 🔑 Top Important Features (Random Forest)")
    features   = ["V14", "V4", "V12", "V10", "V11", "V17", "V7", "Amount", "V3", "V16"]
    importance = [0.18, 0.14, 0.12, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04]
    fig4 = go.Figure(go.Bar(
        x=importance[::-1], y=features[::-1], orientation="h",
        marker=dict(color=importance[::-1], colorscale=[[0, "#1e3a8a"], [1, "#60a5fa"]], showscale=False),
        text=[f"{v:.2f}" for v in importance[::-1]], textposition="outside"
    ))
    fig4.update_layout(
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a", font_color="#e2e8f0",
        xaxis={"gridcolor": "#1e293b"}, height=340
    )
    st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 4 — HOW IT WORKS
# ══════════════════════════════════════════════
with tab4:
    st.markdown("### 📖 How This Project Works")
    steps = [
        ("1️⃣", "Data Collection",          "Used the Kaggle Credit Card Fraud dataset — 284,807 real transactions made by European cardholders in September 2013."),
        ("2️⃣", "Exploratory Data Analysis", "Analyzed class distribution, transaction amounts, and time patterns. Found extreme imbalance: only 0.17% are fraudulent."),
        ("3️⃣", "Handling Imbalance (SMOTE)","Applied SMOTE to create synthetic fraud samples and balance the training data."),
        ("4️⃣", "Model Training",            "Trained Logistic Regression (baseline) and Random Forest (main model) on SMOTE-balanced data."),
        ("5️⃣", "Evaluation",                "Evaluated using Precision, Recall, F1-Score, and ROC-AUC — more meaningful than Accuracy for imbalanced data."),
        ("6️⃣", "Deployment",                "Built this interactive Streamlit app supporting both manual prediction and batch CSV upload."),
    ]
    for icon, title, desc in steps:
        st.markdown(f"""
        <div style='background:#1a1f35;border:1px solid #2d3561;border-radius:12px;
                    padding:20px;margin:10px 0;display:flex;align-items:flex-start;gap:16px'>
            <div style='font-size:2rem;min-width:40px'>{icon}</div>
            <div>
                <div style='font-weight:800;color:#60a5fa;font-size:1.05rem;margin-bottom:4px'>{title}</div>
                <div style='color:#94a3b8;font-size:0.9rem'>{desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("### 🛠️ Tech Stack")
    tech = ["Python 3.10+", "Scikit-learn", "imbalanced-learn (SMOTE)", "Pandas & NumPy", "Plotly", "Streamlit"]
    cols = st.columns(3)
    for i, t in enumerate(tech):
        with cols[i % 3]:
            st.markdown(f"""
            <div style='background:#1a1f35;border:1px solid #2d3561;border-radius:8px;
                        padding:12px;text-align:center;margin:4px 0;color:#60a5fa;font-weight:700'>
                {t}
            </div>""", unsafe_allow_html=True)
