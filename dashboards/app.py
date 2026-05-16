"""Streamlit interactive forecasting dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.synthetic import SyntheticDataGenerator
from src.utils.config import PROJECT_ROOT, load_config

st.set_page_config(
    page_title="FMCG Demand Forecasting",
    page_icon="📦",
    layout="wide",
)

config = load_config()


@st.cache_data
def load_sample_data():
    gen = SyntheticDataGenerator(n_skus=20, n_dcs=5, n_days=365)
    return gen.generate()


def main():
    st.title("AI-Powered Multi-Horizon Demand Forecasting")
    st.caption("FMCG Supply Chain Analytics Platform")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Forecasts", "SKU Analytics", "Warehouse", "Model Performance", "Explainability"]
    )

    data = load_sample_data()
    sales = data["sales_history"]

    with tab1:
        st.subheader("Demand Forecast Visualization")
        col1, col2, col3 = st.columns(3)
        sku = col1.selectbox("SKU", sales["sku_id"].unique())
        dc = col2.selectbox("Distribution Center", sales["dc_id"].unique())
        horizon = col3.slider("Forecast Horizon (days)", 7, 28, 14)

        series = sales[(sales["sku_id"] == sku) & (sales["dc_id"] == dc)].sort_values("date")
        ma7 = series["demand"].rolling(7).mean()
        ma28 = series["demand"].rolling(28).mean()
        last_val = series["demand"].iloc[-horizon:].values
        forecast = np.full(horizon, ma7.iloc[-1] if not np.isnan(ma7.iloc[-1]) else series["demand"].mean())

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series["date"], y=series["demand"], name="Actual", mode="lines"))
        fig.add_trace(go.Scatter(x=series["date"], y=ma7, name="MA-7", line=dict(dash="dot")))
        future_dates = pd.date_range(series["date"].max() + pd.Timedelta(days=1), periods=horizon)
        fig.add_trace(go.Scatter(x=future_dates, y=forecast, name="Forecast", line=dict(dash="dash", color="red")))
        fig.update_layout(title=f"Demand Forecast: {sku} @ {dc}", xaxis_title="Date", yaxis_title="Demand")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("SKU-Level Analytics")
        sku_summary = sales.groupby("sku_id").agg(
            total_demand=("demand", "sum"),
            avg_demand=("demand", "mean"),
            volatility=("demand", "std"),
        ).reset_index()
        fig = px.bar(sku_summary.head(15), x="sku_id", y="total_demand", title="Top SKUs by Demand")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Warehouse / DC Analytics")
        dc_summary = sales.groupby("dc_id").agg(total=("demand", "sum"), avg=("demand", "mean")).reset_index()
        fig = px.pie(dc_summary, values="total", names="dc_id", title="Demand by Distribution Center")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Model Performance")
        report_path = PROJECT_ROOT / "reports" / "model_comparison.csv"
        if report_path.exists():
            comparison = pd.read_csv(report_path, index_col=0)
            st.dataframe(comparison.style.highlight_min(subset=["wmape"], axis=0))
            fig = px.bar(comparison.reset_index(), x="index", y="wmape", title="WMAPE by Model")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run `python scripts/train.py` to generate model comparison metrics.")

        st.subheader("Drift Monitoring")
        drift_report = PROJECT_ROOT / "reports" / "evidently" / "data_drift_report.html"
        if drift_report.exists():
            st.success("Latest drift report available")
            st.markdown(f"[View Drift Report]({drift_report})")
        else:
            st.warning("No drift report yet. Run monitoring pipeline.")

    with tab5:
        st.subheader("SHAP & Explainability")
        shap_path = PROJECT_ROOT / "reports" / "shap_global_importance.csv"
        if shap_path.exists():
            shap_df = pd.read_csv(shap_path)
            fig = px.bar(shap_df.head(15), x="shap_importance", y="feature", orientation="h", title="SHAP Feature Importance")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Train ML models and run SHAP analysis to view feature importance.")

        attn_path = PROJECT_ROOT / "reports" / "attention_heatmap.png"
        if attn_path.exists():
            st.image(str(attn_path), caption="TFT Attention Heatmap")


if __name__ == "__main__":
    main()
