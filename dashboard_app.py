# file: dashboard_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

from data_preprocessing import parse_dates
from feature_engineering import build_customer_features
from segmentation_model import rfm_score, cluster_customers

st.set_page_config(layout="wide", page_title="Customer Segmentation Dashboard")

# -------------- Helpers --------------
@st.cache_data(show_spinner=False)
def load_file(uploaded_file: bytes, name: str, sheet: str | int | None):
    if name.lower().endswith(".xlsx"):
        return pd.read_excel(uploaded_file, sheet_name=sheet)
    return pd.read_csv(uploaded_file)

@st.cache_data(show_spinner=False)
def process_data(raw_df: pd.DataFrame, n_clusters: int, features: list[str]):
    df = parse_dates(raw_df.copy())
    agg = build_customer_features(df)
    agg = rfm_score(agg)
    agg, sil, centers = cluster_customers(
    agg, 
    n_clusters=n_clusters, 
    features=features, 
    scale_data=scale_data, 
    remove_outlier=remove_outlier)
    return agg, sil, centers


# -------------- Sidebar --------------
st.title("Customer Segmentation Dashboard")
with st.sidebar:
    st.header("Data Source")
    use_sample = st.toggle("Use sample dataset (data/data.xlsx)", value=True)
    uploaded = None
    sheet_name = None
    if not use_sample:
        uploaded = st.file_uploader("Upload your sales file", type=["xlsx", "csv"])
        if uploaded is not None and uploaded.name.lower().endswith(".xlsx"):
            # Try to list sheets quickly
            try:
                xls = pd.ExcelFile(uploaded)
                sheet_name = st.selectbox("Sheet", options=xls.sheet_names)
            except Exception:
                sheet_name = None
    else:
        sample_path = Path(__file__).resolve().parents[1] / "data" / "data.xlsx"
        if not sample_path.exists():
            st.warning("Sample file not found at data/data.xlsx. Please upload your file.")
            use_sample = False

    st.header("Model Settings")
    n_clusters = st.slider("Clusters (k)", min_value=2, max_value=10, value=4, step=1)
    default_features = ['frequency','monetary','unique_skus','coupon_rate','total_qty']
    features = st.multiselect("Clustering Features", default_features, default=default_features)
    st.write("**Data Cleaning Options**")
    scale_data = st.toggle("Normalize features (Z-score scaling)", value=True)
    remove_outlier = st.toggle("Remove outliers (IQR method)", value=True)
    run_btn = st.button("Run Segmentation", type="primary")



# -------------- Load data --------------
if use_sample:
    sample_path = Path(__file__).resolve().parents[1] / "data" / "data.xlsx"
    try:
        raw_df = pd.read_excel(sample_path)
    except Exception as e:
        st.error(f"Failed to load sample: {e}")
        st.stop()
else:
    if uploaded is None:
        st.info("Upload a file to begin.")
        st.stop()
    try:
        raw_df = load_file(uploaded, uploaded.name, sheet_name)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

st.success(f"Data loaded: {raw_df.shape[0]:,} rows, {raw_df.shape[1]:,} columns")


# -------------- Run processing --------------
if run_btn or use_sample:
    try:
        agg, sil, centers = process_data(raw_df, n_clusters, features if features else default_features)
    except Exception as e:
        st.error(f"Processing error: {e}")
        st.stop()

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Customers", f"{agg.shape[0]:,}")
    with c2:
        st.metric("Clusters", f"{agg['cluster'].nunique()}" )
    with c3:
        st.metric("Silhouette", f"{sil:.2f}")
    with c4:
        st.metric("Segments", f"{agg['rfm_segment'].nunique() if 'rfm_segment' in agg else 0}")

    # Segment selection (user preference)
    seg_options = sorted(agg['rfm_segment'].unique()) if 'rfm_segment' in agg else []
    selected_segments = st.multiselect("Segments to view", options=seg_options, default=seg_options)
    filtered = agg[agg['rfm_segment'].isin(selected_segments)] if 'rfm_segment' in agg and selected_segments else agg
    st.metric("Users in selected segments", f"{filtered.shape[0]:,}")

    # Tabs
    tab_overview, tab_clusters, tab_dists, tab_table = st.tabs(["Overview", "Clusters", "Distributions", "Data Table"])

    with tab_overview:
        st.subheader("Segment Distribution")
        if 'rfm_segment' in agg:
            seg_counts = agg['rfm_segment'].value_counts().reset_index()
            seg_counts.columns = ['segment', 'count']
            fig = px.bar(seg_counts, x='segment', y='count', text='count', color='segment')
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Monetary vs Frequency by Cluster")
        fig2 = px.scatter(filtered, x='monetary', y='frequency', color='cluster', hover_data=['CustomerId'])
        st.plotly_chart(fig2, use_container_width=True)

    with tab_clusters:
        st.subheader("Cluster Explorer")
        numeric_cols = [c for c in filtered.columns if pd.api.types.is_numeric_dtype(filtered[c])]
        x_axis = st.selectbox("X Axis", options=numeric_cols, index=max(0, numeric_cols.index('frequency') if 'frequency' in numeric_cols else 0))
        y_axis = st.selectbox("Y Axis", options=numeric_cols, index=max(0, numeric_cols.index('monetary') if 'monetary' in numeric_cols else 0))
        color_by = st.selectbox("Color By", options=['cluster', 'rfm_segment'] if 'rfm_segment' in filtered else ['cluster'])
        fig3 = px.scatter(filtered, x=x_axis, y=y_axis, color=color_by, hover_data=['CustomerId'])
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Cluster Profile (mean values)")
        prof = filtered.groupby('cluster')[features if features else default_features].mean().round(2).reset_index()
        st.dataframe(prof, use_container_width=True)

        st.subheader("Cluster Centers (original units)")
        st.dataframe(centers.round(2), use_container_width=True)


    with tab_dists:
        st.subheader("Feature Distributions")
        cols = st.multiselect("Select features", options=features if features else default_features, default=features if features else default_features)
        n = len(cols)
        if n:
            rows = int(np.ceil(n / 2))
            for i in range(rows):
                cc = st.columns(2)
                for j in range(2):
                    idx = i * 2 + j
                    if idx >= n: break
                    col = cols[idx]
                    with cc[j]:
                        fig = px.histogram(filtered, x=col, nbins=30)
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least one feature to visualize.")

    with tab_table:
        st.subheader("Customers Table")
        sortable_cols = ['cluster', 'monetary', 'frequency']
        sort_by = [c for c in sortable_cols if c in filtered.columns]
        st.dataframe(filtered.sort_values(sort_by).reset_index(drop=True), use_container_width=True)

else:
    st.info("Set parameters and click Run Segmentation.")
