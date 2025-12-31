# Customer Segmentation Dashboard

This project provides a **Customer Segmentation Dashboard** built with **Streamlit**, designed for analyzing customer purchase behavior, RFM segmentation, and clustering based on purchase patterns and category preferences.

---

## 🚀 Features

### 1. RFM Analysis
- Computes **Recency, Frequency, Monetary (RFM)** scores.
- Assigns customers into segments such as `Champion`, `Loyal`, `Potential`, `At Risk`, `Hibernating`.
- Interactive bar chart showing the distribution of RFM segments.
- Filterable table of RFM metrics with download option.

### 2. Cluster Analysis
- Performs **KMeans clustering** on customer features (`frequency`, `monetary`, `unique_skus`, `coupon_rate`, `total_qty`).
- Shows cluster counts and allows filtering by clusters.
- Displays **cluster details** and **category-based cluster analysis**.
- Interactive tables with download options.
- Visualizations of cluster centers and mean profiles.

### 3. Customer Frequency Report
- Lists customers based on purchase frequency:
  - 1-time buyers
  - 2-time buyers
  - 3-5 times buyers
- Coupon usage analysis.
- Filterable tables with download capability.

---

## 📂 File Structure
├── app.py                 # Streamlit dashboard using precomputed Excel data
├── dashboard_app.py       # Full dashboard with feature engineering & clustering
├── segmentation_model.py  # Functions for RFM scoring, clustering, and outlier removal
├── feature_engineering.py # Build customer-level features from raw sales data
├── data/                  # Sample dataset (Excel/CSV)
├── requirements.txt       # Python dependencies
└── README.md

