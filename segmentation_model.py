# file: segmentation_model.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ---------- RFM Scoring ----------
def rfm_score(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RFM-style scores based on recency, frequency, and monetary features.
    """
    rfm = agg.copy()

    # Handle recency
    if 'recency_days' in rfm.columns and rfm['recency_days'].notna().sum() > 0:
        rfm['r_rank'] = pd.qcut(
            rfm['recency_days'].rank(method='first', ascending=True),
            5, labels=[5,4,3,2,1]
        )
    else:
        rfm['r_rank'] = 3

    # Handle missing columns
    rfm['frequency'] = rfm.get('frequency', 0)
    rfm['monetary'] = rfm.get('monetary', 0.0)

    # Frequency & Monetary ranks
    rfm['f_rank'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
    rfm['m_rank'] = pd.qcut(rfm['monetary'].rank(method='first'), 5, labels=[1,2,3,4,5])

    rfm['rfm_sum'] = (
        rfm['r_rank'].astype(int) + rfm['f_rank'].astype(int) + rfm['m_rank'].astype(int)
    )

    def label(row):
        total = row['rfm_sum']
        if total >= 13: return 'Champion'
        if total >= 10: return 'Loyal'
        if total >= 7: return 'Potential'
        if total >= 4: return 'At Risk'
        return 'Hibernating'

    rfm['rfm_segment'] = rfm.apply(label, axis=1)
    return rfm


# ---------- Outlier Removal ----------
def remove_outliers(df, features):
    """
    Removes outliers using IQR (Interquartile Range) method.
    """
    Q1 = df[features].quantile(0.25)
    Q3 = df[features].quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((df[features] < (Q1 - 1.5 * IQR)) | (df[features] > (Q3 + 1.5 * IQR))).any(axis=1)
    return df.loc[mask].reset_index(drop=True)


# ---------- Clustering ----------
def cluster_customers(
    df,
    n_clusters: int = 4,
    features: list[str] = None,
    scale_data: bool = True,
    remove_outlier: bool = True
):
    """
    Performs KMeans clustering with optional scaling and outlier removal.
    Returns: 
      - df_clustered (with 'cluster')
      - silhouette score
      - cluster centers (in original units)
    """
    if not features:
        features = ['frequency', 'monetary', 'unique_skus', 'coupon_rate', 'total_qty']

    # Clean
    df_clean = df.copy()
    if remove_outlier:
        df_clean = remove_outliers(df_clean, features)

    X_raw = df_clean[features].copy()  # keep original scale

    # Scale for clustering
    if scale_data:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
    else:
        X_scaled = X_raw.values
        scaler = None

    # KMeans clustering
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = km.fit_predict(X_scaled)
    df_clean['cluster'] = clusters

    # Evaluate
    sil = silhouette_score(X_scaled, clusters) if len(set(clusters)) > 1 else np.nan

    # Get cluster centers back in original units
    if scale_data:
        centers_real = pd.DataFrame(
            scaler.inverse_transform(km.cluster_centers_),
            columns=features
        )
    else:
        centers_real = pd.DataFrame(km.cluster_centers_, columns=features)

    centers_real['cluster'] = range(n_clusters)

    return df_clean, sil, centers_real
