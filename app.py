import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------
# Load Data
# ----------------------
file_path = r"C:\Users\s.khalafi\.cursor\segment customer new\mergedf2.xlsx"
df = pd.read_excel(file_path)


# ----------------------
# Streamlit Layout
# ----------------------
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")
st.title("📊 Customer Segmentation Dashboard")

# ----------------------
# TAB 1: RFM Dashboard
# ----------------------
tab1, tab2, tab3 = st.tabs(["RFM Analysis", "Cluster Analysis", "Customer Frequency Report"])

with tab1:
    st.header("📌 RFM Overview")

    # KPI
    st.metric(label="Total Customers", value=f"{len(df)}")

    # ----------------------
    # RFM Category Bar Chart (Smaller)
    # ----------------------
    st.subheader("RFM Category Distribution")
    rfm_counts = df['RFM_Category_x'].value_counts()

# کوچک‌سازی مطمئن
    fig, ax = plt.subplots(figsize=(10, 3))   # خیلی کوچک
    ax.bar(rfm_counts.index, rfm_counts.values)
    plt.xticks(rotation=90,fontsize=6)

# قرار دادن نمودار داخل ستون باریک (100% مؤثر)
    col1, col2 = st.columns([10, 3])
    with col1:
        st.pyplot(fig, use_container_width=True)


    # ----------------------
    # Filterable RFM Table
    # ----------------------
    st.subheader("RFM Table (Filtered)")

    selected_rfm = st.multiselect("Select RFM Categories", options=df['RFM_Category_x'].unique())

    if selected_rfm:
        filtered_df = df[df['RFM_Category_x'].isin(selected_rfm)]
    else:
        filtered_df = df

    rfm_table = filtered_df[['customer_id','CustomerFullName', 'mobile_x', 'R_x', 'F_x', 'M_x', 'RFM_Score', 'RFM_Category_x']]
    st.dataframe(rfm_table, use_container_width=True)

    st.download_button("Download Filtered RFM Data", rfm_table.to_csv(index=False).encode('utf-8-sig'), "rfm_filtered.csv")

# ----------------------
# TAB 2: Cluster Analysis
# ----------------------
with tab2:
    st.header("🔍 Cluster Preference Analysis")

    # Cluster Counts (Smaller Chart)
    st.subheader("Cluster Counts")
    fig2, ax2 = plt.subplots(figsize=(10, 3.5))
    ax2.bar(df['Cluster'].value_counts().sort_index().index,
            df['Cluster'].value_counts().sort_index().values)
    st.pyplot(fig2)

    cluster_info = """
    0: ['آرایشی بهداشتی'],
    1: ['زنانه'],
    2: ['لوازم خانه'],
    3: ['مردانه'],
    4: ['سوپرمارکت'],
    5: ['بچه گانه', 'زنانه', 'آرایشی بهداشتی'],
    6: ['آرایشی بهداشتی', 'زنانه'],
    7: ['آرایشی بهداشتی', 'سوپرمارکت', 'لوازم خانه'],
    8: ['ورزشی'],
    9: ['الکترونیک'],
    10: ['آرایشی بهداشتی', 'مردانه', 'زنانه'],
    11: ['بچه گانه'],
    12: ['آرایشی بهداشتی', 'مردانه', 'بچه گانه', 'زنانه', 'سوپرمارکت', 'لوازم خانه', 'ورزشی'],
    13: ['آرایشی بهداشتی', 'لوازم خانه'],
    14: ['زنانه', 'مردانه']
    """
    st.subheader("Kmean Clustering")
    st.text_area("Cluster Details", cluster_info, height=250)


    # Cluster Interpretation Table
    #st.subheader("Cluster Interpretation Table")
    #cluster_table = df[['customer_id', 'CustomerFullName', 'mobile_x','Cluster', 'Cluster_Interpretation']]
    #st.dataframe(cluster_table, use_container_width=True)

    #st.download_button("Download Cluster Table", cluster_table.to_csv(index=False), "cluster_table.csv")


    selected_cluster = st.multiselect("Select Cluster", options=df['Cluster'].unique())

    if selected_cluster:
        filtered_dfC = df[df['Cluster'].isin(selected_cluster)]
    else:
        filtered_dfC = df

    Cluster_table = filtered_dfC[['customer_id', 'CustomerFullName', 'mobile_x','Cluster', 'Cluster_Interpretation']]
    st.dataframe(Cluster_table, use_container_width=True)
    st.download_button("Download Filtered Kmean Cluster Data", Cluster_table.to_csv(index=False).encode('utf-8-sig'), "Cluster_table.csv")


    #Tops Clusters Table
    st.subheader("Clusters Based on Customers Category Purchase Frequency")
    cluster_table_top = df[['customer_id', 'CustomerFullName', 'mobile_x','TopCategory', 'Top3Categories']]
    st.dataframe(cluster_table_top, use_container_width=True)

    st.download_button("Download Category Purchase Frequency Table", cluster_table_top.to_csv(index=False).encode('utf-8-sig'), "cluster_table_top.csv")

# ----------------------
# TAB 3: Customer Frequency Report
# ----------------------
with tab3:
    st.header("📘 Customer Purchase Frequency Report")

    one_buy = df[df['Frequency'] == 1]
    two_buy = df[df['Frequency'] == 2]
    three_five_buy = df[(df['Frequency'] >= 3) & (df['Frequency'] <= 5)]

    st.subheader("1-Time Buyers")
    st.write(f"Count: {len(one_buy)}")
    st.dataframe(one_buy[['customer_id','CustomerFullName', 'mobile_x', 'Frequency']])
    st.download_button("Download 1-time Buyers", one_buy.to_csv(index=False).encode('utf-8-sig'), "1_time_buyers.csv")

    st.subheader("2-Time Buyers")
    st.write(f"Count: {len(two_buy)}")
    st.dataframe(two_buy[['customer_id','CustomerFullName', 'mobile_x', 'Frequency']])
    st.download_button("Download 2-time Buyers", two_buy.to_csv(index=False).encode('utf-8-sig'), "2_time_buyers.csv")

    st.subheader("3-5 Time Buyers")
    st.write(f"Count: {len(three_five_buy)}")
    st.dataframe(three_five_buy[['customer_id','CustomerFullName', 'mobile_x', 'Frequency']])
    st.download_button("Download 3-5-time Buyers", three_five_buy.to_csv(index=False).encode('utf-8-sig'), "3_5_time_buyers.csv")

    # ----------------------
    # Coupon Usage (Smaller Chart)
    # ----------------------
    st.subheader("Coupon Usage Analysis")
    coupon_counts = df['CouponUse_x'].value_counts()

    fig2, ax2 = plt.subplots(figsize=(3, 2))
    ax2.bar(coupon_counts.index.astype(str), coupon_counts.values)

    col1, col2 = st.columns([1, 4])  # ستون باریک + فضای خالی
    with col1:
        st.pyplot(fig2, use_container_width=True)


    coupon_table = df[['customer_id','CustomerFullName', 'mobile_x', 'CouponUse_x']]
    st.dataframe(coupon_table)
    st.download_button("Download Coupon Table", coupon_table.to_csv(index=False).encode('utf-8-sig'), "coupon_table.csv")
