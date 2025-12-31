# file: feature_engineering.py
import pandas as pd
import numpy as np

def build_customer_features(df, reference_date=None):
    if reference_date is None:
        reference_date = df['order_date'].max() + pd.Timedelta(days=1)

    revenue_col = next((c for c in df.columns if 'FinalPrice' in c or 'price' in c.lower()), None)
    df['revenue'] = pd.to_numeric(df.get(revenue_col, 0), errors='coerce').fillna(0)
    df['Qty'] = pd.to_numeric(df.get('Sum of QtyFinal', 0), errors='coerce').fillna(0)

    agg = df.groupby('CustomerId').agg(
        first_purchase=('order_date', 'min'),
        last_purchase=('order_date', 'max'),
        frequency=('OrderId', 'nunique'),
        monetary=('revenue', 'sum'),
        total_qty=('Qty', 'sum'),
        unique_skus=('sku', 'nunique'),
        unique_categories=('Category Level1', 'nunique')
    ).reset_index()

    agg['recency_days'] = (reference_date - agg['last_purchase']).dt.days
    agg['customer_tenure_days'] = (agg['last_purchase'] - agg['first_purchase']).dt.days.fillna(0)

    # نرخ استفاده از کوپن
    coupons = df[df['DiscountDescription'].notna()].groupby('CustomerId').size().rename('coupon_orders')
    agg = agg.merge(coupons, on='CustomerId', how='left').fillna({'coupon_orders': 0})
    agg['coupon_rate'] = agg['coupon_orders'] / agg['frequency'].replace(0, np.nan)
    agg['coupon_rate'] = agg['coupon_rate'].fillna(0)

    # فاصله میان خریدها
    def avg_days_between(cid):
        dates = df.loc[df['CustomerId'] == cid, 'order_date'].dropna().sort_values()
        if len(dates) <= 1: return np.nan
        return np.mean(dates.diff().dt.days.dropna())

    agg['avg_days_between'] = agg['CustomerId'].apply(avg_days_between)

    # دسته محبوب
    fav = (df.groupby(['CustomerId', 'Category Level1']).size()
           .reset_index(name='n')
           .sort_values(['CustomerId', 'n'], ascending=[True, False])
           .drop_duplicates('CustomerId'))
    agg = agg.merge(fav[['CustomerId', 'Category Level1']], on='CustomerId', how='left')
    agg.rename(columns={'Category Level1': 'favorite_category'}, inplace=True)

    return agg.fillna(0)
