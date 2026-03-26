"""
Generate Table D4, D5, D6 statistics as CSV files into reports/tables/
"""

import os
import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'reports', 'tables')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'processed')
os.makedirs(OUTPUT_DIR, exist_ok=True)

df_biz = pd.read_csv(os.path.join(DATA_DIR, 'pa_biz_features.csv'))

# ── Table D4: User Auxiliary Information After Min-Max Normalization (PA) ──
print('[Table D4] User features...')
df_users = pd.read_csv(os.path.join(DATA_DIR, 'pa_users_feat.csv'))
feat_cols = [c for c in df_users.columns if c != 'user_id']
stats_d4 = df_users[feat_cols].describe().T[['mean', 'std', '25%', '50%', '75%']]
stats_d4.columns = ['Mean', 'Standard Deviation', '25th Percentile', 'Median', '75th Percentile']
stats_d4.index.name = 'Feature'
stats_d4.to_csv(os.path.join(OUTPUT_DIR, 'table_d4_user_features.csv'))
print(f'  {len(feat_cols)} features, {len(df_users):,} users')

# ── Table D5: Business Auxiliary Information (Attributes) ──
print('[Table D5] Business attributes...')
attr_cols = [c for c in df_biz.columns if c.startswith('attr_')]
attr_sums = df_biz[attr_cols].sum()
attr_means = df_biz[attr_cols].mean()
attr_sums = attr_sums.where(attr_sums > 0)
attr_means = attr_means.where(attr_sums.notna())
stats_d5 = pd.DataFrame({'PA Sum': attr_sums, 'PA Mean': attr_means})
stats_d5.index = [c.replace('attr_', 'attributes: ') for c in stats_d5.index]
stats_d5.index.name = 'Attribute'
stats_d5.to_csv(os.path.join(OUTPUT_DIR, 'table_d5_biz_attributes.csv'))
print(f'  {len(attr_cols)} attributes ({stats_d5["PA Sum"].notna().sum()} with data, {stats_d5["PA Sum"].isna().sum()} NaN)')

# ── Table D6: Business Auxiliary Information (Categories) ──
print('[Table D6] Business categories...')
cat_cols = [c for c in df_biz.columns if c.startswith('cat_')]
cat_sums = df_biz[cat_cols].sum()
cat_means = df_biz[cat_cols].mean()
cat_sums = cat_sums.where(cat_sums > 0)
cat_means = cat_means.where(cat_sums.notna())
stats_d6 = pd.DataFrame({'PA Sum': cat_sums, 'PA Mean': cat_means})
stats_d6.index = [c.replace('cat_', 'categories: ') for c in stats_d6.index]
stats_d6.index.name = 'Category'
stats_d6.to_csv(os.path.join(OUTPUT_DIR, 'table_d6_biz_categories.csv'))
print(f'  {len(cat_cols)} categories ({stats_d6["PA Sum"].notna().sum()} with data, {stats_d6["PA Sum"].isna().sum()} NaN)')

print(f'\nDone! All CSV files in: {OUTPUT_DIR}')
