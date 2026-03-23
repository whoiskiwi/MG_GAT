"""Main entry: run the full Yelp PA preprocessing pipeline."""

import os
from src.data.filter import filter_pa_businesses, filter_pa_reviews, extract_user_ids
from src.data.features import extract_business_features, extract_user_features, load_checkin_data
from src.data.split import time_split

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'raw')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'data', 'processed')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Filter all PA businesses
df_biz_all = filter_pa_businesses(DATA_DIR)
pa_biz_ids = set(df_biz_all['business_id'])

# Step 2: Filter PA reviews within 2009-2018
df_reviews = filter_pa_reviews(DATA_DIR, pa_biz_ids, year_start=2009, year_end=2018)

# Narrow down to only businesses that actually have reviews in 2009-2018
reviewed_biz_ids = set(df_reviews['business_id'])
df_biz = df_biz_all[df_biz_all['business_id'].isin(reviewed_biz_ids)].copy()
print(f'  PA businesses with reviews in 2009-2018: {len(df_biz)}')

# Step 3
pa_user_ids = extract_user_ids(df_reviews)

# Step 4
df_users = extract_user_features(DATA_DIR, pa_user_ids)

# Step 5
df_checkins = load_checkin_data(DATA_DIR, reviewed_biz_ids)

# Step 5b: 商家特征（需要 checkin 数据一起传入）
df_biz_features = extract_business_features(df_biz, df_checkins)

# Step 6: Split by year (train=2009-2016, val=2017, test=2018)
splits = time_split(df_reviews)

# Save
print('\nSaving...')
df_biz.to_csv(os.path.join(OUTPUT_DIR, 'pa_businesses.csv'), index=False)
df_biz_features.to_csv(os.path.join(OUTPUT_DIR, 'pa_biz_features.csv'), index=False)
df_users.to_csv(os.path.join(OUTPUT_DIR, 'pa_users.csv'), index=False)
df_checkins.to_csv(os.path.join(OUTPUT_DIR, 'pa_checkins.csv'), index=False)
for name, df in splits.items():
    df.to_csv(os.path.join(OUTPUT_DIR, f'pa_reviews_{name}.csv'), index=False)

print('\nDone! Output:')
for f in sorted(os.listdir(OUTPUT_DIR)):
    size_mb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024 / 1024
    print(f'  {f}: {size_mb:.1f} MB')
