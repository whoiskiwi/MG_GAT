"""Step 1-3: Filter PA businesses, reviews, and extract user IDs."""

import json
import os
import pandas as pd


def filter_pa_businesses(data_dir: str) -> pd.DataFrame:
    """Step 1: Filter businesses in Pennsylvania."""
    pa_businesses = []
    total = 0
    path = os.path.join(data_dir, 'yelp_academic_dataset_business.json')

    with open(path, 'r') as f:
        for line in f:
            total += 1
            biz = json.loads(line)
            if biz.get('state') == 'PA':
                pa_businesses.append(biz)

    df = pd.DataFrame(pa_businesses)
    print(f'[Step 1] Total businesses: {total} | PA businesses: {len(df)}')
    print(f'  Top cities: {df["city"].value_counts().head(5).to_dict()}')
    return df


def filter_pa_reviews(data_dir: str, pa_biz_ids: set,
                      year_start: int = 2009, year_end: int = 2018) -> pd.DataFrame:
    """Step 2: Filter reviews belonging to PA businesses within [year_start, year_end]."""
    pa_reviews = []
    total = 0
    path = os.path.join(data_dir, 'yelp_academic_dataset_review.json')

    with open(path, 'r') as f:
        for line in f:
            total += 1
            review = json.loads(line)
            if review['business_id'] in pa_biz_ids:
                year = int(review['date'][:4])
                if year_start <= year <= year_end:
                    pa_reviews.append(review)
            if total % 1_000_000 == 0:
                print(f'  ...scanned {total:,} reviews')

    df = pd.DataFrame(pa_reviews)
    df['date'] = pd.to_datetime(df['date'])
    print(f'[Step 2] Total reviews: {total:,} | PA reviews ({year_start}-{year_end}): {len(df):,}')
    print(f'  Date range: {df["date"].min()} ~ {df["date"].max()}')
    print(f'  Unique users: {df["user_id"].nunique()} | Unique businesses: {df["business_id"].nunique()}')
    return df


def extract_user_ids(df_reviews: pd.DataFrame) -> set:
    """Step 3: Extract unique user IDs from PA reviews."""
    user_ids = set(df_reviews['user_id'])
    reviews_per_user = df_reviews.groupby('user_id').size()
    print(f'[Step 3] Unique users: {len(user_ids):,}')
    print(f'  Users with >=5 reviews: {(reviews_per_user >= 5).sum()}')
    print(f'  Users with >=10 reviews: {(reviews_per_user >= 10).sum()}')
    return user_ids
