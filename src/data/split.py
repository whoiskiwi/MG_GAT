"""Step 6: Time-based train/val/test split by year."""

import pandas as pd


def time_split(df_reviews: pd.DataFrame,
               train_years=(2009, 2016), val_year=2017, test_year=2018):
    """Split reviews by year: train=2009-2016, val=2017, test=2018."""
    df = df_reviews.copy()
    df['year'] = df['date'].dt.year

    splits = {
        'train': df[(df['year'] >= train_years[0]) & (df['year'] <= train_years[1])].drop(columns='year'),
        'val': df[df['year'] == val_year].drop(columns='year'),
        'test': df[df['year'] == test_year].drop(columns='year'),
    }

    print('[Step 6] Year-based split:')
    for name, df_split in splits.items():
        print(f'  {name:>5}: {len(df_split):>10,} reviews | {df_split["date"].min()} ~ {df_split["date"].max()} | users={df_split["user_id"].nunique()} biz={df_split["business_id"].nunique()}')

    return splits
