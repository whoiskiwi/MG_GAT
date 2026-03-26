"""
Feature extraction aligned with the paper:
- User features:   33 dims  [Table D4]
- Business features: attributes(93) + categories(946) + hours(14) + checkin(144) + location(2)
- Checkin:         144-dim hourly bins  [Appendix D1]

Paper reference: Leng, Liu, Ruiz — SSRN 3696092
"""

import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ─────────────────────────────────────────────
# USER FEATURES  [Table D4, Appendix D1]
# ─────────────────────────────────────────────

# ── Table D4 schema: feature names (from paper) ──────────────────────
# Values are extracted dynamically from the dataset, not hardcoded.
COMPLIMENT_FIELDS = ['cool', 'cute', 'funny', 'hot', 'list',
                     'more', 'note', 'photos', 'plain', 'profile', 'writer']
VOTE_FIELDS = ['cool', 'funny', 'useful']
ELITE_YEARS = [str(y) for y in range(2005, 2019)]  # 14 years + elite_None = 15

def extract_user_features(data_dir: str, target_user_ids: set) -> pd.DataFrame:
    """
    Extract 33 user features per Table D4.

    Columns (33 total):
      compliments (11): cool, cute, funny, hot, list, more,
                        note, photos, plain, profile, writer
      votes       (3):  cool, funny, useful
      profile     (1):  fans
      yelping_since(3): year, month, day
      elite      (15):  elite_2005 ... elite_2018, elite_None

    Paper: Table D4 "User Auxiliary Information After Min-Max Normalization (PA)"
    Also stores: user_id, friends (for graph construction, not a feature)
    """
    users = []
    total = 0
    path = os.path.join(data_dir, 'yelp_academic_dataset_user.json')

    with open(path, 'r') as f:
        for line in f:
            total += 1
            u = json.loads(line)
            if u['user_id'] not in target_user_ids:
                if total % 500_000 == 0:
                    print(f'  ...scanned {total:,} users')
                continue

            # ── yelping_since ──────────────────────────────
            # Table D4: "yelping since year/month/day" — 3 columns
            ys = str(u.get('yelping_since', '2010-01-01'))[:10]
            try:
                dt = datetime.strptime(ys, '%Y-%m-%d')
                ys_year  = dt.year
                ys_month = dt.month
                ys_day   = dt.day
            except Exception:
                ys_year, ys_month, ys_day = 2010, 1, 1

            # ── elite ──────────────────────────────────────
            # Table D4: one column per year 2005–2018 + elite_None
            # Value = 1 if user was elite that year, else 0
            elite_raw = str(u.get('elite', ''))
            elite_set = set(e.strip() for e in elite_raw.split(',') if e.strip())
            elite_cols = {f'elite_{yr}': (1 if yr in elite_set else 0)
                          for yr in ELITE_YEARS}
            # elite_None: 1 if user has never been elite
            elite_cols['elite_None'] = 1 if len(elite_set) == 0 else 0

            row = {'user_id': u['user_id']}

            # compliments (11) — Table D4
            for f in COMPLIMENT_FIELDS:
                row[f'compliment_{f}'] = u.get(f'compliment_{f}', 0) or 0

            # votes (3) — Table D4
            for f in VOTE_FIELDS:
                row[f'votes_{f}'] = u.get(f, 0) or 0

            # profile (1) — Table D4
            row['fans'] = u.get('fans', 0) or 0

            # yelping_since (3) — Table D4
            row['yelping_since_year']  = ys_year
            row['yelping_since_month'] = ys_month
            row['yelping_since_day']   = ys_day

            # friends — for graph construction, NOT a feature column
            row['friends'] = u.get('friends', '') or ''

            # elite (15) — Table D4
            row.update(elite_cols)
            users.append(row)

            if total % 500_000 == 0:
                print(f'  ...scanned {total:,} users')

    df = pd.DataFrame(users)
    print(f'[User features] Total scanned: {total:,} | Matched: {len(df):,}')

    # ── Min-Max normalise the 33 feature columns ───────────────────────────
    # Paper: Table D4 title "After Min-Max Normalization"
    feature_cols = [c for c in df.columns
                    if c not in ('user_id', 'friends')]
    assert len(feature_cols) == 33, f"Expected 33 feature cols, got {len(feature_cols)}: {feature_cols}"

    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols].values.astype(np.float32))

    print(f'  Feature columns ({len(feature_cols)}): {feature_cols}')

    # ── Compute Table D4 descriptive statistics ───────────────────────────
    stats = df[feature_cols].describe().T[['mean', 'std', '25%', '50%', '75%']]
    stats.columns = ['Mean', 'Standard Deviation', '25th Percentile', 'Median', '75th Percentile']
    print('\n[Table D4] User Auxiliary Information After Min-Max Normalization:')
    print(stats.to_string())

    return df


# ─────────────────────────────────────────────
# BUSINESS FEATURES  [Appendix D1, Table D5-D6]
# ─────────────────────────────────────────────

DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


def _parse_hour(s: str) -> float:
    """'9:0' or '9:00' → float hours (0–24)."""
    try:
        h, m = str(s).split(':')
        return int(h) + int(m) / 60
    except Exception:
        return 0.0


def _flatten_attrs(attrs: dict) -> dict:
    """
    Flatten nested attribute dict into flat key-value pairs.
    e.g. {'Ambience': "{'casual': True, 'classy': False}"}
      → {'Ambience_casual': 1, 'Ambience_classy': 0}
    Paper: Table D5 shows keys like "Ambience: casual", "BusinessParking: lot"
    [Table D5]
    """
    import ast
    flat = {}
    if not isinstance(attrs, dict):
        return flat
    for k, v in attrs.items():
        if v is None:
            continue
        # Try to parse string-encoded dicts e.g. "{'casual': True}"
        if isinstance(v, str) and v.startswith('{'):
            try:
                parsed = ast.literal_eval(v)
                if isinstance(parsed, dict):
                    for sub_k, sub_v in parsed.items():
                        flat[f'{k}_{sub_k}'] = sub_v
                    continue
            except Exception:
                pass
        flat[k] = v
    return flat


def _build_attribute_vocab(biz_list: list) -> list:
    """Collect all flattened attribute keys across PA businesses."""
    keys = set()
    for b in biz_list:
        attrs = b.get('attributes') or {}
        if isinstance(attrs, dict):
            keys.update(_flatten_attrs(attrs).keys())
    return sorted(keys)


def _build_category_vocab(biz_list: list) -> list:
    """Collect all category tokens across PA businesses."""
    cats = set()
    for b in biz_list:
        for c in (b.get('categories') or '').split(','):
            c = c.strip()
            if c:
                cats.add(c)
    return sorted(cats)


def extract_business_features(df_biz: pd.DataFrame,
                               df_checkins: pd.DataFrame) -> pd.DataFrame:
    """
    Extract business features per Appendix D1:
      - Attributes:  one-hot, 93 cols for PA       [Table D5]
      - Categories:  one-hot multi-label, 946 cols  [Table D6]
      - Hours:       14 cols (open/close each day)  [Appendix D1]
      - Check-in:    144-dim hourly bins             [Appendix D1]
      - Location:    2 cols (lat, lon)               [Appendix D1]

    Returns DataFrame indexed by business_id with all feature columns.
    """
    biz_records = df_biz.to_dict('records')

    # Build vocabularies from PA businesses
    attr_vocab = _build_attribute_vocab(biz_records)
    cat_vocab  = _build_category_vocab(biz_records)
    print(f'  Attribute vocab size: {len(attr_vocab)}')
    print(f'  Category vocab size:  {len(cat_vocab)}')

    # Build checkin lookup: business_id → 144-dim array
    # Paper: "We aggregate the check-ins into 144 hourly bins in a week"
    # [Appendix D1]
    # 144 = 7 days × 24 hours
    checkin_lookup = {}
    for _, row in df_checkins.iterrows():
        bins = np.zeros(144, dtype=np.float32)
        for d in str(row.get('dates', '')).split(','):
            d = d.strip()
            if not d:
                continue
            try:
                dt = datetime.strptime(d, '%Y-%m-%d %H:%M:%S')
                day_idx  = dt.weekday()   # 0=Mon … 6=Sun
                hour_idx = dt.hour        # 0–23
                bins[day_idx * 24 + hour_idx] += 1
            except Exception:
                continue
        total = bins.sum()
        if total > 0:
            bins /= total   # normalise
        checkin_lookup[row['business_id']] = bins

    rows = []
    for b in biz_records:
        bid = b['business_id']
        row = {'business_id': bid}

        # ── Attributes (one-hot) ──────────────────────────────────────────
        # Paper: "one-hot (i.e., one-of-K) encoding" [Appendix D1]
        # NaN / False / "False" / "None" → 0
        attrs = _flatten_attrs(b.get('attributes') or {})
        for k in attr_vocab:
            v = attrs.get(k)
            row[f'attr_{k}'] = 0 if v in (None, False, 'False', 'None', 'no', 'No', 'u\'none\'', 'none') else 1

        # ── Categories (multi-label one-hot) ─────────────────────────────
        # Paper: "each business may belong to multiple categories" [Appendix D1]
        biz_cats = set(c.strip() for c in (b.get('categories') or '').split(','))
        for c in cat_vocab:
            row[f'cat_{c}'] = 1 if c in biz_cats else 0

        # ── Hours (14 dims: open+close for each of 7 days) ───────────────
        # Paper: "operation hours contain information about when businesses
        #         open and close on each day of the week" [Appendix D1]
        # Normalised to [0, 1] by dividing by 24
        hours = b.get('hours') or {}
        if not isinstance(hours, dict):
            hours = {}
        for day in DAYS:
            slot = str(hours.get(day) or '0:0-0:0').split('-')
            row[f'hours_{day}_open']  = _parse_hour(slot[0]) / 24
            row[f'hours_{day}_close'] = _parse_hour(slot[-1]) / 24

        # ── Check-in bins (144 dims) ──────────────────────────────────────
        bins = checkin_lookup.get(bid, np.zeros(144, dtype=np.float32))
        for i, v in enumerate(bins):
            row[f'checkin_{i}'] = float(v)

        # ── Location (2 dims) ─────────────────────────────────────────────
        # Paper: "The location data, in latitude and longitude" [Appendix D1]
        row['latitude']  = float(b.get('latitude')  or 0.0)
        row['longitude'] = float(b.get('longitude') or 0.0)

        rows.append(row)

    df_feat = pd.DataFrame(rows)

    # ── Normalise location columns ────────────────────────────────────────
    for col in ['latitude', 'longitude']:
        mn, mx = df_feat[col].min(), df_feat[col].max()
        if mx > mn:
            df_feat[col] = (df_feat[col] - mn) / (mx - mn)

    # ── Summary ──────────────────────────────────────────────────────────
    attr_cols = [c for c in df_feat.columns if c.startswith('attr_')]
    cat_cols  = [c for c in df_feat.columns if c.startswith('cat_')]
    n_attr    = len(attr_cols)
    n_cat     = len(cat_cols)
    n_hours   = 14
    n_checkin = 144
    n_loc     = 2
    total_dim = n_attr + n_cat + n_hours + n_checkin + n_loc
    print(f'[Business features] {len(df_feat)} businesses')
    print(f'  attr={n_attr} + cat={n_cat} + hours={n_hours} + checkin={n_checkin} + loc={n_loc} = {total_dim} dims')

    # ── Compute Table D5 statistics (Attributes) ─────────────────────────
    # PA Sum = 0 → NaN (attribute key exists but never appears in PA data)
    attr_sums = df_feat[attr_cols].sum()
    attr_means = df_feat[attr_cols].mean()
    attr_sums = attr_sums.where(attr_sums > 0)    # 0 → NaN
    attr_means = attr_means.where(attr_sums.notna())  # match NaN
    attr_stats = pd.DataFrame({'PA Sum': attr_sums, 'PA Mean': attr_means})
    print(f'\n[Table D5] Business Auxiliary Information (Attributes) — {n_attr} cols:')
    print(attr_stats.to_string())

    # ── Compute Table D6 statistics (Categories) ─────────────────────────
    cat_sums = df_feat[cat_cols].sum()
    cat_means = df_feat[cat_cols].mean()
    cat_sums = cat_sums.where(cat_sums > 0)
    cat_means = cat_means.where(cat_sums.notna())
    cat_stats = pd.DataFrame({'PA Sum': cat_sums, 'PA Mean': cat_means})
    print(f'\n[Table D6] Business Auxiliary Information (Categories) — {n_cat} cols:')
    print(cat_stats.to_string())

    return df_feat


# ─────────────────────────────────────────────
# CHECKIN RAW LOAD  [Appendix D1]
# ─────────────────────────────────────────────

def load_checkin_data(data_dir: str, pa_biz_ids: set) -> pd.DataFrame:
    """
    Load raw checkin dates for PA businesses.
    The 144-dim binning is done inside extract_business_features().
    Paper: "We aggregate the check-ins into 144 hourly bins in a week" [Appendix D1]
    """
    checkins = []
    total = 0
    path = os.path.join(data_dir, 'yelp_academic_dataset_checkin.json')

    with open(path, 'r') as f:
        for line in f:
            total += 1
            rec = json.loads(line)
            if rec['business_id'] in pa_biz_ids:
                checkins.append({
                    'business_id': rec['business_id'],
                    'dates': rec.get('date', ''),
                })

    df = pd.DataFrame(checkins)
    print(f'[Checkin] Total: {total:,} | PA: {len(df):,} | Without checkin: {len(pa_biz_ids) - len(df)}')
    return df