"""Business geo proximity graph [Appendix D1]."""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import BallTree


def build_geo_graph(df_biz: pd.DataFrame, biz2idx: dict, k: int = 10) -> sp.csr_matrix:
    """
    Geo proximity graph: k-NN by haversine distance.
    Paper: "geographical proximity captures spatial dependencies" [Appendix D1]
    """
    n = len(biz2idx)

    biz_lookup = df_biz.set_index('business_id')
    coords = np.zeros((n, 2), dtype=np.float64)
    valid_mask = np.zeros(n, dtype=bool)

    for bid, idx in biz2idx.items():
        if bid in biz_lookup.index:
            lat = biz_lookup.loc[bid, 'latitude']
            lon = biz_lookup.loc[bid, 'longitude']
            if lat and lon and not np.isnan(float(lat)) and not np.isnan(float(lon)):
                coords[idx, 0] = float(lat)
                coords[idx, 1] = float(lon)
                valid_mask[idx] = True

    valid_idx = np.where(valid_mask)[0]
    valid_coords = np.radians(coords[valid_idx])

    tree = BallTree(valid_coords, metric='haversine')
    _, neighbors = tree.query(valid_coords, k=k + 1)

    rows, cols = [], []
    for li, gi in enumerate(valid_idx):
        for ni in neighbors[li, 1:]:
            gj = valid_idx[ni]
            rows += [gi, gj]; cols += [gj, gi]

    data = np.ones(len(rows), dtype=np.float32)
    G = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    G.data[:] = 1

    print(f'[G_b geo]  Businesses: {n:,} | Avg degree: {G.nnz/n:.1f}')
    return G
