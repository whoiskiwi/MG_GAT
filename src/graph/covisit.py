"""Business co-visitation graph [Appendix D1]."""

from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp


def build_covisit_graph(df_train: pd.DataFrame, biz2idx: dict, k: int = 10) -> sp.csr_matrix:
    """
    Co-visitation graph: k-NN by shared customer count (train set only).
    Paper: "Co-visitation data provide additional predictive value,
            reflecting economic ties based on consumer behavior" [Appendix D1]
    Only uses training set to avoid data leakage. [Appendix D2]
    """
    n = len(biz2idx)

    user_biz = defaultdict(set)
    total_rows = len(df_train)
    for i, (_, row) in enumerate(df_train.iterrows()):
        if i % 200000 == 0:
            print(f'  ...scanning reviews {i:,}/{total_rows:,} ({100*i/total_rows:.1f}%)')
        u = row['user_id']
        b = biz2idx.get(row['business_id'])
        if b is not None:
            user_biz[u].add(b)

    print(f'  ...computing co-visit pairs for {len(user_biz):,} users')
    covisit = defaultdict(int)
    for ui, biz_set in enumerate(user_biz.values()):
        if ui % 50000 == 0:
            print(f'  ...user {ui:,}/{len(user_biz):,} ({100*ui/len(user_biz):.1f}%)')
        biz_list = list(biz_set)
        for i in range(len(biz_list)):
            for j in range(i + 1, len(biz_list)):
                a, b = biz_list[i], biz_list[j]
                covisit[(a, b)] += 1
                covisit[(b, a)] += 1

    biz_neighbors = defaultdict(list)
    for (a, b), cnt in covisit.items():
        biz_neighbors[a].append((cnt, b))

    rows, cols = [], []
    for a, nbrs in biz_neighbors.items():
        top_k = sorted(nbrs, reverse=True)[:k]
        for _, b in top_k:
            rows += [a, b]; cols += [b, a]

    data = np.ones(len(rows), dtype=np.float32)
    G = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    G.data[:] = 1

    print(f'[G_b covisit] Businesses: {n:,} | Avg degree: {G.nnz/n:.1f}')
    return G
