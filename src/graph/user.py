"""User friendship graph G_u [Appendix D1]."""

import numpy as np
import pandas as pd
import scipy.sparse as sp


def build_user_graph(df_users: pd.DataFrame, usr2idx: dict) -> sp.csr_matrix:
    """
    Build undirected user friendship graph from Yelp friends field.
    Paper: "we can build a friendship network, where a connection indicates
            that the two users are friends" [Appendix D1]
    """
    n = len(usr2idx)
    rows, cols = [], []

    total = len(df_users)
    for i, (_, row) in enumerate(df_users.iterrows()):
        if i % 50000 == 0:
            print(f'  ...processing user {i:,}/{total:,} ({100*i/total:.1f}%)')
        uid = row['user_id']
        u = usr2idx.get(uid)
        if u is None:
            continue
        friends_str = str(row.get('friends', '') or '')
        if not friends_str or friends_str == 'None':
            continue
        for f in friends_str.split(','):
            f = f.strip()
            v = usr2idx.get(f)
            if v is not None and u != v:
                rows.append(u); cols.append(v)
                rows.append(v); cols.append(u)

    data = np.ones(len(rows), dtype=np.float32)
    G_u = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    G_u.data[:] = 1

    n_edges = G_u.nnz // 2
    avg_deg = G_u.nnz / n
    print(f'[G_u] Users: {n:,} | Edges: {n_edges:,} | Avg degree: {avg_deg:.3f}')
    return G_u
