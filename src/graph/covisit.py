"""Business co-visitation graph [Appendix D1]."""

import numpy as np
import pandas as pd
import scipy.sparse as sp


def build_covisit_graph(df_train: pd.DataFrame, biz2idx: dict, k: int = 10) -> sp.csr_matrix:
    """
    Co-visitation graph: k-NN by shared customer count (train set only).
    Paper: "Co-visitation data provide additional predictive value,
            reflecting economic ties based on consumer behavior" [Appendix D1]
    Only uses training set to avoid data leakage. [Appendix D2]

    Uses sparse matrix multiplication: C = X^T @ X, where X is the binary
    user-business matrix. C[i,j] = number of users who visited both i and j.
    Top-k neighbors per business, then symmetrized with OR logic (consistent
    with geo/category/user graphs).
    """
    n = len(biz2idx)

    # Build binary user-business matrix X (n_users x n_biz)
    uid2row = {}
    u_rows, b_cols = [], []
    for _, row in df_train.iterrows():
        u = row['user_id']
        b = biz2idx.get(row['business_id'])
        if b is None:
            continue
        if u not in uid2row:
            uid2row[u] = len(uid2row)
        u_rows.append(uid2row[u])
        b_cols.append(b)

    n_users = len(uid2row)
    X = sp.csr_matrix(
        (np.ones(len(u_rows), dtype=np.float32), (u_rows, b_cols)),
        shape=(n_users, n),
    )
    print(f'  ...user-business matrix: {X.shape} | nnz={X.nnz:,}')

    # C[i,j] = number of shared visitors between business i and j
    print('  ...computing co-visit counts via X^T @ X')
    C = (X.T @ X).tocsr()   # (n_biz x n_biz)
    C.setdiag(0)
    C.eliminate_zeros()

    # Top-k directed edges per business
    print(f'  ...selecting top-{k} neighbors per business')
    rows, cols = [], []
    for i in range(n):
        start, end = C.indptr[i], C.indptr[i + 1]
        if start == end:
            continue
        nbr_idx = C.indices[start:end]
        nbr_cnt = C.data[start:end]
        if len(nbr_cnt) > k:
            top = np.argpartition(nbr_cnt, -k)[-k:]
            nbr_idx = nbr_idx[top]
        rows.extend([i] * len(nbr_idx))
        cols.extend(nbr_idx.tolist())

    data = np.ones(len(rows), dtype=np.float32)
    G = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    # Symmetrize with OR logic (same as geo/category/user graphs)
    G = G + G.T
    G.data[:] = 1

    print(f'[G_b covisit] Businesses: {n:,} | Avg degree: {G.nnz/n:.1f}')
    return G
