"""Business shared-category graph [Appendix D1]."""

import numpy as np
import pandas as pd
import scipy.sparse as sp


def build_category_graph(df_biz: pd.DataFrame, biz2idx: dict, k: int = 10) -> sp.csr_matrix:
    """
    Shared category graph: k-NN by number of shared categories.
    Uses sparse matrix multiplication: shared = M @ M.T where M is (n_biz x n_cat).
    """
    n = len(biz2idx)
    biz_lookup = df_biz.set_index('business_id')

    # Build category vocabulary
    cat_vocab = {}
    cat_idx = 0
    rows_m, cols_m = [], []

    for bid, idx in biz2idx.items():
        if bid not in biz_lookup.index:
            continue
        cats = str(biz_lookup.loc[bid, 'categories'] or '')
        for c in cats.split(','):
            c = c.strip()
            if not c:
                continue
            if c not in cat_vocab:
                cat_vocab[c] = cat_idx
                cat_idx += 1
            rows_m.append(idx)
            cols_m.append(cat_vocab[c])

    n_cats = len(cat_vocab)
    print(f'  Category vocab: {n_cats} | Building sparse matrix...')

    # M: (n_biz x n_cat) binary matrix
    M = sp.csr_matrix(
        (np.ones(len(rows_m), dtype=np.float32), (rows_m, cols_m)),
        shape=(n, n_cats)
    )

    # shared_count = M @ M.T → (n_biz x n_biz), entry (i,j) = # shared categories
    print(f'  Computing M @ M.T ...')
    shared = M.dot(M.T).tocsr()
    shared.setdiag(0)  # remove self-loops
    shared.eliminate_zeros()

    # For each business, keep top-k neighbors by shared category count
    print(f'  Extracting top-{k} neighbors per business...')
    rows_g, cols_g = [], []
    for i in range(n):
        row_start = shared.indptr[i]
        row_end = shared.indptr[i + 1]
        if row_start == row_end:
            continue
        js = shared.indices[row_start:row_end]
        vs = shared.data[row_start:row_end]
        if len(js) <= k:
            top_js = js
        else:
            top_idx = np.argpartition(vs, -k)[-k:]
            top_js = js[top_idx]
        for j in top_js:
            rows_g += [i, j]; cols_g += [j, i]

    data = np.ones(len(rows_g), dtype=np.float32)
    G = sp.csr_matrix((data, (rows_g, cols_g)), shape=(n, n))
    G.data[:] = 1

    print(f'[G_b cat]  Businesses: {n:,} | Avg degree: {G.nnz/n:.1f}')
    return G
