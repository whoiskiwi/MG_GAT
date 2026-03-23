"""
Build implicit features via SVD on binarized rating matrix.

Paper: Appendix D1 "Implicit Features"
  X_bina[u,v] = 1 if X[u,v] > 0 else 0
  SVD: X_bina = U(0) Sigma B(0)
  S_u,imp = U(0) * Sigma^(1/2)   shape: (n_users, ki)
  S_b,imp = B(0)^T * Sigma^(1/2) shape: (n_biz, ki)

Only uses training set to avoid data leakage. [Appendix D2]
ki is a hyperparameter. [Appendix D1]
"""

import os
import pickle

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler


def build_implicit_features(
    processed_dir: str,
    graphs_dir: str,
    output_dir: str,
    ki: int = 32,
    random_state: int = 42,
):
    """
    Build and save implicit features for users and businesses.

    Args:
        processed_dir: directory with pa_reviews_train.csv
        graphs_dir:    directory with usr2idx.pkl, biz2idx.pkl
        output_dir:    where to save implicit feature arrays
        ki:            SVD latent dimension (hyperparameter)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load index maps
    with open(os.path.join(graphs_dir, 'usr2idx.pkl'), 'rb') as f:
        usr2idx = pickle.load(f)
    with open(os.path.join(graphs_dir, 'biz2idx.pkl'), 'rb') as f:
        biz2idx = pickle.load(f)

    n_users = len(usr2idx)
    n_biz   = len(biz2idx)
    print(f'Users: {n_users:,} | Businesses: {n_biz:,} | ki={ki}')

    # Load training reviews only (avoid data leakage) [Appendix D2]
    print('Loading training reviews...')
    df_train = pd.read_csv(os.path.join(processed_dir, 'pa_reviews_train.csv'))
    print(f'  Train reviews: {len(df_train):,}')

    # Build binarized rating matrix
    # Paper: "X_bina[u,v] = 1 if X[u,v] > 0 else 0" [Appendix D1]
    print('Building binarized rating matrix...')
    rows, cols = [], []
    for _, row in df_train.iterrows():
        u = usr2idx.get(row['user_id'])
        b = biz2idx.get(row['business_id'])
        if u is not None and b is not None:
            rows.append(u)
            cols.append(b)

    data = np.ones(len(rows), dtype=np.float32)
    X_bina = sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_biz))
    print(f'  Matrix shape: {X_bina.shape} | Non-zeros: {X_bina.nnz:,}')

    # SVD decomposition
    # Paper: "X_bina = U(0) Sigma B(0)" [Appendix D1]
    print(f'Running TruncatedSVD with ki={ki}...')
    svd = TruncatedSVD(n_components=ki, random_state=random_state)
    U = svd.fit_transform(X_bina)      # (n_users, ki)
    S = svd.singular_values_            # (ki,)
    V = svd.components_.T               # (n_biz, ki)

    sqrt_S = np.sqrt(S)

    # Paper: "S_u,imp = U(0) * Sigma^(1/2)" [Appendix D1]
    Su_imp = (U * sqrt_S).astype(np.float32)
    # Paper: "S_b,imp = B(0)^T * Sigma^(1/2)" [Appendix D1]
    Sb_imp = (V * sqrt_S).astype(np.float32)

    print(f'  Su_imp shape: {Su_imp.shape}')
    print(f'  Sb_imp shape: {Sb_imp.shape}')

    # Normalise (consistent with explicit features)
    Su_imp = MinMaxScaler().fit_transform(Su_imp).astype(np.float32)
    Sb_imp = MinMaxScaler().fit_transform(Sb_imp).astype(np.float32)

    # Save
    np.save(os.path.join(output_dir, 'Su_imp.npy'), Su_imp)
    np.save(os.path.join(output_dir, 'Sb_imp.npy'), Sb_imp)

    print(f'\nDone! Saved to {output_dir}:')
    print(f'  Su_imp.npy: {Su_imp.nbytes / 1024 / 1024:.1f} MB  shape={Su_imp.shape}')
    print(f'  Sb_imp.npy: {Sb_imp.nbytes / 1024 / 1024:.1f} MB  shape={Sb_imp.shape}')

    return Su_imp, Sb_imp
