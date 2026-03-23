"""Orchestrator: build all graphs and save."""

import os
import pickle

import pandas as pd
import scipy.sparse as sp

from .user import build_user_graph
from .geo import build_geo_graph
from .covisit import build_covisit_graph
from .category import build_category_graph

K = 10  # [Appendix D1: "k is set at ten for each edge type"]


def build_all_graphs(processed_dir: str, output_dir: str):
    """Load processed data, build all graphs, save to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    print('Loading processed data...')
    df_users = pd.read_csv(os.path.join(processed_dir, 'pa_users.csv'))
    df_biz   = pd.read_csv(os.path.join(processed_dir, 'pa_businesses.csv'))
    df_train = pd.read_csv(os.path.join(processed_dir, 'pa_reviews_train.csv'))

    usr2idx = {uid: i for i, uid in enumerate(df_users['user_id'].unique())}
    biz2idx = {bid: i for i, bid in enumerate(df_biz['business_id'].unique())}
    print(f'Users: {len(usr2idx):,} | Businesses: {len(biz2idx):,}')

    print('\nBuilding user friendship graph...')
    G_u = build_user_graph(df_users, usr2idx)

    print('\nBuilding business geo graph...')
    G_b_geo = build_geo_graph(df_biz, biz2idx, k=K)

    print('\nBuilding business co-visitation graph (train only)...')
    G_b_covisit = build_covisit_graph(df_train, biz2idx, k=K)

    print('\nBuilding business shared-category graph...')
    G_b_cat = build_category_graph(df_biz, biz2idx, k=K)

    print('\nSaving...')
    sp.save_npz(os.path.join(output_dir, 'user_graph.npz'), G_u)
    sp.save_npz(os.path.join(output_dir, 'biz_graph_geo.npz'),     G_b_geo)
    sp.save_npz(os.path.join(output_dir, 'biz_graph_covisit.npz'), G_b_covisit)
    sp.save_npz(os.path.join(output_dir, 'biz_graph_cat.npz'),     G_b_cat)

    with open(os.path.join(output_dir, 'usr2idx.pkl'), 'wb') as f:
        pickle.dump(usr2idx, f)
    with open(os.path.join(output_dir, 'biz2idx.pkl'), 'wb') as f:
        pickle.dump(biz2idx, f)

    print('\nDone! Files saved:')
    for fname in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, fname)
        print(f'  {fname}: {os.path.getsize(fpath)/1024/1024:.1f} MB')

    return G_u, G_b_geo, G_b_covisit, G_b_cat, usr2idx, biz2idx
