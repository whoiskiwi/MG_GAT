"""
Dataset statistics analysis script.
Compares this dataset with the paper's PA (Pennsylvania) dataset statistics.
Read-only: does not modify any existing files.
"""

import pandas as pd
import numpy as np
import scipy.sparse as sp
import pickle
import os

BASE = os.path.dirname(os.path.abspath(__file__))
PROCESSED = os.path.join(BASE, "data", "processed")
GRAPHS   = os.path.join(BASE, "data", "graphs")

# ── 1. Load rating splits ────────────────────────────────────────────────────
print("Loading rating splits...")
df_train = pd.read_csv(os.path.join(PROCESSED, "pa_reviews_train.csv"))
df_val   = pd.read_csv(os.path.join(PROCESSED, "pa_reviews_val.csv"))
df_test  = pd.read_csv(os.path.join(PROCESSED, "pa_reviews_test.csv"))

# ── 2. Load index maps ───────────────────────────────────────────────────────
with open(os.path.join(GRAPHS, "usr2idx.pkl"), "rb") as f:
    usr2idx = pickle.load(f)
with open(os.path.join(GRAPHS, "biz2idx.pkl"), "rb") as f:
    biz2idx = pickle.load(f)

n_users = len(usr2idx)
n_biz   = len(biz2idx)

# ── 3. Basic scale ───────────────────────────────────────────────────────────
total_ratings = len(df_train) + len(df_val) + len(df_test)

# ── 4. Sparsity ──────────────────────────────────────────────────────────────
sparsity = 1 - total_ratings / (n_users * n_biz)
avg_per_user = total_ratings / n_users
avg_per_biz  = total_ratings / n_biz

# ── 5. Graph degrees ─────────────────────────────────────────────────────────
G_u   = sp.load_npz(os.path.join(GRAPHS, "user_graph.npz"))
G_geo = sp.load_npz(os.path.join(GRAPHS, "biz_graph_geo.npz"))
G_cov = sp.load_npz(os.path.join(GRAPHS, "biz_graph_covisit.npz"))
G_cat = sp.load_npz(os.path.join(GRAPHS, "biz_graph_cat.npz"))

avg_degree_user = G_u.nnz / n_users
avg_degree_biz  = (G_geo.nnz + G_cov.nnz + G_cat.nnz) / 3 / n_biz

# ── 6. Rating distribution ───────────────────────────────────────────────────
df_all = pd.concat([df_train, df_val, df_test])
mean_stars = df_all["stars"].mean()
std_stars  = df_all["stars"].std()

# ── 7. Print comparison table ────────────────────────────────────────────────
COL_W = 28

def row(label, mine, paper, note=""):
    print(f"  {label:<{COL_W}} {str(mine):>18}   {str(paper):>18}   {note}")

print()
print("=" * 85)
print("  Dataset Statistics Comparison: This Work  vs  Paper (PA)")
print("=" * 85)
print(f"  {'Metric':<{COL_W}} {'This Dataset':>18}   {'Paper (PA)':>18}   Notes")
print("-" * 85)

# 1. Basic scale
row("# Users",          f"{n_users:,}",       "76,865")
row("# Businesses",     f"{n_biz:,}",         "10,966")
row("Total ratings",    f"{total_ratings:,}",  "260,350")
row("Train ratings",    f"{len(df_train):,}",  "~180,000")
row("Val ratings",      f"{len(df_val):,}",    "~40,000")
row("Test ratings",     f"{len(df_test):,}",   "~40,000")

print("-" * 85)

# 2. Sparsity
row("Matrix sparsity",      f"{sparsity:.6f}",     "—")
row("Avg ratings / user",   f"{avg_per_user:.3f}", "3.387")
row("Avg ratings / biz",    f"{avg_per_biz:.3f}",  "23.742")

print("-" * 85)

# 3. Network density
row("User graph avg degree", f"{avg_degree_user:.3f}", "5.557")
row("Biz graph avg degree",  f"{avg_degree_biz:.1f}",  "30.0")

print("-" * 85)

# 4. Rating distribution
row("Mean stars",  f"{mean_stars:.3f}", "3.728")
row("Std  stars",  f"{std_stars:.3f}",  "1.384")

print("=" * 85)
print()

# ── 8. Interpretation ────────────────────────────────────────────────────────
