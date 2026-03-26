# MG-GAT — Multi-Graph Graph Attention Network

PyTorch reproduction of the rating-prediction model from:

> **Interpretable Recommendations and User-Centric Explanations with Geometric Deep Learning**
> Leng, Liu, Ruiz — SSRN 3696092

Stage 1 only (rating prediction). The LLM explanation stage (Algorithm 2 / Section 4) and the fourth business graph (LLM perceptual map) are excluded by design.

**Best result**: Test RMSE **1.4086** on 2024 Yelp PA (320K users).
**Paper result**: Test RMSE **1.249** on 2019 Yelp PA (77K users).
Gap analysis in [`doc/experiments.md`](doc/experiments.md).

---

## Repository layout

```
MG-GAT/
├── run_preprocess.py       # Step 1 — filter, extract features, time-split
├── run_graph.py            # Step 2 — build 4 sparse adjacency graphs
├── run_implicit.py         # Step 3 — SVD implicit features (ki=32)
├── analyze_dataset.py      # Dataset statistics (read-only)
├── generate_tables.py      # Export feature stats to CSV
├── plot_results.py         # Experiment-progression figures
├── plot_report.py          # 7-dimension comparison report figures
│
├── src/
│   ├── data/
│   │   ├── filter.py       # Filter Yelp → PA subset
│   │   ├── features.py     # User (33-dim) & business (1396-dim) feature extraction
│   │   └── split.py        # Time-based train/val/test split
│   ├── graph/
│   │   ├── build.py        # Graph-build orchestrator
│   │   ├── user.py         # Friendship graph
│   │   ├── geo.py          # Geographic k-NN (haversine, k=10)
│   │   ├── covisit.py      # Co-visitation k-NN (k=10)
│   │   └── category.py     # Shared-category k-NN (k=10)
│   ├── features/
│   │   └── implicit.py     # TruncatedSVD on binarised rating matrix
│   └── model/
│       └── mggat.py        # 5-layer MG-GAT (PyTorch)
│
├── kaggle/
│   └── train_mggat.ipynb   # Cloud training notebook (GPU, checkpoint recovery)
│
├── data/
│   ├── raw/                # Yelp JSON files — download separately (8.7 GB)
│   ├── processed/          # Generated CSVs (1.5 GB)
│   ├── graphs/             # Sparse .npz matrices + .pkl ID maps (18 MB)
│   └── implicit/           # SVD feature arrays (43 MB)
│
├── figures/                # PNG charts from plot_*.py
├── reports/tables/         # Feature statistics CSVs
└── doc/
    ├── experiments.md      # Full experiment log (Exp 0–5, hyperopt, ablation)
    └── README.md           # Extended technical notes
```

---

## Requirements

```
Python 3.10+
torch
numpy  scipy  pandas  scikit-learn
matplotlib
```

---

## Pipeline

### 0. Download Yelp Open Dataset

Place the five JSON files in `data/raw/`:

```
yelp_academic_dataset_business.json
yelp_academic_dataset_review.json
yelp_academic_dataset_user.json
yelp_academic_dataset_checkin.json
yelp_academic_dataset_tip.json
```

### 1. Preprocess

```bash
python run_preprocess.py
```

Filters the Pennsylvania subset, extracts features, and produces time-based splits:

| Output | Rows | Description |
|--------|------|-------------|
| `pa_reviews_train.csv` | 820,496 | Reviews 2009–2016 |
| `pa_reviews_val.csv`   | 179,662 | Reviews 2017 |
| `pa_reviews_test.csv`  | 182,968 | Reviews 2018 |
| `pa_users_feat.csv`    | 320,212 | 33-dim explicit user features |
| `pa_biz_features.csv`  | 31,663  | 1396-dim explicit business features |

### 2. Build graphs

```bash
python run_graph.py
```

Builds four sparse adjacency matrices (k=10 for all k-NN graphs):

| Graph | Edges | Method |
|-------|-------|--------|
| User friendship | 1,744,826 | Yelp friends list |
| Business geo | 402,063 | Haversine distance k-NN |
| Business co-visitation | 492,463 | Shared-customer k-NN |
| Business category | 585,718 | Category-overlap k-NN |

### 3. Compute implicit features

```bash
python run_implicit.py
```

Runs TruncatedSVD (ki=32) on the binarised training matrix.
Outputs: `Su_imp.npy` (320,212 × 32) and `Sb_imp.npy` (31,663 × 32).

### 4. Train

Upload the processed data to Kaggle and run `kaggle/train_mggat.ipynb` on a GPU instance.

Files to upload: rating CSVs, feature CSVs, `usr2idx.pkl`, `biz2idx.pkl`, four `.npz` graphs, `Su_imp.npy`, `Sb_imp.npy`.

For checkpoint recovery on session expiry: save the notebook version, then add its output as a new input dataset named `mggat-ckpt`.

### 5. Analysis and figures

```bash
python analyze_dataset.py   # dataset statistics
python generate_tables.py   # Tables D4–D6 feature stats → reports/tables/
python plot_results.py      # experiment progression → figures/
python plot_report.py       # 7-figure comparison report → figures/
```

---

## Model architecture

The model follows Equations 2–7 of the paper:

| Layer | Equation | Operation |
|-------|----------|-----------|
| 1 | Eq. 2 | Linear projection: `su → d0`, `sb → d0` |
| 2 | Eq. 3 | Multi-graph attention (per-graph ω weights, softmax) |
| 3 | Eq. 4 | Weighted neighbourhood aggregation |
| 4 | Eq. 5 | Non-linear transform + residual (ELU) → `d1` |
| 5 | Eq. 6 | Final embeddings + graph Laplacian regularisation → `kf` |
| Pred | Eq. 7 | `sigmoid(U·B^T + biases) × (r_max − r_min) + r_min` |

Best hyperparameters (Hyperopt, 50 trials):

```
theta1=0.01  theta2=0.1  kf=128  d0=64  d1=64  lr=5e-3  actv1=elu
```

---

## Results

### Experiment progression

| Experiment | Val RMSE | Test RMSE |
|------------|----------|-----------|
| Exp 0 — Baseline | 1.4189 | 1.4593 |
| Exp 2 — Strong reg | 1.3656 | 1.4098 |
| Exp 3 — Small model | 1.4034 | 1.4449 |
| Exp 4 — Small + strong reg | 1.3735 | 1.4162 |
| **Final — Hyperopt** | **1.3891** | **1.4322** |
| Paper (2019 dataset) | — | 1.249 |

### Ablation study (Table 2)

| Configuration | Ours | Paper |
|---------------|------|-------|
| Full MG-GAT | 1.4322 | 1.249 |
| NIG removed | 1.4226 ↓ | 1.303 ↑ |
| FR removed | 1.4336 ↑ | 1.305 ↑ |
| Uniform graph weighting | 1.4292 ↓ | 1.280 ↑ |
| No auxiliary info | 1.4428 ↑ | 1.312 ↑ |
| Pure MF | 1.4648 ↑ | 1.405 ↑ |

↑ = worse than full, ↓ = better than full. NIG and uniform weighting reverse direction vs the paper — attributed to higher data sparsity weakening attention learning.

### Gap decomposition (1.4322 − 1.249 = 0.183)

| Factor | Estimated contribution |
|--------|----------------------|
| Dataset scale & sparsity (4.2× more users) | ~0.08 |
| Business graph density (15.6 vs 30.0 avg degree) | ~0.05 |
| Missing 4th graph (LLM perceptual map) | ~0.02 |
| Hyperparameter search coverage (~23%) | ~0.01 |
| Residual / random seed | ~0.02 |

---

## Figures

`python plot_report.py` generates seven figures in `figures/`:

| File | Content |
|------|---------|
| `figA_baseline_comparison.png` | Horizontal bar — all models sorted by RMSE |
| `figB_ablation_study.png` | RMSE + ΔRMSE + direction reversal table |
| `figC_dataset_statistics.png` | Scale and sparsity: 2024 vs 2019 dataset |
| `figD_gap_waterfall.png` | Waterfall decomposition of the 0.183 gap |
| `figE_training_curves.png` | Val RMSE vs epoch for 5 configurations |
| `figF_hyperopt_search.png` | 50-trial search scatter + RMSE distribution |
| `figG_user_filtering.png` | Filtering ablation comparison card |
