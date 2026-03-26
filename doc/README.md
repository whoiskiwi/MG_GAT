# MG-GAT: Multi-Graph Graph Attention Network

Reproduction of the MG-GAT model (Stage 1) from:

> Leng, Liu, Ruiz. *Interpretable Recommendations and User-Centric Explanations with Geometric Deep Learning*. SSRN 3696092.

Task: rating prediction on the Yelp Pennsylvania dataset using a five-layer graph attention architecture with multiple business similarity graphs.

## Project Structure

```
MG-GAT/
├── run_preprocess.py        # Step 1: filter Yelp data, extract features, time-split
├── run_graph.py             # Step 2: build user & business graphs
├── run_implicit.py          # Step 3: compute implicit features via SVD
├── plot_results.py          # Plot experiment results
├── experiments.md           # Experiment log & reproduction report
├── src/
│   ├── data/
│   │   ├── filter.py        # Filter PA businesses & reviews
│   │   ├── features.py      # Extract user & business features
│   │   └── split.py         # Time-based train/val/test split
│   ├── graph/
│   │   ├── build.py         # Graph construction orchestrator
│   │   ├── user.py          # User friendship graph
│   │   ├── geo.py           # Business geographic k-NN graph
│   │   ├── covisit.py       # Business co-visitation graph
│   │   └── category.py      # Business shared-category graph
│   ├── features/
│   │   └── implicit.py      # SVD implicit features
│   └── model/
│       └── mggat.py         # MG-GAT model (PyTorch)
└── data/
    ├── raw/                 # Yelp Open Dataset JSON files (not tracked)
    ├── processed/           # Filtered CSVs (PA subset)
    ├── graphs/              # Sparse adjacency matrices (.npz) & ID mappings (.pkl)
    └── implicit/            # SVD implicit feature matrices (.npy)
```

## Requirements

- Python 3.10+
- PyTorch
- NumPy, SciPy, Pandas, scikit-learn

## Usage

### 1. Data Preprocessing

Place the Yelp Open Dataset JSON files in `data/raw/`, then run:

```bash
python run_preprocess.py
```

Filters PA businesses and reviews (2009-2018), extracts user/business features, and performs a time-based split:

| Split | Period | Reviews |
|-------|--------|---------|
| Train | 2009-2016 | 820,496 |
| Val   | 2017 | 179,662 |
| Test  | 2018 | 182,968 |

### 2. Build Graphs

```bash
python run_graph.py
```

Constructs 4 graphs (k=10 for business graphs):

| Graph | Description |
|-------|-------------|
| `user_graph.npz` | User friendship graph |
| `biz_graph_geo.npz` | Geographic proximity (Haversine k-NN) |
| `biz_graph_covisit.npz` | Shared customers |
| `biz_graph_cat.npz` | Shared categories |

### 3. Compute Implicit Features

```bash
python run_implicit.py
```

Performs SVD on the binarized training rating matrix to produce 32-dim implicit features for users and businesses.

### 4. Train Model

See `experiments.md` for hyperparameters and training details.

## Data Files

Binary files require Python to inspect:

```python
import numpy as np, pickle

# .npz (sparse matrix) - adjacency matrices
data = np.load('data/graphs/biz_graph_geo.npz', allow_pickle=True)
print(data.keys(), data['shape'])

# .pkl (pickle) - ID mapping dicts
with open('data/graphs/biz2idx.pkl', 'rb') as f:
    biz2idx = pickle.load(f)

# .npy (numpy array) - implicit features
Su = np.load('data/implicit/Su_imp.npy')
print(Su.shape)  # (320212, 32)
```

## Data Source

This project uses the [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/).

Due to file size limitations, the following data directories are **not included** in this repository and need to be generated locally:

| Directory | Size | Description | How to generate |
|-----------|------|-------------|-----------------|
| `data/raw/` | ~8.7 GB | Yelp Open Dataset JSON files | Download from [Yelp](https://business.yelp.com/data/resources/open-dataset/) |
| `data/processed/` | ~1.5 GB | Filtered PA subset CSVs | `python run_preprocess.py` |

The following data directories **are included** in this repository:

| Directory | Size | Description |
|-----------|------|-------------|
| `data/graphs/` | ~18 MB | Sparse adjacency matrices & ID mappings |
| `data/implicit/` | ~43 MB | SVD implicit feature matrices |

### Reproducing the data from scratch

```bash
# 1. Download Yelp Open Dataset and place JSON files in data/raw/
#    Required files: yelp_academic_dataset_business.json,
#                    yelp_academic_dataset_review.json,
#                    yelp_academic_dataset_user.json

# 2. Preprocess: filter PA data, extract features, time-split → data/processed/
python run_preprocess.py

# 3. Build graphs → data/graphs/
python run_graph.py

# 4. Compute implicit features → data/implicit/
python run_implicit.py
```

## Results

| Model | Test RMSE |
|-------|-----------|
| MG-GAT (this work) | 1.4086 |
| MG-GAT (paper) | 1.249 |

The gap is primarily due to dataset version differences (320K vs 77K users, much higher sparsity). See `experiments.md` Section 7 for detailed gap analysis.
