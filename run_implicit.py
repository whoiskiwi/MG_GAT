"""Build implicit features via SVD."""

import os
from src.features.implicit import build_implicit_features

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), 'data', 'processed')
GRAPHS_DIR    = os.path.join(os.path.dirname(__file__), 'data', 'graphs')
OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), 'data', 'implicit')

build_implicit_features(
    processed_dir=PROCESSED_DIR,
    graphs_dir=GRAPHS_DIR,
    output_dir=OUTPUT_DIR,
    ki=32,
)
