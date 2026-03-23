"""Build all graphs from processed data."""

import os
from src.graph import build_all_graphs

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), 'data', 'processed')
OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), 'data', 'graphs')

build_all_graphs(PROCESSED_DIR, OUTPUT_DIR)
