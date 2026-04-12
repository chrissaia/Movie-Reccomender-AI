# test_cosine_data_pipeline.py
import os

# Make sure Python can find your src package
import sys
sys.path.append(os.path.abspath("src"))

import pandas as pd

from src.data.load import load_data
from src.data.preprocess import preprocess_data
from src.cosine.build_features import build_features
from src.cosine.similarity_cosine import load_or_compute_similarity
from src.data.export import export_df

# === CONFIG ===
from src.utils.paths import RAW_MOVIES_PATH
from src.utils.paths import COSINE_SIMILARITY_PATH, MOVIE_NAMES_PATH, COSINE_FEATURES_PATH

def main():
    print("=== Testing Cosine Data Pipeline: Load → Preprocess → Build Features ===")

    # 1. Load Data
    print("\n[1] Loading data...")
    df = load_data(RAW_MOVIES_PATH)
    print(f"Data loaded. Shape: {df.shape}")
    print(df.head(3))

    # 2. Preprocess
    print("\n[2] Preprocessing data...")
    df_clean = preprocess_data(df)
    print(f"Data after preprocessing. Shape: {df_clean.shape}")
    print(df_clean.head(3))

    # 3. Build Features
    print("\n[3] Building features...")
    df_features, movie_names = build_features(df_clean)
    print(f"Data after feature engineering. Shape: {df_features.shape}")
    print(df_features.head(3))

    # 4. Create Similarity Matrix
    print("\n[4] Creating Similarity Matrix...")
    similarity_matrix = load_or_compute_similarity(df_features, COSINE_SIMILARITY_PATH)
    print(f"Similarity matrix successfully loaded. Shape: {similarity_matrix.shape}")

    # 5. Export data
    print("\n[6] Exporting features.. ")
    export_df(pd.DataFrame(df_features), COSINE_FEATURES_PATH)
    export_df(pd.DataFrame(movie_names), MOVIE_NAMES_PATH)
    export_df(pd.DataFrame(similarity_matrix), COSINE_SIMILARITY_PATH)

    print("\nCosine pipeline completed successfully!")

if __name__ == "__main__":
    main()