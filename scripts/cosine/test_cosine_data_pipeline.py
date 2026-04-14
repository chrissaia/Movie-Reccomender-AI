import os
import sys

sys.path.append(os.path.abspath("src"))

import pandas as pd

from src.data.load import load_data
from src.data.preprocess import preprocess_data
from src.cosine.build_features import build_features
from src.cosine.similarity_cosine import load_or_compute_similarity
from src.data.export import export_df

from src.utils.paths import (
    COSINE_SIMILARITY_PATH,
    MOVIE_NAMES_PATH,
    COSINE_FEATURES_PATH,
    RAW_MOVIES_ENRICHED_PATH,
)
OVERWRITE = True


def main():
    print("=== Testing Cosine Data Pipeline: Load → Preprocess → Build Features ===")

    print("\n[1] Loading data...")
    df = load_data(RAW_MOVIES_ENRICHED_PATH)
    print(f"Data loaded. Shape: {df.shape}")
    print(df.head(3))

    print("\n[2] Preprocessing data...")
    df_clean = preprocess_data(df)
    print(f"Data after preprocessing. Shape: {df_clean.shape}")
    print(df_clean.head(3))

    print("\n[3] Building features...")
    df_features, movie_names = build_features(df_clean)
    print(f"Data after feature engineering. Shape: {df_features.shape}")
    print(df_features.head(3))

    print("\n[4] Creating Similarity Matrix...")
    similarity_matrix = load_or_compute_similarity(
        df_features=df_features,
        output_path=COSINE_SIMILARITY_PATH,
        force_recompute=OVERWRITE,
    )
    print(f"Similarity matrix successfully loaded. Shape: {similarity_matrix.shape}")

    print("\n[5] Exporting features...")
    export_df(df_features, COSINE_FEATURES_PATH, overwrite=OVERWRITE, header=True)
    export_df(pd.DataFrame({"movie_name": movie_names}), MOVIE_NAMES_PATH, overwrite=OVERWRITE, header=True)
    export_df(similarity_matrix, COSINE_SIMILARITY_PATH, overwrite=OVERWRITE, header=True)

    print("\nCosine pipeline completed successfully!")


if __name__ == "__main__":
    main()