# test_cosine_data_pipeline.py
import os

# Make sure Python can find your src package
import sys

import pandas as pd

sys.path.append(os.path.abspath("src"))

from src.data.load import load_data
from src.data.preprocess import preprocess_data
from src.ranking.build_dataset import build_dataset
from src.ranking.similarity_rank import load_or_compute_similarity
from src.data.split import group_train_test_split
from src.data.export import export_df

# === CONFIG ===
from src.utils.paths import RAW_MOVIES_PATH
from src.utils.paths import RANKING_SIMILARITY_PATH

from src.utils.paths import X_TRAIN_PATH, X_TEST_PATH
from src.utils.paths import Y_TRAIN_PATH, Y_TEST_PATH
from src.utils.paths import QID_TRAIN_PATH, QID_TEST_PATH



def main():
    print("=== Testing Cosine Data Pipeline: Load → Preprocess → Build Dataset ===")

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

    # 3. Create Similarity Matrix
    print("\n[3] Creating Similarity Matrix...")
    similarity_matrix = load_or_compute_similarity(df_clean, RANKING_SIMILARITY_PATH)
    print(f"Similarity matrix successfully loaded. Shape: {similarity_matrix.shape}")
    print(similarity_matrix.head(3))

    # 4. Build Features
    print("\n[4] Building features...")
    X, y, qid = build_dataset(df_clean, similarity_matrix)
    print(f"Data after feature engineering. X shape: {X.shape}\n, Y shape: {y.shape}, QID shape: {qid.shape}")
    print(X.head(3))

    # 5. Split Features into train and test
    print("\n[5] Splitting features...")
    X_train, X_test, y_train, y_test, qid_train, qid_test = group_train_test_split(X, y, qid)
    print(f"Data after splitting. "
          f"X_train shape: {X_train.shape}\n, X_test shape: {X_test.shape}, "
          f"y_train shape: {y_train.shape}\n, y_test shape: {y_test.shape}, "
          f"qid_train shape: {qid_train.shape}\n, qid_test shape: {qid_test.shape}, ")

    # 6. Export Features
    print("\n[6] Exporting features.. ")
    export_df(similarity_matrix, RANKING_SIMILARITY_PATH)

    export_df(pd.DataFrame(X_train), X_TRAIN_PATH)
    export_df(pd.DataFrame(X_test), X_TEST_PATH)
    export_df(pd.DataFrame(y_train), Y_TRAIN_PATH)
    export_df(pd.DataFrame(y_test), Y_TEST_PATH)
    export_df(pd.DataFrame(qid_train), QID_TRAIN_PATH)
    export_df(pd.DataFrame(qid_test), QID_TEST_PATH)


    print("\n✅ Phase 1 ranking data pipeline completed successfully!")


if __name__ == "__main__":
    main()