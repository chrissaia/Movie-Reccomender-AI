# test_cosine_data_pipeline.py
import os

# Make sure Python can find your src package
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath("src"))

from src.data.load import load_data
from src.ranking.train import train_model
from src.ranking.tune import tune_model
from src.ranking.evaluate import evaluate_model

# === CONFIG ===
from src.utils.paths import X_TRAIN_PATH, X_TEST_PATH
from src.utils.paths import Y_TRAIN_PATH, Y_TEST_PATH
from src.utils.paths import QID_TRAIN_PATH, QID_TEST_PATH



def main():
    print("=== Testing Cosine Data Pipeline: Load → Tune → Train → Evaluate ===")

    # 1. Load Data
    print("\n[1] Loading data...")
    X_train = load_data(X_TRAIN_PATH)
    X_test = load_data(X_TEST_PATH)
    y_train = load_data(Y_TRAIN_PATH)
    y_test = load_data(Y_TEST_PATH)
    qid_train = load_data(QID_TRAIN_PATH)
    qid_test = load_data(QID_TEST_PATH)

    print(X_train.shape, y_train.shape, qid_train.shape)
    print(y_train[:20].T)
    print(qid_train[:20].T)
    print("unique labels:", np.unique(y_train))
    print(pd.Series(y_train.squeeze()).value_counts())
    qid = qid_train.squeeze()
    print(pd.Series(y_train.squeeze()).value_counts())
    print(f"(qid[1:]).all() - {(qid[1:]).all()}, {(qid[1:])}")

    print(f"Data loaded. "
         f"X_train shape: {X_train.shape}\n, X_test shape: {X_test.shape}, "
          f"y_train shape: {y_train.shape}\n, y_test shape: {y_test.shape}, "
          f"qid_train shape: {qid_train.shape}\n, qid_test shape: {qid_test.shape}, ")


    # 2. Tune
    print("\n[2] Tuning data...")
    best_params = tune_model(X_train, y_train, qid_train, n_trials=20)
    print(f"Best parameters achieved. Shape: {best_params}")

    # 3. Train
    print("\n[3] Training model...")
    model = train_model(X_train, y_train, qid_train, best_params)
    print(f"Model trained")

    # 4. Build Features
    print("\n[4] Evaluate model...")
    scores, metrics, importance = evaluate_model(model, X_test, y_test, qid_test, X_train)
    print(f"Model scores: {scores}")
    print(f"Model metrics: {metrics}")
    print(X_train.columns.tolist())
    print(f"Model importance: {importance}")


    print("\n✅ Phase 2 modeling pipeline completed successfully!")


if __name__ == "__main__":
    main()