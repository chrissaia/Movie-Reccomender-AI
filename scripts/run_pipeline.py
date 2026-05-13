#!/usr/bin/env python3
"""
scripts/run_pipeline.py

End-to-end movie recommender pipeline.
"""

from __future__ import annotations
import os
import sys

sys.path.append(os.path.abspath("src"))

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional
from collections import OrderedDict

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from src.data.load import load_data
from src.data.preprocess import preprocess_data
from src.data.split import group_train_test_split

from src.cosine.build_features import build_features as build_cosine_features
from src.cosine.similarity_cosine import load_or_compute_similarity as cosine_similarity_matrix

from src.ranking.similarity_rank import load_or_compute_similarity as ranking_similarity_matrix
from src.ranking.build_dataset import build_dataset
from src.ranking.tune import tune_model
from src.ranking.train import train_model
from src.ranking.evaluate import evaluate_model
from src.db.sqlite import get_connection
from src.cosine.neighbors import build_topk_neighbors
from src.db.schema import ALL_SCHEMA_STATEMENTS
from src.db.repository import init_schema, replace_movies, replace_neighbors


from src.utils.paths import (
    PROJECT_ROOT,
    RAW_MOVIES_ENRICHED_PATH,
    COSINE_FEATURES_PATH,
    COSINE_SIMILARITY_PATH,
    MOVIE_NAMES_PATH,
    RANKING_SIMILARITY_PATH,
    X_TRAIN_PATH,
    X_TEST_PATH,
    Y_TRAIN_PATH,
    Y_TEST_PATH,
    QID_TRAIN_PATH,
    QID_TEST_PATH,
    ARTIFACTS_DIR,
    SQLITE_DB_PATH,
)


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def flatten_1d(x: Any) -> np.ndarray:
    return np.asarray(x).reshape(-1)


def load_params_json(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"--params_json not found: {p}")
    with p.open("r") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with output_path.open("w") as f:
        json.dump(obj, f, indent=2)


def mlflow_log_model_safe(model: Any, name_or_path: str = "models") -> None:
    try:
        mlflow.sklearn.log_model(sk_model=model, name=name_or_path)
    except TypeError:
        mlflow.sklearn.log_model(model, artifact_path=name_or_path)


def run_cosine_pipeline(args: argparse.Namespace) -> None:
    log = logging.getLogger("pipeline.cosine")

    with mlflow.start_run(run_name="cosine_pipeline"):
        mlflow.log_param("pipeline", "cosine")
        mlflow.log_param("input", args.input)

        log.info("Loading raw data...")
        df = load_data(args.input)
        log.info("Loaded: %d rows x %d cols", df.shape[0], df.shape[1])

        log.info("Preprocessing...")
        df = preprocess_data(df)
        movies_df = df.copy()

        log.info("Building cosine features...")
        df_features, movie_names = build_cosine_features(df)

        ensure_dir(COSINE_FEATURES_PATH.parent)
        ensure_dir(MOVIE_NAMES_PATH.parent)

        df_features.to_csv(COSINE_FEATURES_PATH, index=False)
        pd.DataFrame({"movie_name": movie_names}).to_csv(MOVIE_NAMES_PATH, index=False)

        mlflow.log_artifact(str(COSINE_FEATURES_PATH), artifact_path="processed/cosine")
        mlflow.log_artifact(str(MOVIE_NAMES_PATH), artifact_path="processed/cosine")

        log.info("Computing/loading cosine similarity matrix...")
        sim_df = cosine_similarity_matrix(
            df_features=df_features,
            output_path=COSINE_SIMILARITY_PATH,
            force_recompute=args.force_recompute,
        )

        sim_df.to_csv(COSINE_SIMILARITY_PATH, index=True)
        mlflow.log_artifact(str(COSINE_SIMILARITY_PATH), artifact_path="processed/cosine")

        summary = {
            "pipeline": "cosine",
            "n_movies": len(movie_names),
            "n_features": int(df_features.shape[1]),
            "similarity_shape": list(sim_df.shape),
        }

        mlflow.log_param("n_movies", len(movie_names))
        mlflow.log_param("n_features", int(df_features.shape[1]))
        mlflow.log_text(json.dumps(summary, indent=2), artifact_file="cosine_summary.json")


        # --------------- KNN ---------------
        log.info("Building top-k neighbor table...")
        neighbors_df = build_topk_neighbors(similarity_matrix=sim_df, top_k=args.top_k)

        print("Rebuilding aligned movies metadata...")
        movies_df.columns = movies_df.columns.str.strip()

        numeric_votes = pd.to_numeric(movies_df["votes"], errors="coerce").fillna(0)
        movies_df = movies_df.loc[numeric_votes > 5000].copy()
        movies_df = movies_df.reset_index(drop=True)

        movies_df = movies_df.loc[
            :,
            [
                "name",
                "year",
                "genre",
                "director",
                "writer",
                "star",
                "country",
                "rating",
                "company",
                "score",
                "votes",
                "budget",
                "gross",
                "runtime",
                "tmdb_found",
                "tmdb_id",
                "tmdb_title",
                "tmdb_original_title",
                "tmdb_release_date",
                "tmdb_overview",
                "tmdb_genres",
                "tmdb_keywords",
                "tmdb_cast_top5",
                "tmdb_directors",
                "tmdb_writers",
                "tmdb_popularity",
                "tmdb_vote_average",
                "tmdb_vote_count",
                "tmdb_runtime",
                "tmdb_original_language",
                "tmdb_production_companies",
                "tmdb_production_countries",
                "tmdb_spoken_languages",
            ],
        ].copy()

        movies_df.insert(0, "movie_id", range(len(movies_df)))


        log.info("Writing to SQLite...")
        conn = get_connection(SQLITE_DB_PATH)
        try:
            init_schema(conn, ALL_SCHEMA_STATEMENTS)
            conn.execute("DELETE FROM movie_neighbors;")
            conn.execute("DELETE FROM movies;")

            replace_movies(conn, movies_df)
            replace_neighbors(conn, neighbors_df)
        finally:
            conn.close()

        print("\nCosine pipeline complete.")
        print(f"Movies: {len(movie_names)}")
        print(f"Features: {df_features.shape[1]}")
        print(f"Similarity matrix shape: {sim_df.shape}")



def run_ranking_pipeline(args: argparse.Namespace) -> None:
    log = logging.getLogger("pipeline.ranking")

    base_params: Dict[str, Any] = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "boosting_type": "gbdt",
        "random_state": args.seed,
        "n_jobs": -1,
        "verbosity": -1,
        "lambda_l1": 1e-3,
        "lambda_l2": 1e-3,
        "num_leaves": 64,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "learning_rate": 0.05,
        "n_estimators": 150,
    }
    base_params.update(load_params_json(args.params_json))

    with mlflow.start_run(run_name="ranking_pipeline"):
        mlflow.log_param("pipeline", "ranking")
        mlflow.log_param("input", args.input)
        mlflow.log_param("top_k", args.top_k)
        mlflow.log_param("min_similarity", args.min_similarity if args.min_similarity is not None else "None")
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("tune_enabled", int(args.tune))
        mlflow.log_param("tune_trials", args.tune_trials)
        mlflow.log_param("tune_cv_splits", args.tune_cv_splits)
        mlflow.log_params({f"lgbm__base__{k}": v for k, v in base_params.items()})

        log.info("Loading raw data...")
        df = load_data(args.input)
        log.info("Loaded: %d rows x %d cols", df.shape[0], df.shape[1])

        log.info("Preprocessing...")
        df = preprocess_data(df)


        log.info("Computing/loading ranking similarity matrix...")
        sim_df = ranking_similarity_matrix(
            df_features=df,
            output_path=RANKING_SIMILARITY_PATH,
            force_recompute=args.force_recompute,
        )

        ranking_paths = [X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH, QID_TRAIN_PATH, QID_TEST_PATH]
        log.info("Building ranking dataset...")
        if all(p.exists() for p in ranking_paths) and not args.force_recompute:
            log.info("Files already exist. Loading...")

            # Use list comprehension to call load_data on each path individually
            X_train, X_test, y_train, y_test, qid_train, qid_test = (load_data(p) for p in ranking_paths)

            y_train = y_train.to_numpy()
            y_test = y_test.to_numpy()
            qid_train = qid_train.to_numpy()
            qid_test = qid_test.to_numpy()

        else:
            log.info("Building ranking dataset...")
            X, y, qid = build_dataset(
                df=df,
                similarity_matrix=sim_df,
                top_k=args.top_k,
                min_similarity=args.min_similarity,
            )

            X = pd.DataFrame(X)
            y = flatten_1d(y)
            qid = flatten_1d(qid)

            if len(X) != len(y) or len(X) != len(qid):
                raise ValueError("X, y, and qid must have the same length")

            log.info("Splitting ranking dataset...")
            X_train, X_test, y_train, y_test, qid_train, qid_test = group_train_test_split(
                X, y, qid, test_size=args.test_size, random_state=args.seed
            )

        y_train = flatten_1d(y_train)
        y_test = flatten_1d(y_test)
        qid_train = flatten_1d(qid_train)
        qid_test = flatten_1d(qid_test)

        feature_columns = X_train.columns.tolist()
        ensure_dir(ARTIFACTS_DIR)
        params_path = ARTIFACTS_DIR / "feature_columns.json"
        with params_path.open("w") as f:
            json.dump(feature_columns, f, indent=2)
        mlflow.log_text(json.dumps(feature_columns, indent=2), artifact_file="feature_columns.json")


        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("n_features", int(X_train.shape[1]))
        mlflow.log_param("n_train_queries", int(len(np.unique(qid_train))))
        mlflow.log_param("n_test_queries", int(len(np.unique(qid_test))))

        final_params = dict(base_params)

        if args.tune:
            log.info("Running Optuna tuning...")
            tune_start = time.time()

            best_params = tune_model(
                X_train,
                y_train,
                qid_train,
                n_trials=args.tune_trials,
                n_splits=args.tune_cv_splits,
                seed=args.seed,
            )

            tune_time = time.time() - tune_start
            final_params.update(best_params)

            mlflow.log_metric("tune_time_seconds", tune_time)
            mlflow.log_params({f"lgbm__best__{k}": v for k, v in best_params.items()})

        log.info("Training ranking models...")
        train_start = time.time()
        model = train_model(X_train, y_train, qid_train, final_params)
        train_time = time.time() - train_start
        mlflow.log_metric("train_time_seconds", train_time)

        log.info("Evaluating ranking models...")
        eval_start = time.time()
        scores, metrics, importance = evaluate_model(model, X_test, y_test, qid_test, X_train)
        eval_time = time.time() - eval_start
        mlflow.log_metric("eval_time_seconds", eval_time)

        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))


        print("\nFeature importance:")
        print(importance)
        mlflow.log_dict(importance.to_dict(into=OrderedDict), artifact_file="feature_importance.json")


        ensure_dir(ARTIFACTS_DIR)
        params_path = ARTIFACTS_DIR / "ranking_best_params.json"
        save_json(final_params, params_path)
        mlflow.log_dict(final_params, artifact_file="ranking_best_params.json")



        mlflow_log_model_safe(model, "ranking_model")

        summary = {
            "pipeline": "ranking",
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "n_features": int(X_train.shape[1]),
            "n_train_queries": int(len(np.unique(qid_train))),
            "n_test_queries": int(len(np.unique(qid_test))),
            "metrics": metrics,
            "params": final_params,
        }

        mlflow.log_dict(summary, artifact_file="ranking_summary.json")

        print("\nRanking pipeline complete.")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")


def run(args: argparse.Namespace) -> None:
    tracking_uri = args.mlflow_uri or f"file://{PROJECT_ROOT / 'mlruns'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)

    if args.pipeline == "cosine":
        run_cosine_pipeline(args)
    elif args.pipeline == "ranking":
        run_ranking_pipeline(args)
    elif args.pipeline == "both":
        run_cosine_pipeline(args)
        run_ranking_pipeline(args)
    else:
        raise ValueError(f"Unsupported pipeline: {args.pipeline}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Movie Recommender pipeline")

    p.add_argument(
        "--pipeline",
        type=str,
        choices=["cosine", "ranking", "both"],
        required=True,
        help="Which pipeline to run",
    )
    p.add_argument(
        "--input",
        type=str,
        default=str(RAW_MOVIES_ENRICHED_PATH),
        help="Path to raw movie dataset",
    )
    p.add_argument(
        "--experiment",
        type=str,
        default="Movie Recommender",
        help="MLflow experiment name",
    )
    p.add_argument(
        "--mlflow_uri",
        type=str,
        default=None,
        help="MLflow tracking URI",
    )
    p.add_argument(
        "--params_json",
        type=str,
        default=None,
        help="Optional JSON parameter override file",
    )
    p.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test split size for ranking pipeline",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    p.add_argument(
        "--tune",
        action="store_true",
        help="Enable Optuna tuning for ranking pipeline",
    )
    p.add_argument(
        "--tune_trials",
        type=int,
        default=20,
        help="Number of Optuna trials",
    )
    p.add_argument(
        "--tune_cv_splits",
        type=int,
        default=3,
        help="Number of GroupKFold splits",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k similar candidates per query movie for ranking dataset",
    )
    p.add_argument(
        "--min_similarity",
        type=float,
        default=None,
        help="Optional minimum similarity threshold for ranking dataset",
    )
    p.add_argument(
        "--force_recompute",
        action="store_true",
        help="Recompute similarity matrices even if files already exist",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    run(args)