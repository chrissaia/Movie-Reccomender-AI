"""
INFERENCE PIPELINE - Production Movie Recommendation Serving ensuring Consistency
=================================================================================

This module provides the core inference functionality for the Movie Recommender ranker.
It ensures that serving-time feature construction exactly matches training-time ranker
features, which is CRITICAL for ranking quality in production.

Key Responsibilities:
1. Load the trained ranker and exact feature schema from serving artifacts
2. Query movie metadata and candidate neighbors from SQLite
3. Build deterministic pairwise ranker features for each source/candidate pair
4. Aggregate pairwise signals across multiple liked movies
5. Score candidates with the trained LightGBM ranker
6. Blend cosine retrieval strength with ranker score
7. Convert outputs into user-friendly recommendation payloads

CRITICAL PATTERN: Training/Serving Consistency
- Uses the exact ranker feature column order from training artifacts
- Builds only the 26 ranker features the models expects
- Handles missing metadata safely and deterministically
- Keeps retrieval and inference logic separated but consistent

Production Deployment:
- MODEL_DIR points to serving-time models artifacts
- SQLite DB is read directly for real-time inference
- Optimized for small batch scoring of recommendation candidates
"""

from __future__ import annotations

import json
import logging
import pickle
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from zipp import Path

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
except Exception:  # pragma: no cover
    trace = None
    Status = None
    StatusCode = None

from src.utils.paths import SERVING_MODEL_ARTIFACTS_DIR, SQLITE_DB_PATH
from src.db.sqlite import get_connection
from src.ranking.build_dataset import compute_features
from src.serving.recommend import get_combined_candidates

# -----------------------------------------------------
# Configuration
# -----------------------------------------------------

logger = logging.getLogger(__name__)

DB_PATH = SQLITE_DB_PATH
MODEL_DIR = next(SERVING_MODEL_ARTIFACTS_DIR)
MODEL_PATH = MODEL_DIR / "model.pkl"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.json"

DEFAULT_TOP_K = 10
DEFAULT_CANDIDATE_POOL = 75
DEFAULT_MIN_SUPPORT = 1
RECOMMENDATION_SPLIT_WEIGHT = 0.90 # cosine_sim = RECOMMENDATION_SPLIT_WEIGHT || prediction = (RECOMMENDATION_SPLIT_WEIGHT - 1)

if trace is not None:
    tracer = trace.get_tracer(__name__)
else:  # pragma: no cover
    tracer = None


# -----------------------------------------------------
# Import-time artifact loading
# -----------------------------------------------------

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info("Ranker models loaded successfully from %s", MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load ranker models from {MODEL_PATH}: {e}") from e

try:
    with open(FEATURE_COLUMNS_PATH, "r") as f:
        FEATURE_COLS = json.load(f)
    logger.info("Loaded %s feature columns from %s", len(FEATURE_COLS), FEATURE_COLUMNS_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load feature columns from {FEATURE_COLUMNS_PATH}: {e}") from e

# Source of truth check: the models itself must agree with the feature file
try:
    model_feature_names = [str(x) for x in model.feature_names_in_]
except Exception as e:
    raise RuntimeError(f"Model does not expose feature_names_in_: {e}") from e

if FEATURE_COLS != model_feature_names:
    raise RuntimeError(
        "feature_columns.json does not match models.feature_names_in_. "
        f"feature_columns.json has {len(FEATURE_COLS)} features, "
        f"models expects {len(model_feature_names)} features."
    )


# -----------------------------------------------------
# DB Utilities
# -----------------------------------------------------

def _fetch_rows(query: str, params: tuple | list) -> list[sqlite3.Row]:
    conn = get_connection(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(query, params).fetchall()
    finally:
        conn.close()


def _validate_positive_int(value: int, field_name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _validate_movie_ids(movie_ids: list[int]) -> list[int]:
    if not movie_ids:
        raise ValueError("movie_ids must not be empty")

    cleaned: list[int] = []
    seen = set()

    for movie_id in movie_ids:
        if not isinstance(movie_id, int) or movie_id <= 0:
            raise ValueError("All movie_ids must be positive integers")
        if movie_id not in seen:
            cleaned.append(movie_id)
            seen.add(movie_id)

    return cleaned


def _min_max_normalize(values: list[float]) -> list[float]:
    if not values:
        return []

    arr = np.array(values, dtype=float)
    lo = float(arr.min())
    hi = float(arr.max())

    if hi == lo:
        return [1.0 for _ in values]

    return ((arr - lo) / (hi - lo)).tolist()


# -----------------------------------------------------
# DB Query Helpers
# -----------------------------------------------------

def _fetch_movies_by_ids(movie_ids: list[int]) -> list[dict]:
    cleaned_movie_ids = _validate_movie_ids(movie_ids)
    placeholders = ",".join(["?"] * len(cleaned_movie_ids))

    # HARDCODE IT
    query = f"""
    SELECT
        movie_id,
        name,
        year,
        score,
        votes,
        budget,
        gross,
        runtime,
        director,
        writer,
        star,
        country,
        company,
        genre,
        rating,
        tmdb_genres,
        tmdb_keywords,
        tmdb_cast_top5,
        tmdb_directors,
        tmdb_writers,
        tmdb_overview,
        tmdb_popularity,
        tmdb_vote_average,
        tmdb_vote_count,
        tmdb_runtime,
        tmdb_original_language,
        tmdb_production_companies,
        tmdb_production_countries,
        tmdb_spoken_languages
    FROM movies
    WHERE movie_id IN ({placeholders})
    """

    rows = _fetch_rows(query, cleaned_movie_ids)
    found = [dict(row) for row in rows]

    if len(found) != len(cleaned_movie_ids):
        found_ids = {row["movie_id"] for row in found}
        missing = [movie_id for movie_id in cleaned_movie_ids if movie_id not in found_ids]
        raise ValueError(f"Some movie_ids were not found in the database: {missing}")

    by_id = {row["movie_id"]: row for row in found}
    return [by_id[movie_id] for movie_id in cleaned_movie_ids]


def _fetch_single_candidate_rows(
    movie_id: int,
    candidate_pool: int = 50,
) -> list[dict]:
    _validate_positive_int(movie_id, "movie_id")
    _validate_positive_int(candidate_pool, "candidate_pool")

    query = """
    SELECT
        mn.neighbor_movie_id AS movie_id,
        m.name,
        m.year,
        m.score,
        m.votes,
        m.budget,
        m.gross,
        m.runtime,
        m.director,
        m.writer,
        m.star,
        m.country,
        m.company,
        m.genre,
        m.rating,
        m.tmdb_genres,
        m.tmdb_keywords,
        m.tmdb_cast_top5,
        m.tmdb_directors,
        m.tmdb_writers,
        m.tmdb_overview,
        m.tmdb_popularity,
        m.tmdb_vote_average,
        m.tmdb_vote_count,
        m.tmdb_runtime,
        m.tmdb_original_language,
        m.tmdb_production_companies,
        m.tmdb_production_countries,
        m.tmdb_spoken_languages,
        mn.cosine_score,
        mn.rank
    FROM movie_neighbors mn
    JOIN movies m
      ON mn.neighbor_movie_id = m.movie_id
    WHERE mn.source_movie_id = ?
    ORDER BY mn.rank ASC, mn.cosine_score DESC
    LIMIT ?
    """

    rows = _fetch_rows(query, (movie_id, candidate_pool))
    return [dict(row) for row in rows]



def _fetch_candidate_rows(
    movie_ids: list[int],
    candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    min_support: int = DEFAULT_MIN_SUPPORT,
) -> list[dict]:
    cleaned_movie_ids = _validate_movie_ids(movie_ids)
    _validate_positive_int(candidate_pool, "candidate_pool")
    _validate_positive_int(min_support, "min_support")

    placeholders = ",".join(["?"] * len(cleaned_movie_ids))

    query = f"""
    SELECT
        mn.source_movie_id,
        mn.neighbor_movie_id,
        mn.cosine_score,
        mn.rank,
        m.movie_id,
        m.name,
        m.year,
        m.score,
        m.votes,
        m.budget,
        m.gross,
        m.runtime,
        m.director,
        m.writer,
        m.star,
        m.country,
        m.company,
        m.genre,
        m.rating,
        m.tmdb_genres,
        m.tmdb_keywords,
        m.tmdb_cast_top5,
        m.tmdb_directors,
        m.tmdb_writers,
        m.tmdb_overview,
        m.tmdb_popularity,
        m.tmdb_vote_average,
        m.tmdb_vote_count,
        m.tmdb_runtime,
        m.tmdb_original_language,
        m.tmdb_production_companies,
        m.tmdb_production_countries,
        m.tmdb_spoken_languages
    FROM movie_neighbors mn
    JOIN movies m
      ON mn.neighbor_movie_id = m.movie_id
    WHERE mn.source_movie_id IN ({placeholders})
      AND mn.neighbor_movie_id NOT IN ({placeholders})
      AND mn.rank <= ?
    ORDER BY mn.source_movie_id ASC, mn.rank ASC
    """

    params = cleaned_movie_ids + cleaned_movie_ids + [candidate_pool]
    rows = _fetch_rows(query, params)

    aggregated: dict[int, dict] = defaultdict(
        lambda: {
            "movie": None,
            "scores": [],
            "rank_scores": [],
            "ranks": [],
            "source_movie_ids": set(),
        }
    )

    for row in rows:
        candidate_id = row["neighbor_movie_id"]
        source_movie_id = row["source_movie_id"]
        cosine_score = float(row["cosine_score"])
        rank = int(row["rank"])

        entry = aggregated[candidate_id]
        entry["movie"] = dict(row)
        entry["scores"].append(cosine_score)
        entry["rank_scores"].append(1.0 / rank)
        entry["ranks"].append(rank)
        entry["source_movie_ids"].add(source_movie_id)

    ranked_candidates: list[dict] = []
    total_sources = len(cleaned_movie_ids)

    for _, data in aggregated.items():
        support_count = len(data["source_movie_ids"])
        if support_count < min_support:
            continue

        avg_similarity = sum(data["scores"]) / len(data["scores"])
        avg_rank_score = sum(data["rank_scores"]) / len(data["rank_scores"])
        support_ratio = support_count / total_sources
        best_rank = min(data["ranks"])

        individual_overlap_penalty = max(0.0, (11 - best_rank) / 10) * 0.20

        combined_score = (
            (avg_similarity * 0.50)
            + (support_ratio * 0.35)
            + (avg_rank_score * 0.15)
            - individual_overlap_penalty
        )

        movie_row = data["movie"]
        movie_row["combined_score"] = combined_score
        movie_row["avg_similarity"] = avg_similarity
        movie_row["avg_rank_score"] = avg_rank_score
        movie_row["support_ratio"] = support_ratio
        movie_row["support_count"] = support_count
        movie_row["best_rank"] = best_rank
        movie_row["source_movie_ids"] = sorted(data["source_movie_ids"])

        ranked_candidates.append(movie_row)

    ranked_candidates.sort(key=lambda row: row["combined_score"], reverse=True)
    return ranked_candidates


# -----------------------------------------------------
# Feature Engineering - UNDER CONSTRUCTION
# -----------------------------------------------------

def _build_pair_features(source: dict, candidate: dict) -> dict[str, float]:
    source_series = pd.Series(source)
    candidate_series = pd.Series(candidate)

    features = compute_features(source_series, candidate_series)

    return {
        feature_name: float(features.get(feature_name, 0.0))
        for feature_name in FEATURE_COLS
    }


def _aggregate_pair_features(pair_features: list[dict[str, float]]) -> dict[str, float]:
    if not pair_features:
        return {feature_name: 0.0 for feature_name in FEATURE_COLS}

    aggregated: dict[str, float] = {}

    min_features = {
        "year_diff",
        "score_diff",
        "votes_rel_diff",
        "budget_rel_diff",
        "gross_rel_diff",
        "runtime_rel_diff",
        "tmdb_popularity_rel_diff",
        "tmdb_vote_average_diff",
        "tmdb_vote_count_rel_diff",
    }

    for feature_name in FEATURE_COLS:
        values = [float(features.get(feature_name, 0.0)) for features in pair_features]
        aggregated[feature_name] = min(values) if feature_name in min_features else max(values)

    return aggregated


def _build_ranker_frame(feature_rows: list[dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(feature_rows)
    return df.reindex(columns=FEATURE_COLS, fill_value=0.0)


# -----------------------------------------------------
# Explanation / Summary Helpers
# -----------------------------------------------------

def _build_explanations(agg_features: dict) -> list[str]:
    """
    Build compact user-facing explanation phrases from aggregated feature signals.
    """
    reasons: list[str] = []

    if agg_features.get("genre_jaccard", 0.0) >= 0.30 or agg_features.get("genre_overlap_count", 0.0) >= 2:
        reasons.append("strong genre overlap")

    if agg_features.get("keyword_jaccard", 0.0) >= 0.15 or agg_features.get("keyword_overlap_count", 0.0) >= 2:
        reasons.append("high keyword overlap")

    if agg_features.get("same_decade", 0.0) == 1.0:
        reasons.append("same decade")

    if agg_features.get("cast_overlap_count", 0.0) >= 1:
        reasons.append("shared cast signal")

    if agg_features.get("director_overlap", 0.0) >= 1:
        reasons.append("shared director signal")

    if agg_features.get("writer_overlap", 0.0) >= 1:
        reasons.append("shared writer signal")

    if agg_features.get("rating_match", 0.0) == 1.0:
        reasons.append("similar rating")

    if agg_features.get("company_overlap", 0.0) >= 1:
        reasons.append("shared studio/company signal")

    return reasons[:3]


def _build_taste_summary(selected_movies: list[dict]) -> dict:
    """
    Summarize the user's selected movies into simple, human-readable taste signals.
    """
    genres = Counter()
    ratings = Counter()
    decades = Counter()
    keywords = Counter()

    for movie in selected_movies:
        genre_value = str(movie.get("tmdb_genres") or movie.get("genre") or "")
        for genre in [g.strip().lower() for g in genre_value.split("|") if g.strip()]:
            genres[genre] += 1

        rating = str(movie.get("rating") or "").strip()
        if rating:
            ratings[rating] += 1

        year = movie.get("year")
        if year is not None and str(year).strip() != "":
            decade = (int(float(year)) // 10) * 10
            decades[f"{decade}s"] += 1

        keyword_value = str(movie.get("tmdb_keywords") or "")
        for keyword in [k.strip().lower() for k in keyword_value.split("|") if k.strip()]:
            keywords[keyword] += 1

    return {
        "top_genres": [item[0] for item in genres.most_common(3)],
        "preferred_ratings": [item[0] for item in ratings.most_common(3)],
        "favorite_decades": [item[0] for item in decades.most_common(3)],
        "top_keywords": [item[0] for item in keywords.most_common(5)],
    }


def _build_headline(taste_summary: dict) -> str:
    """
    Convert aggregated taste signals into a simple frontend-friendly headline.
    """
    top_genres = taste_summary.get("top_genres", [])
    if top_genres:
        genre = str(top_genres[0]).strip()
        if genre:
            return f"Because you like {genre.lower()}, here are some good {genre.lower()} movies."
    return "Here are some movies you might like."

# -----------------------------------------------------
# Public Prediction Function
# -----------------------------------------------------

# ==================== SINGLE PREDICTIONS ====================
def predict_single(
    movie_id: int,
    top_k: int = DEFAULT_TOP_K,
    candidate_pool: int = 50,
) -> dict:
    _validate_positive_int(movie_id, "movie_id")
    _validate_positive_int(top_k, "top_k")
    _validate_positive_int(candidate_pool, "candidate_pool")

    selected_movies = _fetch_movies_by_ids([movie_id])
    source_movie = selected_movies[0]

    candidates = _fetch_single_candidate_rows(
        movie_id=movie_id,
        candidate_pool=candidate_pool,
    )

    if not candidates:
        return {
            "movie_id": movie_id,
            "top_k": top_k,
            "recommendations": [],
        }

    candidate_feature_rows: list[dict] = []
    candidate_payload_rows: list[dict] = []

    for candidate in candidates:
        pair_features = [_build_pair_features(source=source_movie, candidate=candidate)]
        agg_features = _aggregate_pair_features(pair_features)

        candidate_feature_rows.append(agg_features)
        candidate_payload_rows.append(
            {
                "movie_id": int(candidate["movie_id"]),
                "title": candidate["name"],
                "cosine_score": float(candidate["cosine_score"]),
                "rank": int(candidate["rank"]),
                "agg_features": agg_features,
            }
        )

    X = _build_ranker_frame(candidate_feature_rows)
    ranker_scores = model.predict(X)

    cosine_norm = _min_max_normalize([row["cosine_score"] for row in candidate_payload_rows])
    ranker_norm = _min_max_normalize([float(score) for score in ranker_scores])

    recommendations: list[dict] = []
    for row, cosine_score_norm, ranker_score_raw, ranker_score_norm in zip(
            candidate_payload_rows,
            cosine_norm,
            ranker_scores,
            ranker_norm,
    ):
        final_score = (RECOMMENDATION_SPLIT_WEIGHT * float(cosine_score_norm)) + ((RECOMMENDATION_SPLIT_WEIGHT - 1) * float(ranker_score_norm))

        recommendations.append(
            {
                "movie_id": row["movie_id"],
                "title": row["title"],
                "cosine_score": round(float(row["cosine_score"]), 6),
                "ranker_score": round(float(ranker_score_raw), 6),
                "ranker_score_norm": round(float(ranker_score_norm), 6),
                "final_score": round(float(final_score), 6),
                "rank": row["rank"],
                "score": round(float(final_score), 6),
                "explanations": _build_explanations(row["agg_features"]),
            }
        )

    final_recommendations = sorted(
        recommendations,
        key=lambda x: x["final_score"],
        reverse=True,
    )[:top_k]

    return {
        "movie_id": movie_id,
        "top_k": top_k,
        "recommendations": final_recommendations,
    }







# ==================== COMBINED PREDICTIONS ====================
def predict_combined(
    movie_ids: list[int],
    top_k: int = DEFAULT_TOP_K,
    candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    min_support: int = DEFAULT_MIN_SUPPORT,
) -> dict:
    """
    Main inference function for movie recommendation serving.

    Pipeline:
    1. Validate selected movie IDs
    2. Fetch selected movie metadata
    3. Retrieve combined cosine candidates from the neighbors table
    4. Build pairwise ranker features for each source/candidate pair
    5. Aggregate to one candidate-level feature vector
    6. Score candidates with the trained LightGBM ranker
    7. Blend cosine retrieval strength with ranker signal
    8. Return headline, taste summary, and ranked recommendations

    Args:
        movie_ids: List of selected/liked movie IDs
        top_k: Number of final recommendations to return
        candidate_pool: Max depth per source movie from precomputed neighbors
        min_support: Minimum number of source movies that must support a candidate

    Returns:
        {
            "movie_ids": [...],
            "top_k": 10,
            "headline": "...",
            "taste_summary": {...},
            "recommendations": [
                {
                    "movie_id": 123,
                    "title": "Aliens",
                    "cosine_score": 0.61,
                    "ranker_score": 1.92,
                    "final_score": 0.74,
                    "support_count": 2,
                    "explanations": [...]
                }
            ]
        }
    """
    cleaned_movie_ids = _validate_movie_ids(movie_ids)
    _validate_positive_int(top_k, "top_k")
    _validate_positive_int(candidate_pool, "candidate_pool")
    _validate_positive_int(min_support, "min_support")

    if tracer is not None:
        with tracer.start_as_current_span("movie_recommendation_inference") as root_span:
            try:
                root_span.set_attribute("request.movie_count", len(cleaned_movie_ids))
                root_span.set_attribute("request.top_k", top_k)
                root_span.set_attribute("request.candidate_pool", candidate_pool)
                root_span.set_attribute("request.min_support", min_support)

                result = _predict_internal(
                    movie_ids=cleaned_movie_ids,
                    top_k=top_k,
                    candidate_pool=candidate_pool,
                    min_support=min_support,
                )

                if Status is not None and StatusCode is not None:
                    root_span.set_status(Status(StatusCode.OK))
                return result
            except Exception as e:
                if hasattr(root_span, "record_exception"):
                    root_span.record_exception(e)
                if Status is not None and StatusCode is not None:
                    root_span.set_status(Status(StatusCode.ERROR))
                raise
    else:
        return _predict_internal(
            movie_ids=cleaned_movie_ids,
            top_k=top_k,
            candidate_pool=candidate_pool,
            min_support=min_support,
        )


def _predict_internal(
    movie_ids: list[int],
    top_k: int,
    candidate_pool: int,
    min_support: int,
) -> dict:
    # -----------------------------------------------------
    # STEP 1: Fetch selected movies
    # -----------------------------------------------------
    selected_movies = _fetch_movies_by_ids(movie_ids)

    # -----------------------------------------------------
    # STEP 2: Retrieve candidate pool from cosine neighbors
    # -----------------------------------------------------
    candidates = get_combined_candidates(
        movie_ids=movie_ids,
        candidate_pool=candidate_pool,
        min_support=min_support,
    )

    if not candidates:
        taste_summary = _build_taste_summary(selected_movies)
        return {
            "movie_ids": movie_ids,
            "top_k": top_k,
            "headline": _build_headline(taste_summary),
            "taste_summary": taste_summary,
            "recommendations": [],
        }

    # -----------------------------------------------------
    # STEP 3: Build candidate-level ranker features
    # -----------------------------------------------------
    candidate_feature_rows: list[dict] = []
    candidate_payload_rows: list[dict] = []

    for candidate in candidates:
        candidate_movie = candidate.movie

        pair_features = [
            _build_pair_features(source=source_movie, candidate=candidate_movie)
            for source_movie in selected_movies
        ]

        agg_features = _aggregate_pair_features(pair_features)
        candidate_feature_rows.append(agg_features)

        candidate_payload_rows.append(
            {
                "movie_id": int(candidate.movie_id),
                "title": candidate.title,
                "combined_score": float(candidate.combined_score),
                "support_count": int(candidate.support_count),
                "source_movie_ids": list(candidate.source_movie_ids),
                "agg_features": agg_features,
            }
        )

    # -----------------------------------------------------
    # STEP 4: Construct models-ready frame and score with ranker
    # -----------------------------------------------------
    X = _build_ranker_frame(candidate_feature_rows)
    ranker_scores = model.predict(X)

    # -----------------------------------------------------
    # STEP 5: Blend cosine retrieval and ranker score
    # -----------------------------------------------------
    ranker_norm = _min_max_normalize([float(score) for score in ranker_scores])

    final_rows: list[dict] = []
    for row, ranker_score_raw, ranker_score_norm in zip(
        candidate_payload_rows,
        ranker_scores,
        ranker_norm,
    ):
        final_score = (0.30 * row["combined_score"]) + (0.70 * ranker_score_norm)

        final_rows.append(
            {
                "movie_id": row["movie_id"],
                "title": row["title"],
                "combined_score": round(float(row["combined_score"]), 6),
                "ranker_score": round(float(ranker_score_raw), 6),
                "final_score": round(float(final_score), 6),
                "score": round(float(final_score), 6),
                "support_count": row["support_count"],
                "explanations": _build_explanations(row["agg_features"]),
            }
        )

    final_rows.sort(key=lambda item: item["final_score"], reverse=True)
    final_rows = final_rows[:top_k]

    # -----------------------------------------------------
    # STEP 6: Build frontend-friendly taste summary
    # -----------------------------------------------------
    taste_summary = _build_taste_summary(selected_movies)
    headline = _build_headline(taste_summary)

    return {
        "movie_ids": movie_ids,
        "top_k": top_k,
        "headline": headline,
        "taste_summary": taste_summary,
        "recommendations": final_rows,
    }