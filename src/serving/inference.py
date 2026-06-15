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
MAX_TASTE_ROWS = 4 # the amount of taste profile rows show
ORGANIZED_ROW_TOP_K = 10 # changes the amount of movies show in the row of the TASTE summary
SOURCE_MOVIE_ROW_TOP_K = 7 # changes the amount of movies show in the row of their selected movies
MAX_DYNAMIC_ORGANIZED_ROWS = 25 # changes the amount of rows

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


def _clean_optional_movie_ids(movie_ids: list[int] | set[int] | None) -> set[int]:
    if not movie_ids:
        return set()

    cleaned = set()
    for movie_id in movie_ids:
        if isinstance(movie_id, int) and movie_id > 0:
            cleaned.add(movie_id)

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

    if not reasons:
        reasons.append("similar story patterns and audience profile")

    return reasons[:3]



def _split_pipe(value) -> list[str]:
    """
    Split a pipe-delimited metadata field into clean lowercase tokens.
    Example:
        "Drama|Thriller|Crime" -> ["drama", "thriller", "crime"]
    """
    if pd.isna(value) or not str(value).strip():
        return []
    return [part.strip().lower() for part in str(value).split("|") if part.strip()]


def _pretty_label(value: str) -> str:
    """
    Convert a raw metadata token into something nicer for frontend text.
    Example:
        "psychological-thriller" -> "Psychological Thriller"
    """
    return value.replace("-", " ").strip().title()

# -----------------------------------------------------
# Taste Profile Recommendations
# -----------------------------------------------------

def _build_taste_summary(selected_movies: list[dict]) -> dict:
    """
    Build a human-readable summary of what the user's selected movies have in common.

    Input:
        selected_movies = a list of movie dicts from your DB / fetch layer

    Output:
        {
            "top_genres": [...],
            "top_keywords": [...],
            "repeated_directors": [...],
            "repeated_cast": [...],
            "year_range": {"min": ..., "max": ...}
        }

    Important:
    This is explanation logic, not ranking logic.
    It helps describe the user's taste in plain English.
    """
    # Counter objects let us count how often genres/keywords/etc appear across the selected movies.
    genre_counter = Counter()
    keyword_counter = Counter()
    director_counter = Counter()
    cast_counter = Counter()

    # Keep track of release years so we can describe the era/range.
    years = []

    # Loop through each movie the user selected and collect metadata signals.
    # Count all ___ across the selected movies.
    for movie in selected_movies:
        genre_counter.update(_split_pipe(movie.get("tmdb_genres", "")))
        keyword_counter.update(_split_pipe(movie.get("tmdb_keywords", "")))
        director_counter.update(_split_pipe(movie.get("tmdb_directors", "")))
        cast_counter.update(_split_pipe(movie.get("tmdb_cast_top5", "")))

        # Safely parse the year and save it if valid.
        year = pd.to_numeric(movie.get("year"), errors="coerce")
        if pd.notna(year):
            years.append(int(year))

    # Find the most common ___ overall
    top_genres = [g for g, _ in genre_counter.most_common(3)]
    repeated_directors = [d for d, c in director_counter.items() if c >= 2]
    repeated_cast = [a for a, c in cast_counter.items() if c >= 2]

    boring_keywords = {
        "based on novel or book",
        "murder",
        "friendship",
        "dramatic",
        "suspenseful",
        "depressing",
        "clinical",
        "cruel",
        "appreciative",
        "exhilarated",
        "voiceover",
        "1980s",
        "1940s",
        "1950s",
        "woman director",
        "independent film",
        "duringcreditsstinger",
        "aftercreditsstinger",
        "sequel",
        "violence",
        "death",
        "love",
    }

    top_keywords = [
        k for k, c in keyword_counter.most_common(15)
        if k not in boring_keywords
    ][:5]

    # Build a basic min/max year range if we have usable years.
    year_range = None
    if years:
        year_range = {"min": min(years), "max": max(years)}

    # Return the structured summary.
    # This is easy to inspect in logs and easy to use in headline generation.
    return {
        "top_genres": top_genres,
        "top_keywords": top_keywords,
        "repeated_directors": repeated_directors,
        "repeated_cast": repeated_cast,
        "year_range": year_range,

        # Used by row-ranking layer
        "genre_counts": dict(genre_counter),
        "keyword_counts": dict(keyword_counter),
        "director_counts": dict(director_counter),
        "cast_counts": dict(cast_counter),
        "selected_count": len(selected_movies),
    }

def _build_headline(taste_summary: dict) -> str:
    """
    Turn the structured taste summary into one simple frontend headline.

    Priority:
    1. Repeated director is strongest, most specific signal
    2. Genre + keyword combo is next best
    3. Genre-only fallback
    4. Generic fallback if nothing useful exists
    """
    genres = taste_summary.get("top_genres", [])
    keywords = taste_summary.get("top_keywords", [])
    directors = taste_summary.get("repeated_directors", [])
    casts = taste_summary.get("repeated_cast", [])

    if casts:
        actor = _pretty_label(casts[0])
        return f"Because you like films with {actor}, here are some strong matches"

    # If the same director shows up multiple times in the selected movies,
    # that is usually a strong and very understandable pattern.
    if directors:
        director = _pretty_label(directors[0])
        return f"Because you like films shaped by {director}, here are some strong matches"

    # Best non-director case:
    # use the top 2 genres plus the strongest repeated keyword/theme.
    if len(genres) >= 2 and keywords:
        return (
            f"Because you like {_pretty_label(genres[0])} and "
            f"{_pretty_label(genres[1])} stories with "
            f"{_pretty_label(keywords[0])}, here are some strong matches"
        )

    # If no repeated keyword exists, still use the strongest 2 genres.
    if len(genres) >= 2:
        return (
            f"Because you like {_pretty_label(genres[0])} and "
            f"{_pretty_label(genres[1])} films, here are some strong matches"
        )

    # If only one genre is reliable, use that.
    if genres:
        return f"Because you like {_pretty_label(genres[0])} films, here are some strong matches"

    # Final fallback in case metadata is weak or missing.
    return "Here are some movies you might like"


def _build_signal_title(signal_kind: str, signal_value: str, taste_summary: dict) -> str:
    label = _pretty_label(signal_value)

    if signal_kind == "director":
        return f"Because You Like Films Shaped by {label}"

    if signal_kind == "cast":
        return f"Because You Like Movies with {label}"

    if signal_kind == "keyword":
        top_genres = taste_summary.get("top_genres", [])
        if top_genres:
            return f"Because You Like {label} {_pretty_label(top_genres[0])}"
        return f"Because You Like {label}"

    if signal_kind == "genre":
        return f"Because You Like {_pretty_label(signal_value)}"

    return f"Because You Like {label}"


def _build_taste_row_signals(taste_summary: dict) -> list[dict]:
    """
    Build possible taste-profile rows and score their importance.

    These are row candidates, not movie candidates.
    The final page-ranking layer decides where these rows appear.
    """
    signals: list[dict] = []

    selected_count = max(int(taste_summary.get("selected_count") or 1), 1)

    genre_counts = taste_summary.get("genre_counts", {})
    keyword_counts = taste_summary.get("keyword_counts", {})
    director_counts = taste_summary.get("director_counts", {})
    cast_counts = taste_summary.get("cast_counts", {})

    def support_ratio(count: int | float) -> float:
        return min(float(count) / selected_count, 1.0)

    def add_signal(
        kind: str,
        value: str,
        count: int | float,
        specificity: float,
        base_strength: float,
    ) -> None:
        if not value:
            return

        support = support_ratio(count)
        signal_strength = min((base_strength * 0.65) + (support * 0.35), 1.0)

        signals.append(
            {
                "kind": kind,
                "value": value,
                "title": _build_signal_title(kind, value, taste_summary),
                "signal_strength": round(signal_strength, 6),
                "support_ratio": round(support, 6),
                "specificity": round(specificity, 6),
            }
        )

    for director in taste_summary.get("repeated_directors", [])[:3]:
        add_signal(
            kind="director",
            value=director,
            count=director_counts.get(director, 1),
            specificity=0.95,
            base_strength=0.95,
        )

    for actor in taste_summary.get("repeated_cast", [])[:3]:
        add_signal(
            kind="cast",
            value=actor,
            count=cast_counts.get(actor, 1),
            specificity=0.90,
            base_strength=0.88,
        )

    for keyword in taste_summary.get("top_keywords", [])[:5]:
        add_signal(
            kind="keyword",
            value=keyword,
            count=keyword_counts.get(keyword, 1),
            specificity=0.82,
            base_strength=0.82,
        )

    for genre in taste_summary.get("top_genres", [])[:4]:
        add_signal(
            kind="genre",
            value=genre,
            count=genre_counts.get(genre, 1),
            specificity=0.55,
            base_strength=0.70,
        )

    signals.sort(
        key=lambda signal: (
            signal["signal_strength"],
            signal["specificity"],
            signal["support_ratio"],
        ),
        reverse=True,
    )

    return signals



def get_constrained_row(
    selected_movies: list[dict],
    movie_ids: list[int],
    signal_kind: str,
    signal_value: str,
    title: str,
    top_k: int = 10,
    candidate_pool: int = 100,
    min_support: int = 1,
    exclude_movie_ids: set[int] | None = None,
) -> dict | None:
    if not signal_value.strip():
        return None

    signal_value = signal_value.strip().lower()

    candidates = get_combined_candidates(
        movie_ids=movie_ids,
        candidate_pool=candidate_pool,
        min_support=min_support,
    )

    if not candidates:
        return None

    filtered_candidates = []
    selected_id_set = set(movie_ids)
    exclude_id_set = _clean_optional_movie_ids(exclude_movie_ids)

    for candidate in candidates:
        candidate_id = int(candidate.movie_id)
        if candidate_id in selected_id_set or candidate_id in exclude_id_set:
            continue

        movie = candidate.movie

        match = False
        if signal_kind == "genre":
            match = signal_value in str(movie.get("tmdb_genres", "")).lower()
        elif signal_kind == "keyword":
            match = signal_value in str(movie.get("tmdb_keywords", "")).lower()
        elif signal_kind == "director":
            match = signal_value in str(movie.get("tmdb_directors", "")).lower()
        elif signal_kind == "cast":
            match = signal_value in str(movie.get("tmdb_cast_top5", "")).lower()

        if match:
            filtered_candidates.append(candidate)

    if not filtered_candidates:
        return None

    candidate_feature_rows: list[dict] = []
    candidate_payload_rows: list[dict] = []

    for candidate in filtered_candidates:
        candidate_movie = candidate.movie

        pair_features = [
            _build_pair_features(source=source_movie, candidate=candidate_movie)
            for source_movie in selected_movies
        ]

        agg_features = _aggregate_pair_features(pair_features)
        candidate_feature_rows.append(agg_features)

        vote_average = float(candidate_movie.get("tmdb_vote_average") or 0.0)
        vote_count = float(candidate_movie.get("tmdb_vote_count") or 0.0)

        candidate_payload_rows.append(
            {
                "movie_id": int(candidate.movie_id),
                "title": candidate.title,
                "combined_score": float(candidate.combined_score),
                "support_count": int(candidate.support_count),
                "vote_average": vote_average,
                "vote_count": vote_count,
                "agg_features": agg_features,
            }
        )

    if not candidate_payload_rows:
        return None

    X = _build_ranker_frame(candidate_feature_rows)
    ranker_scores = model.predict(X)

    ranker_norm = _min_max_normalize([float(score) for score in ranker_scores])
    vote_average_norm = _min_max_normalize([row["vote_average"] for row in candidate_payload_rows])
    vote_count_norm = _min_max_normalize(
        [float(np.log1p(row["vote_count"])) for row in candidate_payload_rows]
    )

    quality_norm = [
        (0.70 * va) + (0.30 * vc)
        for va, vc in zip(vote_average_norm, vote_count_norm)
    ]

    final_rows: list[dict] = []
    for row, ranker_score_raw, ranker_score_norm, quality_score_norm in zip(
        candidate_payload_rows,
        ranker_scores,
        ranker_norm,
        quality_norm,
    ):
        final_score = (
            0.55 * float(ranker_score_norm)
            + 0.25 * float(row["combined_score"])
            + 0.20 * float(quality_score_norm)
        )

        final_rows.append(
            {
                "movie_id": row["movie_id"],
                "title": row["title"],
                "combined_score": round(float(row["combined_score"]), 6),
                "ranker_score": round(float(ranker_score_raw), 6),
                "quality_score": round(float(quality_score_norm), 6),
                "final_score": round(float(final_score), 6),
                "score": round(float(final_score), 6),
                "support_count": row["support_count"],
                "explanations": _build_explanations(row["agg_features"]),
            }
        )

    final_rows.sort(key=lambda item: item["final_score"], reverse=True)
    final_rows = final_rows[:top_k]

    if not final_rows:
        return None

    return {
        "title": title,
        "items": final_rows,
    }



def _public_rec_item(row: dict) -> dict:
    """
    Strip internal scoring fields that the frontend does not need.
    """
    keep = [
        "movie_id",
        "title",
        "score",
        "final_score",
        "combined_score",
        "ranker_score",
        "ranker_score_norm",
        "quality_score",
        "support_count",
        "explanations",
    ]

    return {key: row[key] for key in keep if key in row}


def _average_item_score(items: list[dict]) -> float:
    if not items:
        return 0.0

    scores = [
        float(item.get("score") or item.get("final_score") or 0.0)
        for item in items
    ]

    return sum(scores) / len(scores)


def _item_ids(items: list[dict]) -> set[int]:
    ids = set()

    for item in items:
        movie_id = item.get("movie_id")
        if movie_id is not None:
            ids.add(int(movie_id))

    return ids


def _build_top_picks_row(all_ranked_rows: list[dict], top_k: int) -> dict | None:
    items = [_public_rec_item(row) for row in all_ranked_rows[:top_k]]

    if not items:
        return None

    return {
        "type": "top_picks",
        "title": "Top Picks For You",
        "pinned": True,
        "row_score": 1.0,
        "items": items,
    }


def _build_familiar_but_not_obvious_row(
    all_ranked_rows: list[dict],
    top_k: int,
    exclude_ids: set[int],
) -> dict | None:
    scored = []

    for row in all_ranked_rows:
        movie_id = int(row["movie_id"])
        if movie_id in exclude_ids:
            continue

        final_score = float(row.get("final_score", 0.0))
        ranker_norm = float(row.get("ranker_score_norm", final_score))
        combined_score = float(row.get("combined_score", 0.0))

        # Best zone: strong taste match, but not the most obvious cosine clone.
        not_obvious_score = 1.0 - abs(combined_score - 0.60)
        not_obvious_score = max(0.0, min(not_obvious_score, 1.0))

        engagement_score = (
            0.50 * final_score
            + 0.25 * ranker_norm
            + 0.25 * not_obvious_score
        )

        scored.append((engagement_score, row))

    scored.sort(key=lambda item: item[0], reverse=True)

    items = [_public_rec_item(row) for _, row in scored[:top_k]]

    if not items:
        return None

    return {
        "type": "familiar_but_not_obvious",
        "title": "Familiar, But Not Obvious",
        "pinned": True,
        "row_score": 0.99,
        "items": items,
    }


def _build_hidden_gems_row(
    all_ranked_rows: list[dict],
    top_k: int,
    exclude_ids: set[int],
) -> dict | None:
    if not all_ranked_rows:
        return None

    popularity_values = [
        float(np.log1p(row.get("popularity", 0.0) or 0.0))
        for row in all_ranked_rows
    ]
    vote_average_values = [
        float(row.get("vote_average", 0.0) or 0.0)
        for row in all_ranked_rows
    ]

    popularity_norm = _min_max_normalize(popularity_values)
    vote_average_norm = _min_max_normalize(vote_average_values)

    scored = []

    for row, pop_norm, vote_norm in zip(
        all_ranked_rows,
        popularity_norm,
        vote_average_norm,
    ):
        movie_id = int(row["movie_id"])
        if movie_id in exclude_ids:
            continue

        final_score = float(row.get("final_score", 0.0))
        low_popularity_score = 1.0 - float(pop_norm)

        hidden_gem_score = (
            0.45 * final_score
            + 0.30 * float(vote_norm)
            + 0.25 * low_popularity_score
        )

        scored.append((hidden_gem_score, row))

    scored.sort(key=lambda item: item[0], reverse=True)

    items = [_public_rec_item(row) for _, row in scored[:top_k]]

    if not items:
        return None

    return {
        "type": "hidden_gems",
        "title": "Hidden Gems You Might Like",
        "pinned": True,
        "row_score": 0.98,
        "items": items,
    }


def _build_ranked_taste_rows(
    selected_movies: list[dict],
    movie_ids: list[int],
    taste_summary: dict,
    exclude_movie_ids: set[int] | None = None,
) -> list[dict]:
    rows = []

    for signal in _build_taste_row_signals(taste_summary):
        row = get_constrained_row(
            selected_movies=selected_movies,
            movie_ids=movie_ids,
            signal_kind=signal["kind"],
            signal_value=signal["value"],
            title=signal["title"],
            top_k=ORGANIZED_ROW_TOP_K,
            candidate_pool=100,
            min_support=1,
            exclude_movie_ids=exclude_movie_ids,
        )

        if not row:
            continue

        items = row.get("items", [])
        avg_movie_score = _average_item_score(items)

        row_score = (
            0.35 * float(signal["signal_strength"])
            + 0.25 * avg_movie_score
            + 0.20 * float(signal["specificity"])
            + 0.10 * float(signal["support_ratio"])
            + 0.10 * _row_diversity_value(signal["kind"])
        )

        rows.append(
            {
                "type": "taste_profile",
                "title": row["title"],
                "pinned": False,
                "row_score": round(float(row_score), 6),
                "signal_kind": signal["kind"],
                "signal_value": signal["value"],
                "items": [_public_rec_item(item) for item in items],
            }
        )

    return rows


def _row_diversity_value(signal_kind: str) -> float:
    """
    Keeps the page from becoming only generic genre rows.
    """
    if signal_kind == "director":
        return 0.90
    if signal_kind == "cast":
        return 0.86
    if signal_kind == "keyword":
        return 0.82
    if signal_kind == "genre":
        return 0.60
    return 0.50


def _source_movie_fit_score(source_movie: dict, taste_summary: dict) -> float:
    """
    Measures how well an individual selected movie represents the user's total taste profile.

    Example:
    Avengers should rank high if the whole selected list is Marvel/action/superhero-heavy.
    """
    movie_genres = set(_split_pipe(source_movie.get("tmdb_genres", "")))
    movie_keywords = set(_split_pipe(source_movie.get("tmdb_keywords", "")))
    movie_directors = set(_split_pipe(source_movie.get("tmdb_directors", "")))
    movie_cast = set(_split_pipe(source_movie.get("tmdb_cast_top5", "")))

    top_genres = set(taste_summary.get("top_genres", []))
    top_keywords = set(taste_summary.get("top_keywords", []))
    repeated_directors = set(taste_summary.get("repeated_directors", []))
    repeated_cast = set(taste_summary.get("repeated_cast", []))

    genre_fit = len(movie_genres & top_genres) / max(len(top_genres), 1)
    keyword_fit = len(movie_keywords & top_keywords) / max(len(top_keywords), 1)

    director_fit = 1.0 if movie_directors & repeated_directors else 0.0
    cast_fit = 1.0 if movie_cast & repeated_cast else 0.0

    return min(
        0.45 * genre_fit
        + 0.30 * keyword_fit
        + 0.15 * director_fit
        + 0.10 * cast_fit,
        1.0,
    )


def _build_source_movie_rows(
    selected_movies: list[dict],
    taste_summary: dict,
    exclude_movie_ids: set[int] | None = None,
) -> list[dict]:
    rows = []

    for source_movie in selected_movies:
        source_movie_id = int(source_movie["movie_id"])
        source_title = source_movie.get("name") or source_movie.get("title") or "this movie"

        single = predict_single(
            movie_id=source_movie_id,
            top_k=SOURCE_MOVIE_ROW_TOP_K,
            candidate_pool=50,
            exclude_movie_ids=exclude_movie_ids,
        )

        items = single.get("recommendations", [])
        if not items:
            continue

        source_fit = _source_movie_fit_score(source_movie, taste_summary)
        avg_movie_score = _average_item_score(items)

        row_score = (
            0.45 * source_fit
            + 0.35 * avg_movie_score
            + 0.20 * 0.70
        )

        rows.append(
            {
                "type": "source_movie",
                "title": f"Because You Liked {source_title}",
                "pinned": False,
                "row_score": round(float(row_score), 6),
                "source_movie_id": source_movie_id,
                "items": [_public_rec_item(item) for item in items],
            }
        )

    return rows


def _build_organized_rows(
    all_ranked_rows: list[dict],
    selected_movies: list[dict],
    movie_ids: list[int],
    taste_summary: dict,
    top_k: int,
    exclude_movie_ids: set[int] | None = None,
    max_taste_profile_rows: int = MAX_TASTE_ROWS,
) -> list[dict]:
    """
    Final row-level ranking system.

    Pinned rows always come first.
    Dynamic rows are naturally mixed based on row_score.
    """
    organized_rows: list[dict] = []
    disliked_ids = _clean_optional_movie_ids(exclude_movie_ids)

    top_picks = _build_top_picks_row(all_ranked_rows, top_k)
    if top_picks:
        organized_rows.append(top_picks)

    used_ids = _item_ids(top_picks["items"]) if top_picks else set()

    familiar = _build_familiar_but_not_obvious_row(
        all_ranked_rows=all_ranked_rows,
        top_k=top_k,
        exclude_ids=used_ids,
    )
    if familiar:
        organized_rows.append(familiar)
        used_ids |= _item_ids(familiar["items"])

    hidden_gems = _build_hidden_gems_row(
        all_ranked_rows=all_ranked_rows,
        top_k=top_k,
        exclude_ids=used_ids,
    )
    if hidden_gems:
        organized_rows.append(hidden_gems)
        used_ids |= _item_ids(hidden_gems["items"])

    dynamic_rows = []
    dynamic_rows.extend(
        _build_ranked_taste_rows(
            selected_movies=selected_movies,
            movie_ids=movie_ids,
            taste_summary=taste_summary,
            exclude_movie_ids=disliked_ids,
        )
    )
    dynamic_rows.extend(
        _build_source_movie_rows(
            selected_movies=selected_movies,
            taste_summary=taste_summary,
            exclude_movie_ids=disliked_ids,
        )
    )

    dynamic_rows.sort(key=lambda row: row["row_score"], reverse=True)

    seen_titles = {row["title"] for row in organized_rows}
    taste_profile_count = 0

    for row in dynamic_rows:
        if row["title"] in seen_titles:
            continue

        if row.get("type") == "taste_profile":
            if taste_profile_count >= max_taste_profile_rows:
                continue
            taste_profile_count += 1

        organized_rows.append(row)
        seen_titles.add(row["title"])

        if len(organized_rows) >= 3 + MAX_DYNAMIC_ORGANIZED_ROWS:
            break

    return organized_rows



# -----------------------------------------------------
# Public Prediction Function
# -----------------------------------------------------

# ==================== SINGLE PREDICTIONS ====================
def predict_single(
    movie_id: int,
    top_k: int = DEFAULT_TOP_K,
    candidate_pool: int = 50,
    exclude_movie_ids: list[int] | set[int] | None = None,
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
    exclude_id_set = _clean_optional_movie_ids(exclude_movie_ids)
    candidates = [
        candidate
        for candidate in candidates
        if int(candidate["movie_id"]) not in exclude_id_set
    ]

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
        final_score = ((RECOMMENDATION_SPLIT_WEIGHT * float(cosine_score_norm))
                       + ((1 - RECOMMENDATION_SPLIT_WEIGHT) * float(ranker_score_norm)))

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

    taste_summary = _build_taste_summary(selected_movies)
    headline = _build_headline(taste_summary)

    return {
        "movie_ids": movie_id,
        "top_k": top_k,
        "headline": headline,
        "taste_summary": taste_summary,
        "recommendations": final_recommendations,
    }





# ==================== COMBINED PREDICTIONS ====================
def predict_combined(
    movie_ids: list[int],
    top_k: int = DEFAULT_TOP_K,
    candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    min_support: int = DEFAULT_MIN_SUPPORT,
    exclude_movie_ids: list[int] | set[int] | None = None,
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
    exclude_id_set = _clean_optional_movie_ids(exclude_movie_ids)
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
                root_span.set_attribute("request.exclude_movie_count", len(exclude_id_set))

                result = _predict_internal(
                    movie_ids=cleaned_movie_ids,
                    top_k=top_k,
                    candidate_pool=candidate_pool,
                    min_support=min_support,
                    exclude_movie_ids=exclude_id_set,
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
            exclude_movie_ids=exclude_id_set,
        )


def _predict_internal(
    movie_ids: list[int],
    top_k: int,
    candidate_pool: int,
    min_support: int,
    exclude_movie_ids: set[int] | None = None,
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
    exclude_id_set = _clean_optional_movie_ids(exclude_movie_ids)
    candidates = [
        candidate
        for candidate in candidates
        if int(candidate.movie_id) not in exclude_id_set
    ]

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
                "vote_average": float(candidate_movie.get("tmdb_vote_average") or 0.0),
                "vote_count": float(candidate_movie.get("tmdb_vote_count") or 0.0),
                "popularity": float(candidate_movie.get("tmdb_popularity") or 0.0),
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

                "ranker_score_norm": round(float(ranker_score_norm), 6),
                "vote_average": row.get("vote_average", 0.0),
                "vote_count": row.get("vote_count", 0.0),
                "popularity": row.get("popularity", 0.0),
            }
        )

    final_rows.sort(key=lambda item: item["final_score"], reverse=True)
    all_ranked_rows = sorted(
        final_rows,
        key=lambda item: item["final_score"],
        reverse=True,
    )

    top_recommendations = all_ranked_rows[:top_k]

    # -----------------------------------------------------
    # STEP 6: Build frontend-friendly taste summary
    # -----------------------------------------------------
    taste_summary = _build_taste_summary(selected_movies)
    headline = _build_headline(taste_summary)

    # -----------------------------------------------------
    # STEP 7: Build organized row-level recommendation layout
    # -----------------------------------------------------
    organized_rows = _build_organized_rows(
        all_ranked_rows=all_ranked_rows,
        selected_movies=selected_movies,
        movie_ids=movie_ids,
        taste_summary=taste_summary,
        top_k=top_k,
        exclude_movie_ids=exclude_id_set,
    )

    taste_rows = [
        row for row in organized_rows
        if row.get("type") == "taste_profile"
    ]

    return {
        "movie_ids": movie_ids,
        "top_k": top_k,
        "headline": headline,
        "taste_summary": taste_summary,
        "organized_rows": organized_rows,
        "taste_rows": taste_rows,
        "recommendations": [_public_rec_item(row) for row in top_recommendations],
    }
