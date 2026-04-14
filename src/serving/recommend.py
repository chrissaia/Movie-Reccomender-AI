"""
RECOMMENDATION INFERENCE PIPELINE
================================

This module provides production-ready movie search and recommendation serving.

Key Responsibilities:
1. Query the movies database safely and consistently
2. Serve single-movie recommendations
3. Serve combined recommendations across multiple liked movies
4. Apply deterministic ranking logic for search and aggregation
5. Keep DB access isolated and reusable

Design Principles:
- Clear function boundaries
- Input validation
- Safe parameterized SQL
- Deterministic ranking
- Small, testable helper functions
"""

from __future__ import annotations

import logging
import sqlite3
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

DB_PATH = Path("data/db/movies.db")
DEFAULT_SEARCH_LIMIT = 10
DEFAULT_RECOMMEND_LIMIT = 10
COMBINED_CANDIDATE_POOL = 75


# -----------------------------------------------------
# Data Models
# -----------------------------------------------------

@dataclass(frozen=True)
class SearchResult:
    movie_id: int
    title: str


@dataclass(frozen=True)
class Recommendation:
    movie_id: int
    title: str
    score: float


# -----------------------------------------------------
# DB Utilities
# -----------------------------------------------------

@contextmanager
def get_connection(db_path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    """
    Yield a SQLite connection with row access by column name.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        yield conn
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


def _fetch_rows(query: str, params: tuple | list) -> list[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(query, params).fetchall()


# -----------------------------------------------------
# Search
# -----------------------------------------------------

def search_movies(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[SearchResult]:
    """
    Search movie titles.

    Ranking:
    1. Exact prefix matches first
    2. Earlier substring position first
    3. Higher score first if available
    4. Alphabetical tie-breaker
    """
    _validate_positive_int(limit, "limit")

    if not isinstance(query, str):
        raise ValueError("query must be a string")

    normalized_query = query.strip().lower()
    if not normalized_query:
        return []

    sql = """
    SELECT movie_id, name
    FROM movies
    WHERE LOWER(name) LIKE ?
    ORDER BY
        CASE
            WHEN LOWER(name) LIKE ? THEN 0
            ELSE 1
        END ASC,
        INSTR(LOWER(name), ?) ASC,
        score DESC,
        votes DESC,
        name COLLATE NOCASE ASC
    LIMIT ?
    """

    rows = _fetch_rows(
        sql,
        (
            f"%{normalized_query}%",
            f"{normalized_query}%",
            normalized_query,
            limit,
        ),
    )

    return [
        SearchResult(
            movie_id=row["movie_id"],
            title=row["name"],
        )
        for row in rows
    ]


# -----------------------------------------------------
# Single-Movie Recommendations
# -----------------------------------------------------

def get_recommendations(movie_id: int, top_k: int = DEFAULT_RECOMMEND_LIMIT) -> list[Recommendation]:
    """
    Return top neighbors for one source movie.

    Uses precomputed movie_neighbors ranked by ASC rank.
    """
    _validate_positive_int(movie_id, "movie_id")
    _validate_positive_int(top_k, "top_k")

    query = """
    SELECT
        mn.neighbor_movie_id AS movie_id,
        m.name AS title,
        mn.cosine_score AS score
    FROM movie_neighbors mn
    JOIN movies m
      ON mn.neighbor_movie_id = m.movie_id
    WHERE mn.source_movie_id = ?
    ORDER BY mn.rank ASC, mn.cosine_score DESC
    LIMIT ?
    """

    rows = _fetch_rows(query, (movie_id, top_k))

    return [
        Recommendation(
            movie_id=row["movie_id"],
            title=row["title"],
            score=float(row["score"]),
        )
        for row in rows
    ]


# -----------------------------------------------------
# Combined Recommendations
# -----------------------------------------------------

def get_combined_recommendations(
    movie_ids: list[int],
    top_k: int = DEFAULT_RECOMMEND_LIMIT,
    candidate_pool: int = COMBINED_CANDIDATE_POOL,
    min_support: int | None = None,
) -> list[Recommendation]:
    """
    Build combined recommendations from multiple source movies.

    Strategy:
    1. Pull a wider candidate set per source movie
    2. Aggregate by candidate movie_id
    3. Reward:
       - higher cosine similarity
       - appearing for multiple selected movies
       - stronger rank positions
    4. Penalize one-off neighbors dominating the list

    Final score:
        final_score =
            (avg_similarity * 0.55) +
            (support_ratio * 0.35) +
            (avg_rank_score * 0.10)

    Where:
    - avg_similarity = mean cosine score across supporting source movies
    - support_ratio = support_count / number of selected movies
    - avg_rank_score = mean(1 / rank)
    """
    cleaned_movie_ids = _validate_movie_ids(movie_ids)
    _validate_positive_int(top_k, "top_k")
    _validate_positive_int(candidate_pool, "candidate_pool")

    if min_support is None:
        min_support = 1

    placeholders = ",".join(["?"] * len(cleaned_movie_ids))

    query = f"""
    SELECT
        mn.source_movie_id,
        mn.neighbor_movie_id,
        m.name AS title,
        mn.cosine_score,
        mn.rank
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
            "title": "",
            "scores": [],
            "rank_scores": [],
            "ranks": [],
            "source_movie_ids": set(),
        }
    )

    for row in rows:
        candidate_id = row["neighbor_movie_id"]
        source_movie_id = row["source_movie_id"]
        title = row["title"]
        cosine_score = float(row["cosine_score"])
        rank = int(row["rank"])

        entry = aggregated[candidate_id]
        entry["title"] = title
        entry["scores"].append(cosine_score)
        entry["rank_scores"].append(1.0 / rank)
        entry["ranks"].append(rank)
        entry["source_movie_ids"].add(source_movie_id)

    ranked: list[Recommendation] = []
    total_sources = len(cleaned_movie_ids)

    for candidate_id, data in aggregated.items():
        support_count = len(data["source_movie_ids"])

        if support_count < min_support:
            continue

        avg_similarity = sum(data["scores"]) / len(data["scores"])
        avg_rank_score = sum(data["rank_scores"]) / len(data["rank_scores"])
        support_ratio = support_count / total_sources

        best_rank = min(data["ranks"])

        # penalty if it is already very high in an individual row
        # rank 1 gets biggest penalty, deeper ranks get less
        individual_overlap_penalty = max(0.0, (11 - best_rank) / 10) * 0.20

        final_score = (
                (avg_similarity * 0.50) +
                (support_ratio * 0.35) +
                (avg_rank_score * 0.15) -
                individual_overlap_penalty
        )

        ranked.append(
            Recommendation(
                movie_id=candidate_id,
                title=data["title"],
                score=final_score,
            )
        )

    ranked.sort(key=lambda rec: rec.score, reverse=True)
    return ranked[:top_k]


# -----------------------------------------------------
# Serialization Helpers
# -----------------------------------------------------

def recommendation_to_dict(rec: Recommendation) -> dict:
    return {
        "movie_id": rec.movie_id,
        "title": rec.title,
        "score": rec.score,
    }


def search_result_to_dict(result: SearchResult) -> dict:
    return {
        "movie_id": result.movie_id,
        "title": result.title,
    }