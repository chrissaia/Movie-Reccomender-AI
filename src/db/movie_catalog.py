from __future__ import annotations

import re
import sqlite3

from src.db.sqlite import get_connection
from src.utils.paths import SQLITE_DB_PATH

# oh yeah
def _conn() -> sqlite3.Connection:
    conn = get_connection(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _movie_row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "movie_id": int(row["movie_id"]),
        "title": row["name"],
        "year": row["year"],
        "score": row["score"],
        "votes": row["votes"],
        "runtime": row["runtime"],
        "director": row["director"],
        "writer": row["writer"],
        "star": row["star"],
        "genre": row["genre"],
        "rating": row["rating"],
        "country": row["country"],
        "company": row["company"],
        "tmdb_genres": row["tmdb_genres"],
        "tmdb_keywords": row["tmdb_keywords"],
        "tmdb_cast_top5": row["tmdb_cast_top5"],
        "tmdb_directors": row["tmdb_directors"],
        "tmdb_overview": row["tmdb_overview"],
        "tmdb_vote_average": row["tmdb_vote_average"],
        "tmdb_vote_count": row["tmdb_vote_count"],
        "tmdb_runtime": row["tmdb_runtime"],
        "tmdb_popularity": row["tmdb_popularity"],
    }


def get_movie_detail(movie_id: int) -> dict | None:
    conn = _conn()
    try:
        row = conn.execute(
            """
            SELECT *
            FROM movies
            WHERE movie_id = ?
            """,
            (movie_id,),
        ).fetchone()

        return _movie_row_to_dict(row) if row else None
    finally:
        conn.close()


def _tokens(query: str) -> list[str]:
    stop_words = {
        "a", "an", "and", "by", "for", "from", "i", "in", "me", "movie",
        "movies", "of", "please", "show", "the", "to", "with", "want", "like",
    }
    words = re.findall(r"[a-z0-9']+", query.lower())
    return [word for word in words if len(word) > 2 and word not in stop_words]


def discover_movies(query: str, limit: int = 30) -> list[dict]:
    terms = _tokens(query)
    if not terms:
        return []

    conn = _conn()
    try:
        clauses = []
        params: list[str | int] = []
        searchable = [
            "LOWER(name)",
            "LOWER(genre)",
            "LOWER(director)",
            "LOWER(writer)",
            "LOWER(star)",
            "LOWER(rating)",
            "LOWER(tmdb_genres)",
            "LOWER(tmdb_keywords)",
            "LOWER(tmdb_cast_top5)",
            "LOWER(tmdb_directors)",
            "LOWER(tmdb_overview)",
        ]

        for term in terms:
            term_clauses = [f"{column} LIKE ?" for column in searchable]
            clauses.append("(" + " OR ".join(term_clauses) + ")")
            params.extend([f"%{term}%"] * len(searchable))

        params.append(limit)
        rows = conn.execute(
            f"""
            SELECT *,
                (COALESCE(score, 0) * 0.45)
                + (COALESCE(tmdb_vote_average, 0) * 0.35)
                + (CASE WHEN COALESCE(tmdb_vote_count, votes, 0) > 1000 THEN 1 ELSE 0 END * 0.20) AS discovery_score
            FROM movies
            WHERE {' OR '.join(clauses)}
            ORDER BY discovery_score DESC, COALESCE(tmdb_popularity, 0) DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

        return [_movie_row_to_dict(row) for row in rows]
    finally:
        conn.close()
