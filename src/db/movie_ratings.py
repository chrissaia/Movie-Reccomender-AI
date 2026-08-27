from __future__ import annotations

from datetime import datetime, timezone

from src.db.sqlite import get_connection
from src.utils.paths import SQLITE_DB_PATH


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_movie_rating_tables() -> None:
    conn = get_connection(SQLITE_DB_PATH)

    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_movie_ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                movie_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                rating REAL NOT NULL,
                description TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(user_id, movie_id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_disliked_movies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                movie_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(user_id, movie_id)
            )
        """)
        # One-time cleanup for the removed Watchlist feature.
        conn.execute("DROP TABLE IF EXISTS user_watchlist_movies")
        conn.commit()
    finally:
        conn.close()


def upsert_movie_rating(
    user_id: str,
    movie_id: int,
    title: str,
    rating: float,
    description: str | None,
) -> dict:
    now = _now()
    cleaned_description = description.strip() if description else None

    conn = get_connection(SQLITE_DB_PATH)
    try:
        conn.execute(
            """
            INSERT INTO user_movie_ratings (
                user_id,
                movie_id,
                title,
                rating,
                description,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, movie_id)
            DO UPDATE SET
                title = excluded.title,
                rating = excluded.rating,
                description = excluded.description,
                updated_at = excluded.updated_at
            """,
            (
                user_id,
                movie_id,
                title.strip(),
                rating,
                cleaned_description,
                now,
                now,
            ),
        )
        conn.commit()

        return {
            "movie_id": movie_id,
            "title": title.strip(),
            "rating": rating,
            "description": cleaned_description,
            "updatedAt": now,
        }
    finally:
        conn.close()


def dislike_movie(user_id: str, movie_id: int, title: str) -> dict:
    now = _now()
    cleaned_title = title.strip()

    conn = get_connection(SQLITE_DB_PATH)
    try:
        conn.execute(
            """
            INSERT INTO user_disliked_movies (
                user_id,
                movie_id,
                title,
                created_at
            )
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, movie_id)
            DO UPDATE SET
                title = excluded.title
            """,
            (user_id, movie_id, cleaned_title, now),
        )
        conn.commit()

        return {
            "movie_id": movie_id,
            "title": cleaned_title,
            "createdAt": now,
        }
    finally:
        conn.close()


def list_disliked_movie_ids(user_id: str) -> list[int]:
    conn = get_connection(SQLITE_DB_PATH)
    try:
        rows = conn.execute(
            """
            SELECT movie_id
            FROM user_disliked_movies
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (user_id,),
        ).fetchall()

        return [int(row[0]) for row in rows]
    finally:
        conn.close()
