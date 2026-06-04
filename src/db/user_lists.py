from __future__ import annotations

import uuid
from datetime import datetime, timezone

from src.db.sqlite import get_connection
from src.utils.paths import SQLITE_DB_PATH


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_user_list_tables() -> None:
    conn = get_connection(SQLITE_DB_PATH)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_lists (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_list_movies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                list_id TEXT NOT NULL,
                movie_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                position INTEGER NOT NULL,
                FOREIGN KEY (list_id) REFERENCES user_lists(id) ON DELETE CASCADE
            )
        """)
        conn.commit()
    finally:
        conn.close()


def list_user_lists(user_id: str) -> list[dict]:
    conn = get_connection(SQLITE_DB_PATH)
    conn.row_factory = None
    try:
        lists = conn.execute(
            """
            SELECT id, name, created_at
            FROM user_lists
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (user_id,),
        ).fetchall()

        output = []

        for list_id, name, created_at in lists:
            movies = conn.execute(
                """
                SELECT movie_id, title
                FROM user_list_movies
                WHERE list_id = ?
                ORDER BY position ASC
                """,
                (list_id,),
            ).fetchall()

            output.append({
                "id": list_id,
                "name": name,
                "createdAt": created_at,
                "movies": [
                    {"movie_id": movie_id, "title": title}
                    for movie_id, title in movies
                ],
            })

        return output
    finally:
        conn.close()


def create_user_list(user_id: str, name: str, movies: list[dict]) -> dict:
    list_id = str(uuid.uuid4())
    created_at = _now()

    conn = get_connection(SQLITE_DB_PATH)
    try:
        conn.execute(
            """
            INSERT INTO user_lists (id, user_id, name, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (list_id, user_id, name.strip(), created_at),
        )

        for position, movie in enumerate(movies):
            conn.execute(
                """
                INSERT INTO user_list_movies (list_id, movie_id, title, position)
                VALUES (?, ?, ?, ?)
                """,
                (list_id, int(movie["movie_id"]), movie["title"], position),
            )

        conn.commit()

        return {
            "id": list_id,
            "name": name.strip(),
            "createdAt": created_at,
            "movies": movies,
        }
    finally:
        conn.close()


def update_user_list(user_id: str, list_id: str, name: str | None, movies: list[dict] | None) -> dict:
    conn = get_connection(SQLITE_DB_PATH)
    try:
        exists = conn.execute(
            "SELECT id FROM user_lists WHERE id = ? AND user_id = ?",
            (list_id, user_id),
        ).fetchone()

        if not exists:
            raise ValueError("List not found")

        if name is not None:
            conn.execute(
                "UPDATE user_lists SET name = ? WHERE id = ? AND user_id = ?",
                (name.strip(), list_id, user_id),
            )

        if movies is not None:
            conn.execute("DELETE FROM user_list_movies WHERE list_id = ?", (list_id,))

            for position, movie in enumerate(movies):
                conn.execute(
                    """
                    INSERT INTO user_list_movies (list_id, movie_id, title, position)
                    VALUES (?, ?, ?, ?)
                    """,
                    (list_id, int(movie["movie_id"]), movie["title"], position),
                )

        conn.commit()
    finally:
        conn.close()

    return next(item for item in list_user_lists(user_id) if item["id"] == list_id)


def delete_user_list(user_id: str, list_id: str) -> None:
    conn = get_connection(SQLITE_DB_PATH)
    try:
        conn.execute(
            "DELETE FROM user_lists WHERE id = ? AND user_id = ?",
            (list_id, user_id),
        )
        conn.commit()
    finally:
        conn.close()