from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone

from src.db.sqlite import get_connection
from src.utils.paths import SQLITE_DB_PATH


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_shared_list_tables() -> None:
    conn = get_connection(SQLITE_DB_PATH)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS shared_lists (
                id TEXT PRIMARY KEY,
                owner_user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS shared_list_members (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                shared_list_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(shared_list_id, user_id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS shared_list_movies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                shared_list_id TEXT NOT NULL,
                movie_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                added_by_user_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(shared_list_id, movie_id)
            )
        """)

        conn.commit()
    finally:
        conn.close()


def _are_friends(conn: sqlite3.Connection, user_a: str, user_b: str) -> bool:
    row = conn.execute(
        """
        SELECT id
        FROM friendships
        WHERE status = 'accepted'
          AND (
            (requester_user_id = ? AND receiver_user_id = ?)
            OR
            (requester_user_id = ? AND receiver_user_id = ?)
          )
        """,
        (user_a, user_b, user_b, user_a),
    ).fetchone()

    return row is not None


def _get_member_role(conn: sqlite3.Connection, shared_list_id: str, user_id: str) -> str | None:
    row = conn.execute(
        """
        SELECT role
        FROM shared_list_members
        WHERE shared_list_id = ? AND user_id = ?
        """,
        (shared_list_id, user_id),
    ).fetchone()

    if not row:
        return None

    return row["role"]


def _require_member(conn: sqlite3.Connection, shared_list_id: str, user_id: str) -> str:
    role = _get_member_role(conn, shared_list_id, user_id)

    if role is None:
        raise ValueError("Shared list not found")

    return role


def _require_editor(conn: sqlite3.Connection, shared_list_id: str, user_id: str) -> str:
    role = _require_member(conn, shared_list_id, user_id)

    if role not in {"owner", "editor"}:
        raise ValueError("You do not have permission to edit this shared list")

    return role


def _require_owner(conn: sqlite3.Connection, shared_list_id: str, user_id: str) -> None:
    role = _require_member(conn, shared_list_id, user_id)

    if role != "owner":
        raise ValueError("Only the owner can do this")


def _profile_for(conn: sqlite3.Connection, user_id: str) -> dict | None:
    row = conn.execute(
        """
        SELECT user_id, email, name
        FROM user_profiles
        WHERE user_id = ?
        """,
        (user_id,),
    ).fetchone()

    if not row:
        return None

    return {
        "user_id": row["user_id"],
        "email": row["email"],
        "name": row["name"],
    }


def _shared_list_to_dict(conn: sqlite3.Connection, shared_list_id: str) -> dict:
    shared = conn.execute(
        """
        SELECT *
        FROM shared_lists
        WHERE id = ?
        """,
        (shared_list_id,),
    ).fetchone()

    if not shared:
        raise ValueError("Shared list not found")

    members = conn.execute(
        """
        SELECT user_id, role, created_at
        FROM shared_list_members
        WHERE shared_list_id = ?
        ORDER BY role ASC, created_at ASC
        """,
        (shared_list_id,),
    ).fetchall()

    movies = conn.execute(
        """
        SELECT movie_id, title, added_by_user_id, position, created_at
        FROM shared_list_movies
        WHERE shared_list_id = ?
        ORDER BY position ASC
        """,
        (shared_list_id,),
    ).fetchall()

    return {
        "id": shared["id"],
        "name": shared["name"],
        "owner_user_id": shared["owner_user_id"],
        "createdAt": shared["created_at"],
        "updatedAt": shared["updated_at"],
        "members": [
            {
                "user_id": row["user_id"],
                "role": row["role"],
                "createdAt": row["created_at"],
                "profile": _profile_for(conn, row["user_id"]),
            }
            for row in members
        ],
        "movies": [
            {
                "movie_id": row["movie_id"],
                "title": row["title"],
                "added_by_user_id": row["added_by_user_id"],
                "position": row["position"],
                "createdAt": row["created_at"],
            }
            for row in movies
        ],
    }


def list_shared_lists(user_id: str) -> list[dict]:
    conn = get_connection(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        rows = conn.execute(
            """
            SELECT sl.id
            FROM shared_lists sl
            JOIN shared_list_members m
              ON sl.id = m.shared_list_id
            WHERE m.user_id = ?
            ORDER BY sl.updated_at DESC
            """,
            (user_id,),
        ).fetchall()

        return [_shared_list_to_dict(conn, row["id"]) for row in rows]
    finally:
        conn.close()


def get_shared_list(user_id: str, shared_list_id: str) -> dict:
    conn = get_connection(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        _require_member(conn, shared_list_id, user_id)
        return _shared_list_to_dict(conn, shared_list_id)
    finally:
        conn.close()


def create_shared_list(
    owner_user_id: str,
    name: str,
    member_user_ids: list[str],
    movies: list[dict],
) -> dict:
    now = _now()
    shared_list_id = str(uuid.uuid4())

    conn = get_connection(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        conn.execute(
            """
            INSERT INTO shared_lists (id, owner_user_id, name, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (shared_list_id, owner_user_id, name.strip(), now, now),
        )

        conn.execute(
            """
            INSERT INTO shared_list_members (shared_list_id, user_id, role, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (shared_list_id, owner_user_id, "owner", now),
        )

        unique_members = []
        seen = {owner_user_id}

        for member_user_id in member_user_ids:
            if member_user_id in seen:
                continue

            if not _are_friends(conn, owner_user_id, member_user_id):
                raise ValueError("You can only create shared lists with accepted friends")

            unique_members.append(member_user_id)
            seen.add(member_user_id)

        for member_user_id in unique_members:
            conn.execute(
                """
                INSERT INTO shared_list_members (shared_list_id, user_id, role, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (shared_list_id, member_user_id, "editor", now),
            )

        for position, movie in enumerate(movies):
            conn.execute(
                """
                INSERT OR IGNORE INTO shared_list_movies (
                    shared_list_id,
                    movie_id,
                    title,
                    added_by_user_id,
                    position,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    shared_list_id,
                    int(movie["movie_id"]),
                    movie["title"],
                    owner_user_id,
                    position,
                    now,
                ),
            )

        conn.commit()
        return _shared_list_to_dict(conn, shared_list_id)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def update_shared_list_name(user_id: str, shared_list_id: str, name: str) -> dict:
    now = _now()

    conn = get_connection(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        _require_editor(conn, shared_list_id, user_id)

        conn.execute(
            """
            UPDATE shared_lists
            SET name = ?, updated_at = ?
            WHERE id = ?
            """,
            (name.strip(), now, shared_list_id),
        )

        conn.commit()
        return _shared_list_to_dict(conn, shared_list_id)
    finally:
        conn.close()


def add_shared_list_member(
    owner_user_id: str,
    shared_list_id: str,
    member_user_id: str,
    role: str = "editor",
) -> dict:
    if role not in {"editor", "viewer"}:
        raise ValueError("Role must be editor or viewer")

    now = _now()

    conn = get_connection(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        _require_owner(conn, shared_list_id, owner_user_id)

        if not _are_friends(conn, owner_user_id, member_user_id):
            raise ValueError("You can only add accepted friends to shared lists")

        conn.execute(
            """
            INSERT OR IGNORE INTO shared_list_members (
                shared_list_id,
                user_id,
                role,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (shared_list_id, member_user_id, role, now),
        )

        conn.execute(
            """
            UPDATE shared_lists
            SET updated_at = ?
            WHERE id = ?
            """,
            (now, shared_list_id),
        )

        conn.commit()
        return _shared_list_to_dict(conn, shared_list_id)
    finally:
        conn.close()


def remove_shared_list_member(
    current_user_id: str,
    shared_list_id: str,
    member_user_id: str,
) -> dict:
    now = _now()

    conn = get_connection(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        current_role = _require_member(conn, shared_list_id, current_user_id)

        if member_user_id == current_user_id:
            if current_role == "owner":
                raise ValueError("Owner cannot leave. Delete the shared list instead.")
        else:
            _require_owner(conn, shared_list_id, current_user_id)

        conn.execute(
            """
            DELETE FROM shared_list_members
            WHERE shared_list_id = ? AND user_id = ?
            """,
            (shared_list_id, member_user_id),
        )

        conn.execute(
            """
            UPDATE shared_lists
            SET updated_at = ?
            WHERE id = ?
            """,
            (now, shared_list_id),
        )

        conn.commit()
        return _shared_list_to_dict(conn, shared_list_id)
    finally:
        conn.close()


def add_shared_list_movie(
    user_id: str,
    shared_list_id: str,
    movie_id: int,
    title: str,
) -> dict:
    now = _now()

    conn = get_connection(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        _require_editor(conn, shared_list_id, user_id)

        max_position = conn.execute(
            """
            SELECT MAX(position) AS max_position
            FROM shared_list_movies
            WHERE shared_list_id = ?
            """,
            (shared_list_id,),
        ).fetchone()["max_position"]

        next_position = 0 if max_position is None else int(max_position) + 1

        conn.execute(
            """
            INSERT OR IGNORE INTO shared_list_movies (
                shared_list_id,
                movie_id,
                title,
                added_by_user_id,
                position,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (shared_list_id, movie_id, title, user_id, next_position, now),
        )

        conn.execute(
            """
            UPDATE shared_lists
            SET updated_at = ?
            WHERE id = ?
            """,
            (now, shared_list_id),
        )

        conn.commit()
        return _shared_list_to_dict(conn, shared_list_id)
    finally:
        conn.close()


def remove_shared_list_movie(
    user_id: str,
    shared_list_id: str,
    movie_id: int,
) -> dict:
    now = _now()

    conn = get_connection(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        _require_editor(conn, shared_list_id, user_id)

        conn.execute(
            """
            DELETE FROM shared_list_movies
            WHERE shared_list_id = ? AND movie_id = ?
            """,
            (shared_list_id, movie_id),
        )

        conn.execute(
            """
            UPDATE shared_lists
            SET updated_at = ?
            WHERE id = ?
            """,
            (now, shared_list_id),
        )

        conn.commit()
        return _shared_list_to_dict(conn, shared_list_id)
    finally:
        conn.close()


def delete_shared_list(user_id: str, shared_list_id: str) -> None:
    conn = get_connection(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        _require_owner(conn, shared_list_id, user_id)

        conn.execute(
            "DELETE FROM shared_list_movies WHERE shared_list_id = ?",
            (shared_list_id,),
        )
        conn.execute(
            "DELETE FROM shared_list_members WHERE shared_list_id = ?",
            (shared_list_id,),
        )
        conn.execute(
            "DELETE FROM shared_lists WHERE id = ?",
            (shared_list_id,),
        )

        conn.commit()
    finally:
        conn.close()


def convert_personal_list_to_shared(
    owner_user_id: str,
    personal_list_id: str,
    member_user_ids: list[str],
) -> dict:
    now = _now()
    shared_list_id = str(uuid.uuid4())

    conn = get_connection(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        personal = conn.execute(
            """
            SELECT *
            FROM user_lists
            WHERE id = ? AND user_id = ?
            """,
            (personal_list_id, owner_user_id),
        ).fetchone()

        if not personal:
            raise ValueError("Personal list not found")

        movies = conn.execute(
            """
            SELECT movie_id, title, position
            FROM user_list_movies
            WHERE list_id = ?
            ORDER BY position ASC
            """,
            (personal_list_id,),
        ).fetchall()

        if not member_user_ids:
            raise ValueError("Choose at least one friend to share with")

        unique_members = []
        seen = {owner_user_id}

        for member_user_id in member_user_ids:
            if member_user_id in seen:
                continue

            if not _are_friends(conn, owner_user_id, member_user_id):
                raise ValueError("You can only share lists with accepted friends")

            unique_members.append(member_user_id)
            seen.add(member_user_id)

        conn.execute(
            """
            INSERT INTO shared_lists (id, owner_user_id, name, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (shared_list_id, owner_user_id, personal["name"], now, now),
        )

        conn.execute(
            """
            INSERT INTO shared_list_members (shared_list_id, user_id, role, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (shared_list_id, owner_user_id, "owner", now),
        )

        for member_user_id in unique_members:
            conn.execute(
                """
                INSERT INTO shared_list_members (shared_list_id, user_id, role, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (shared_list_id, member_user_id, "editor", now),
            )

        for movie in movies:
            conn.execute(
                """
                INSERT INTO shared_list_movies (
                    shared_list_id,
                    movie_id,
                    title,
                    added_by_user_id,
                    position,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    shared_list_id,
                    int(movie["movie_id"]),
                    movie["title"],
                    owner_user_id,
                    int(movie["position"]),
                    now,
                ),
            )

        # Delete original personal list after successful shared-list creation.
        conn.execute(
            "DELETE FROM user_list_movies WHERE list_id = ?",
            (personal_list_id,),
        )
        conn.execute(
            "DELETE FROM user_lists WHERE id = ? AND user_id = ?",
            (personal_list_id, owner_user_id),
        )

        conn.commit()
        return _shared_list_to_dict(conn, shared_list_id)

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()