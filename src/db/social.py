from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone

from src.db.sqlite import get_connection
from src.utils.paths import SQLITE_DB_PATH


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_profile(row: sqlite3.Row) -> dict:
    return {
        "user_id": row["user_id"],
        "email": row["email"],
        "name": row["name"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def ensure_social_tables() -> None:
    conn = get_connection(SQLITE_DB_PATH)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                email TEXT,
                name TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS friendships (
                id TEXT PRIMARY KEY,
                requester_user_id TEXT NOT NULL,
                receiver_user_id TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(requester_user_id, receiver_user_id)
            )
        """)

        conn.commit()
    finally:
        conn.close()


def upsert_user_profile(user_id: str, email: str | None, name: str | None) -> dict:
    now = _now()

    conn = get_connection(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        existing = conn.execute(
            "SELECT * FROM user_profiles WHERE user_id = ?",
            (user_id,),
        ).fetchone()

        if existing:
            conn.execute(
                """
                UPDATE user_profiles
                SET email = ?, name = ?, updated_at = ?
                WHERE user_id = ?
                """,
                (email, name, now, user_id),
            )
        else:
            conn.execute(
                """
                INSERT INTO user_profiles (user_id, email, name, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, email, name, now, now),
            )

        conn.commit()

        row = conn.execute(
            "SELECT * FROM user_profiles WHERE user_id = ?",
            (user_id,),
        ).fetchone()

        return _row_to_profile(row)
    finally:
        conn.close()


def search_user_profiles(current_user_id: str, query: str, limit: int = 10) -> list[dict]:
    cleaned = query.strip().lower()

    if not cleaned:
        return []

    conn = get_connection(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        rows = conn.execute(
            """
            SELECT *
            FROM user_profiles
            WHERE user_id != ?
              AND (
                lower(coalesce(email, '')) LIKE ?
                OR lower(coalesce(name, '')) LIKE ?
              )
            ORDER BY name ASC, email ASC
            LIMIT ?
            """,
            (current_user_id, f"%{cleaned}%", f"%{cleaned}%", limit),
        ).fetchall()

        return [_row_to_profile(row) for row in rows]
    finally:
        conn.close()


def create_friend_request(requester_user_id: str, receiver_user_id: str) -> dict:
    if requester_user_id == receiver_user_id:
        raise ValueError("You cannot add yourself as a friend")

    now = _now()

    user_a, user_b = sorted([requester_user_id, receiver_user_id])

    conn = get_connection(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        existing = conn.execute(
            """
            SELECT *
            FROM friendships
            WHERE (requester_user_id = ? AND receiver_user_id = ?)
               OR (requester_user_id = ? AND receiver_user_id = ?)
            """,
            (requester_user_id, receiver_user_id, receiver_user_id, requester_user_id),
        ).fetchone()

        if existing:
            return _friendship_to_dict(existing, requester_user_id)

        friendship_id = str(uuid.uuid4())

        conn.execute(
            """
            INSERT INTO friendships (
                id,
                requester_user_id,
                receiver_user_id,
                status,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                friendship_id,
                requester_user_id,
                receiver_user_id,
                "pending",
                now,
                now,
            ),
        )

        conn.commit()

        row = conn.execute(
            "SELECT * FROM friendships WHERE id = ?",
            (friendship_id,),
        ).fetchone()

        return _friendship_to_dict(row, requester_user_id)
    finally:
        conn.close()


def list_friends_and_requests(user_id: str) -> dict:
    conn = get_connection(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        rows = conn.execute(
            """
            SELECT *
            FROM friendships
            WHERE requester_user_id = ?
               OR receiver_user_id = ?
            ORDER BY updated_at DESC
            """,
            (user_id, user_id),
        ).fetchall()

        friends = []
        incoming_requests = []
        outgoing_requests = []

        for row in rows:
            item = _friendship_to_dict(row, user_id)

            if row["status"] == "accepted":
                friends.append(item)
            elif row["status"] == "pending" and row["receiver_user_id"] == user_id:
                incoming_requests.append(item)
            elif row["status"] == "pending" and row["requester_user_id"] == user_id:
                outgoing_requests.append(item)

        return {
            "friends": friends,
            "incoming_requests": incoming_requests,
            "outgoing_requests": outgoing_requests,
        }
    finally:
        conn.close()


def accept_friend_request(user_id: str, friendship_id: str) -> dict:
    now = _now()

    conn = get_connection(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        row = conn.execute(
            "SELECT * FROM friendships WHERE id = ?",
            (friendship_id,),
        ).fetchone()

        if not row:
            raise ValueError("Friend request not found")

        if row["receiver_user_id"] != user_id:
            raise ValueError("Only the receiver can accept this request")

        conn.execute(
            """
            UPDATE friendships
            SET status = 'accepted', updated_at = ?
            WHERE id = ?
            """,
            (now, friendship_id),
        )

        conn.commit()

        updated = conn.execute(
            "SELECT * FROM friendships WHERE id = ?",
            (friendship_id,),
        ).fetchone()

        return _friendship_to_dict(updated, user_id)
    finally:
        conn.close()


def delete_friendship(user_id: str, friendship_id: str) -> None:
    conn = get_connection(SQLITE_DB_PATH)

    try:
        deleted = conn.execute(
            """
            DELETE FROM friendships
            WHERE id = ?
              AND (
                requester_user_id = ?
                OR receiver_user_id = ?
              )
            """,
            (friendship_id, user_id, user_id),
        ).rowcount

        conn.commit()

        if deleted == 0:
            raise ValueError("Friendship not found")
    finally:
        conn.close()


def _get_profile(conn: sqlite3.Connection, user_id: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM user_profiles WHERE user_id = ?",
        (user_id,),
    ).fetchone()

    if not row:
        return None

    return _row_to_profile(row)


def _friendship_to_dict(row: sqlite3.Row, current_user_id: str) -> dict:
    conn = get_connection(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        other_user_id = (
            row["receiver_user_id"]
            if row["requester_user_id"] == current_user_id
            else row["requester_user_id"]
        )

        other_profile = _get_profile(conn, other_user_id)

        return {
            "id": row["id"],
            "requester_user_id": row["requester_user_id"],
            "receiver_user_id": row["receiver_user_id"],
            "other_user_id": other_user_id,
            "other_user": other_profile,
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
    finally:
        conn.close()