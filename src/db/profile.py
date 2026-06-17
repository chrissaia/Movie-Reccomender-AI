from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone

from src.db.sqlite import get_connection
from src.utils.paths import SQLITE_DB_PATH


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def _dump_list(value: list[str] | None) -> str:
    return json.dumps(value or [])


def _unique_values(values: list[str]) -> list[str]:
    seen = set()
    unique = []

    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized.lower() in seen:
            continue

        seen.add(normalized.lower())
        unique.append(normalized)

    return unique


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name = ?
        """,
        (table_name,),
    ).fetchone()

    return row is not None


def _split_pipe(value) -> list[str]:
    if value is None:
        return []
    return [part.strip().lower() for part in str(value).split("|") if part.strip()]


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, sql_type: str) -> None:
    existing = conn.execute(f"PRAGMA table_info({table})").fetchall()
    existing_names = {row[1] for row in existing}

    if column not in existing_names:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")


def _get_shared_lists_with_movies_for_user(
    conn: sqlite3.Connection,
    user_id: str,
) -> list[dict]:
    rows = conn.execute(
        """
        SELECT 
            sl.id,
            sl.name,
            sl.owner_user_id,
            sl.created_at,
            sl.updated_at
        FROM shared_lists sl
        JOIN shared_list_members m
          ON sl.id = m.shared_list_id
        WHERE m.user_id = ?
        ORDER BY sl.updated_at DESC
        """,
        (user_id,),
    ).fetchall()

    lists = []

    for row in rows:
        movies = conn.execute(
            """
            SELECT movie_id, title, position
            FROM shared_list_movies
            WHERE shared_list_id = ?
            ORDER BY position ASC
            """,
            (row["id"],),
        ).fetchall()

        lists.append(
            {
                "id": row["id"],
                "name": row["name"],
                "kind": "shared",
                "owner_user_id": row["owner_user_id"],
                "createdAt": row["created_at"],
                "updatedAt": row["updated_at"],
                "movies": [
                    {
                        "movie_id": int(movie["movie_id"]),
                        "title": movie["title"],
                    }
                    for movie in movies
                ],
            }
        )

    return lists


def ensure_profile_tables() -> None:
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

        _add_column_if_missing(conn, "user_profiles", "bio", "TEXT")
        _add_column_if_missing(conn, "user_profiles", "avatar_url", "TEXT")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_onboarding_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL UNIQUE,
                favorite_genres TEXT NOT NULL DEFAULT '[]',
                disliked_genres TEXT NOT NULL DEFAULT '[]',
                preferred_moods TEXT NOT NULL DEFAULT '[]',
                preferred_pacing TEXT NOT NULL DEFAULT '[]',
                preferred_decades TEXT NOT NULL DEFAULT '[]',
                favorite_movies TEXT NOT NULL DEFAULT '[]',
                disliked_movies TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        conn.commit()
    finally:
        conn.close()


def update_profile(user_id: str, name: str | None, bio: str | None, avatar_url: str | None) -> dict:
    now = _now()

    conn = get_connection(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        existing = conn.execute(
            "SELECT user_id FROM user_profiles WHERE user_id = ?",
            (user_id,),
        ).fetchone()

        if existing:
            conn.execute(
                """
                UPDATE user_profiles
                SET name = COALESCE(?, name),
                    bio = ?,
                    avatar_url = ?,
                    updated_at = ?
                WHERE user_id = ?
                """,
                (name, bio, avatar_url, now, user_id),
            )
        else:
            conn.execute(
                """
                INSERT INTO user_profiles (
                    user_id, email, name, bio, avatar_url, created_at, updated_at
                )
                VALUES (?, NULL, ?, ?, ?, ?, ?)
                """,
                (user_id, name, bio, avatar_url, now, now),
            )

        conn.commit()
        return get_profile_home(user_id)["profile"]
    finally:
        conn.close()


def update_onboarding_preferences(user_id: str, payload: dict) -> dict:
    now = _now()

    fields = {
        "favorite_genres": _dump_list(payload.get("favorite_genres")),
        "disliked_genres": _dump_list(payload.get("disliked_genres")),
        "preferred_moods": _dump_list(payload.get("preferred_moods")),
        "preferred_pacing": _dump_list(payload.get("preferred_pacing")),
        "preferred_decades": _dump_list(payload.get("preferred_decades")),
        "favorite_movies": _dump_list(payload.get("favorite_movies")),
        "disliked_movies": _dump_list(payload.get("disliked_movies")),
    }

    conn = get_connection(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        existing = conn.execute(
            "SELECT id FROM user_onboarding_preferences WHERE user_id = ?",
            (user_id,),
        ).fetchone()

        if existing:
            conn.execute(
                """
                UPDATE user_onboarding_preferences
                SET favorite_genres = ?,
                    disliked_genres = ?,
                    preferred_moods = ?,
                    preferred_pacing = ?,
                    preferred_decades = ?,
                    favorite_movies = ?,
                    disliked_movies = ?,
                    updated_at = ?
                WHERE user_id = ?
                """,
                (
                    fields["favorite_genres"],
                    fields["disliked_genres"],
                    fields["preferred_moods"],
                    fields["preferred_pacing"],
                    fields["preferred_decades"],
                    fields["favorite_movies"],
                    fields["disliked_movies"],
                    now,
                    user_id,
                ),
            )
        else:
            conn.execute(
                """
                INSERT INTO user_onboarding_preferences (
                    user_id,
                    favorite_genres,
                    disliked_genres,
                    preferred_moods,
                    preferred_pacing,
                    preferred_decades,
                    favorite_movies,
                    disliked_movies,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    fields["favorite_genres"],
                    fields["disliked_genres"],
                    fields["preferred_moods"],
                    fields["preferred_pacing"],
                    fields["preferred_decades"],
                    fields["favorite_movies"],
                    fields["disliked_movies"],
                    now,
                    now,
                ),
            )

        conn.commit()
        return get_profile_home(user_id)["onboarding_preferences"]
    finally:
        conn.close()


def get_profile_home(user_id: str) -> dict:
    conn = get_connection(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        profile = _get_profile(conn, user_id)
        onboarding = _get_onboarding(conn, user_id)
        disliked_movie_titles = _get_disliked_movie_titles(conn, user_id)
        if disliked_movie_titles:
            onboarding = {
                **onboarding,
                "disliked_movies": _unique_values(
                    onboarding.get("disliked_movies", []) + disliked_movie_titles
                ),
            }
        personal_lists = _get_personal_lists(conn, user_id)
        shared_lists = _get_shared_lists(conn, user_id)
        friends = _get_friends(conn, user_id)
        watchlist = _get_watchlist(conn, user_id)

        movie_ids = _collect_movie_ids(conn, user_id)
        taste_summary = _build_taste_summary(conn, movie_ids)

        return {
            "profile": profile,
            "onboarding_preferences": onboarding,
            "stats": {
                "saved_lists_count": len(personal_lists),
                "shared_lists_count": len(shared_lists),
                "friends_count": len(friends),
                "unique_movies_count": len(set(movie_ids)),
            },
            "taste_summary": taste_summary,
            "lists": personal_lists,
            "shared_lists": shared_lists,
            "friends": friends,
            "watchlist": watchlist,
            "recent_activity": _get_recent_activity(conn, user_id),
        }
    finally:
        conn.close()


def get_friend_profile(user_id: str, viewer_user_id: str) -> dict | None:
    conn = get_connection(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        if user_id != viewer_user_id and not _are_accepted_friends(
            conn,
            user_id,
            viewer_user_id,
        ):
            return None

        profile = _get_profile(conn, user_id)
        movie_ids = _collect_movie_ids(conn, user_id)
        viewer_movie_ids = _collect_movie_ids(conn, viewer_user_id)
        taste_summary = _build_taste_summary(conn, movie_ids)
        viewer_taste_summary = _build_taste_summary(conn, viewer_movie_ids)
        shared_lists = _get_shared_lists(conn, user_id)
        friends = _get_friends(conn, user_id)
        taste_match = _build_taste_match(
            conn=conn,
            viewer_user_id=viewer_user_id,
            friend_user_id=user_id,
            viewer_movie_ids=viewer_movie_ids,
            friend_movie_ids=movie_ids,
            viewer_taste_summary=viewer_taste_summary,
            friend_taste_summary=taste_summary,
        )

        return {
            "profile": {
                "user_id": profile["user_id"],
                "name": profile["name"],
                "bio": profile["bio"],
                "avatar_url": profile["avatar_url"],
            },
            "taste_summary": taste_summary,
            "taste_match": taste_match,
            "stats": {
                "shared_lists_count": len(shared_lists),
                "friends_count": len(friends),
                "unique_movies_count": len(set(movie_ids)),
            },
            "shared_lists": shared_lists[:4],
        }
    finally:
        conn.close()


def get_friend_lists(user_id: str, viewer_user_id: str) -> dict | None:
    conn = get_connection(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        if user_id != viewer_user_id and not _are_accepted_friends(
            conn,
            user_id,
            viewer_user_id,
        ):
            return None

        profile = _get_profile(conn, user_id)

        personal_lists = [
            {
                **list_item,
                "kind": "personal",
            }
            for list_item in _get_personal_lists_with_movies(conn, user_id)
        ]

        shared_lists = _get_shared_lists_with_movies_for_user(conn, user_id)

        return {
            "profile": {
                "user_id": profile["user_id"],
                "name": profile["name"],
                "avatar_url": profile["avatar_url"],
            },
            "lists": personal_lists + shared_lists,
        }
    finally:
        conn.close()


def get_friend_list_for_copy(
    user_id: str,
    viewer_user_id: str,
    list_id: str,
) -> dict | None:
    conn = get_connection(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        if user_id != viewer_user_id and not _are_accepted_friends(
            conn,
            user_id,
            viewer_user_id,
        ):
            return None

        lists = _get_personal_lists_with_movies(conn, user_id)
        return next((item for item in lists if item["id"] == list_id), None)
    finally:
        conn.close()


def get_friend_taste_match(user_id: str, viewer_user_id: str) -> dict | None:
    conn = get_connection(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        if user_id != viewer_user_id and not _are_accepted_friends(
            conn,
            user_id,
            viewer_user_id,
        ):
            return None

        friend_movie_ids = _collect_movie_ids(conn, user_id)
        viewer_movie_ids = _collect_movie_ids(conn, viewer_user_id)

        return _build_taste_match(
            conn=conn,
            viewer_user_id=viewer_user_id,
            friend_user_id=user_id,
            viewer_movie_ids=viewer_movie_ids,
            friend_movie_ids=friend_movie_ids,
            viewer_taste_summary=_build_taste_summary(conn, viewer_movie_ids),
            friend_taste_summary=_build_taste_summary(conn, friend_movie_ids),
        )
    finally:
        conn.close()


def _get_profile(conn: sqlite3.Connection, user_id: str) -> dict:
    row = conn.execute(
        """
        SELECT user_id, email, name, bio, avatar_url, created_at, updated_at
        FROM user_profiles
        WHERE user_id = ?
        """,
        (user_id,),
    ).fetchone()

    if not row:
        now = _now()
        return {
            "user_id": user_id,
            "email": None,
            "name": None,
            "bio": None,
            "avatar_url": None,
            "created_at": now,
            "updated_at": now,
        }

    return dict(row)


def _are_accepted_friends(
    conn: sqlite3.Connection,
    first_user_id: str,
    second_user_id: str,
) -> bool:
    row = conn.execute(
        """
        SELECT id
        FROM friendships
        WHERE status = 'accepted'
          AND (
            (requester_user_id = ? AND receiver_user_id = ?)
            OR (requester_user_id = ? AND receiver_user_id = ?)
          )
        LIMIT 1
        """,
        (first_user_id, second_user_id, second_user_id, first_user_id),
    ).fetchone()

    return row is not None


def _get_onboarding(conn: sqlite3.Connection, user_id: str) -> dict:
    row = conn.execute(
        """
        SELECT *
        FROM user_onboarding_preferences
        WHERE user_id = ?
        """,
        (user_id,),
    ).fetchone()

    if not row:
        return {
            "favorite_genres": [],
            "disliked_genres": [],
            "preferred_moods": [],
            "preferred_pacing": [],
            "preferred_decades": [],
            "favorite_movies": [],
            "disliked_movies": [],
        }

    return {
        "favorite_genres": _json_list(row["favorite_genres"]),
        "disliked_genres": _json_list(row["disliked_genres"]),
        "preferred_moods": _json_list(row["preferred_moods"]),
        "preferred_pacing": _json_list(row["preferred_pacing"]),
        "preferred_decades": _json_list(row["preferred_decades"]),
        "favorite_movies": _json_list(row["favorite_movies"]),
        "disliked_movies": _json_list(row["disliked_movies"]),
    }


def _get_disliked_movie_titles(conn: sqlite3.Connection, user_id: str) -> list[str]:
    if not _table_exists(conn, "user_disliked_movies"):
        return []

    rows = conn.execute(
        """
        SELECT title
        FROM user_disliked_movies
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT 20
        """,
        (user_id,),
    ).fetchall()

    return [row["title"] for row in rows if row["title"]]


def _get_watchlist(conn: sqlite3.Connection, user_id: str) -> list[dict]:
    if not _table_exists(conn, "user_watchlist_movies"):
        return []

    rows = conn.execute(
        """
        SELECT movie_id, title, status, created_at, updated_at
        FROM user_watchlist_movies
        WHERE user_id = ?
        ORDER BY updated_at DESC
        """,
        (user_id,),
    ).fetchall()

    return [
        {
            "movie_id": int(row["movie_id"]),
            "title": row["title"],
            "status": row["status"],
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
        }
        for row in rows
    ]


def _get_personal_lists_with_movies(conn: sqlite3.Connection, user_id: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT id, name, created_at
        FROM user_lists
        WHERE user_id = ?
        ORDER BY created_at DESC
        """,
        (user_id,),
    ).fetchall()

    lists = []
    for row in rows:
        movies = conn.execute(
            """
            SELECT movie_id, title
            FROM user_list_movies
            WHERE list_id = ?
            ORDER BY position ASC
            """,
            (row["id"],),
        ).fetchall()

        lists.append({
            "id": row["id"],
            "name": row["name"],
            "createdAt": row["created_at"],
            "movies": [
                {
                    "movie_id": int(movie["movie_id"]),
                    "title": movie["title"],
                }
                for movie in movies
            ],
        })

    return lists


def _get_personal_lists(conn: sqlite3.Connection, user_id: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT id, name, created_at
        FROM user_lists
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT 6
        """,
        (user_id,),
    ).fetchall()

    return [
        {
            "id": row["id"],
            "name": row["name"],
            "createdAt": row["created_at"],
        }
        for row in rows
    ]


def _get_shared_lists(conn: sqlite3.Connection, user_id: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT sl.id, sl.name, sl.owner_user_id, sl.created_at, sl.updated_at
        FROM shared_lists sl
        JOIN shared_list_members m
          ON sl.id = m.shared_list_id
        WHERE m.user_id = ?
        ORDER BY sl.updated_at DESC
        LIMIT 6
        """,
        (user_id,),
    ).fetchall()

    return [
        {
            "id": row["id"],
            "name": row["name"],
            "owner_user_id": row["owner_user_id"],
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
        }
        for row in rows
    ]


def _get_friends(conn: sqlite3.Connection, user_id: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT *
        FROM friendships
        WHERE status = 'accepted'
          AND (requester_user_id = ? OR receiver_user_id = ?)
        ORDER BY updated_at DESC
        LIMIT 8
        """,
        (user_id, user_id),
    ).fetchall()

    friends = []

    for row in rows:
        other_user_id = (
            row["receiver_user_id"]
            if row["requester_user_id"] == user_id
            else row["requester_user_id"]
        )

        profile = conn.execute(
            """
            SELECT user_id, email, name
            FROM user_profiles
            WHERE user_id = ?
            """,
            (other_user_id,),
        ).fetchone()

        friends.append({
            "friendship_id": row["id"],
            "user_id": other_user_id,
            "name": profile["name"] if profile else None,
            "email": profile["email"] if profile else None,
        })

    return friends


def _collect_movie_ids(conn: sqlite3.Connection, user_id: str) -> list[int]:
    personal = conn.execute(
        """
        SELECT lm.movie_id
        FROM user_list_movies lm
        JOIN user_lists ul
          ON lm.list_id = ul.id
        WHERE ul.user_id = ?
        """,
        (user_id,),
    ).fetchall()

    shared = conn.execute(
        """
        SELECT sm.movie_id
        FROM shared_list_movies sm
        JOIN shared_list_members mem
          ON sm.shared_list_id = mem.shared_list_id
        WHERE mem.user_id = ?
        """,
        (user_id,),
    ).fetchall()

    return [int(row["movie_id"]) for row in personal + shared]


def _collect_saved_movies(conn: sqlite3.Connection, user_id: str) -> list[dict]:
    personal = conn.execute(
        """
        SELECT lm.movie_id, lm.title
        FROM user_list_movies lm
        JOIN user_lists ul
          ON lm.list_id = ul.id
        WHERE ul.user_id = ?
        """,
        (user_id,),
    ).fetchall()

    shared = conn.execute(
        """
        SELECT sm.movie_id, sm.title
        FROM shared_list_movies sm
        JOIN shared_list_members mem
          ON sm.shared_list_id = mem.shared_list_id
        WHERE mem.user_id = ?
        """,
        (user_id,),
    ).fetchall()

    movies_by_id = {}
    for row in personal + shared:
        movie_id = int(row["movie_id"])
        if movie_id not in movies_by_id:
            movies_by_id[movie_id] = {
                "movie_id": movie_id,
                "title": row["title"],
            }

    return list(movies_by_id.values())


def _overlap(left: list[str], right: list[str], limit: int = 8) -> list[str]:
    right_values = {item.lower() for item in right}
    matches = []

    for item in left:
        if item.lower() in right_values and item not in matches:
            matches.append(item)

        if len(matches) >= limit:
            break

    return matches


def _jaccard(left: list[str], right: list[str]) -> float:
    left_values = {item.lower() for item in left}
    right_values = {item.lower() for item in right}

    if not left_values and not right_values:
        return 0.0

    return len(left_values & right_values) / max(len(left_values | right_values), 1)


def _unique_ints(values: list[int], limit: int) -> list[int]:
    unique = []
    seen = set()

    for value in values:
        movie_id = int(value)
        if movie_id in seen:
            continue

        seen.add(movie_id)
        unique.append(movie_id)

        if len(unique) >= limit:
            break

    return unique


def _build_taste_match(
    conn: sqlite3.Connection,
    viewer_user_id: str,
    friend_user_id: str,
    viewer_movie_ids: list[int],
    friend_movie_ids: list[int],
    viewer_taste_summary: dict,
    friend_taste_summary: dict,
) -> dict:
    overlap = {
        "genres": _overlap(
            viewer_taste_summary.get("favorite_genres", []),
            friend_taste_summary.get("favorite_genres", []),
        ),
        "actors": _overlap(
            viewer_taste_summary.get("favorite_actors", []),
            friend_taste_summary.get("favorite_actors", []),
        ),
        "directors": _overlap(
            viewer_taste_summary.get("favorite_directors", []),
            friend_taste_summary.get("favorite_directors", []),
        ),
        "keywords": _overlap(
            viewer_taste_summary.get("favorite_keywords", []),
            friend_taste_summary.get("favorite_keywords", []),
        ),
    }

    weighted_score = (
        0.35
        * _jaccard(
            viewer_taste_summary.get("favorite_genres", []),
            friend_taste_summary.get("favorite_genres", []),
        )
        + 0.25
        * _jaccard(
            viewer_taste_summary.get("favorite_actors", []),
            friend_taste_summary.get("favorite_actors", []),
        )
        + 0.25
        * _jaccard(
            viewer_taste_summary.get("favorite_directors", []),
            friend_taste_summary.get("favorite_directors", []),
        )
        + 0.15
        * _jaccard(
            viewer_taste_summary.get("favorite_keywords", []),
            friend_taste_summary.get("favorite_keywords", []),
        )
    )

    viewer_saved_movies = _collect_saved_movies(conn, viewer_user_id)
    friend_saved_movies = _collect_saved_movies(conn, friend_user_id)
    viewer_saved_by_id = {movie["movie_id"]: movie for movie in viewer_saved_movies}
    friend_saved_by_id = {movie["movie_id"]: movie for movie in friend_saved_movies}
    shared_movie_ids = [
        movie_id for movie_id in viewer_saved_by_id if movie_id in friend_saved_by_id
    ]

    curated_seed_movie_ids = _unique_ints(
        shared_movie_ids + viewer_movie_ids[:8] + friend_movie_ids[:8],
        limit=12,
    )

    return {
        "similarity_score": round(float(weighted_score), 4),
        "overlap": overlap,
        "shared_saved_movies": [
            viewer_saved_by_id[movie_id] for movie_id in shared_movie_ids[:8]
        ],
        "curated_seed_movie_ids": curated_seed_movie_ids,
    }


def _build_taste_summary(conn: sqlite3.Connection, movie_ids: list[int]) -> dict:
    if not movie_ids:
        return {
            "favorite_genres": [],
            "favorite_directors": [],
            "favorite_actors": [],
            "favorite_keywords": [],
        }

    placeholders = ",".join(["?"] * len(set(movie_ids)))

    rows = conn.execute(
        f"""
        SELECT genre, director, star, tmdb_genres, tmdb_directors, tmdb_cast_top5, tmdb_keywords
        FROM movies
        WHERE movie_id IN ({placeholders})
        """,
        tuple(set(movie_ids)),
    ).fetchall()

    genre_counter = Counter()
    director_counter = Counter()
    actor_counter = Counter()
    keyword_counter = Counter()

    for row in rows:
        genre_counter.update(_split_pipe(row["tmdb_genres"]) or _split_pipe(row["genre"]))
        director_counter.update(_split_pipe(row["tmdb_directors"]) or _split_pipe(row["director"]))
        actor_counter.update(_split_pipe(row["tmdb_cast_top5"]) or _split_pipe(row["star"]))
        keyword_counter.update(_split_pipe(row["tmdb_keywords"]))

    return {
        "favorite_genres": [item for item, _ in genre_counter.most_common(6)],
        "favorite_directors": [item for item, _ in director_counter.most_common(6)],
        "favorite_actors": [item for item, _ in actor_counter.most_common(6)],
        "favorite_keywords": [item for item, _ in keyword_counter.most_common(8)],
    }


def _get_recent_activity(conn: sqlite3.Connection, user_id: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT id, name, created_at
        FROM user_lists
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT 5
        """,
        (user_id,),
    ).fetchall()

    return [
        {
            "type": "created_list",
            "label": f"Created list: {row['name']}",
            "createdAt": row["created_at"],
        }
        for row in rows
    ]
