from __future__ import annotations

import sqlite3

from src.serving.query_intent import QueryIntent
from src.serving.semantic_discovery import semantic_discover_movies


CREATE_MOVIES = """
CREATE TABLE movies (
    movie_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    year INTEGER,
    genre TEXT,
    director TEXT,
    writer TEXT,
    star TEXT,
    country TEXT,
    rating TEXT,
    company TEXT,
    score REAL,
    votes REAL,
    budget REAL,
    gross REAL,
    runtime REAL,
    tmdb_found INTEGER,
    tmdb_id INTEGER,
    tmdb_title TEXT,
    tmdb_original_title TEXT,
    tmdb_release_date TEXT,
    tmdb_overview TEXT,
    tmdb_genres TEXT,
    tmdb_keywords TEXT,
    tmdb_cast_top5 TEXT,
    tmdb_directors TEXT,
    tmdb_writers TEXT,
    tmdb_popularity REAL,
    tmdb_vote_average REAL,
    tmdb_vote_count REAL,
    tmdb_runtime REAL,
    tmdb_original_language TEXT,
    tmdb_production_companies TEXT,
    tmdb_production_countries TEXT,
    tmdb_spoken_languages TEXT
);
"""


def make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(CREATE_MOVIES)

    rows = [
        (
            1,
            "The Dark Knight",
            2008,
            "Action",
            "Christopher Nolan",
            "Jonathan Nolan",
            "Christian Bale",
            "United States",
            "PG-13",
            "Warner Bros.",
            9.0,
            2000000,
            None,
            None,
            152,
            1,
            155,
            "The Dark Knight",
            "The Dark Knight",
            "2008-07-18",
            "Batman faces a frightening criminal mastermind in a dark city.",
            "Drama|Crime|Action|Thriller",
            "superhero|dark|scary|crime|chaos",
            "Christian Bale|Heath Ledger|Aaron Eckhart",
            "Christopher Nolan",
            "Jonathan Nolan",
            90,
            8.5,
            30000,
            152,
            "en",
            "Warner Bros.",
            "United States",
            "English",
        ),
        (
            2,
            "Scary Movie",
            2000,
            "Comedy",
            "Keenen Ivory Wayans",
            "Shawn Wayans",
            "Anna Faris",
            "United States",
            "R",
            "Dimension Films",
            6.3,
            200000,
            None,
            None,
            88,
            1,
            4247,
            "Scary Movie",
            "Scary Movie",
            "2000-07-07",
            "A funny parody of scary horror movies.",
            "Comedy|Horror",
            "funny|scary|parody|horror",
            "Anna Faris|Regina Hall",
            "Keenen Ivory Wayans",
            "Shawn Wayans",
            55,
            6.4,
            9000,
            88,
            "en",
            "Dimension Films",
            "United States",
            "English",
        ),
        (
            3,
            "Moon",
            2009,
            "Drama",
            "Duncan Jones",
            "Nathan Parker",
            "Sam Rockwell",
            "United Kingdom",
            "R",
            "Stage 6 Films",
            7.8,
            350000,
            None,
            None,
            97,
            1,
            17431,
            "Moon",
            "Moon",
            "2009-06-12",
            "A lonely astronaut nears the end of his time on a lunar base.",
            "Science Fiction|Drama",
            "space|loneliness|astronaut|moon",
            "Sam Rockwell",
            "Duncan Jones",
            "Nathan Parker",
            40,
            7.6,
            8000,
            97,
            "en",
            "Stage 6 Films",
            "United Kingdom",
            "English",
        ),
        (
            4,
            "Cry Wolf",
            2005,
            "Thriller",
            "Jeff Wadlow",
            "Becky Mode",
            "Lindy Booth",
            "United States",
            "PG-13",
            "Universal Pictures",
            5.8,
            1000,
            None,
            None,
            90,
            1,
            12345,
            "Cry Wolf",
            "Cry Wolf",
            "2005-07-08",
            "Teenagers tell a heartbreaking lie that leads to grief and sorrow.",
            "Drama|Thriller",
            "heartbreaking|grief|sorrow",
            "Lindy Booth",
            "Jeff Wadlow",
            "Becky Mode",
            10,
            5.8,
            1000,
            90,
            "en",
            "Universal Pictures",
            "United States",
            "English",
        ),
    ]

    conn.executemany(
        """
        INSERT INTO movies VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        rows,
    )
    conn.commit()
    return conn


def test_person_matches_are_strong_but_not_required() -> None:
    conn = make_conn()
    try:
        results = semantic_discover_movies(
            conn,
            'christopher nolan funny scary !!!! " OR 1=1 --',
            3,
        )
    finally:
        conn.close()

    assert results[0]["title"] == "The Dark Knight"
    assert "Christopher Nolan" in results[0]["match_reasons"]


def test_funny_scary_query_finds_horror_comedy() -> None:
    conn = make_conn()
    try:
        results = semantic_discover_movies(conn, "funny and scary movies", 3)
    finally:
        conn.close()

    assert results[0]["title"] == "Scary Movie"
    assert {"Funny", "Scary"}.issubset(set(results[0]["match_reasons"]))


def test_space_loneliness_query_uses_overview_and_keywords() -> None:
    conn = make_conn()
    try:
        results = semantic_discover_movies(conn, "space movie with loneliness", 3)
    finally:
        conn.close()

    assert results[0]["title"] == "Moon"
    assert "Space" in results[0]["match_reasons"]


def test_cry_is_treated_as_a_sad_drama_vibe_not_a_title_match() -> None:
    conn = make_conn()
    try:
        results = semantic_discover_movies(conn, "cry", 3)
    finally:
        conn.close()

    assert results[0]["title"] == "Cry Wolf"
    assert "Sad" in results[0]["match_reasons"]
    assert "Drama" in results[0]["match_reasons"]


def test_laugh_is_treated_as_a_comedy_vibe() -> None:
    conn = make_conn()
    try:
        results = semantic_discover_movies(conn, "laugh", 3)
    finally:
        conn.close()

    assert results[0]["title"] == "Scary Movie"
    assert "Comedy" in results[0]["match_reasons"]


def test_llm_intent_can_add_people_to_plain_language(monkeypatch) -> None:
    conn = make_conn()

    def fake_intent(_: str) -> QueryIntent:
        return QueryIntent(
            people=["Christopher Nolan"],
            genres=["thriller"],
            moods=["scary"],
            keywords=["dark"],
            query_rewrite="dark scary thriller directed by Christopher Nolan",
        )

    monkeypatch.setattr(
        "src.serving.semantic_discovery.parse_query_intent",
        fake_intent,
    )

    try:
        results = semantic_discover_movies(conn, "something dark and scary", 3)
    finally:
        conn.close()

    assert results[0]["title"] == "The Dark Knight"
    assert "Christopher Nolan" in results[0]["match_reasons"]
