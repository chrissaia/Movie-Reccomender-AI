from __future__ import annotations

import sqlite3
import pandas as pd


def init_schema(conn: sqlite3.Connection, statements: list[str]) -> None:
    for stmt in statements:
        conn.execute(stmt)
    conn.commit()


def replace_movies(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    required_cols = [
        "movie_id",
        "name",
        "year",
        "genre",
        "director",
        "writer",
        "star",
        "country",
        "rating",
        "company",
        "score",
        "votes",
        "budget",
        "gross",
        "runtime",
        "tmdb_found",
        "tmdb_id",
        "tmdb_title",
        "tmdb_original_title",
        "tmdb_release_date",
        "tmdb_overview",
        "tmdb_genres",
        "tmdb_keywords",
        "tmdb_cast_top5",
        "tmdb_directors",
        "tmdb_writers",
        "tmdb_popularity",
        "tmdb_vote_average",
        "tmdb_vote_count",
        "tmdb_runtime",
        "tmdb_original_language",
        "tmdb_production_companies",
        "tmdb_production_countries",
        "tmdb_spoken_languages",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"movies dataframe missing required columns: {missing}")

    conn.execute("DELETE FROM movies;")

    rows = df[required_cols].itertuples(index=False, name=None)
    conn.executemany(
        """
        INSERT INTO movies (
            movie_id, name, year, genre, director, writer, star, country,
            rating, company, score, votes, budget, gross, runtime,
            tmdb_found, tmdb_id, tmdb_title, tmdb_original_title,
            tmdb_release_date, tmdb_overview, tmdb_genres, tmdb_keywords,
            tmdb_cast_top5, tmdb_directors, tmdb_writers, tmdb_popularity,
            tmdb_vote_average, tmdb_vote_count, tmdb_runtime,
            tmdb_original_language, tmdb_production_companies,
            tmdb_production_countries, tmdb_spoken_languages
        )
        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        rows,
    )
    conn.commit()


def replace_neighbors(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    required_cols = [
        "source_movie_id",
        "neighbor_movie_id",
        "rank",
        "cosine_score",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"neighbors dataframe missing required columns: {missing}")

    conn.execute("DELETE FROM movie_neighbors;")

    rows = df[required_cols].itertuples(index=False, name=None)
    conn.executemany(
        """
        INSERT INTO movie_neighbors (
            source_movie_id, neighbor_movie_id, rank, cosine_score
        )
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


def replace_movie_search_index(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS movie_search_fts USING fts5(
            movie_id UNINDEXED,
            title,
            people,
            genres,
            keywords,
            overview,
            search_text
        );
        """
    )
    conn.execute("DELETE FROM movie_search_fts;")
    conn.execute(
        """
        INSERT INTO movie_search_fts (
            movie_id,
            title,
            people,
            genres,
            keywords,
            overview,
            search_text
        )
        SELECT
            movie_id,
            COALESCE(name, '') || ' ' || COALESCE(tmdb_title, ''),
            COALESCE(director, '') || ' ' ||
                COALESCE(writer, '') || ' ' ||
                COALESCE(star, '') || ' ' ||
                COALESCE(tmdb_cast_top5, '') || ' ' ||
                COALESCE(tmdb_directors, '') || ' ' ||
                COALESCE(tmdb_writers, ''),
            COALESCE(genre, '') || ' ' || COALESCE(tmdb_genres, ''),
            COALESCE(tmdb_keywords, '') || ' ' ||
                COALESCE(rating, '') || ' ' ||
                COALESCE(country, '') || ' ' ||
                COALESCE(company, ''),
            COALESCE(tmdb_overview, ''),
            COALESCE(name, '') || ' ' ||
                COALESCE(tmdb_title, '') || ' ' ||
                COALESCE(genre, '') || ' ' ||
                COALESCE(tmdb_genres, '') || ' ' ||
                COALESCE(director, '') || ' ' ||
                COALESCE(writer, '') || ' ' ||
                COALESCE(star, '') || ' ' ||
                COALESCE(tmdb_cast_top5, '') || ' ' ||
                COALESCE(tmdb_directors, '') || ' ' ||
                COALESCE(tmdb_keywords, '') || ' ' ||
                COALESCE(tmdb_overview, '')
        FROM movies;
        """
    )
    conn.commit()


def get_neighbors_for_movie(
    conn: sqlite3.Connection,
    source_movie_id: int,
    top_k: int = 10,
) -> pd.DataFrame:
    query = """
    SELECT
        mn.source_movie_id,
        mn.neighbor_movie_id,
        mn.rank,
        mn.cosine_score,
        m.name AS neighbor_name,
        m.year AS neighbor_year,
        m.tmdb_genres,
        m.tmdb_keywords,
        m.tmdb_vote_average,
        m.tmdb_vote_count
    FROM movie_neighbors mn
    JOIN movies m
      ON mn.neighbor_movie_id = m.movie_id
    WHERE mn.source_movie_id = ?
    ORDER BY mn.rank ASC
    LIMIT ?
    """
    return pd.read_sql_query(query, conn, params=(source_movie_id, top_k))
