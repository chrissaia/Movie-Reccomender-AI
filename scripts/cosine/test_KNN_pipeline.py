#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pandas as pd

from src.data.export import export_df
from src.data.load import load_data
from src.data.preprocess import preprocess_data
from src.cosine.build_features import build_features
from src.cosine.similarity_cosine import load_or_compute_similarity
from src.cosine.neighbors import build_topk_neighbors
from src.db.sqlite import get_connection
from src.db.schema import ALL_SCHEMA_STATEMENTS
from src.db.repository import (
    init_schema,
    replace_neighbors,
    replace_movies,
)
from src.utils.paths import (
    RAW_MOVIES_ENRICHED_PATH,
    COSINE_SIMILARITY_PATH,
    SQLITE_DB_PATH,
    MOVIE_NAMES_PATH,
    COSINE_FEATURES_PATH,
)

OVERWRITE = True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build movie neighbor table in SQLite")
    p.add_argument("--input", type=str, default=str(RAW_MOVIES_ENRICHED_PATH))
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--force_recompute", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading raw data...")
    df = load_data(args.input)

    print("Preprocessing...")
    df = preprocess_data(df)

    print("Building cosine features...")
    df_features, movie_names = build_features(df)

    print("Rebuilding aligned movies metadata...")
    movies_df = df.copy()
    movies_df.columns = movies_df.columns.str.strip()

    numeric_votes = pd.to_numeric(movies_df["votes"], errors="coerce").fillna(0)
    movies_df = movies_df.loc[numeric_votes > 5000].copy()
    movies_df = movies_df.reset_index(drop=True)

    movies_df = movies_df.loc[
        :,
        [
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
        ],
    ].copy()

    movies_df.insert(0, "movie_id", range(len(movies_df)))

    print("Computing/loading cosine similarity...")
    sim = load_or_compute_similarity(
        df_features=df_features,
        output_path=COSINE_SIMILARITY_PATH,
        force_recompute=args.force_recompute,
    )

    print("Building top-k neighbor table...")
    neighbors_df = build_topk_neighbors(similarity_matrix=sim, top_k=args.top_k)

    print("Writing to SQLite...")
    conn = get_connection(SQLITE_DB_PATH)
    try:
        init_schema(conn, ALL_SCHEMA_STATEMENTS)
        conn.execute("DELETE FROM movie_neighbors;")
        conn.execute("DELETE FROM movies;")

        replace_movies(conn, movies_df)
        replace_neighbors(conn, neighbors_df)
    finally:
        conn.close()

    print(f"Saved {len(movies_df)} movies")
    print(f"Saved {len(neighbors_df)} neighbor rows")
    print(f"Database: {SQLITE_DB_PATH}")

    print("Exporting files...")
    export_df(df_features, COSINE_FEATURES_PATH, overwrite=OVERWRITE, header=True)
    export_df(pd.DataFrame({"movie_name": movie_names}), MOVIE_NAMES_PATH, overwrite=OVERWRITE, header=True)
    export_df(sim, COSINE_SIMILARITY_PATH, overwrite=OVERWRITE, header=True)


if __name__ == "__main__":
    main()