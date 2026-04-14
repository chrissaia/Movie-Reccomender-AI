from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


def _safe_split_pipe(series: pd.Series, prefix: str) -> pd.DataFrame:
    cleaned = series.fillna("").astype(str).str.lower().str.split("|")
    exploded = cleaned.explode().str.strip()
    exploded = exploded[exploded.ne("")]

    if exploded.empty:
        return pd.DataFrame(index=series.index)

    dummies = pd.get_dummies(exploded, prefix=prefix, dtype=float)
    dummies = dummies.groupby(level=0).max()
    return dummies.reindex(series.index, fill_value=0.0)


def _build_movie_feature_matrix(init_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build and clean the dataframe for similarity computation.

    Parameters
    ----------
    init_df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        cleaned dataframe
    pd.DataFrame
        dataframe with one-hot, normalized and scaled values
    """
    df = init_df.copy().reset_index(drop=True)

    numeric_base = ["year", "score", "votes", "budget", "gross", "runtime"]
    tmdb_numeric = ["tmdb_popularity", "tmdb_vote_average", "tmdb_vote_count", "tmdb_runtime"]
    categorical_base = ["genre", "director", "writer", "star", "country", "rating", "company"]
    pipe_cols = [
        "tmdb_genres",
        "tmdb_keywords",
        "tmdb_cast_top5",
        "tmdb_directors",
        "tmdb_writers",
        "tmdb_production_companies",
        "tmdb_production_countries",
        "tmdb_spoken_languages",
    ]

    for col in numeric_base + tmdb_numeric:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median() if not df[col].dropna().empty else 0.0)

    for col in categorical_base:
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = df[col].fillna("unknown").astype(str).str.lower()

    for col in pipe_cols:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.lower()

    base_cat = pd.get_dummies(df[categorical_base], dummy_na=False, dtype=float)

    multi_hot_parts = []
    for col in pipe_cols:
        multi_hot_parts.append(_safe_split_pipe(df[col], prefix=col))

    multi_hot = pd.concat(multi_hot_parts, axis=1) if multi_hot_parts else pd.DataFrame(index=df.index)

    scaler = MinMaxScaler()
    X_num = pd.DataFrame(
        scaler.fit_transform(df[numeric_base + tmdb_numeric]),
        columns=numeric_base + tmdb_numeric,
        index=df.index,
    )

    X = pd.concat([base_cat, multi_hot, X_num], axis=1)
    return df, X


def load_or_compute_similarity(
    df_features: pd.DataFrame,
    output_path: str | Path,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Load a similarity matrix from disk if it exists, otherwise compute and save it.

    Parameters
    ----------
    df_features : pd.DataFrame
        Feature dataframe used to build the similarity matrix
    output_path : str | Path
        Where the similarity CSV should live
    round_digits : int
        Number of decimal places to round similarity scores
    force_recompute : bool
        If True, ignore any existing file and rebuild the matrix

    Returns
    -------
    pd.DataFrame
        Similarity matrix as a dataframe
    """
    output_path = Path(output_path)

    if output_path.exists() and not force_recompute:
        print("File already exists. Loading file...")
        similarity_df = pd.read_csv(output_path, index_col=0)
        similarity_df = similarity_df.apply(pd.to_numeric, errors="raise")
        return similarity_df

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Building similarity matrix...")
    df_model, X = _build_movie_feature_matrix(df_features)

    print("Computing cosine similarity...")
    sim_matrix = cosine_similarity(X)

    similarity_df = pd.DataFrame(
        sim_matrix,
        index=df_model.index,
        columns=df_model.index,
    )
    similarity_df.to_csv(output_path, index=True)

    return similarity_df
