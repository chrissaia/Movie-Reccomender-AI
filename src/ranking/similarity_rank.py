from __future__ import annotations
import os

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

def _build_movie_feature_matrix(init_df):
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

    cat_cols = ["genre", "director", "writer", "star", "country", "rating", "company"]
    num_cols = ["year", "score", "votes", "budget", "gross", "runtime"]

    for col in cat_cols:
        df[col] = df[col].fillna("unknown").astype(str).str.lower()

    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    # one-hot encode categoricals
    X_cat = pd.get_dummies(df[cat_cols], dummy_na=False)

    # scale numerics
    scaler = MinMaxScaler()
    X_num = pd.DataFrame(
        scaler.fit_transform(df[num_cols]),
        columns=num_cols,
        index=df.index
    )

    # combine
    X = pd.concat([X_cat, X_num], axis=1)

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
        similarity_df = pd.read_csv(output_path, header=None)
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
        columns=df_model.index
    )

    return similarity_df

