from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


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

    print("Computing cosine similarity...")
    sim = cosine_similarity(df_features)
    similarity_df = pd.DataFrame(sim)
    similarity_df.to_csv(output_path, index=True)

    return similarity_df