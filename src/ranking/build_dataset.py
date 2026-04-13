from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


REQUIRED_MOVIE_COLUMNS = {
    "name",
    "year",
    "score",
    "votes",
    "budget",
    "gross",
    "runtime",
    "director",
    "writer",
    "star",
    "country",
    "company",
    "genre",
    "rating",
}


def _safe_string(value: Any) -> str:
    """Convert a value to a clean lowercase string for text comparison."""
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float safely."""
    if value is None or value == "" or pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """Convert a value to int safely."""
    if value is None or value == "" or pd.isna(value):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def jaccard_text(a: Any, b: Any) -> float:
    """
    Compute Jaccard similarity between two text values split on whitespace.
    Returns a value in [0, 1].
    """
    sa = set(_safe_string(a).split())
    sb = set(_safe_string(b).split())

    union_size = len(sa | sb)
    if union_size == 0:
        return 0.0

    return len(sa & sb) / union_size


def relative_diff(a: Any, b: Any) -> float:
    """
    Calculate relative numeric difference between two values.

    Returns:
        abs(a - b) / max(1.0, max(abs(a), abs(b)))
    """
    a_f = _safe_float(a)
    b_f = _safe_float(b)
    return abs(a_f - b_f) / max(1.0, max(abs(a_f), abs(b_f)))


def compute_features(movie_a: pd.Series, movie_b: pd.Series) -> dict[str, float]:
    """
    Compute pairwise features for two movies.

    Assumes both rows contain the required movie columns.
    """
    return {
        "title_overlap": jaccard_text(movie_a["name"], movie_b["name"]),
        "year_diff": abs(_safe_int(movie_a["year"]) - _safe_int(movie_b["year"])),
        "score_diff": abs(_safe_float(movie_a["score"]) - _safe_float(movie_b["score"])),
        "votes_rel_diff": relative_diff(movie_a["votes"], movie_b["votes"]),
        "budget_rel_diff": relative_diff(movie_a["budget"], movie_b["budget"]),
        "gross_rel_diff": relative_diff(movie_a["gross"], movie_b["gross"]),
        "runtime_rel_diff": relative_diff(movie_a["runtime"], movie_b["runtime"]),
        "director_match": float(_safe_string(movie_a["director"]) == _safe_string(movie_b["director"])),
        "writer_match": float(_safe_string(movie_a["writer"]) == _safe_string(movie_b["writer"])),
        "star_match": float(_safe_string(movie_a["star"]) == _safe_string(movie_b["star"])),
        "country_match": float(_safe_string(movie_a["country"]) == _safe_string(movie_b["country"])),
        "company_match": float(_safe_string(movie_a["company"]) == _safe_string(movie_b["company"])),
        "genre_match": float(_safe_string(movie_a["genre"]) == _safe_string(movie_b["genre"])),
        "rating_match": float(_safe_string(movie_a["rating"]) == _safe_string(movie_b["rating"])),
    }


def make_label(similarity: float) -> int:
    """
    Bucket a similarity score into a relevance label.

    Labels:
        4: similarity >= 0.90
        3: similarity >= 0.80
        2: similarity >= 0.65
        1: similarity >= 0.50
        0: similarity < 0.50
    """
    if similarity >= 0.90:
        return 4
    if similarity >= 0.80:
        return 3
    if similarity >= 0.65:
        return 2
    if similarity >= 0.50:
        return 1
    return 0


def _validate_inputs(df: pd.DataFrame, similarity_matrix: pd.DataFrame, top_k: int) -> None:
    """Validate dataset and similarity matrix inputs."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if not isinstance(similarity_matrix, pd.DataFrame):
        raise TypeError("similarity_matrix must be a pandas DataFrame")

    missing_cols = REQUIRED_MOVIE_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"df is missing required columns: {sorted(missing_cols)}")

    if len(df) == 0:
        raise ValueError("df is empty")

    n_rows, n_cols = similarity_matrix.shape
    if abs(n_rows - n_cols) > 1:
        raise ValueError("similarity_matrix must be square")

    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")

    if similarity_matrix.isna().any().any():
        raise ValueError("similarity_matrix contains NaN values")


def build_dataset(
    df: pd.DataFrame,
    similarity_matrix: pd.DataFrame,
    top_k: int = 50,
    min_similarity: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build a learning-to-rank style dataset from movie pairs.

    For each movie i:
    - rank all candidate movies by similarity
    - skip self-pair
    - keep the top_k most similar candidates
    - compute pairwise features
    - assign a relevance label from the similarity score
    - use i as the query id (qid)

    Parameters
    ----------
    df : pd.DataFrame
        Movie metadata DataFrame.
    similarity_matrix : pd.DataFrame
        Square similarity matrix aligned to df row order.
    top_k : int, default=50
        Number of candidate movies to keep per query movie.
    min_similarity : float | None, default=None
        Optional minimum similarity threshold. If provided, candidates below
        this value are excluded.

    Returns
    -------
    X : pd.DataFrame
        Pairwise feature matrix.
    y : pd.DataFrame
        Relevance labels.
    qid : pd.DataFrame
        Query ids, one per row in X/y.
    """
    _validate_inputs(df, similarity_matrix, top_k)

    n = len(df)
    top_k = min(top_k, max(0, n - 1))

    X_rows: list[dict[str, float]] = []
    y_rows: list[int] = []
    qid_rows: list[int] = []

    # Faster access than repeated iloc lookups inside nested loops
    movie_rows = [df.iloc[i] for i in range(n)]
    sim_values = similarity_matrix.to_numpy()

    for i in range(n):
        sims = sim_values[i]

        # Descending sort by similarity
        candidate_idx = np.argsort(sims)[::-1]

        kept = 0
        for j in candidate_idx:
            if j == i:
                continue

            sim = float(sims[j])

            if min_similarity is not None and sim < min_similarity:
                continue

            features = compute_features(movie_rows[i], movie_rows[j])
            label = make_label(sim)

            X_rows.append(features)
            y_rows.append(label)
            qid_rows.append(i)

            kept += 1
            if kept >= top_k:
                break

    X = pd.DataFrame(X_rows)
    y = pd.DataFrame(y_rows, dtype=np.int32)
    qid = pd.DataFrame(qid_rows, dtype=np.int32)

    return X, y, qid