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
    "tmdb_genres",
    "tmdb_keywords",
    "tmdb_cast_top5",
    "tmdb_directors",
    "tmdb_writers",
    "tmdb_overview",
    "tmdb_popularity",
    "tmdb_vote_average",
    "tmdb_vote_count",
    "tmdb_runtime",
    "tmdb_original_language",
    "tmdb_production_companies",
    "tmdb_production_countries",
    "tmdb_spoken_languages",
}


def _safe_string(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "" or pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None or value == "" or pd.isna(value):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _split_pipe(value: Any) -> set[str]:
    text = _safe_string(value)
    if not text:
        return set()
    return {part.strip() for part in text.split("|") if part.strip()}


def _jaccard_sets(a: set[str], b: set[str]) -> float:
    """
    Compute Jaccard similarity between two text values split on whitespace.
    Returns a value in [0, 1].
    """
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _overlap_count(a: set[str], b: set[str]) -> int:
    return len(a & b)


def jaccard_text(a: Any, b: Any) -> float:
    """
    Compute Jaccard similarity between two text values split on whitespace.
    Returns a value in [0, 1].
    """
    sa = set(_safe_string(a).split())
    sb = set(_safe_string(b).split())
    return _jaccard_sets(sa, sb)


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
    genres_a = _split_pipe(movie_a["tmdb_genres"])
    genres_b = _split_pipe(movie_b["tmdb_genres"])

    keywords_a = _split_pipe(movie_a["tmdb_keywords"])
    keywords_b = _split_pipe(movie_b["tmdb_keywords"])

    cast_a = _split_pipe(movie_a["tmdb_cast_top5"])
    cast_b = _split_pipe(movie_b["tmdb_cast_top5"])

    directors_a = _split_pipe(movie_a["tmdb_directors"])
    directors_b = _split_pipe(movie_b["tmdb_directors"])

    writers_a = _split_pipe(movie_a["tmdb_writers"])
    writers_b = _split_pipe(movie_b["tmdb_writers"])

    countries_a = _split_pipe(movie_a["tmdb_production_countries"])
    countries_b = _split_pipe(movie_b["tmdb_production_countries"])

    languages_a = _split_pipe(movie_a["tmdb_spoken_languages"])
    languages_b = _split_pipe(movie_b["tmdb_spoken_languages"])

    companies_a = _split_pipe(movie_a["tmdb_production_companies"])
    companies_b = _split_pipe(movie_b["tmdb_production_companies"])

    year_a = _safe_int(movie_a["year"])
    year_b = _safe_int(movie_b["year"])

    return {
        "title_overlap": jaccard_text(movie_a["name"], movie_b["name"]),
        "overview_overlap": jaccard_text(movie_a["tmdb_overview"], movie_b["tmdb_overview"]),
        "year_diff": abs(year_a - year_b),
        "same_decade": float(year_a // 10 == year_b // 10),
        "score_diff": abs(_safe_float(movie_a["score"]) - _safe_float(movie_b["score"])),
        "votes_rel_diff": relative_diff(movie_a["votes"], movie_b["votes"]),
        "budget_rel_diff": relative_diff(movie_a["budget"], movie_b["budget"]),
        "gross_rel_diff": relative_diff(movie_a["gross"], movie_b["gross"]),
        "runtime_rel_diff": relative_diff(
            movie_a["tmdb_runtime"] if pd.notna(movie_a["tmdb_runtime"]) else movie_a["runtime"],
            movie_b["tmdb_runtime"] if pd.notna(movie_b["tmdb_runtime"]) else movie_b["runtime"],
        ),
        "tmdb_popularity_rel_diff": relative_diff(movie_a["tmdb_popularity"], movie_b["tmdb_popularity"]),
        "tmdb_vote_average_diff": abs(
            _safe_float(movie_a["tmdb_vote_average"]) - _safe_float(movie_b["tmdb_vote_average"])
        ),
        "tmdb_vote_count_rel_diff": relative_diff(movie_a["tmdb_vote_count"], movie_b["tmdb_vote_count"]),
        "genre_overlap_count": _overlap_count(genres_a, genres_b),
        "genre_jaccard": _jaccard_sets(genres_a, genres_b),
        "keyword_overlap_count": _overlap_count(keywords_a, keywords_b),
        "keyword_jaccard": _jaccard_sets(keywords_a, keywords_b),
        "cast_overlap_count": _overlap_count(cast_a, cast_b),
        "cast_jaccard": _jaccard_sets(cast_a, cast_b),
        "director_overlap": _overlap_count(directors_a, directors_b),
        "writer_overlap": _overlap_count(writers_a, writers_b),
        "country_overlap": _overlap_count(countries_a, countries_b),
        "language_overlap": _overlap_count(languages_a, languages_b),
        "company_overlap": _overlap_count(companies_a, companies_b),
        "original_language_match": float(
            _safe_string(movie_a["tmdb_original_language"]) == _safe_string(movie_b["tmdb_original_language"])
        ),
        "rating_match": float(_safe_string(movie_a["rating"]) == _safe_string(movie_b["rating"])),
        "legacy_genre_match": float(_safe_string(movie_a["genre"]) == _safe_string(movie_b["genre"])),
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
    if n_rows != n_cols:
        raise ValueError("similarity_matrix must be square")

    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")

    if similarity_matrix.isna().any().any():
        raise ValueError("similarity_matrix contains NaN values")

    if len(df) != n_rows:
        raise ValueError("df and similarity_matrix must be aligned")







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

    movie_rows = [df.iloc[i] for i in range(n)]
    sim_values = similarity_matrix.to_numpy()

    for i in range(n):
        sims = sim_values[i]
        candidate_idx = np.argsort(sims)[::-1]

        kept = 0
        for j in candidate_idx:
            if j == i:
                continue

            sim = float(sims[j])

            if min_similarity is not None and sim < min_similarity:
                continue

            X_rows.append(compute_features(movie_rows[i], movie_rows[j]))
            y_rows.append(make_label(sim))
            qid_rows.append(i)

            kept += 1
            if kept >= top_k:
                break

    X = pd.DataFrame(X_rows)
    y = pd.DataFrame(y_rows, dtype=np.int32)
    qid = pd.DataFrame(qid_rows, dtype=np.int32)

    return X, y, qid

