from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def _validate_inputs(
    movies_liked: Sequence[str],
    similarity_matrix: np.ndarray,
    movie_names: Sequence[str],
) -> None:
    if not movies_liked:
        raise ValueError("movies_liked must not be empty")

    if len(movie_names) == 0:
        raise ValueError("movie_names must not be empty")

    if similarity_matrix.ndim != 2:
        raise ValueError("similarity_matrix must be 2D")

    if similarity_matrix.shape[0] - similarity_matrix.shape[1] > 1:
        raise ValueError("similarity_matrix must be square")

    if similarity_matrix.shape[0] != len(movie_names):
        raise ValueError("similarity_matrix shape must match number of movie_names")

    missing = [movie for movie in movies_liked if movie not in movie_names]
    if missing:
        raise ValueError(f"Movies not found in movie_names: {missing}")


def _normalize_movie_names(movie_names: Sequence[str] | pd.Series | pd.DataFrame) -> list[str]:
    if isinstance(movie_names, pd.DataFrame):
        values = movie_names.iloc[:, 0].tolist()
    elif isinstance(movie_names, pd.Series):
        values = movie_names.tolist()
    else:
        values = list(movie_names)

    return [str(v).strip() for v in values]


def _get_movie_indices(movies_liked: Sequence[str], movie_names: Sequence[str]) -> list[int]:
    name_to_idx = {name: idx for idx, name in enumerate(movie_names)}
    return [name_to_idx[movie] for movie in movies_liked]


def _format_recommendations(
    similarities: dict[int, float],
    movie_names: Sequence[str],
    movies_liked: Sequence[str],
) -> list[str]:
    liked_set = set(movies_liked)
    sorted_items = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    rec_list = []
    for movie_idx, score in sorted_items:
        movie_name = movie_names[movie_idx]
        if movie_name not in liked_set:
            rec_list.append(f"{movie_name}...{round(score, 2)}% match")

    return rec_list


def individual_recommend(
    movies_liked: Sequence[str],
    similarity_matrix: pd.DataFrame | np.ndarray,
    movie_names: Sequence[str] | pd.Series | pd.DataFrame,
    per_movie_top_k: int = 3,
) -> list[str]:
    """
    Recommend movies by taking the top-k neighbors of each liked movie individually.

    For each liked movie:
    - find its top similar movies
    - combine all candidates
    - keep the best similarity score seen for each candidate

    Parameters
    ----------
    movies_liked : Sequence[str]
        List of liked movie titles
    similarity_matrix : pd.DataFrame
        NxN similarity matrix
    movie_names : list
        Movie names aligned to similarity_matrix rows
    per_movie_top_k : int
        Number of neighbors to take per liked movie

    Returns
    -------
    list[str]
        Formatted recommendation strings
    """
    movie_names = _normalize_movie_names(movie_names)
    similarity_matrix = np.asarray(similarity_matrix)

    _validate_inputs(movies_liked, similarity_matrix, movie_names)

    if per_movie_top_k <= 0:
        raise ValueError("per_movie_top_k must be > 0")

    indices = _get_movie_indices(movies_liked, movie_names)
    similarities: dict[int, float] = {}

    for movie_idx in indices:
        sims = similarity_matrix[movie_idx]
        candidate_idx = np.argsort(sims)[::-1]
        candidate_idx = [idx for idx in candidate_idx if idx != movie_idx][:per_movie_top_k]

        for rec_idx in candidate_idx:
            score = float(round(100 * sims[rec_idx], 2))
            similarities[rec_idx] = max(similarities.get(rec_idx, 0.0), score)

    return _format_recommendations(similarities, movie_names, movies_liked)


def combined_recommend(
    movies_liked: Sequence[str],
    similarity_matrix: pd.DataFrame | np.ndarray,
    movie_names: Sequence[str] | pd.Series | pd.DataFrame,
    top_k: int = 10,
) -> list[str]:
    """
    Recommend movies by averaging similarity rows across all liked movies.

    This creates a group preference profile, then returns the top-k matches.

    Parameters
    ----------
    movies_liked : Sequence[str]
        List of liked movie titles
    similarity_matrix : pd.DataFrame
        NxN similarity matrix
    movie_names : pd.DataFrame
        Movie names aligned to similarity_matrix rows
    top_k : int
        Number of recommendations to return

    Returns
    -------
    list[str]
        Formatted recommendation strings
    """
    movie_names = _normalize_movie_names(movie_names)
    similarity_matrix = np.asarray(similarity_matrix)

    _validate_inputs(movies_liked, similarity_matrix, movie_names)

    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    indices = _get_movie_indices(movies_liked, movie_names)
    similarity_row = np.mean(similarity_matrix[indices], axis=0)

    liked_set = set(indices)
    candidate_idx = np.argsort(similarity_row)[::-1]
    candidate_idx = [idx for idx in candidate_idx if idx not in liked_set][:top_k]

    similarities = {
        idx: float(round(100 * similarity_row[idx], 2))
        for idx in candidate_idx
    }

    return _format_recommendations(similarities, movie_names, movies_liked)