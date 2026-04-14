from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


TEXT_JOIN_TOKEN = " "


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _split_pipe_text(value) -> list[str]:
    text = _clean_text(value)
    if not text:
        return []
    return [part.strip().lower() for part in text.split("|") if part.strip()]


def _normalize_numeric(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce").fillna(0.0)
    max_val = series.max()
    min_val = series.min()

    if max_val == min_val:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)

    return (series - min_val) / (max_val - min_val)


def _bucket_country(country: str) -> str:
    country = _clean_text(country)

    if country in {"united states", "canada"}:
        return "uscan"
    if country in {"united kingdom", "ireland"}:
        return "gb"
    if country in {"australia", "new zealand"}:
        return "oceania"
    if country in {
        "france", "germany", "spain", "italy", "denmark",
        "sweden", "norway", "netherlands", "belgium", "finland",
        "switzerland", "austria"
    }:
        return "eu"
    if country in {"japan", "hong kong", "china", "south korea", "taiwan"}:
        return "east_asia"

    return "other"


def _build_text_blob(row: pd.Series) -> str:
    parts: list[str] = []

    parts.extend(_split_pipe_text(row.get("tmdb_genres", "")))
    parts.extend(_split_pipe_text(row.get("tmdb_keywords", "")))
    parts.extend(_split_pipe_text(row.get("tmdb_cast_top5", "")))
    parts.extend(_split_pipe_text(row.get("tmdb_directors", "")))
    parts.extend(_split_pipe_text(row.get("tmdb_writers", "")))

    overview = _clean_text(row.get("tmdb_overview", ""))
    if overview:
        parts.append(overview)

    # fallback legacy fields if tmdb fields are missing
    for col in ["genre", "director", "writer", "star", "company", "rating"]:
        val = _clean_text(row.get(col, ""))
        if val:
            parts.append(val)

    return TEXT_JOIN_TOKEN.join(parts)


def build_features(
    df: pd.DataFrame,
    min_votes: int = 5000,
    max_tfidf_features: int = 3000,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build cosine retrieval features using TMDB-enriched metadata.

    Output:
    - dense numeric features
    - TF-IDF text features from overview/genres/keywords/cast/crew
    - movie_names aligned to feature rows
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    # keep only usable rows
    numeric_votes = pd.to_numeric(df["votes"], errors="coerce").fillna(0)
    df = df.loc[numeric_votes > min_votes].copy()

    # prefer TMDB runtime/popularity/votes if present
    if "tmdb_runtime" not in df.columns:
        df["tmdb_runtime"] = np.nan
    if "tmdb_popularity" not in df.columns:
        df["tmdb_popularity"] = np.nan
    if "tmdb_vote_count" not in df.columns:
        df["tmdb_vote_count"] = np.nan
    if "tmdb_vote_average" not in df.columns:
        df["tmdb_vote_average"] = np.nan
    if "tmdb_release_date" not in df.columns:
        df["tmdb_release_date"] = np.nan
    if "tmdb_overview" not in df.columns:
        df["tmdb_overview"] = ""
    if "tmdb_genres" not in df.columns:
        df["tmdb_genres"] = ""
    if "tmdb_keywords" not in df.columns:
        df["tmdb_keywords"] = ""
    if "tmdb_cast_top5" not in df.columns:
        df["tmdb_cast_top5"] = ""
    if "tmdb_directors" not in df.columns:
        df["tmdb_directors"] = ""
    if "tmdb_writers" not in df.columns:
        df["tmdb_writers"] = ""

    df["runtime_final"] = pd.to_numeric(df["tmdb_runtime"], errors="coerce").fillna(
        pd.to_numeric(df["runtime"], errors="coerce")
    ).fillna(0)

    df["popularity_final"] = pd.to_numeric(df["tmdb_popularity"], errors="coerce").fillna(0)
    df["vote_count_final"] = pd.to_numeric(df["tmdb_vote_count"], errors="coerce").fillna(
        pd.to_numeric(df["votes"], errors="coerce")
    ).fillna(0)

    df["vote_average_final"] = pd.to_numeric(df["tmdb_vote_average"], errors="coerce").fillna(
        pd.to_numeric(df["score"], errors="coerce")
    ).fillna(0)

    df["gross_final"] = pd.to_numeric(df["gross"], errors="coerce").fillna(0)
    df["budget_final"] = pd.to_numeric(df["budget"], errors="coerce").fillna(0)
    df["year_final"] = pd.to_numeric(df["year"], errors="coerce").fillna(0)

    df["country_bucket"] = df["country"].apply(_bucket_country)

    movie_names = [f"{name} ({int(year)})" for name, year in zip(df["name"], df["year_final"])]

    numeric_features = pd.DataFrame(
        {
            "year_norm": _normalize_numeric(df["year_final"]),
            "runtime_norm": _normalize_numeric(df["runtime_final"]),
            "vote_average_norm": _normalize_numeric(df["vote_average_final"]),
            "vote_count_norm": _normalize_numeric(np.log1p(df["vote_count_final"])),
            "popularity_norm": _normalize_numeric(np.log1p(df["popularity_final"])),
            "gross_norm": _normalize_numeric(np.log1p(df["gross_final"])),
            "budget_norm": _normalize_numeric(np.log1p(df["budget_final"])),
        },
        index=df.index,
    )

    country_dummies = pd.get_dummies(df["country_bucket"], prefix="country", dtype=float)

    text_corpus = df.apply(_build_text_blob, axis=1)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_tfidf_features,
        ngram_range=(1, 2),
    )
    tfidf_matrix = vectorizer.fit_transform(text_corpus)
    tfidf_df = pd.DataFrame.sparse.from_spmatrix(
        tfidf_matrix,
        index=df.index,
        columns=[f"tfidf_{c}" for c in vectorizer.get_feature_names_out()],
    )

    feature_df = pd.concat(
        [
            numeric_features.reset_index(drop=True),
            country_dummies.reset_index(drop=True),
            tfidf_df.reset_index(drop=True),
        ],
        axis=1,
    )

    return feature_df, movie_names