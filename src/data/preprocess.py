import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal preprocessing for movie dataset + TMDB.
    """
    df = df.copy()

    # Clean column names
    df.columns = df.columns.str.strip()

    # Strip whitespace from string/object columns
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        df[col] = df[col].fillna("").astype(str).str.strip()

    # Convert numeric columns (base + TMDB)
    numeric_cols = [
        "year", "score", "votes", "gross", "runtime", "budget",
        "tmdb_popularity", "tmdb_vote_average", "tmdb_vote_count", "tmdb_runtime"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalize boolean
    if "tmdb_found" in df.columns:
        df["tmdb_found"] = df["tmdb_found"].astype(int)

    return df