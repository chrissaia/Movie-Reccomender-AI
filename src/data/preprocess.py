import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal preprocessing for the movie recommendation dataset.

    What this does:
    - strips whitespace from column names
    - strips whitespace from string values
    - converts numeric columns to numeric types where needed

    What this does NOT do:
    - drop NaNs
    - filter votes
    - create popularityScore
    - bucket/director/writer/star
    - encode genre/country
    - normalize features

    All of that belongs in build_features.py
    """
    df = df.copy()

    # Clean column names
    df.columns = df.columns.str.strip()

    # Strip whitespace from string/object columns
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip()

    # Convert known numeric columns
    numeric_cols = ["year", "score", "votes", "gross", "runtime"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df