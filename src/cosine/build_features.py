import pandas as pd



def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Build movie recommendation features.

    Steps performed:
    1. Strip column whitespace
    2. Drop rows with any missing values
    3. Keep only rows with votes > 5000 and only the needed columns
    4. Create popularityScore = score * votes
    5. Drop score, votes, and name
    6. Bucket director, writer, and star by frequency
    7. Remove countries with <= 8 occurrences
    8. Map countries into regional buckets
    9. One-hot encode country and genre as list columns
    10. Normalize gross, popularityScore, runtime, and year
    11. Normalize director, writer, and star bucket values

    Returns
    -------
    pd.DataFrame
        Feature-engineered dataframe
    """

    df = df.copy()

    # tidy headers
    df.columns = df.columns.str.strip()

    # drop all rows with NaN
    df = df.dropna(axis=0)

    # keep only rows with votes > 5000 and only desired columns
    df = df.loc[
        df["votes"] > 5000,
        ["name", "genre", "year", "score", "votes", "director", "writer", "star", "country", "gross", "runtime"]
    ].copy()

    # create popularity score
    df["popularityScore"] = df["score"] * df["votes"]

    # delete unwanted columns
    df = df.drop(columns=["score", "votes"])

    # --------------------------------------------------
    # bucket director / writer / star by occurrence count
    # --------------------------------------------------
    director_counts = df["director"].value_counts()
    writer_counts = df["writer"].value_counts()
    star_counts = df["star"].value_counts()

    df["director"] = df["director"].apply(lambda x: (x, director_counts[x] // 10))
    df["writer"] = df["writer"].apply(lambda x: (x, writer_counts[x] // 10))
    df["star"] = df["star"].apply(lambda x: (x, star_counts[x] // 10))

    # remove countries with <= 8 occurrences
    df = df[df.groupby("country")["country"].transform("count") > 8].copy()

    # country mapping
    def map_country(country: str) -> str:
        if country in ["United States", "Canada"]:
            return "USCan"
        elif country in ["United Kingdom", "Ireland"]:
            return "GB"
        elif country in ["Australia", "New Zealand"]:
            return "Oceania"
        elif country in ["France", "Germany", "Spain", "Italy", "Denmark", "Sweden", "Norway", "Netherlands"]:
            return "EU"
        elif country in ["Japan", "Hong Kong", "China", "South Korea"]:
            return "East Asia"
        else:
            return "Other"

    df["country"] = df["country"].apply(map_country)

    # one-hot list encoding for country
    countries = ["USCan", "GB", "Oceania", "EU", "East Asia", "Other"]
    df["country"] = df["country"].apply(
        lambda country: [1 if country == c else 0 for c in countries].index(1) / len(country)
    )

    # one-hot list encoding for genre
    genres = [
        "Comedy", "Action", "Drama", "Crime", "Biography",
        "Adventure", "Animation", "Horror", "Fantasy",
        "Mystery", "Thriller", "Family", "Romance",
        "Sci-Fi", "Music"
    ]

    df["genre"] = df["genre"].apply(
        lambda genre: [1 if genre == g else 0 for g in genres].index(1) / len(genres)
    )


    # create movie_names list
    movie_names = []
    for mov, year in zip(list(df["name"]), list(df["year"])):
        movie_names.append(f"{mov} ({year})")

    df = df.drop(columns=["name"])


    # --------------------------------------------------
    # continuous normalization
    # --------------------------------------------------
    df["gross"] = round(df["gross"] / df["gross"].max(), 2)

    max_popularity = df["popularityScore"].max()
    if max_popularity != 0:
        df["popularityScore"] = df["popularityScore"].apply(
            lambda value: round(value / max_popularity, 2)
        )
    else:
        df["popularityScore"] = 0.0

    max_runtime = df["runtime"].max()
    if max_runtime != 0:
        df["runtime"] = df["runtime"].apply(lambda value: round(value / max_runtime, 2))
    else:
        df["runtime"] = 0.0

    max_year = df["year"].max()
    min_year = df["year"].min()
    diff = max_year - min_year

    if diff != 0:
        df["year"] = df["year"].apply(lambda value: round((max_year - value) / diff, 2))
    else:
        df["year"] = 0.0

    # --------------------------------------------------
    # discrete normalization
    # --------------------------------------------------
    df["writer"] = df["writer"].apply(lambda x: round(x[1] / 4, 2))
    df["star"] = df["star"].apply(lambda x: round(x[1] / 4, 2))
    df["director"] = df["director"].apply(lambda x: round(x[1] / 4, 2))


    return df, movie_names