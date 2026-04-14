from __future__ import annotations


DROP_MOVIES_TABLE = "DROP TABLE IF EXISTS movies;"
DROP_NEIGHBORS_TABLE = "DROP TABLE IF EXISTS movie_neighbors;"


CREATE_MOVIES_TABLE = """
CREATE TABLE IF NOT EXISTS movies (
    movie_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    year INTEGER,
    genre TEXT,
    director TEXT,
    writer TEXT,
    star TEXT,
    country TEXT,
    rating TEXT,
    company TEXT,
    score REAL,
    votes REAL,
    budget REAL,
    gross REAL,
    runtime REAL,
    tmdb_found INTEGER,
    tmdb_id INTEGER,
    tmdb_title TEXT,
    tmdb_original_title TEXT,
    tmdb_release_date TEXT,
    tmdb_overview TEXT,
    tmdb_genres TEXT,
    tmdb_keywords TEXT,
    tmdb_cast_top5 TEXT,
    tmdb_directors TEXT,
    tmdb_writers TEXT,
    tmdb_popularity REAL,
    tmdb_vote_average REAL,
    tmdb_vote_count REAL,
    tmdb_runtime REAL,
    tmdb_original_language TEXT,
    tmdb_production_companies TEXT,
    tmdb_production_countries TEXT,
    tmdb_spoken_languages TEXT
);
"""


CREATE_NEIGHBORS_TABLE = """
CREATE TABLE IF NOT EXISTS movie_neighbors (
    source_movie_id INTEGER NOT NULL,
    neighbor_movie_id INTEGER NOT NULL,
    rank INTEGER NOT NULL,
    cosine_score REAL NOT NULL,
    PRIMARY KEY (source_movie_id, neighbor_movie_id),
    FOREIGN KEY (source_movie_id) REFERENCES movies(movie_id),
    FOREIGN KEY (neighbor_movie_id) REFERENCES movies(movie_id)
);
"""

CREATE_NEIGHBORS_SOURCE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_movie_neighbors_source
ON movie_neighbors(source_movie_id, rank);
"""

CREATE_NEIGHBORS_SCORE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_movie_neighbors_score
ON movie_neighbors(source_movie_id, cosine_score DESC);
"""

CREATE_MOVIES_TMDB_ID_INDEX = """
CREATE INDEX IF NOT EXISTS idx_movies_tmdb_id
ON movies(tmdb_id);
"""

CREATE_MOVIES_NAME_YEAR_INDEX = """
CREATE INDEX IF NOT EXISTS idx_movies_name_year
ON movies(name, year);
"""


ALL_SCHEMA_STATEMENTS = [
    DROP_NEIGHBORS_TABLE,
    DROP_MOVIES_TABLE,
    CREATE_MOVIES_TABLE,
    CREATE_NEIGHBORS_TABLE,
    CREATE_NEIGHBORS_SOURCE_INDEX,
    CREATE_NEIGHBORS_SCORE_INDEX,
    CREATE_MOVIES_TMDB_ID_INDEX,
    CREATE_MOVIES_NAME_YEAR_INDEX,
]