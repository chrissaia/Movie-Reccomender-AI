from __future__ import annotations


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
    runtime REAL
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


ALL_SCHEMA_STATEMENTS = [
    CREATE_MOVIES_TABLE,
    CREATE_NEIGHBORS_TABLE,
    CREATE_NEIGHBORS_SOURCE_INDEX,
    CREATE_NEIGHBORS_SCORE_INDEX,
]