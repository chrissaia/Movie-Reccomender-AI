# src/utils/paths.py

from pathlib import Path

# Root of project
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data folders
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Raw data
RAW_MOVIES_PATH = RAW_DATA_DIR / "movies.csv"
RAW_MOVIES_ENRICHED_PATH = RAW_DATA_DIR / "movies_enriched.csv"


# Cosine pipeline
COSINE_DIR = PROCESSED_DATA_DIR / "cosine"
COSINE_FEATURES_PATH = COSINE_DIR / "movies_features.csv"
COSINE_SIMILARITY_PATH = COSINE_DIR / "similarity_cosine.csv"
MOVIE_NAMES_PATH = COSINE_DIR / "movie_names.csv"

# Ranking pipeline
RANKING_DIR = PROCESSED_DATA_DIR / "ranking"
RANKING_SIMILARITY_PATH = RANKING_DIR / "similarity_rank.csv"
X_TRAIN_PATH = RANKING_DIR / "X_train.csv"
X_TEST_PATH = RANKING_DIR / "X_test.csv"
Y_TRAIN_PATH = RANKING_DIR / "Y_train.csv"
Y_TEST_PATH = RANKING_DIR / "Y_test.csv"
QID_TRAIN_PATH = RANKING_DIR / "QID_train.csv"
QID_TEST_PATH = RANKING_DIR / "QID_test.csv"

# Models
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Database
DB_DIR = DATA_DIR / "db"
SQLITE_DB_PATH = DB_DIR / "movies.db"