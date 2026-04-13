#!/usr/bin/env python3
from __future__ import annotations

from src.db.sqlite import get_connection
from src.db.schema import ALL_SCHEMA_STATEMENTS
from src.db.repository import init_schema
from src.utils.paths import SQLITE_DB_PATH


def main() -> None:
    conn = get_connection(SQLITE_DB_PATH)
    try:
        init_schema(conn, ALL_SCHEMA_STATEMENTS)
        print(f"Initialized database at: {SQLITE_DB_PATH}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()