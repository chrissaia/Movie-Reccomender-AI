import sqlite3

def get_recommendations(movie_id: int, top_k: int = 5):
    conn = sqlite3.connect("data/db/movies.db")

    query = """
    SELECT m.name, mn.cosine_score
    FROM movie_neighbors mn
    JOIN movies m
      ON mn.neighbor_movie_id = m.movie_id
    WHERE mn.source_movie_id = ?
    ORDER BY mn.rank ASC
    LIMIT ?
    """

    rows = conn.execute(query, (movie_id, top_k)).fetchall()
    conn.close()

    return rows

def search_movies(query: str, limit: int = 10):
    conn = sqlite3.connect("data/db/movies.db")

    sql = """
    SELECT movie_id, name
    FROM movies
    WHERE LOWER(name) LIKE ?
    LIMIT ?
    """

    rows = conn.execute(sql, (f"%{query.lower()}%", limit)).fetchall()
    conn.close()

    return rows

def get_combined_recommendations(movie_ids: list[int], top_k: int = 10):
    conn = sqlite3.connect("data/db/movies.db")

    placeholders = ",".join(["?"] * len(movie_ids))
    query = f"""
    SELECT
        mn.neighbor_movie_id,
        m.name,
        AVG(mn.cosine_score) as avg_score
    FROM movie_neighbors mn
    JOIN movies m
      ON mn.neighbor_movie_id = m.movie_id
    WHERE mn.source_movie_id IN ({placeholders})
      AND mn.neighbor_movie_id NOT IN ({placeholders})
    GROUP BY mn.neighbor_movie_id, m.name
    ORDER BY avg_score DESC
    LIMIT ?
    """

    params = movie_ids + movie_ids + [top_k]
    rows = conn.execute(query, params).fetchall()
    conn.close()

    return rows