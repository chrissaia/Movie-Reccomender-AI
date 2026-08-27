from __future__ import annotations

import math
import re
import sqlite3
from dataclasses import dataclass

from src.serving.query_intent import QueryIntent, parse_query_intent


STOP_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "by",
    "film",
    "films",
    "for",
    "from",
    "give",
    "i",
    "in",
    "is",
    "like",
    "me",
    "movie",
    "movies",
    "of",
    "please",
    "show",
    "that",
    "the",
    "to",
    "with",
}


GENRE_SYNONYMS = {
    "action": ["action", "martial arts", "fight", "chase"],
    "adventure": ["adventure", "quest", "journey"],
    "animation": ["animation", "animated"],
    "comedy": ["comedy", "funny", "humor", "hilarious", "satire", "laugh", "laughter", "giggle"],
    "crime": ["crime", "criminal", "detective", "heist", "mafia"],
    "drama": ["drama", "dramatic", "emotional", "sad", "cry", "crying", "tears"],
    "fantasy": ["fantasy", "magic", "myth"],
    "horror": ["horror", "scary", "spooky", "creepy", "haunting"],
    "mystery": ["mystery", "detective", "whodunit", "puzzle"],
    "romance": ["romance", "romantic", "love"],
    "science fiction": ["science fiction", "sci fi", "sci-fi", "space", "future"],
    "thriller": ["thriller", "suspense", "tense", "psychological"],
    "western": ["western", "cowboy"],
}


MOOD_SYNONYMS = {
    "action-packed": ["action", "fight", "martial arts", "chase", "explosion"],
    "dark": ["dark", "bleak", "grim", "noir", "tragic"],
    "funny": ["funny", "comedy", "humor", "hilarious", "satire", "laugh", "laughter", "giggle"],
    "mind-bending": ["mind-bending", "mind bending", "twist", "surreal", "dream", "memory"],
    "romantic": ["romantic", "romance", "love", "relationship"],
    "scary": ["scary", "spooky", "creepy", "horror", "haunting", "terror"],
    "sad": ["sad", "cry", "crying", "tears", "heartbreaking", "melancholy", "grief", "sorrow"],
    "slow": ["slow", "quiet", "meditative", "patient"],
    "space": ["space", "planet", "astronaut", "alien", "galaxy"],
}


MOOD_GENRES = {
    "action-packed": ["action"],
    "dark": ["drama", "thriller"],
    "funny": ["comedy"],
    "mind-bending": ["mystery", "science fiction"],
    "romantic": ["romance", "drama"],
    "scary": ["horror"],
    "sad": ["drama"],
    "slow": ["drama"],
    "space": ["science fiction"],
    "uplifting": ["drama", "comedy"],
    "feel-good": ["comedy"],
    "hopeful": ["drama"],
    "bittersweet": ["drama", "romance"],
    "nostalgic": ["drama"],
    "cozy": ["comedy", "romance"],
    "whimsical": ["fantasy", "comedy"],
    "epic": ["adventure", "fantasy"],
    "mysterious": ["mystery", "thriller"],
    "suspenseful": ["thriller", "mystery"],
    "gritty": ["crime", "drama"],
    "disturbing": ["horror", "thriller"],
    "thought-provoking": ["drama", "science fiction"],
    "contemplative": ["drama"],
    "adventurous": ["adventure"],
}


@dataclass(frozen=True)
class ParsedQuery:
    raw: str
    terms: list[str]
    phrases: list[str]
    genres: list[str]
    moods: list[str]
    people: list[str]


def _normalize(value: object) -> str:
    if value is None:
        return ""
    return str(value).lower()


def _title_case(label: str) -> str:
    return " ".join(part.capitalize() for part in label.split())


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        cleaned = value.strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            deduped.append(cleaned)
    return deduped


def _tokens(query: str) -> list[str]:
    words = re.findall(r"[a-z0-9']+", query.lower())
    return _dedupe(
        [word for word in words if len(word) > 2 and word not in STOP_WORDS]
    )


def _matching_labels(query: str, mapping: dict[str, list[str]]) -> list[str]:
    lower = query.lower()
    labels = []
    for label, synonyms in mapping.items():
        if any(re.search(rf"\b{re.escape(term)}\b", lower) for term in synonyms):
            labels.append(label)
    return labels


def _extract_people(query: str) -> list[str]:
    people = []
    patterns = [
        r"\bby\s+([A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){0,3})",
        r"\bfrom\s+([A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){0,3})",
        r"\bstarring\s+([A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){0,3})",
        r"\bwith\s+([A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){1,3})",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, query):
            candidate = match.group(1).strip()
            if candidate.lower() not in STOP_WORDS:
                people.append(candidate)

    for match in re.finditer(r"\b([A-Z][A-Za-z'-]+\s+[A-Z][A-Za-z'-]+)\b", query):
        people.append(match.group(1).strip())

    return _dedupe(people)


def _adjacent_phrases(query: str) -> list[str]:
    words = _tokens(query)
    return [" ".join(pair) for pair in zip(words, words[1:])]


def _genres_for_moods(moods: list[str]) -> list[str]:
    return _dedupe(
        [genre for mood in moods for genre in MOOD_GENRES.get(mood, [])]
    )


def parse_query(query: str) -> ParsedQuery:
    terms = _tokens(query)
    genres = _matching_labels(query, GENRE_SYNONYMS)
    moods = _matching_labels(query, MOOD_SYNONYMS)
    genres = _dedupe([*genres, *_genres_for_moods(moods)])
    people = _dedupe(_extract_people(query) + _adjacent_phrases(query))
    phrases = people + genres + moods

    for label in genres + moods:
        terms.extend(
            term
            for synonym in (GENRE_SYNONYMS.get(label, []) + MOOD_SYNONYMS.get(label, []))
            for term in _tokens(synonym)
        )

    return ParsedQuery(
        raw=query.strip(),
        terms=_dedupe(terms),
        phrases=_dedupe(phrases),
        genres=_dedupe(genres),
        moods=_dedupe(moods),
        people=_dedupe(people),
    )


def _merge_intent(parsed: ParsedQuery, intent: QueryIntent | None) -> ParsedQuery:
    if intent is None:
        return parsed

    intent_text = " ".join(
        [
            *(intent.people or []),
            *(intent.genres or []),
            *(intent.moods or []),
            *(intent.keywords or []),
        ]
    )
    intent_terms = _tokens(intent_text)
    merged_moods = _dedupe([*parsed.moods, *[mood.lower() for mood in intent.moods]])
    merged_genres = _dedupe(
        [
            *parsed.genres,
            *[genre.lower() for genre in intent.genres],
            *_genres_for_moods(merged_moods),
        ]
    )

    return ParsedQuery(
        raw=parsed.raw,
        terms=_dedupe([*parsed.terms, *intent_terms]),
        phrases=_dedupe(
            [
                *parsed.phrases,
                *intent.people,
                *intent.genres,
                *intent.moods,
                *intent.keywords,
            ]
        ),
        genres=merged_genres,
        moods=merged_moods,
        people=_dedupe([*parsed.people, *intent.people]),
    )


def _fts_query(parsed: ParsedQuery) -> str:
    parts = []
    for phrase in parsed.people:
        safe_words = _tokens(phrase)
        if safe_words:
            parts.append('"' + " ".join(safe_words) + '"')
    parts.extend(parsed.terms[:12])
    return " OR ".join(_dedupe(parts))


def _like_conditions(columns: list[str], terms: list[str]) -> tuple[str, list[str]]:
    clauses = []
    params = []
    for term in terms:
        term_clauses = [f"LOWER({column}) LIKE ?" for column in columns]
        clauses.append("(" + " OR ".join(term_clauses) + ")")
        params.extend([f"%{term.lower()}%"] * len(columns))
    return " OR ".join(clauses), params


def ensure_search_index(conn: sqlite3.Connection) -> None:
    exists = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table'
          AND name = 'movie_search_fts'
        """
    ).fetchone()
    if not exists:
        rebuild_search_index(conn)
        return

    movie_count = conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0]
    fts_count = conn.execute("SELECT COUNT(*) FROM movie_search_fts").fetchone()[0]
    if movie_count != fts_count:
        rebuild_search_index(conn)


def rebuild_search_index(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS movie_search_fts USING fts5(
            movie_id UNINDEXED,
            title,
            people,
            genres,
            keywords,
            overview,
            search_text
        );
        """
    )
    conn.execute("DELETE FROM movie_search_fts;")
    conn.execute(
        """
        INSERT INTO movie_search_fts (
            movie_id,
            title,
            people,
            genres,
            keywords,
            overview,
            search_text
        )
        SELECT
            movie_id,
            COALESCE(name, '') || ' ' || COALESCE(tmdb_title, ''),
            COALESCE(director, '') || ' ' ||
                COALESCE(writer, '') || ' ' ||
                COALESCE(star, '') || ' ' ||
                COALESCE(tmdb_cast_top5, '') || ' ' ||
                COALESCE(tmdb_directors, '') || ' ' ||
                COALESCE(tmdb_writers, ''),
            COALESCE(genre, '') || ' ' || COALESCE(tmdb_genres, ''),
            COALESCE(tmdb_keywords, '') || ' ' ||
                COALESCE(rating, '') || ' ' ||
                COALESCE(country, '') || ' ' ||
                COALESCE(company, ''),
            COALESCE(tmdb_overview, ''),
            COALESCE(name, '') || ' ' ||
                COALESCE(tmdb_title, '') || ' ' ||
                COALESCE(genre, '') || ' ' ||
                COALESCE(tmdb_genres, '') || ' ' ||
                COALESCE(director, '') || ' ' ||
                COALESCE(writer, '') || ' ' ||
                COALESCE(star, '') || ' ' ||
                COALESCE(tmdb_cast_top5, '') || ' ' ||
                COALESCE(tmdb_directors, '') || ' ' ||
                COALESCE(tmdb_keywords, '') || ' ' ||
                COALESCE(tmdb_overview, '')
        FROM movies;
        """
    )
    conn.commit()


def _candidate_rows(
    conn: sqlite3.Connection,
    parsed: ParsedQuery,
    candidate_limit: int,
) -> list[sqlite3.Row]:
    ensure_search_index(conn)
    conn.row_factory = sqlite3.Row
    fts = _fts_query(parsed)

    if fts:
        try:
            rows = conn.execute(
                """
                SELECT m.*, bm25(movie_search_fts) AS fts_rank
                FROM movie_search_fts
                JOIN movies m ON m.movie_id = movie_search_fts.movie_id
                WHERE movie_search_fts MATCH ?
                ORDER BY fts_rank ASC
                LIMIT ?
                """,
                (fts, candidate_limit),
            ).fetchall()
            if rows:
                return rows
        except sqlite3.OperationalError:
            pass

    searchable = [
        "name",
        "genre",
        "director",
        "writer",
        "star",
        "rating",
        "tmdb_genres",
        "tmdb_keywords",
        "tmdb_cast_top5",
        "tmdb_directors",
        "tmdb_overview",
    ]
    where_sql, params = _like_conditions(searchable, parsed.terms)
    if not where_sql:
        return []

    rows = conn.execute(
        f"""
        SELECT *, 0 AS fts_rank
        FROM movies
        WHERE {where_sql}
        LIMIT ?
        """,
        [*params, candidate_limit],
    ).fetchall()
    return rows


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(term.lower() in text for term in terms)


def _row_score_and_reasons(row: sqlite3.Row, parsed: ParsedQuery) -> tuple[float, list[str]]:
    title = _normalize(row["name"])
    people_text = " ".join(
        _normalize(row[column])
        for column in ["director", "writer", "star", "tmdb_cast_top5", "tmdb_directors"]
    )
    genre_text = " ".join(_normalize(row[column]) for column in ["genre", "tmdb_genres"])
    keyword_text = _normalize(row["tmdb_keywords"])
    overview_text = _normalize(row["tmdb_overview"])
    all_text = " ".join([title, people_text, genre_text, keyword_text, overview_text])

    score = 0.0
    reasons: list[str] = []
    all_mood_terms = {
        synonym
        for mood in parsed.moods
        for synonym in MOOD_SYNONYMS.get(mood, [mood])
    }

    for person in parsed.people:
        person_lower = person.lower()
        if person_lower in people_text:
            score += 16.0
            reasons.append(_title_case(person))
        elif person_lower in all_text:
            score += 7.0
            reasons.append(_title_case(person))

    for genre in parsed.genres:
        genre_terms = GENRE_SYNONYMS.get(genre, [genre])
        if _contains_any(genre_text, genre_terms):
            score += 4.0
            reasons.append(_title_case(genre))
        elif _contains_any(keyword_text + " " + overview_text, genre_terms):
            score += 1.5
            reasons.append(_title_case(genre))

    for mood in parsed.moods:
        mood_synonyms = MOOD_SYNONYMS.get(mood, [mood])
        if _contains_any(keyword_text + " " + overview_text + " " + genre_text, mood_synonyms):
            score += 5.0
            reasons.append(_title_case(mood))

    for term in parsed.terms:
        if term in all_mood_terms:
            continue
        if term in title:
            score += 0.5
        elif term in genre_text or term in people_text:
            score += 0.75
        elif term in keyword_text:
            score += 0.35
        elif term in overview_text:
            score += 0.75

    vote_average = row["tmdb_vote_average"] or row["score"] or 0
    vote_count = row["tmdb_vote_count"] or row["votes"] or 0
    popularity = row["tmdb_popularity"] or 0
    score += float(vote_average) * 0.22
    score += min(math.log10(float(vote_count) + 1), 5) * 0.25
    score += min(float(popularity), 120) * 0.01

    return score, _dedupe(reasons[:6])


def semantic_discover_movies(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 30,
) -> list[dict]:
    parsed = _merge_intent(parse_query(query), parse_query_intent(query))
    if not parsed.terms and not parsed.people:
        return []

    rows = _candidate_rows(conn, parsed, max(limit * 8, 80))
    scored = []
    for row in rows:
        score, reasons = _row_score_and_reasons(row, parsed)
        scored.append((score, reasons, row))

    scored.sort(
        key=lambda item: (
            item[0],
            item[2]["tmdb_popularity"] or 0,
            item[2]["tmdb_vote_average"] or item[2]["score"] or 0,
        ),
        reverse=True,
    )

    movies = []
    for score, reasons, row in scored[:limit]:
        movies.append(
            {
                "movie_id": int(row["movie_id"]),
                "title": row["name"],
                "year": row["year"],
                "score": row["score"],
                "votes": row["votes"],
                "runtime": row["runtime"],
                "director": row["director"],
                "writer": row["writer"],
                "star": row["star"],
                "genre": row["genre"],
                "rating": row["rating"],
                "country": row["country"],
                "company": row["company"],
                "tmdb_genres": row["tmdb_genres"],
                "tmdb_keywords": row["tmdb_keywords"],
                "tmdb_cast_top5": row["tmdb_cast_top5"],
                "tmdb_directors": row["tmdb_directors"],
                "tmdb_overview": row["tmdb_overview"],
                "tmdb_vote_average": row["tmdb_vote_average"],
                "tmdb_vote_count": row["tmdb_vote_count"],
                "tmdb_runtime": row["tmdb_runtime"],
                "tmdb_popularity": row["tmdb_popularity"],
                "semantic_score": round(score, 3),
                "match_reasons": reasons,
            }
        )
    return movies
