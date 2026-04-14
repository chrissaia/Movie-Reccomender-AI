import os
import time
import requests
import pandas as pd

API_KEY = "5d1421370bac86b6d373cf6be6f0e942"
INPUT_CSV = "../data/raw/movies.csv"
OUTPUT_CSV = "../data/raw/movies_enriched.csv"

BASE_URL = "https://api.themoviedb.org/3"
HEADERS = {"accept": "application/json"}

session = requests.Session()
session.headers.update(HEADERS)


def tmdb_get(path: str, params: dict | None = None) -> dict:
    params = params or {}
    params["api_key"] = API_KEY
    r = session.get(f"{BASE_URL}{path}", params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def search_movie(title: str, year) -> dict | None:
    params = {
        "query": title,
        "year": int(year) if pd.notna(year) else None,
        "include_adult": "false",
        "language": "en-US",
    }
    data = tmdb_get("/search/movie", params=params)
    results = data.get("results", [])
    return results[0] if results else None


def get_movie_bundle(tmdb_id: int) -> dict:
    return tmdb_get(
        f"/movie/{tmdb_id}",
        params={
            "language": "en-US",
            "append_to_response": "credits,keywords",
        },
    )


def pick_crew_names(crew: list[dict], jobs: set[str], limit: int = 3) -> list[str]:
    names = []
    seen = set()
    for person in crew:
        if person.get("job") in jobs:
            name = person.get("name")
            if name and name not in seen:
                names.append(name)
                seen.add(name)
        if len(names) >= limit:
            break
    return names


def enrich_row(row: pd.Series) -> dict:
    title = str(row["name"]).strip()
    year = row["year"]

    try:
        hit = search_movie(title, year)
        if not hit:
            return {"tmdb_found": False}

        tmdb_id = hit["id"]
        details = get_movie_bundle(tmdb_id)

        genres = [g["name"] for g in details.get("genres", [])]
        keywords_block = details.get("keywords", {})
        keywords = [k["name"] for k in keywords_block.get("keywords", [])]

        credits = details.get("credits", {})
        cast = [c["name"] for c in credits.get("cast", [])[:5]]
        crew = credits.get("crew", [])

        directors = pick_crew_names(crew, {"Director"}, limit=2)
        writers = pick_crew_names(crew, {"Writer", "Screenplay", "Story", "Novel", "Author"}, limit=5)

        return {
            "tmdb_found": True,
            "tmdb_id": tmdb_id,
            "tmdb_title": details.get("title"),
            "tmdb_original_title": details.get("original_title"),
            "tmdb_release_date": details.get("release_date"),
            "tmdb_overview": details.get("overview"),
            "tmdb_genres": "|".join(genres),
            "tmdb_keywords": "|".join(keywords),
            "tmdb_cast_top5": "|".join(cast),
            "tmdb_directors": "|".join(directors),
            "tmdb_writers": "|".join(writers),
            "tmdb_popularity": details.get("popularity"),
            "tmdb_vote_average": details.get("vote_average"),
            "tmdb_vote_count": details.get("vote_count"),
            "tmdb_runtime": details.get("runtime"),
            "tmdb_original_language": details.get("original_language"),
            "tmdb_production_companies": "|".join(
                c["name"] for c in details.get("production_companies", [])
            ),
            "tmdb_production_countries": "|".join(
                c["iso_3166_1"] for c in details.get("production_countries", [])
            ),
            "tmdb_spoken_languages": "|".join(
                l["english_name"] for l in details.get("spoken_languages", [])
            ),
        }

    except Exception as e:
        return {
            "tmdb_found": False,
            "tmdb_error": str(e),
        }


def main():
    df = pd.read_csv(INPUT_CSV)
    enriched_rows = []

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        enriched = enrich_row(row)
        enriched_rows.append(enriched)

        if i % 50 == 0:
            print(f"Processed {i}/{len(df)}")

        time.sleep(0.15)

    enriched_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(enriched_rows)], axis=1)
    enriched_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    #main() - DONE USING THIS - IT TAKES TOO LONG
    pass