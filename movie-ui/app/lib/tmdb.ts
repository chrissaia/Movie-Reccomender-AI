import {
  FALLBACK_POSTER,
  TMDB_API_KEY,
  TMDB_POSTER_W185_BASE,
  TMDB_POSTER_W342_BASE,
} from "./config";

type TmdbSearchResult = {
  title?: string;
  poster_path?: string;
  overview?: string;
  release_date?: string;
};

async function searchTmdb(title: string): Promise<TmdbSearchResult | undefined> {
  if (!TMDB_API_KEY) return undefined;

  try {
    const res = await fetch(
      `https://api.themoviedb.org/3/search/movie?query=${encodeURIComponent(
        title
      )}&api_key=${TMDB_API_KEY}`
    );

    const data = await res.json();
    return (
      data?.results?.find(
        (movie: TmdbSearchResult) =>
          String(movie.title ?? "").toLowerCase() === title.toLowerCase()
      ) ?? data?.results?.[0]
    );
  } catch {
    return undefined;
  }
}

export async function getTmdbPoster(title: string): Promise<string> {
  const match = await searchTmdb(title);
  if (!match?.poster_path) return FALLBACK_POSTER;
  return `${TMDB_POSTER_W185_BASE}${match.poster_path}`;
}

export async function getTmdbOptionalPoster(
  title: string
): Promise<string | undefined> {
  const match = await searchTmdb(title);
  if (!match?.poster_path) return undefined;
  return `${TMDB_POSTER_W185_BASE}${match.poster_path}`;
}

export async function getTmdbDetails(title: string) {
  const match = await searchTmdb(title);

  return {
    poster: match?.poster_path
      ? `${TMDB_POSTER_W342_BASE}${match.poster_path}`
      : FALLBACK_POSTER,
    overview: match?.overview ?? "",
    year: match?.release_date ? String(match.release_date).slice(0, 4) : "",
  };
}
