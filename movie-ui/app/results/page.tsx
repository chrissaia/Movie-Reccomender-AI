"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

type SelectedMovie = {
  movie_id: number;
  title: string;
};

type RawRec = {
  movie_id?: number;
  title?: string;
  name?: string;
  score?: number;
};

type MovieDetails = {
  title: string;
  score?: number;
  poster?: string;
  plot?: string;
  director?: string;
  actors?: string;
  genre?: string;
  year?: string;
  imdbRating?: string;
};

type RowData = {
  sourceTitle: string;
  items: MovieDetails[];
};

const OMDB_API_KEY = "e07c39f3";
const TMDB_API_KEY = "5d1421370bac86b6d373cf6be6f0e942";
const TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500";
const FALLBACK_POSTER = "/no-poster.png";

async function enrichMovie(rec: RawRec): Promise<MovieDetails> {
  const title = rec.title || rec.name || "Unknown Title";

  try {
    const searchRes = await fetch(
      `https://api.themoviedb.org/3/search/movie?query=${encodeURIComponent(title)}&api_key=${TMDB_API_KEY}`
    );
    const searchData = await searchRes.json();

    const match =
      searchData?.results?.find(
        (m: any) => String(m.title).toLowerCase() === title.toLowerCase()
      ) || searchData?.results?.[0];

    if (!match) {
      return {
        title,
        score: rec.score,
        poster: FALLBACK_POSTER,
        plot: "No description available.",
        director: "Unknown",
        actors: "Unknown",
      };
    }

    const detailsRes = await fetch(
      `https://api.themoviedb.org/3/movie/${match.id}?api_key=${TMDB_API_KEY}&append_to_response=credits`
    );
    const details = await detailsRes.json();

    return {
      title,
      score: rec.score,
      poster: details.poster_path
        ? `${TMDB_IMAGE_BASE}${details.poster_path}`
        : FALLBACK_POSTER,
      plot: details.overview || "No description available.",
      director:
        details.credits?.crew?.find((p: any) => p.job === "Director")?.name || "Unknown",
      actors:
        details.credits?.cast?.slice(0, 4).map((p: any) => p.name).join(", ") || "Unknown",
      genre: details.genres?.map((g: any) => g.name).join(", "),
      year: details.release_date?.slice(0, 4),
      imdbRating: undefined,
    };
  } catch {
    return {
      title,
      score: rec.score,
      poster: FALLBACK_POSTER,
      plot: "No description available.",
      director: "Unknown",
      actors: "Unknown",
    };
  }
}

function MovieCard({ movie }: { movie: MovieDetails }) {
  return (
    <div className="movie-card">
      <div className="poster-shell">
        <img
          src={movie.poster || FALLBACK_POSTER}
          alt={movie.title}
          className="poster-img"
        />
        <div className="poster-overlay">
          <div className="overlay-title">{movie.title}</div>
          <div className="overlay-meta">
            {[movie.year, movie.genre, movie.imdbRating ? `IMDb ${movie.imdbRating}` : null]
              .filter(Boolean)
              .join(" • ")}
          </div>
          <div className="overlay-plot">{movie.plot}</div>
          <div className="overlay-extra">
            <div><strong>Director:</strong> {movie.director}</div>
            <div><strong>Cast:</strong> {movie.actors}</div>
            {movie.score !== undefined && (
              <div><strong>Score:</strong> {movie.score.toFixed(3)}</div>
            )}
          </div>
        </div>
      </div>
      <div className="card-label">{movie.title}</div>
    </div>
  );
}

function MovieRow({ title, items }: { title: string; items: MovieDetails[] }) {
  const [open, setOpen] = useState(true);

  return (
    <section style={{ marginBottom: 34 }}>
      <button className="row-toggle" onClick={() => setOpen(!open)}>
        <h2 className="row-title">{title}</h2>
        <span className="row-link">{open ? "Hide" : "Show"}</span>
      </button>

      {open && (
        <div className="row-scroll">
          {items.map((movie, i) => (
            <MovieCard key={`${movie.title}-${i}`} movie={movie} />
          ))}
        </div>
      )}
    </section>
  );
}

export default function ResultsPage() {
  const router = useRouter();
  const [selected, setSelected] = useState<SelectedMovie[]>([]);
  const [rows, setRows] = useState<RowData[]>([]);
  const [combined, setCombined] = useState<MovieDetails[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const raw = localStorage.getItem("selectedMovies");
    if (!raw) {
      router.push("/");
      return;
    }

    const parsed: SelectedMovie[] = JSON.parse(raw);
    setSelected(parsed);

    const load = async () => {
      try {
        const combinedRes = await fetch("http://127.0.0.1:8000/recommend/combined", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            movie_ids: parsed.map((m) => m.movie_id),
            top_k: 10,
          }),
        });
        const combinedData = await combinedRes.json();

        const combinedRaw: RawRec[] = Array.isArray(combinedData)
          ? combinedData
          : Array.isArray(combinedData.recommendations)
          ? combinedData.recommendations
          : [];

        const combinedItems = await Promise.all(combinedRaw.map(enrichMovie));
        setCombined(combinedItems);

        const loadedRows = await Promise.all(
          parsed.map(async (movie) => {
            const res = await fetch("http://127.0.0.1:8000/recommend/by-movie", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                movie_id: movie.movie_id,
                top_k: 5,
              }),
            });

            const data = await res.json();

            const rawItems: RawRec[] = Array.isArray(data)
              ? data
              : Array.isArray(data.recommendations)
              ? data.recommendations
              : [];

            const items = await Promise.all(rawItems.map(enrichMovie));

            return {
              sourceTitle: movie.title,
              items,
            };
          })
        );

        setRows(loadedRows);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    load();
  }, [router]);

  return (
    <main className="results-page">
      <style>{`
        .results-page {
          min-height: 100vh;
          padding: 28px 24px 56px;
          background: linear-gradient(135deg, #020617 0%, #0f172a 35%, #1e3a8a 100%);
        }

        .results-wrap {
          max-width: 1400px;
          margin: 0 auto;
        }

        .back-btn {
          border: 1px solid rgba(255,255,255,0.12);
          background: rgba(255,255,255,0.06);
          color: #e2e8f0;
          border-radius: 999px;
          padding: 10px 16px;
          cursor: pointer;
          font-weight: 700;
          margin-bottom: 24px;
        }

        .hero-title {
          font-size: clamp(1.8rem, 3.4vw, 3.2rem);
          line-height: 1.08;
          font-weight: 900;
          letter-spacing: -0.05em;
          color: #f8fafc;
          margin: 0 0 10px 0;
          max-width: 1100px;
        }

        .hero-sub {
          color: rgba(255,255,255,0.72);
          font-size: 17px;
          margin-bottom: 24px;
        }

        .chip-wrap {
          display: flex;
          flex-wrap: wrap;
          gap: 10px;
          margin-bottom: 30px;
        }

        .chip {
          background: rgba(255,255,255,0.08);
          color: #e2e8f0;
          padding: 10px 14px;
          border-radius: 999px;
          border: 1px solid rgba(255,255,255,0.08);
          font-weight: 600;
        }

        .row-toggle {
          width: 100%;
          background: transparent;
          border: none;
          padding: 0;
          margin-bottom: 14px;
          display: flex;
          align-items: center;
          justify-content: space-between;
          cursor: pointer;
        }

        .row-title {
          color: #f8fafc;
          font-size: clamp(1.05rem, 1.8vw, 1.45rem);
          font-weight: 800;
          letter-spacing: -0.03em;
          margin: 0;
          text-align: left;
        }

        .row-link {
          color: #93c5fd;
          font-size: 14px;
          font-weight: 700;
        }

        .row-scroll {
          display: flex;
          gap: 16px;
          overflow-x: auto;
          padding-bottom: 8px;
        }

        .movie-card {
          width: 190px;
          flex: 0 0 auto;
          transition: transform 0.22s ease;
        }

        .movie-card:hover {
          transform: translateY(-6px) scale(1.03);
          z-index: 10;
        }

        .movie-card:hover .poster-overlay {
          opacity: 1;
        }

        .poster-shell {
          position: relative;
          border-radius: 16px;
          overflow: hidden;
          background: rgba(255,255,255,0.05);
          border: 1px solid rgba(255,255,255,0.08);
          box-shadow: 0 16px 40px rgba(0,0,0,0.34);
        }

        .poster-img {
          width: 100%;
          aspect-ratio: 2 / 3;
          object-fit: cover;
          display: block;
        }

        .poster-overlay {
          position: absolute;
          inset: 0;
          opacity: 0;
          transition: opacity 0.22s ease;
          padding: 14px;
          display: flex;
          flex-direction: column;
          justify-content: flex-end;
          background: linear-gradient(to top, rgba(2,6,23,0.98) 0%, rgba(2,6,23,0.92) 48%, rgba(2,6,23,0.15) 100%);
        }

        .overlay-title {
          color: #f8fafc;
          font-weight: 800;
          font-size: 16px;
          margin-bottom: 6px;
        }

        .overlay-meta {
          color: rgba(255,255,255,0.7);
          font-size: 12px;
          margin-bottom: 8px;
          line-height: 1.4;
        }

        .overlay-plot {
          color: rgba(255,255,255,0.84);
          font-size: 12.5px;
          line-height: 1.45;
          margin-bottom: 8px;
          display: -webkit-box;
          -webkit-line-clamp: 5;
          -webkit-box-orient: vertical;
          overflow: hidden;
        }

        .overlay-extra {
          color: #cbd5e1;
          font-size: 11.5px;
          line-height: 1.5;
        }

        .card-label {
          margin-top: 10px;
          color: #f8fafc;
          font-weight: 700;
          font-size: 14px;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .loading {
          color: #cbd5e1;
          font-size: 18px;
        }
      `}</style>

      <div className="results-wrap">
        <button className="back-btn" onClick={() => router.push("/")}>
          ← Back
        </button>

        <h1 className="hero-title">Your Movie Matches</h1>
        <p className="hero-sub">Hover a poster to see details.</p>

        <div className="chip-wrap">
          {selected.map((movie) => (
            <div key={movie.movie_id} className="chip">
              {movie.title}
            </div>
          ))}
        </div>

        {loading ? (
          <div className="loading">Loading recommendations...</div>
        ) : (
          <>
            {combined.length > 0 && (
              <MovieRow title="Top Picks For You" items={combined} />
            )}

            {rows.map((row) => (
              <MovieRow
                key={row.sourceTitle}
                title={`Because you liked ${row.sourceTitle}`}
                items={row.items}
              />
            ))}
          </>
        )}
      </div>
    </main>
  );
}