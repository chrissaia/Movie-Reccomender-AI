"use client";

import { Show, SignInButton, UserButton } from "@clerk/nextjs";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

type TasteSummary = {
  top_genres?: string[];
  top_keywords?: string[];
  repeated_directors?: string[];
  repeated_cast?: string[];
  year_range?: {
    min: number;
    max: number;
  } | null;
};

type CombinedResponse = {
  headline?: string;
  taste_summary?: TasteSummary;
  taste_rows?: {
    title: string;
    items: RawRec[];
  }[];
  recommendations?: RawRec[];
};

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
  sourceTitle?: string;
  title?: string;
  items: MovieDetails[];
};

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

  const [showSaveModal, setShowSaveModal] = useState(false);
  const [listName, setListName] = useState("");
  const [saveMessage, setSaveMessage] = useState("");

  const [headline, setHeadline] = useState("");
  const [tasteSummary, setTasteSummary] = useState<TasteSummary | null>(null);
  const [tasteRows, setTasteRows] = useState<RowData[]>([]);

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
        const combinedData: CombinedResponse = await combinedRes.json();
        console.log("combinedData", combinedData);

        setHeadline(combinedData.headline || "");
        setTasteSummary(combinedData.taste_summary || null);

        const combinedRaw: RawRec[] = Array.isArray(combinedData)
          ? combinedData
          : Array.isArray(combinedData.recommendations)
          ? combinedData.recommendations
          : [];

        const combinedItems = await Promise.all(combinedRaw.map(enrichMovie));
        setCombined(combinedItems);
        const loadedTasteRows: RowData[] = await Promise.all(
        (combinedData.taste_rows || []).map(async (row) => {
            const items = await Promise.all((row.items || []).map(enrichMovie));
            return {
              title: row.title,
              items,
            };
          })
        );

        setTasteRows(loadedTasteRows);

        const loadedRows = await Promise.all(
          parsed.map(async (movie) => {
            const res = await fetch("http://127.0.0.1:8000/recommend/by-movie", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                movie_id: movie.movie_id,
                top_k: 7,
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

  const saveList = () => {
    const trimmed = listName.trim();
    if (!trimmed || selected.length === 0) return;

    const existing = localStorage.getItem("savedMovieLists");
    const parsed = existing ? JSON.parse(existing) : [];

    const newList = {
      id: crypto.randomUUID(),
      name: trimmed,
      movies: selected,
      createdAt: new Date().toISOString(),
    };

    const updated = [newList, ...parsed];
    localStorage.setItem("savedMovieLists", JSON.stringify(updated));

    setSaveMessage(`Saved "${trimmed}"`);
    setShowSaveModal(false);
    setListName("");

    setTimeout(() => setSaveMessage(""), 2500);
  };

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

        .top-bar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 16px;
          margin-bottom: 24px;
        }

        .top-actions {
          display: flex;
          align-items: center;
          gap: 12px;
          flex-wrap: wrap;
        }

        .top-user {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .back-btn,
        .pill-btn,
        .save-btn {
          border: 1px solid rgba(255,255,255,0.12);
          background: rgba(255,255,255,0.06);
          color: #e2e8f0;
          border-radius: 999px;
          padding: 10px 16px;
          cursor: pointer;
          font-weight: 700;
        }

        .back-btn:hover,
        .pill-btn:hover,
        .save-btn:hover {
          background: rgba(255,255,255,0.10);
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

        .selected-row {
          margin-bottom: 30px;
        }

        .chip-wrap {
          display: flex;
          flex-wrap: wrap;
          gap: 10px;
          flex: 1;
          min-width: 0;
        }

        .save-side {
          display: flex;
          flex-direction: column;
          align-items: flex-end;
          gap: 8px;
          flex: 0 0 auto;
        }

        .save-list-primary {
          border: none;
          background: linear-gradient(135deg, #3b82f6, #2563eb);
          color: white;
          border-radius: 999px;
          padding: 10px 18px;
          cursor: pointer;
          font-weight: 700;
          box-shadow: 0 14px 30px rgba(37, 99, 235, 0.24);
        }

        .save-list-primary:hover {
          filter: brightness(1.05);
        }

        .save-msg {
          color: #93c5fd;
          font-size: 14px;
          font-weight: 700;
        }

        @media (max-width: 900px) {
          .selected-row {
            flex-direction: column;
            align-items: stretch;
          }

          .save-side {
            align-items: flex-start;
          }
        }

        .chip {
          background: rgba(255,255,255,0.08);
          color: #e2e8f0;
          padding: 10px 14px;
          border-radius: 999px;
          border: 1px solid rgba(255,255,255,0.08);
          font-weight: 600;
        }

        .save-list-row {
          display: flex;
          align-items: center;
          gap: 12px;
          margin-bottom: 30px;
          flex-wrap: wrap;
        }

        .save-msg {
          color: #93c5fd;
          font-size: 14px;
          font-weight: 700;
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

        .modal-backdrop {
          position: fixed;
          inset: 0;
          background: rgba(2,6,23,0.72);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 100;
        }

        .modal-card {
          width: min(460px, calc(100vw - 32px));
          background: #0f172a;
          border: 1px solid rgba(255,255,255,0.08);
          border-radius: 20px;
          padding: 22px;
          box-shadow: 0 24px 60px rgba(0,0,0,0.45);
        }

        .modal-title {
          color: #f8fafc;
          font-size: 22px;
          font-weight: 800;
          margin: 0 0 10px 0;
        }

        .modal-sub {
          color: rgba(255,255,255,0.72);
          font-size: 14px;
          margin-bottom: 16px;
        }

        .modal-input {
          width: 100%;
          padding: 14px 16px;
          border-radius: 14px;
          border: 1px solid rgba(255,255,255,0.12);
          background: rgba(255,255,255,0.06);
          color: #f8fafc;
          outline: none;
          font-size: 16px;
          margin-bottom: 16px;
        }

        .modal-actions {
          display: flex;
          justify-content: flex-end;
          gap: 10px;
        }

        .modal-secondary {
          border: 1px solid rgba(255,255,255,0.12);
          background: transparent;
          color: #cbd5e1;
          border-radius: 999px;
          padding: 10px 16px;
          cursor: pointer;
          font-weight: 700;
        }

        .modal-primary {
          border: none;
          background: linear-gradient(135deg, #3b82f6, #2563eb);
          color: white;
          border-radius: 999px;
          padding: 10px 16px;
          cursor: pointer;
          font-weight: 700;
        }

        @media (max-width: 640px) {
          .top-bar {
            flex-direction: row;
            align-items: center;
          }

          .top-actions {
            flex-wrap: wrap;
          }
        }

        .taste-panel {
          border: 1px solid rgba(255,255,255,0.08);
          background: rgba(255,255,255,0.05);
          border-radius: 24px;
          padding: 20px;
          margin-bottom: 28px;
          box-shadow: 0 16px 40px rgba(0,0,0,0.28);
        }

        .taste-title {
          color: #f8fafc;
          font-size: 22px;
          font-weight: 800;
          margin: 0 0 10px 0;
        }

        .taste-headline {
          color: #dbeafe;
          font-size: 16px;
          font-weight: 600;
          line-height: 1.6;
          margin-bottom: 16px;
        }

        .taste-grid {
          display: flex;
          flex-wrap: wrap;
          gap: 10px;
        }

        .taste-chip {
          background: rgba(255,255,255,0.08);
          color: #e2e8f0;
          padding: 10px 14px;
          border-radius: 999px;
          border: 1px solid rgba(255,255,255,0.08);
          font-weight: 600;
          font-size: 14px;
        }

        .taste-panel {
          border: 1px solid rgba(255,255,255,0.08);
          background: rgba(255,255,255,0.05);
          border-radius: 24px;
          padding: 20px;
          margin-bottom: 28px;
          box-shadow: 0 16px 40px rgba(0,0,0,0.28);
        }

        .taste-title {
          color: #f8fafc;
          font-size: 22px;
          font-weight: 800;
          margin: 0 0 10px 0;
        }

        .taste-headline {
          color: #dbeafe;
          font-size: 16px;
          font-weight: 600;
          line-height: 1.6;
          margin-bottom: 16px;
        }

        .taste-grid {
          display: flex;
          flex-wrap: wrap;
          gap: 10px;
        }

        .taste-chip {
          background: rgba(255,255,255,0.08);
          color: #e2e8f0;
          padding: 10px 14px;
          border-radius: 999px;
          border: 1px solid rgba(255,255,255,0.08);
          font-weight: 600;
          font-size: 14px;
        }

        .hero-meta-row {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 16px;
          margin-bottom: 24px;
        }

        .hero-sub {
          color: rgba(255,255,255,0.72);
          font-size: 17px;
          margin-bottom: 0;
        }

        .hero-actions {
          display: flex;
          justify-content: flex-end;
          align-items: center;
          gap: 12px;
          flex: 0 0 auto;
        }

      `}</style>

      <div className="results-wrap">
        <div className="top-bar">
          <div className="top-actions">
            <button className="back-btn" onClick={() => router.push("/")}>
              ← Back
            </button>
          </div>


          <div className="absolute top-6 right-6 z-20 flex items-center gap-3">
            <button
              onClick={() => router.push("/lists")}
              className="border border-white/12 bg-white/4 text-slate-200 rounded-full px-4 py-1.5 cursor-pointer font-bold"
            >
              My Lists
            </button>
            <Show
              when="signed-out"
              fallback={<UserButton />}
            >
              <SignInButton mode="modal">
                <button className="pill-btn">Sign In</button>
              </SignInButton>
            </Show>
          </div>
        </div>
        <h1 className="hero-title">Your Movie Matches</h1>
        <div className="hero-meta-row">
          <p className="hero-sub">Hover a poster to see details.</p>
          <div className="hero-actions">
            <Show
              when="signed-out"
              fallback={
                <button className="save-list-primary" onClick={() => setShowSaveModal(true)}>
                  Save List
                </button>
              }
            >
              <SignInButton mode="modal">
                <button className="save-list-primary">Sign in to Save</button>
              </SignInButton>
            </Show>

            {saveMessage && <div className="save-msg">{saveMessage}</div>}
          </div>
        </div>
        <div className="selected-row">
          <div className="chip-wrap">
            {selected.map((movie) => (
              <div key={movie.movie_id} className="chip">
                {movie.title}
              </div>
            ))}
          </div>
        </div>
        {loading ? (
          <div className="loading">Loading recommendations...</div>
        ) : (
          <>
              {combined.length > 0 && (
                <MovieRow title="Top Picks For You" items={combined} />
              )}

              {tasteRows.map((row, i) => (
                <MovieRow
                  key={`${row.title}-${i}`}
                  title={row.title || "Because you like this"}
                  items={row.items}
                />
              ))}

              {rows.map((row) => (
                <MovieRow
                  key={row.sourceTitle}
                  title={`Because you liked ${row.sourceTitle}`}
                  items={row.items}
                />
              ))}
            </>
        )}

        {showSaveModal && (
          <div className="modal-backdrop" onClick={() => setShowSaveModal(false)}>
            <div className="modal-card" onClick={(e) => e.stopPropagation()}>
              <h3 className="modal-title">Save this movie list</h3>
              <div className="modal-sub">
                Save your selected favorites as a named list.
              </div>

              <input
                className="modal-input"
                value={listName}
                onChange={(e) => setListName(e.target.value)}
                placeholder="e.g. Drama with friends"
              />

              <div className="modal-actions">
                <button
                  className="modal-secondary"
                  onClick={() => setShowSaveModal(false)}
                >
                  Cancel
                </button>
                <button className="modal-primary" onClick={saveList}>
                  Save
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}