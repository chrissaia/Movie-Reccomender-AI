"use client";

import { useUser } from "@clerk/nextjs";
import { useEffect, useMemo, useState } from "react";

import AppHeader from "../components/AppHeader";
import MovieDetailModal from "../components/MovieDetailModal";
import { API_BASE_URL, FALLBACK_POSTER } from "../lib/config";
import { formatScore } from "../lib/format";
import { logHandledError } from "../lib/log";
import { getTmdbDetails } from "../lib/tmdb";
import type {
  CombinedRecommendationResponse,
  OrganizedRow,
  RecommendationItem,
  SavedMovie,
} from "../types";

const DISLIKED_MOVIES_STORAGE_KEY = "dislikedRecommendationMovieIds";

function readStoredDislikedMovieIds() {
  try {
    const raw = localStorage.getItem(DISLIKED_MOVIES_STORAGE_KEY);
    const parsed: unknown = raw ? JSON.parse(raw) : [];

    return Array.isArray(parsed)
      ? parsed.filter(
          (movieId): movieId is number => typeof movieId === "number",
        )
      : [];
  } catch (err) {
    logHandledError("Stored disliked movies parse failed", err);
    return [];
  }
}

async function enrichRows(rows: OrganizedRow[]): Promise<OrganizedRow[]> {
  return Promise.all(
    rows.map(async (row) => ({
      ...row,
      items: await Promise.all(
        row.items.map(async (item) => {
          const details = await getTmdbDetails(item.title);
          return {
            ...item,
            ...details,
          };
        }),
      ),
    })),
  );
}

function rowBadge(type: string) {
  if (type === "top_picks") return "Best overall";
  if (type === "familiar_but_not_obvious") return "Smart discovery";
  if (type === "hidden_gems") return "Underrated";
  if (type === "taste_profile") return "Taste pattern";
  if (type === "source_movie") return "Movie anchor";
  return "Recommended";
}

export default function ResultsPage() {
  const { user, isSignedIn } = useUser();
  const userId = user?.id;

  const [selected, setSelected] = useState<SavedMovie[]>([]);
  const [rows, setRows] = useState<OrganizedRow[]>([]);
  const [headline, setHeadline] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [dislikedMovieIds, setDislikedMovieIds] = useState<Set<number>>(
    () => new Set(),
  );

  const [showSaveModal, setShowSaveModal] = useState(false);
  const [listName, setListName] = useState("");
  const [saveMessage, setSaveMessage] = useState("");
  const [activeMovie, setActiveMovie] = useState<RecommendationItem | null>(
    null,
  );
  const selectedMovieIds = useMemo(
    () => selected.map((movie) => movie.movie_id),
    [selected],
  );
  const selectedMovieIdKey = selectedMovieIds.join(",");

  useEffect(() => {
    const raw = localStorage.getItem("selectedMovies");
    const parsed: SavedMovie[] = raw ? JSON.parse(raw) : [];
    setSelected(parsed);
  }, []);

  useEffect(() => {
    const loadRecommendations = async () => {
      const movieIds = selectedMovieIdKey
        ? selectedMovieIdKey.split(",").map((movieId) => Number(movieId))
        : [];

      if (movieIds.length === 0) {
        setLoading(false);
        return;
      }

      setLoading(true);
      setError("");

      try {
        const headers: HeadersInit = {
          "Content-Type": "application/json",
        };

        if (isSignedIn && userId) {
          headers["X-User-Id"] = userId;
        }

        const res = await fetch(`${API_BASE_URL}/recommend/combined`, {
          method: "POST",
          headers,
          body: JSON.stringify({
            movie_ids: movieIds,
            top_k: 10,
          }),
        });

        if (!res.ok) {
          throw new Error("Recommendation request failed");
        }

        const data: CombinedRecommendationResponse = await res.json();

        const organizedRows =
          data.organized_rows && data.organized_rows.length > 0
            ? data.organized_rows
            : [
                {
                  type: "top_picks",
                  title: "Top Picks For You",
                  pinned: true,
                  row_score: 1,
                  items: data.recommendations ?? [],
                },
              ];

        const enriched = await enrichRows(organizedRows);

        setRows(enriched);
        setHeadline(data.headline ?? "");
      } catch (err) {
        logHandledError("Recommendation load failed", err);
        setError("Could not load recommendations.");
      } finally {
        setLoading(false);
      }
    };

    loadRecommendations();
  }, [isSignedIn, selectedMovieIdKey, userId]);

  useEffect(() => {
    const loadDislikedMovies = async () => {
      if (!isSignedIn || !userId) {
        setDislikedMovieIds(new Set(readStoredDislikedMovieIds()));
        return;
      }

      try {
        const res = await fetch(`${API_BASE_URL}/profile/disliked-movies`, {
          headers: { "X-User-Id": userId },
        });

        if (!res.ok) {
          throw new Error("Disliked movies request failed");
        }

        const data: { movie_ids?: number[] } = await res.json();
        setDislikedMovieIds(new Set(data.movie_ids ?? []));
      } catch (err) {
        logHandledError("Disliked movies load failed", err);
      }
    };

    loadDislikedMovies();
  }, [isSignedIn, userId]);

  const visibleRows = useMemo(
    () =>
      rows
        .map((row) => ({
          ...row,
          items: row.items.filter(
            (movie) => !dislikedMovieIds.has(movie.movie_id),
          ),
        }))
        .filter((row) => row.items.length > 0),
    [rows, dislikedMovieIds],
  );

  const dislikeRecommendation = async (movie: RecommendationItem) => {
    setDislikedMovieIds((current) => {
      const next = new Set(current);
      next.add(movie.movie_id);

      if (!isSignedIn || !userId) {
        localStorage.setItem(
          DISLIKED_MOVIES_STORAGE_KEY,
          JSON.stringify(Array.from(next)),
        );
      }

      return next;
    });

    if (!isSignedIn || !userId) return;

    try {
      const res = await fetch(`${API_BASE_URL}/profile/disliked-movies`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-User-Id": userId,
        },
        body: JSON.stringify({
          movie_id: movie.movie_id,
          title: movie.title,
        }),
      });

      if (!res.ok) {
        throw new Error("Disliked movie save failed");
      }
    } catch (err) {
      logHandledError("Disliked movie save failed", err);
      setSaveMessage("Hidden here, but could not save dislike.");
    }
  };

  const saveList = async () => {
    if (!isSignedIn || !userId) {
      setSaveMessage("Please sign in first.");
      return;
    }

    const trimmed = listName.trim();
    if (!trimmed || selected.length === 0) return;

    try {
      const res = await fetch(`${API_BASE_URL}/user/lists`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-User-Id": userId,
        },
        body: JSON.stringify({
          name: trimmed,
          movies: selected.map((movie) => ({
            movie_id: movie.movie_id,
            title: movie.title,
          })),
        }),
      });

      if (!res.ok) {
        throw new Error("Failed to save list");
      }

      setSaveMessage(`Saved "${trimmed}"`);
      setShowSaveModal(false);
      setListName("");
      setTimeout(() => setSaveMessage(""), 2500);
    } catch (err) {
      logHandledError("List save failed", err);
      setSaveMessage("Could not save list.");
    }
  };

  return (
    <main className="results-page">
      <style>{`
        .results-page {
          min-height: 100vh;
          padding: 28px 24px 64px;
          background:
            radial-gradient(circle at 18% 12%, rgba(185, 28, 28, 0.18), transparent 34%),
            radial-gradient(circle at 82% 8%, rgba(127, 29, 29, 0.16), transparent 30%),
            linear-gradient(135deg, #070607 0%, #1a0b10 38%, #450a0a 100%);
          color: #f8fafc;
        }

        .wrap {
          max-width: 1320px;
          margin: 0 auto;
        }

        .top-bar {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 16px;
          margin-bottom: 32px;
        }

        .top-actions {
          display: flex;
          align-items: center;
          gap: 12px;
          flex-wrap: wrap;
        }

        .pill-btn {
          border: 1px solid rgba(255,255,255,0.12);
          background: rgba(255,255,255,0.06);
          color: #e2e8f0;
          border-radius: 999px;
          padding: 10px 16px;
          cursor: pointer;
          font-weight: 800;
        }

        .pill-btn:hover {
          background: rgba(255,255,255,0.10);
        }

        .primary-btn {
          border: none;
          background: linear-gradient(135deg, #3b82f6, #2563eb);
          color: white;
          border-radius: 999px;
          padding: 10px 16px;
          cursor: pointer;
          font-weight: 800;
        }

        .hero {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          gap: 20px;
          margin-bottom: 30px;
        }

        .hero-copy {
          min-width: 0;
        }

        .hero-title {
          font-size: clamp(2rem, 4vw, 4.1rem);
          line-height: 1.02;
          font-weight: 950;
          letter-spacing: -0.06em;
          margin: 0 0 12px;
        }

        .headline {
          color: rgba(255,255,255,0.74);
          font-size: 17px;
          max-width: 900px;
          line-height: 1.55;
          margin-bottom: 18px;
        }

        .selected-row {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
        }

        .selected-chip {
          background: rgba(255,255,255,0.08);
          border: 1px solid rgba(255,255,255,0.10);
          color: #e2e8f0;
          border-radius: 999px;
          padding: 8px 12px;
          font-size: 13px;
          font-weight: 700;
        }

        .save-box {
          display: flex;
          flex-direction: column;
          align-items: flex-end;
          gap: 8px;
          min-width: 180px;
        }

        .save-message {
          color: #93c5fd;
          font-size: 14px;
          font-weight: 800;
        }

        .row-section {
          margin-bottom: 34px;
        }

        .row-head {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 14px;
          margin-bottom: 14px;
        }

        .row-title {
          font-size: clamp(1.25rem, 2vw, 1.9rem);
          font-weight: 900;
          letter-spacing: -0.04em;
          margin: 0;
        }

        .row-badge {
          border: 1px solid rgba(147,197,253,0.18);
          color: #bfdbfe;
          background: rgba(59,130,246,0.12);
          border-radius: 999px;
          padding: 7px 11px;
          font-size: 12px;
          font-weight: 900;
          white-space: nowrap;
        }

        .movie-row {
          display: grid;
          grid-auto-flow: column;
          grid-auto-columns: minmax(165px, 190px);
          gap: 16px;
          overflow-x: auto;
          padding-bottom: 12px;
          scroll-snap-type: x mandatory;
        }

        .movie-card {
          position: relative;
          scroll-snap-align: start;
          border-radius: 18px;
          overflow: hidden;
          background: rgba(255,255,255,0.06);
          border: 1px solid rgba(255,255,255,0.08);
          box-shadow: 0 16px 40px rgba(0,0,0,0.28);
          min-height: 285px;
        }

        .poster {
          width: 100%;
          aspect-ratio: 2 / 3;
          object-fit: cover;
          display: block;
          background: rgba(255,255,255,0.04);
        }

        .dislike-btn {
          position: absolute;
          top: 10px;
          right: 10px;
          z-index: 4;
          display: grid;
          place-items: center;
          width: 32px;
          height: 32px;
          border: 1px solid rgba(255,255,255,0.16);
          border-radius: 999px;
          background: rgba(15,23,42,0.78);
          color: #fecaca;
          cursor: pointer;
          font-size: 14px;
          line-height: 1;
          box-shadow: 0 10px 24px rgba(0,0,0,0.28);
          backdrop-filter: blur(8px);
          opacity: 0;
          pointer-events: none;
          transition:
            background 160ms ease,
            transform 160ms ease,
            opacity 160ms ease,
            border-color 160ms ease;
        }

        .movie-card:hover .dislike-btn,
        .dislike-btn:focus-visible {
          opacity: 1;
          pointer-events: auto;
        }

        .dislike-btn:hover,
        .dislike-btn:focus-visible {
          background: rgba(127,29,29,0.88);
          border-color: rgba(254,202,202,0.36);
          transform: translateY(-1px);
        }

        .card-base {
          padding: 10px;
        }

        .movie-title {
          font-size: 14px;
          font-weight: 850;
          line-height: 1.3;
          margin: 0 0 5px;
        }

        .movie-meta {
          color: rgba(255,255,255,0.62);
          font-size: 12px;
          font-weight: 700;
        }

        .card-hover {
          position: absolute;
          inset: 0;
          padding: 14px;
          display: flex;
          flex-direction: column;
          justify-content: flex-end;
          opacity: 0;
          transition: opacity 160ms ease;
          background: linear-gradient(to top, rgba(2,6,23,0.98), rgba(2,6,23,0.78), rgba(2,6,23,0.18));
        }

        .movie-card:hover .card-hover {
          opacity: 1;
        }

        .hover-title {
          font-size: 16px;
          font-weight: 950;
          margin-bottom: 8px;
        }

        .explain-list {
          display: flex;
          flex-direction: column;
          gap: 5px;
          color: rgba(255,255,255,0.78);
          font-size: 12px;
          line-height: 1.35;
        }

        .loading,
        .empty,
        .error {
          border: 1px solid rgba(255,255,255,0.08);
          background: rgba(255,255,255,0.05);
          border-radius: 24px;
          padding: 24px;
          color: #cbd5e1;
        }

        .modal-backdrop {
          position: fixed;
          inset: 0;
          background: rgba(0,0,0,0.66);
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 20px;
          z-index: 50;
        }

        .modal {
          width: min(480px, 100%);
          border: 1px solid rgba(255,255,255,0.10);
          background: #0f172a;
          border-radius: 24px;
          padding: 22px;
          box-shadow: 0 24px 80px rgba(0,0,0,0.45);
        }

        .modal-title {
          font-size: 24px;
          font-weight: 900;
          margin: 0 0 12px;
        }

        .modal-input {
          width: 100%;
          border: 1px solid rgba(255,255,255,0.14);
          background: rgba(255,255,255,0.07);
          color: #f8fafc;
          border-radius: 16px;
          padding: 14px 15px;
          outline: none;
          font-size: 16px;
          margin-bottom: 16px;
        }

        .modal-actions {
          display: flex;
          justify-content: flex-end;
          gap: 10px;
        }

        @media (max-width: 760px) {
          .hero {
            flex-direction: column;
          }

          .save-box {
            align-items: flex-start;
          }

          .movie-row {
            grid-auto-columns: minmax(145px, 160px);
          }
        }

      `}</style>

      <div className="wrap">
        <AppHeader
          leading={{ label: "← Search", href: "/" }}
          actions={[
            { label: "My Friends", href: "/friends" },
            { label: "My Lists", href: "/lists" },
          ]}
        />

        <section className="hero">
          <div className="hero-copy">
            <h1 className="hero-title">Your movie map is ready.</h1>

            {headline && <div className="headline">{headline}</div>}

            <div className="selected-row">
              {selected.map((movie) => (
                <span className="selected-chip" key={movie.movie_id}>
                  {movie.title}
                </span>
              ))}
            </div>
          </div>

          <div className="save-box">
            <button
              className="primary-btn"
              onClick={() => setShowSaveModal(true)}
              disabled={selected.length === 0}
            >
              Save List
            </button>
            {saveMessage && <div className="save-message">{saveMessage}</div>}
          </div>
        </section>

        {loading && <div className="loading">Loading recommendations...</div>}

        {!loading && error && <div className="error">{error}</div>}

        {!loading && !error && selected.length === 0 && (
          <div className="empty">
            No movies selected yet. Go back and choose a few favorites.
          </div>
        )}

        {!loading &&
          !error &&
          visibleRows.map((row) => (
            <section className="row-section" key={`${row.type}-${row.title}`}>
              <div className="row-head">
                <h2 className="row-title">{row.title}</h2>
                <span className="row-badge">{rowBadge(row.type)}</span>
              </div>

              <div className="movie-row">
                {row.items.map((movie) => (
                  <article
                    className="movie-card"
                    key={`${row.title}-${movie.movie_id}`}
                    onClick={() => setActiveMovie(movie)}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") setActiveMovie(movie);
                    }}
                  >
                    <button
                      type="button"
                      className="dislike-btn"
                      aria-label={`Dislike ${movie.title}`}
                      title="Dislike"
                      onClick={(event) => {
                        event.stopPropagation();
                        dislikeRecommendation(movie);
                      }}
                    >
                      👎
                    </button>

                    <img
                      className="poster"
                      src={movie.poster || FALLBACK_POSTER}
                      alt={movie.title}
                    />

                    <div className="card-base">
                      <h3 className="movie-title">{movie.title}</h3>
                      <div className="movie-meta">
                        {movie.year ? `${movie.year} · ` : ""}
                        {formatScore(movie.score ?? movie.final_score)}
                      </div>
                    </div>

                    <div className="card-hover">
                      <div className="hover-title">{movie.title}</div>
                      <div className="explain-list">
                        {(movie.explanations ?? [])
                          .slice(0, 3)
                          .map((reason) => (
                            <div key={reason}>• {reason}</div>
                          ))}
                        {movie.overview && (
                          <div>{movie.overview.slice(0, 130)}...</div>
                        )}
                      </div>
                    </div>
                  </article>
                ))}
              </div>
            </section>
          ))}
      </div>

      <MovieDetailModal
        movie={activeMovie}
        onClose={() => setActiveMovie(null)}
      />

      {showSaveModal && (
        <div className="modal-backdrop">
          <div className="modal">
            <h2 className="modal-title">Save this movie list</h2>
            <input
              className="modal-input"
              value={listName}
              onChange={(event) => setListName(event.target.value)}
              placeholder="Drama with friends"
            />

            <div className="modal-actions">
              <button
                className="pill-btn"
                onClick={() => setShowSaveModal(false)}
              >
                Cancel
              </button>
              <button className="primary-btn" onClick={saveList}>
                Save
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
