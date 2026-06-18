"use client";

import { useEffect, useState } from "react";
import { useUser } from "@clerk/nextjs";

import { API_BASE_URL, FALLBACK_POSTER } from "../lib/config";
import { logHandledError } from "../lib/log";

type MovieLite = {
  movie_id: number;
  title: string;
  poster?: string;
  overview?: string;
  year?: string;
};

type MovieDetail = MovieLite & {
  score?: number;
  votes?: number;
  runtime?: number;
  director?: string;
  writer?: string;
  star?: string;
  genre?: string;
  rating?: string;
  country?: string;
  company?: string;
  tmdb_genres?: string;
  tmdb_keywords?: string;
  tmdb_cast_top5?: string;
  tmdb_directors?: string;
  tmdb_overview?: string;
  tmdb_vote_average?: number;
  tmdb_vote_count?: number;
  tmdb_runtime?: number;
};

type Props = {
  movie: MovieLite | null;
  onClose: () => void;
};

function starFillPercent(rating: number, starNumber: number) {
  if (rating >= starNumber) return 100;
  if (rating >= starNumber - 0.5) return 50;
  return 0;
}

export default function MovieDetailModal({ movie, onClose }: Props) {
  const { user, isSignedIn } = useUser();
  const userId = user?.id;
  const [detail, setDetail] = useState<MovieDetail | null>(null);
  const [rating, setRating] = useState(3);
  const [note, setNote] = useState("");
  const [message, setMessage] = useState("");

  useEffect(() => {
    const load = async () => {
      if (!movie) return;

      setDetail(movie);
      setMessage("");

      try {
        const res = await fetch(`${API_BASE_URL}/movies/${movie.movie_id}`);
        if (!res.ok) return;
        const data = await res.json();
        setDetail({ ...movie, ...data, title: data.title ?? movie.title });
      } catch (err) {
        logHandledError("Movie detail load failed", err);
      }
    };

    load();
  }, [movie]);

  if (!movie || !detail) return null;

  const saveRating = async () => {
    if (!isSignedIn || !userId) {
      setMessage("Sign in to rate this movie.");
      return;
    }

    try {
      const res = await fetch(`${API_BASE_URL}/profile/movie-ratings`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-User-Id": userId,
        },
        body: JSON.stringify({
          movie_id: movie.movie_id,
          title: movie.title,
          rating,
          description: note,
        }),
      });

      setMessage(res.ok ? "Rating saved." : "Could not save rating.");
    } catch (err) {
      logHandledError("Movie rating save failed", err);
      setMessage("Could not reach the rating service.");
    }
  };

  const addToWatchlist = async () => {
    if (!isSignedIn || !userId) {
      setMessage("Sign in to add this to your watchlist.");
      return;
    }

    try {
      const res = await fetch(`${API_BASE_URL}/profile/watchlist`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-User-Id": userId,
        },
        body: JSON.stringify({
          movie_id: movie.movie_id,
          title: movie.title,
          status: "planned",
        }),
      });

      setMessage(res.ok ? "Added to watchlist." : "Could not add to watchlist.");
    } catch (err) {
      logHandledError("Watchlist save failed", err);
      setMessage("Could not reach the watchlist service.");
    }
  };

  return (
    <div className="movie-modal-backdrop" onClick={onClose}>
      <style>{`
        .movie-modal-backdrop { position: fixed; inset: 0; z-index: 80; display: grid; place-items: center; padding: 20px; background: rgba(0,0,0,0.72); }
        .movie-modal { width: min(860px, 100%); max-height: min(760px, 92vh); overflow-y: auto; border: 1px solid rgba(255,255,255,0.12); border-radius: 24px; background: #0f172a; color: #f8fafc; box-shadow: 0 30px 90px rgba(0,0,0,0.55); }
        .movie-modal-grid { display: grid; grid-template-columns: minmax(180px, 260px) minmax(0, 1fr); gap: 22px; padding: 22px; }
        .movie-modal-poster { width: 100%; border-radius: 16px; aspect-ratio: 2 / 3; object-fit: cover; background: rgba(255,255,255,0.06); }
        .movie-modal-title { font-size: clamp(1.8rem, 4vw, 3rem); line-height: 1.05; font-weight: 950; margin: 0 0 10px; }
        .movie-modal-meta { color: rgba(255,255,255,0.68); font-weight: 750; margin-bottom: 14px; }
        .movie-modal-overview { color: rgba(255,255,255,0.76); line-height: 1.55; margin-bottom: 16px; }
        .movie-modal-facts { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin-bottom: 18px; }
        .movie-fact { border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; background: rgba(255,255,255,0.06); padding: 10px; }
        .movie-fact-label { color: #c4b5fd; font-size: 11px; font-weight: 950; text-transform: uppercase; margin-bottom: 4px; }
        .movie-actions { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-top: 16px; }
        .movie-star-row { display: flex; gap: 4px; margin: 10px 0; }
        .movie-star { position: relative; width: 32px; height: 32px; border: 0; background: transparent; color: rgba(255,255,255,0.24); cursor: pointer; font-size: 30px; line-height: 1; padding: 0; }
        .movie-star-fill { position: absolute; left: 0; top: 0; overflow: hidden; color: #fbbf24; pointer-events: none; }
        .movie-note { width: 100%; min-height: 76px; border: 1px solid rgba(255,255,255,0.12); border-radius: 14px; background: rgba(255,255,255,0.07); color: #f8fafc; padding: 12px; resize: vertical; }
        .movie-close { position: sticky; top: 0; float: right; margin: 12px 12px 0 0; width: 36px; height: 36px; border-radius: 999px; border: 1px solid rgba(255,255,255,0.16); background: rgba(15,23,42,0.84); color: #f8fafc; cursor: pointer; z-index: 2; }
        @media (max-width: 720px) { .movie-modal-grid { grid-template-columns: 1fr; } .movie-modal-poster { max-width: 240px; } .movie-modal-facts { grid-template-columns: 1fr; } }
      `}</style>
      <div className="movie-modal" onClick={(event) => event.stopPropagation()}>
        <button className="movie-close" onClick={onClose} aria-label="Close">×</button>
        <div className="movie-modal-grid">
          <img className="movie-modal-poster" src={movie.poster || FALLBACK_POSTER} alt={movie.title} />
          <div>
            <h2 className="movie-modal-title">{detail.title}</h2>
            <div className="movie-modal-meta">
              {[detail.year, detail.rating, detail.tmdb_runtime || detail.runtime ? `${detail.tmdb_runtime || detail.runtime} min` : ""].filter(Boolean).join(" · ")}
            </div>
            <p className="movie-modal-overview">
              {detail.tmdb_overview || detail.overview || "No overview available yet."}
            </p>
            <div className="movie-modal-facts">
              <div className="movie-fact"><div className="movie-fact-label">Director</div>{detail.tmdb_directors || detail.director || "Unknown"}</div>
              <div className="movie-fact"><div className="movie-fact-label">Cast</div>{detail.tmdb_cast_top5 || detail.star || "Unknown"}</div>
              <div className="movie-fact"><div className="movie-fact-label">Genres</div>{detail.tmdb_genres || detail.genre || "Unknown"}</div>
              <div className="movie-fact"><div className="movie-fact-label">Audience Score</div>{detail.tmdb_vote_average || detail.score || "N/A"}</div>
            </div>
            <div className="movie-star-row" aria-label="Rate movie">
              {[1, 2, 3, 4, 5].map((star) => (
                <button
                  className="movie-star"
                  key={star}
                  type="button"
                  onClick={(event) => {
                    const rect = event.currentTarget.getBoundingClientRect();
                    const half = event.clientX - rect.left < rect.width / 2;
                    setRating(half ? star - 0.5 : star);
                  }}
                >
                  ☆
                  <span className="movie-star-fill" style={{ width: `${starFillPercent(rating, star)}%` }}>★</span>
                </button>
              ))}
            </div>
            <textarea className="movie-note" value={note} onChange={(event) => setNote(event.target.value)} placeholder="Add a quick note about your rating..." />
            <div className="movie-actions">
              <button className="primary-btn" onClick={saveRating}>Save Rating</button>
              <button className="pill-btn" onClick={addToWatchlist}>Add to Watchlist</button>
              {message && <span className="movie-modal-meta">{message}</span>}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
