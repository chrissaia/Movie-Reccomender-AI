"use client";

import { Show, SignInButton, UserButton } from "@clerk/nextjs";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

type SavedMovie = {
  movie_id: number;
  title: string;
};

type SavedList = {
  id: string;
  name: string;
  movies: SavedMovie[];
  createdAt: string;
};

function formatDate(value: string) {
  try {
    return new Date(value).toLocaleDateString();
  } catch {
    return value;
  }
}

export default function ListsPage() {
  const router = useRouter();
  const [lists, setLists] = useState<SavedList[]>([]);

  useEffect(() => {
    const raw = localStorage.getItem("savedMovieLists");
    const parsed: SavedList[] = raw ? JSON.parse(raw) : [];
    setLists(parsed);
  }, []);

  const deleteList = (id: string) => {
    const updated = lists.filter((list) => list.id !== id);
    setLists(updated);
    localStorage.setItem("savedMovieLists", JSON.stringify(updated));
  };

  const seeResults = (list: SavedList) => {
    localStorage.setItem("selectedMovies", JSON.stringify(list.movies));
    router.push("/results");
  };

  const editList = (list: SavedList) => {
      router.push(`/edit-list?id=${list.id}`);
  };

  return (
    <main className="lists-page">
      <style>{`
        .lists-page {
          min-height: 100vh;
          padding: 28px 24px 56px;
          background: linear-gradient(135deg, #020617 0%, #0f172a 35%, #1e3a8a 100%);
        }

        .lists-wrap {
          max-width: 1200px;
          margin: 0 auto;
          position: relative;
        }

        .top-bar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 16px;
          margin-bottom: 28px;
        }

        .top-actions {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .top-user {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .pill-btn {
          border: 1px solid rgba(255,255,255,0.12);
          background: rgba(255,255,255,0.06);
          color: #e2e8f0;
          border-radius: 999px;
          padding: 10px 16px;
          cursor: pointer;
          font-weight: 700;
        }

        .pill-btn:hover {
          background: rgba(255,255,255,0.10);
        }

        .hero-title {
          font-size: clamp(1.8rem, 3.4vw, 3.2rem);
          line-height: 1.08;
          font-weight: 900;
          letter-spacing: -0.05em;
          color: #f8fafc;
          margin: 0 0 10px 0;
          max-width: 900px;
        }

        .hero-sub {
          color: rgba(255,255,255,0.72);
          font-size: 17px;
          margin-bottom: 28px;
        }

        .empty-state {
          border: 1px solid rgba(255,255,255,0.08);
          background: rgba(255,255,255,0.05);
          border-radius: 24px;
          padding: 28px;
          color: #cbd5e1;
          box-shadow: 0 16px 40px rgba(0,0,0,0.28);
        }

        .empty-title {
          color: #f8fafc;
          font-size: 22px;
          font-weight: 800;
          margin: 0 0 8px 0;
        }

        .empty-copy {
          color: rgba(255,255,255,0.72);
          line-height: 1.6;
          margin-bottom: 16px;
        }

        .list-grid {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 18px;
        }

        @media (max-width: 1100px) {
          .list-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
          }
        }

        @media (max-width: 700px) {
          .list-grid {
            grid-template-columns: 1fr;
          }
        }

        .list-card {
          border: 1px solid rgba(255,255,255,0.08);
          background: rgba(255,255,255,0.05);
          border-radius: 24px;
          padding: 20px;
          box-shadow: 0 16px 40px rgba(0,0,0,0.28);
          min-height: 320px;

          display: flex;
          flex-direction: column;
        }

        .list-title {
          color: #f8fafc;
          font-size: 22px;
          font-weight: 800;
          margin: 0 0 10px 0;
          letter-spacing: -0.03em;
        }

        .list-meta {
          color: rgba(255,255,255,0.72);
          font-size: 14px;
          margin-bottom: 14px;
        }

        .movie-preview {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
          margin-bottom: 18px;
          min-height: 88px;
          align-items: flex-start;
        }

        .movie-chip {
          background: rgba(255,255,255,0.08);
          color: #e2e8f0;
          padding: 8px 12px;
          border-radius: 999px;
          border: 1px solid rgba(255,255,255,0.08);
          font-size: 13px;
          font-weight: 600;
          max-width: 100%;
          display: inline-flex;
          align-items: center;
          flex: 0 0 auto;
        }

        .more-chip {
          background: rgba(59,130,246,0.12);
          color: #bfdbfe;
          padding: 8px 12px;
          border-radius: 999px;
          border: 1px solid rgba(147,197,253,0.18);
          font-size: 13px;
          font-weight: 700;
          cursor: pointer;
        }

        .list-actions {
          display: flex;
          flex-wrap: wrap;
          gap: 10px;
          margin-top: auto;
          padding-top: 16px;
        }

        .primary-btn {
          border: none;
          background: linear-gradient(135deg, #3b82f6, #2563eb);
          color: white;
          border-radius: 999px;
          padding: 10px 16px;
          cursor: pointer;
          font-weight: 700;
        }

        .secondary-btn {
          border: 1px solid rgba(255,255,255,0.12);
          background: transparent;
          color: #cbd5e1;
          border-radius: 999px;
          padding: 10px 16px;
          cursor: pointer;
          font-weight: 700;
        }

        .secondary-btn:hover,
        .primary-btn:hover {
          filter: brightness(1.05);
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
      `}</style>

      <div className="lists-wrap">
        <div className="top-bar">
          <div className="top-actions">
            <button className="pill-btn" onClick={() => router.push("/")}>
              ← Home
            </button>
          </div>

          <div className="top-user">
            <Show when="signed-out" fallback={<UserButton />}>
              <SignInButton mode="modal">
                <button className="pill-btn">Sign In</button>
              </SignInButton>
            </Show>
          </div>
        </div>

        <h1 className="hero-title">My Lists</h1>
        <p className="hero-sub">
          Save your favorite starting points and come back to them anytime.
        </p>

        {lists.length === 0 ? (
          <div className="empty-state">
            <h2 className="empty-title">No saved lists yet</h2>
            <div className="empty-copy">
              Go generate some recommendations, then save a list from the results page.
            </div>
            <button className="primary-btn" onClick={() => router.push("/")}>
              Find Movies
            </button>
          </div>
        ) : (
          <div className="list-grid">
            {lists.map((list) => (
              <div key={list.id} className="list-card">
                <h2 className="list-title">{list.name}</h2>
                <div className="list-meta">
                  {list.movies.length} movies • Saved {formatDate(list.createdAt)}
                </div>

                <div className="movie-preview">
                  {list.movies.slice(0, 4).map((movie) => (
                    <div key={`${list.id}-${movie.movie_id}`} className="movie-chip">
                      {movie.title}
                    </div>
                  ))}

                  {list.movies.length > 4 && (
                    <button className="more-chip" onClick={() => editList(list)}>
                      +{list.movies.length - 4} more
                    </button>
                  )}
                </div>

                <div className="list-actions">
                  <button className="primary-btn" onClick={() => seeResults(list)}>
                    See Results
                  </button>
                  <button className="secondary-btn" onClick={() => editList(list)}>
                    Edit List
                  </button>
                  <button className="secondary-btn" onClick={() => deleteList(list.id)}>
                    Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </main>
  );
}