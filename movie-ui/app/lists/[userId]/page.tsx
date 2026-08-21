"use client";

import { SignInButton, useUser } from "@clerk/nextjs";
import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";

import AppHeader from "../../components/AppHeader";
import { API_BASE_URL } from "../../lib/config";
import { formatDate } from "../../lib/format";
import { logHandledError } from "../../lib/log";
import type { SavedList, SavedMovie } from "../../types";

type FriendList = SavedList & {
  kind?: "personal" | "shared";
  owner_user_id?: string;
  updatedAt?: string;
};

type FriendListsPayload = {
  profile: {
    user_id: string;
    name: string | null;
    avatar_url: string | null;
  };
  lists: FriendList[];
};

export default function FriendListsPage() {
  const params = useParams<{ userId: string }>();
  const router = useRouter();
  const { user, isLoaded, isSignedIn } = useUser();
  const viewerUserId = user?.id;

  const [payload, setPayload] = useState<FriendListsPayload | null>(null);
  const [myLists, setMyLists] = useState<SavedList[]>([]);
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState("");
  const [copyMessages, setCopyMessages] = useState<Record<string, string>>({});
  const [addMessages, setAddMessages] = useState<Record<string, string>>({});
  const [pickerMovie, setPickerMovie] = useState<SavedMovie | null>(null);

  useEffect(() => {
    const loadLists = async () => {
      if (!isLoaded) return;

      if (!isSignedIn || !viewerUserId || !params.userId) {
        setLoading(false);
        return;
      }

      setLoading(true);
      setMessage("");

      try {
        const [friendRes, mineRes] = await Promise.all([
          fetch(`${API_BASE_URL}/profiles/${params.userId}/lists`, {
            headers: { "X-User-Id": viewerUserId },
          }),
          fetch(`${API_BASE_URL}/user/lists`, {
            headers: { "X-User-Id": viewerUserId },
          }),
        ]);

        if (!friendRes.ok) {
          setMessage("These lists are not available.");
          setPayload(null);
          return;
        }

        setPayload(await friendRes.json());
        setMyLists(mineRes.ok ? await mineRes.json() : []);
      } catch (err) {
        logHandledError("Friend lists load failed", err);
        setMessage("Could not reach the list service.");
      } finally {
        setLoading(false);
      }
    };

    loadLists();
  }, [isLoaded, isSignedIn, params.userId, viewerUserId]);

  const displayName = payload?.profile.name || "Movie Friend";

  const seeResults = (list: FriendList) => {
    localStorage.setItem("selectedMovies", JSON.stringify(list.movies));
    router.push("/results");
  };

  const setCopyMessage = (listId: string, value: string) => {
    setCopyMessages((current) => ({ ...current, [listId]: value }));
  };

  const setAddMessage = (movieId: number, value: string) => {
    setAddMessages((current) => ({ ...current, [String(movieId)]: value }));
  };

  const copyList = async (list: FriendList) => {
    if (!viewerUserId || !params.userId) return;

    setCopyMessage(list.id, "Copying...");

    try {
      const res = await fetch(
        `${API_BASE_URL}/profiles/${params.userId}/lists/${list.id}/copy`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-User-Id": viewerUserId,
          },
          body: JSON.stringify({ name: `Copy of ${list.name}` }),
        },
      );

      setCopyMessage(
        list.id,
        res.ok
          ? `Copied "${list.name}" to your lists.`
          : "Could not copy that list.",
      );

      if (res.ok) {
        const created = await res.json();
        setMyLists((current) => [created, ...current]);
      }
    } catch (err) {
      logHandledError("Friend list copy failed", err);
      setCopyMessage(list.id, "Could not reach the list service.");
    }
  };

  const addMovieToList = async (targetList: SavedList) => {
    if (!viewerUserId || !pickerMovie) return;

    const alreadySaved = targetList.movies.some(
      (movie) => movie.movie_id === pickerMovie.movie_id,
    );

    if (alreadySaved) {
      setAddMessage(pickerMovie.movie_id, `Already in "${targetList.name}".`);
      return;
    }

    const nextMovies = [...targetList.movies, pickerMovie];
    setAddMessage(pickerMovie.movie_id, `Adding to "${targetList.name}"...`);

    try {
      const res = await fetch(`${API_BASE_URL}/user/lists/${targetList.id}`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          "X-User-Id": viewerUserId,
        },
        body: JSON.stringify({
          name: targetList.name,
          movies: nextMovies,
        }),
      });

      if (!res.ok) {
        setAddMessage(pickerMovie.movie_id, "Could not add that movie.");
        return;
      }

      const updated = await res.json();
      setMyLists((current) =>
        current.map((list) => (list.id === updated.id ? updated : list)),
      );
      setAddMessage(pickerMovie.movie_id, `Added to "${targetList.name}".`);
      setPickerMovie(null);
    } catch (err) {
      logHandledError("Friend list movie add failed", err);
      setAddMessage(pickerMovie.movie_id, "Could not reach the list service.");
    }
  };

  return (
    <main className="friend-lists-page">
      <style>{`
        .friend-lists-page {
          min-height: 100vh;
          padding: 28px 24px 56px;
          background:
            radial-gradient(circle at 18% 12%, rgba(59, 130, 246, 0.20), transparent 34%),
            radial-gradient(circle at 82% 8%, rgba(30, 64, 175, 0.18), transparent 30%),
            linear-gradient(135deg, #020617 0%, #0f172a 35%, #1e3a8a 100%);
          color: #f8fafc;
        }

        .lists-wrap {
          max-width: 1200px;
          margin: 0 auto;
        }

        .hero-title {
          font-size: clamp(1.8rem, 3.4vw, 3.2rem);
          line-height: 1.08;
          font-weight: 950;
          letter-spacing: -0.05em;
          margin: 32px 0 10px 0;
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

        .list-grid {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 18px;
        }

        .list-card {
          position: relative;
          border: 1px solid rgba(255,255,255,0.08);
          background: rgba(255,255,255,0.05);
          border-radius: 24px;
          padding: 20px;
          box-shadow: 0 16px 40px rgba(0,0,0,0.28);
          min-height: 300px;
          display: flex;
          flex-direction: column;
        }

        .shared-badge {
          position: absolute;
          top: 18px;
          right: 18px;
          display: inline-flex;
          align-items: center;
          gap: 6px;
          max-width: calc(100% - 36px);
          border: 1px solid rgba(147,197,253,0.22);
          background: rgba(59,130,246,0.16);
          color: #dbeafe;
          border-radius: 999px;
          padding: 7px 10px;
          font-size: 12px;
          font-weight: 900;
        }

        .list-title {
          font-size: 22px;
          font-weight: 900;
          margin: 0 0 10px;
          padding-right: 116px;
          letter-spacing: -0.03em;
        }

        .list-card:not(.is-shared) .list-title {
          padding-right: 0;
        }

        .list-meta {
          color: rgba(255,255,255,0.72);
          font-size: 14px;
          margin-bottom: 14px;
        }

        .movie-preview {
          display: grid;
          grid-template-columns: 1fr;
          gap: 8px;
          margin-bottom: 18px;
          align-items: flex-start;
        }

        .movie-card {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 10px;
          background: rgba(255,255,255,0.08);
          color: #e2e8f0;
          padding: 10px 12px;
          border-radius: 14px;
          border: 1px solid rgba(255,255,255,0.08);
          font-size: 13px;
          font-weight: 800;
        }

        .movie-title {
          min-width: 0;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .mini-btn {
          flex: 0 0 auto;
          border: 1px solid rgba(255,255,255,0.12);
          background: rgba(255,255,255,0.06);
          color: #dbeafe;
          border-radius: 999px;
          padding: 7px 10px;
          cursor: pointer;
          font-size: 12px;
          font-weight: 900;
        }

        .more-chip {
          justify-self: start;
          background: rgba(59,130,246,0.12);
          color: #bfdbfe;
          border-color: rgba(147,197,253,0.18);
          font-weight: 800;
          padding: 8px 12px;
          border-radius: 999px;
          border: 1px solid rgba(255,255,255,0.08);
          font-size: 13px;
        }

        .list-actions {
          display: flex;
          flex-wrap: wrap;
          gap: 10px;
          margin-top: auto;
          padding-top: 16px;
        }

        .copy-area {
          margin-top: auto;
        }

        .inline-message {
          width: 100%;
          min-height: 18px;
          margin: 8px 0 0;
          color: #bfdbfe;
          font-size: 13px;
          font-weight: 800;
        }

        .primary-btn,
        .secondary-btn {
          border-radius: 999px;
          padding: 10px 16px;
          cursor: pointer;
          font-weight: 800;
        }

        .primary-btn {
          border: none;
          background: linear-gradient(135deg, #3b82f6, #2563eb);
          color: white;
        }

        .secondary-btn {
          border: 1px solid rgba(255,255,255,0.12);
          background: transparent;
          color: #cbd5e1;
        }

        .picker-backdrop {
          position: fixed;
          inset: 0;
          z-index: 40;
          display: grid;
          place-items: center;
          padding: 20px;
          background: rgba(2,6,23,0.72);
        }

        .picker-panel {
          width: min(480px, 100%);
          border: 1px solid rgba(255,255,255,0.12);
          background: #111827;
          border-radius: 20px;
          padding: 20px;
          box-shadow: 0 24px 80px rgba(0,0,0,0.42);
        }

        .picker-head {
          display: flex;
          justify-content: space-between;
          gap: 16px;
          align-items: flex-start;
          margin-bottom: 16px;
        }

        .picker-title {
          margin: 0 0 4px;
          font-size: 20px;
          font-weight: 950;
        }

        .picker-sub {
          color: rgba(255,255,255,0.68);
          font-size: 14px;
        }

        .picker-close {
          border: 1px solid rgba(255,255,255,0.12);
          background: rgba(255,255,255,0.06);
          color: #e5e7eb;
          border-radius: 999px;
          cursor: pointer;
          font-weight: 900;
          height: 34px;
          width: 34px;
        }

        .picker-list {
          display: grid;
          gap: 10px;
        }

        .picker-list button {
          width: 100%;
          text-align: left;
          border: 1px solid rgba(255,255,255,0.10);
          background: rgba(255,255,255,0.06);
          color: #f8fafc;
          border-radius: 14px;
          padding: 12px 14px;
          cursor: pointer;
          font-weight: 900;
        }

        @media (max-width: 1100px) {
          .list-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }

        @media (max-width: 700px) {
          .list-grid { grid-template-columns: 1fr; }
          .list-title { padding-right: 0; }
          .shared-badge { position: static; margin-bottom: 12px; }
        }
      `}</style>

      <div className="lists-wrap">
        <AppHeader
          leading={{ label: "← Profile", href: `/profile/${params.userId}` }}
          actions={[{ label: "My Lists", href: "/lists" }]}
        />

        <section className="hero">
          <h1 className="hero-title">{displayName}&apos;s movie lists.</h1>
          <div className="hero-sub">
            View their saved lists, copy one into your account, or use a list
            for recommendations.
          </div>
        </section>

        {!isSignedIn && isLoaded && (
          <div className="empty-state">
            <h2>Sign in to view friend lists.</h2>
            <SignInButton mode="modal">
              <button className="primary-btn">Sign In</button>
            </SignInButton>
          </div>
        )}

        {isSignedIn && loading && (
          <div className="empty-state">Loading lists...</div>
        )}
        {message && (
          <div className="empty-state" style={{ marginBottom: 18 }}>
            {message}
          </div>
        )}

        {isSignedIn && !loading && payload && payload.lists.length === 0 && (
          <div className="empty-state">No lists available yet.</div>
        )}

        {isSignedIn && !loading && payload && payload.lists.length > 0 && (
          <div className="list-grid">
            {payload.lists.map((list) => {
              const preview = list.movies.slice(0, 5);
              const remaining = Math.max(
                list.movies.length - preview.length,
                0,
              );
              const isShared = list.kind === "shared";

              return (
                <article
                  className={`list-card ${isShared ? "is-shared" : ""}`}
                  key={list.id}
                >
                  {isShared && (
                    <div className="shared-badge" aria-label="Shared list">
                      <span>👥</span>
                      <span>Shared</span>
                    </div>
                  )}

                  <h2 className="list-title">{list.name}</h2>
                  <div className="list-meta">
                    {list.movies.length} movies · Saved{" "}
                    {formatDate(list.createdAt)}
                  </div>

                  <div className="movie-preview">
                    {preview.map((movie) => (
                      <div
                        className="movie-card"
                        key={`${list.id}-${movie.movie_id}`}
                      >
                        <span className="movie-title">{movie.title}</span>
                        <button
                          className="mini-btn"
                          onClick={() => setPickerMovie(movie)}
                        >
                          Add to List
                        </button>
                      </div>
                    ))}
                    {remaining > 0 && (
                      <span className="more-chip">+{remaining} more</span>
                    )}
                  </div>

                  <div className="copy-area">
                    <div className="list-actions">
                      <button
                        className="secondary-btn"
                        onClick={() => copyList(list)}
                      >
                        Copy
                      </button>
                      <button
                        className="primary-btn"
                        onClick={() => seeResults(list)}
                      >
                        See Results
                      </button>
                    </div>
                    {copyMessages[list.id] && (
                      <div className="inline-message">
                        {copyMessages[list.id]}
                      </div>
                    )}
                  </div>
                </article>
              );
            })}
          </div>
        )}
      </div>

      {pickerMovie && (
        <div className="picker-backdrop" role="dialog" aria-modal="true">
          <div className="picker-panel">
            <div className="picker-head">
              <div>
                <h2 className="picker-title">Add to list</h2>
                <div className="picker-sub">{pickerMovie.title}</div>
              </div>
              <button
                className="picker-close"
                onClick={() => setPickerMovie(null)}
                aria-label="Close"
              >
                ×
              </button>
            </div>

            {myLists.length === 0 ? (
              <div className="empty-state">
                You do not have any personal lists yet.
              </div>
            ) : (
              <div className="picker-list">
                {myLists.map((list) => (
                  <button key={list.id} onClick={() => addMovieToList(list)}>
                    {list.name} · {list.movies.length} movies
                  </button>
                ))}
              </div>
            )}

            {addMessages[String(pickerMovie.movie_id)] && (
              <div className="inline-message">
                {addMessages[String(pickerMovie.movie_id)]}
              </div>
            )}
          </div>
        </div>
      )}
    </main>
  );
}
