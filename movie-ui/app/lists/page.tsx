"use client";

import { SignInButton, UserButton, useUser } from "@clerk/nextjs";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

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

type SharedMember = {
  user_id: string;
  role: string;
  profile: {
    user_id: string;
    email: string | null;
    name: string | null;
  } | null;
};

type SharedList = {
  id: string;
  name: string;
  owner_user_id: string;
  createdAt: string;
  updatedAt: string;
  members: SharedMember[];
  movies: SavedMovie[];
};

type UnifiedList = SavedList & {
  kind: "personal" | "shared";
  sharedWith?: string[];
  owner_user_id?: string;
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
  const { user, isSignedIn, isLoaded } = useUser();
  const userId = user?.id;

  const [lists, setLists] = useState<UnifiedList[]>([]);
  const [loading, setLoading] = useState(true);

  const getSharedNames = (members: SharedMember[]) => {
  return members
    .filter((member) => member.user_id !== userId)
    .map(
      (member) =>
        member.profile?.name || member.profile?.email || "Friend"
    );
};

  useEffect(() => {
    const loadLists = async () => {
      if (!isLoaded) return;

      if (!isSignedIn || !userId) {
        setLists([]);
        setLoading(false);
        return;
      }

      setLoading(true);

      try {
        const [personalRes, sharedRes] = await Promise.all([
          fetch(`${API_BASE_URL}/user/lists`, {
            headers: { "X-User-Id": userId },
          }),
          fetch(`${API_BASE_URL}/shared-lists`, {
            headers: { "X-User-Id": userId },
          }),
        ]);

        if (!personalRes.ok) throw new Error("Failed to load personal lists");
        if (!sharedRes.ok) throw new Error("Failed to load shared lists");

        const personalData: SavedList[] = await personalRes.json();
        const sharedData: SharedList[] = await sharedRes.json();

        const personalLists: UnifiedList[] = personalData.map((list) => ({
          ...list,
          kind: "personal",
        }));

        const sharedLists: UnifiedList[] = sharedData.map((list) => ({
          id: list.id,
          name: list.name,
          movies: list.movies.map((movie) => ({
            movie_id: movie.movie_id,
            title: movie.title,
          })),
          createdAt: list.createdAt,
          kind: "shared",
          sharedWith: getSharedNames(list.members),
          owner_user_id: list.owner_user_id,
        }));

        setLists([...sharedLists, ...personalLists]);


      } catch (err) {
        console.error(err);
        setLists([]);
      } finally {
        setLoading(false);
      }
    };

    loadLists();
  }, [isLoaded, isSignedIn, userId]);

  const deleteList = async (list: UnifiedList) => {
      if (!isSignedIn || !userId) return;

      const endpoint =
        list.kind === "shared"
          ? `${API_BASE_URL}/shared-lists/${list.id}`
          : `${API_BASE_URL}/user/lists/${list.id}`;

      try {
        const res = await fetch(endpoint, {
          method: "DELETE",
          headers: {
            "X-User-Id": userId,
          },
        });

        if (!res.ok) throw new Error("Failed to delete list");

        setLists((prev) => prev.filter((item) => item.id !== list.id));
      } catch (err) {
        console.error(err);
      }
  };

  const createNewList = async () => {
      if (!isSignedIn || !userId) return;

      const res = await fetch(`${API_BASE_URL}/user/lists`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-User-Id": userId,
        },
        body: JSON.stringify({
          name: "Untitled List",
          movies: [],
        }),
      });

      if (!res.ok) return;

      const created = await res.json();
      router.push(`/edit-list?kind=personal&listId=${created.id}`);
  };

  const seeResults = (list: SavedList) => {
    localStorage.setItem("selectedMovies", JSON.stringify(list.movies));
    router.push("/results");
  };

  const editList = (list: SavedList) => {
    router.push(`/edit-list?kind=${list.kind}&listId=${list.id}`);
  };

  return (
    <main className="lists-page">
      <style>{`
        .lists-page {
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

        .secondary-btn {
          border: 1px solid rgba(255,255,255,0.12);
          background: transparent;
          color: #cbd5e1;
          border-radius: 999px;
          padding: 10px 16px;
          cursor: pointer;
          font-weight: 800;
        }

        .danger-btn {
          border: 1px solid rgba(248,113,113,0.26);
          background: rgba(127,29,29,0.18);
          color: #fecaca;
          border-radius: 999px;
          padding: 10px 16px;
          cursor: pointer;
          font-weight: 800;
        }

        .hero-title {
          font-size: clamp(1.8rem, 3.4vw, 3.2rem);
          line-height: 1.08;
          font-weight: 950;
          letter-spacing: -0.05em;
          margin: 0 0 10px 0;
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
          font-size: 22px;
          font-weight: 900;
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
          font-size: 22px;
          font-weight: 900;
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
          font-weight: 700;
        }

        .more-chip {
          background: rgba(59,130,246,0.12);
          color: #bfdbfe;
          padding: 8px 12px;
          border-radius: 999px;
          border: 1px solid rgba(147,197,253,0.18);
          font-size: 13px;
          font-weight: 800;
        }

        .list-actions {
          display: flex;
          flex-wrap: wrap;
          gap: 10px;
          margin-top: auto;
          padding-top: 16px;
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
          position: relative;
        }

        .shared-badge {
          position: absolute;
          top: 14px;
          right: 14px;
          display: inline-flex;
          align-items: center;
          gap: 6px;
          border: 1px solid rgba(196, 181, 253, 0.28);
          background: rgba(124, 58, 237, 0.18);
          color: #ddd6fe;
          border-radius: 999px;
          padding: 7px 10px;
          font-size: 12px;
          font-weight: 900;
          max-width: 180px;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .hero-sub-row {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 16px;
          margin-bottom: 28px;
        }

        .hero-sub {
          margin-bottom: 0;
        }

        @media (max-width: 760px) {
          .hero-sub-row {
            flex-direction: column;
            align-items: flex-start;
          }
        }

      `}</style>

      <div className="lists-wrap">
        <div className="top-bar">
          <button className="pill-btn" onClick={() => router.push("/")}>
            ← Search
          </button>

          <div className="top-actions">
            <button className="pill-btn" onClick={() => router.push("/friends")}>
              My Friends
            </button>

            {!isSignedIn && (
              <SignInButton mode="modal">
                <button className="primary-btn">Sign In</button>
              </SignInButton>
            )}

            {isSignedIn && <UserButton />}
          </div>
        </div>
        <section className="hero">
          <div className="hero-copy">
            <h1 className="hero-title">Your saved movie lists.</h1>

            <div className="hero-sub-row">
              <div className="hero-sub">
                Keep your favorite taste profiles organized and replay recommendations anytime.
              </div>

              <button className="primary-btn" onClick={createNewList}>
                Add New List
              </button>
            </div>
          </div>
        </section>


        {!isSignedIn && isLoaded && (
          <div className="empty-state">
            <h2 className="empty-title">Sign in to save movie lists.</h2>
            <p className="empty-copy">
              Your lists now live in the backend, so they are tied to your account.
            </p>
            <SignInButton mode="modal">
              <button className="primary-btn">Sign In</button>
            </SignInButton>
          </div>
        )}

        {isSignedIn && loading && <div className="empty-state">Loading lists...</div>}

        {isSignedIn && !loading && lists.length === 0 && (
          <div className="empty-state">
            <h2 className="empty-title">No saved lists yet.</h2>
            <p className="empty-copy">
              Search for movies, get recommendations, then save your selected favorites as a list.
            </p>
            <button className="primary-btn" onClick={() => router.push("/")}>
              Start Searching
            </button>
          </div>
        )}

        {isSignedIn && !loading && lists.length > 0 && (
          <div className="list-grid">
            {lists.map((list) => {
              const preview = list.movies.slice(0, 5);
              const remaining = Math.max(list.movies.length - preview.length, 0);

              return (
                <article className="list-card" key={list.id}>
                  {list.kind === "shared" && list.sharedWith && list.sharedWith.length > 0 && (
                      <div className="shared-badge">
                        <span>👥</span>
                        <span>+ {list.sharedWith.join(", ")}</span>
                      </div>
                  )}
                  <h2 className="list-title">{list.name}</h2>

                  <div className="list-meta">
                    {list.movies.length} movies · Saved {formatDate(list.createdAt)}
                  </div>

                  <div className="movie-preview">
                    {preview.map((movie) => (
                      <span className="movie-chip" key={movie.movie_id}>
                        {movie.title}
                      </span>
                    ))}

                    {remaining > 0 && (
                      <span className="more-chip">+{remaining} more</span>
                    )}
                  </div>

                  <div className="list-actions">
                    <button className="primary-btn" onClick={() => seeResults(list)}>
                      See Results
                    </button>
                    <button className="secondary-btn" onClick={() => editList(list)}>
                      Edit
                    </button>
                    <button
                      className="danger-btn"
                      onClick={() => {
                        const confirmed = window.confirm(
                          `Are you sure you want to delete "${list.name}"?`
                        );

                        if (confirmed) {
                          deleteList(list);
                        }
                      }}
                    >
                      Delete
                    </button>
                  </div>
                </article>
              );
            })}
          </div>
        )}
      </div>
    </main>
  );
}