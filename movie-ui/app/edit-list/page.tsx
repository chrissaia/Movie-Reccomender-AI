"use client";

import { Show, SignInButton, UserButton } from "@clerk/nextjs";
import { useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

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

type SearchMovie = {
  movie_id: number;
  title: string;
};

type MovieSearchResult = SearchMovie & {
  poster?: string;
};

const TMDB_API_KEY = "5d1421370bac86b6d373cf6be6f0e942";
const TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w185";
const FALLBACK_POSTER = "/no-poster.png";

function formatDate(value: string) {
  try {
    return new Date(value).toLocaleDateString();
  } catch {
    return value;
  }
}

async function getTmdbPoster(title: string): Promise<string | undefined> {
  try {
    const res = await fetch(
      `https://api.themoviedb.org/3/search/movie?query=${encodeURIComponent(title)}&api_key=${TMDB_API_KEY}`
    );
    const data = await res.json();

    const match =
      data?.results?.find(
        (m: { title?: string }) =>
          String(m.title ?? "").toLowerCase() === title.toLowerCase()
      ) || data?.results?.[0];

    if (!match?.poster_path) return undefined;
    return `${TMDB_IMAGE_BASE}${match.poster_path}`;
  } catch {
    return undefined;
  }
}

export default function EditListsPage() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const [lists, setLists] = useState<SavedList[]>([]);
  const [selectedListId, setSelectedListId] = useState<string | null>(null);
  const [draftName, setDraftName] = useState("");

  const [query, setQuery] = useState("");
  const [results, setResults] = useState<MovieSearchResult[]>([]);
  const [loadingSearch, setLoadingSearch] = useState(false);
  const [saveMessage, setSaveMessage] = useState("");

  useEffect(() => {
    const raw = localStorage.getItem("savedMovieLists");
    const parsed: SavedList[] = raw ? JSON.parse(raw) : [];
    setLists(parsed);

    const listIdFromUrl = searchParams.get("listId");
    if (listIdFromUrl && parsed.some((list) => list.id === listIdFromUrl)) {
      setSelectedListId(listIdFromUrl);
      const found = parsed.find((list) => list.id === listIdFromUrl);
      setDraftName(found?.name ?? "");
    } else if (parsed.length > 0) {
      setSelectedListId(parsed[0].id);
      setDraftName(parsed[0].name);
    }
  }, [searchParams]);

  useEffect(() => {
    const runSearch = async () => {
      if (!query.trim()) {
        setResults([]);
        return;
      }

      setLoadingSearch(true);
      try {
        const res = await fetch(
          `http://127.0.0.1:8000/movies/search?q=${encodeURIComponent(query)}`
        );
        const data = await res.json();
        const baseResults: SearchMovie[] = Array.isArray(data) ? data : [];

        const withPosters = await Promise.all(
          baseResults.map(async (movie) => ({
            ...movie,
            poster: await getTmdbPoster(movie.title),
          }))
        );

        setResults(withPosters);
      } catch (err) {
        console.error(err);
        setResults([]);
      } finally {
        setLoadingSearch(false);
      }
    };

    const timeout = setTimeout(runSearch, 250);
    return () => clearTimeout(timeout);
  }, [query]);

  const persistLists = (nextLists: SavedList[]) => {
    setLists(nextLists);
    localStorage.setItem("savedMovieLists", JSON.stringify(nextLists));
  };

  const selectedList = useMemo(
    () => lists.find((list) => list.id === selectedListId) ?? null,
    [lists, selectedListId]
  );

  const filteredSearchResults = useMemo(() => {
    if (!selectedList) return results;
    const existingIds = new Set(selectedList.movies.map((movie) => movie.movie_id));
    return results.filter((movie) => !existingIds.has(movie.movie_id));
  }, [results, selectedList]);

  const chooseList = (list: SavedList) => {
    setSelectedListId(list.id);
    setDraftName(list.name);
    setSaveMessage("");
  };

  const saveRename = () => {
    if (!selectedList) return;

    const trimmed = draftName.trim();
    if (!trimmed) return;

    const nextLists = lists.map((list) =>
      list.id === selectedList.id ? { ...list, name: trimmed } : list
    );
    persistLists(nextLists);

    setSaveMessage(`Saved "${trimmed}"`);
    setTimeout(() => setSaveMessage(""), 2200);
  };

  const removeMovie = (movieId: number) => {
    if (!selectedList) return;

    const nextLists = lists.map((list) =>
      list.id === selectedList.id
        ? { ...list, movies: list.movies.filter((movie) => movie.movie_id !== movieId) }
        : list
    );

    persistLists(nextLists);
  };

  const addMovie = (movie: SearchMovie) => {
    if (!selectedList) return;

    const nextLists = lists.map((list) =>
      list.id === selectedList.id
        ? {
            ...list,
            movies: [...list.movies, { movie_id: movie.movie_id, title: movie.title }],
          }
        : list
    );

    persistLists(nextLists);
    setQuery("");
    setResults([]);
  };

  const deleteList = () => {
    if (!selectedList) return;

    const nextLists = lists.filter((list) => list.id !== selectedList.id);
    persistLists(nextLists);

    if (nextLists.length > 0) {
      setSelectedListId(nextLists[0].id);
      setDraftName(nextLists[0].name);
    } else {
      setSelectedListId(null);
      setDraftName("");
    }

    setSaveMessage("");
  };

  const seeResults = () => {
    if (!selectedList || selectedList.movies.length === 0) return;
    localStorage.setItem("selectedMovies", JSON.stringify(selectedList.movies));
    router.push("/results");
  };

  return (
    <main className="edit-page">
      <style>{`
        .edit-page {
          min-height: 140vh;
          padding: 28px 24px 56px;
          background: linear-gradient(135deg, #020617 0%, #0f172a 35%, #1e3a8a 100%);
        }

        .edit-wrap {
          max-width: 1440px;
          margin: 0 auto;
        }

        .top-bar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 16px;
          margin-bottom: 28px;
        }

        .top-left {
          display: flex;
          align-items: center;
          gap: 12px;
          flex-wrap: wrap;
        }

        .top-right {
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
          font-size: 14px;
        }

        .pill-btn:hover {
          background: rgba(255,255,255,0.10);
        }

        .hero-title {
          font-size: clamp(1.9rem, 3.4vw, 3.3rem);
          line-height: 1.08;
          font-weight: 900;
          letter-spacing: -0.05em;
          color: #f8fafc;
          margin: 0 0 10px 0;
        }

        .hero-sub {
          color: rgba(255,255,255,0.72);
          font-size: 17px;
          margin-bottom: 28px;
          max-width: 900px;
        }

        .layout {
          display: grid;
          grid-template-columns: 280px minmax(0, 1fr);
          gap: 18px;
          align-items: start;
        }

        .main-stack {
          display: flex;
          flex-direction: column;
          gap: 18px;
          min-width: 0;
        }

        .panel {
          border: 1px solid rgba(255,255,255,0.08);
          background: rgba(255,255,255,0.05);
          border-radius: 24px;
          padding: 20px;
          box-shadow: 0 16px 40px rgba(0,0,0,0.28);
          min-width: 0;
        }

        .panel-title {
          color: #f8fafc;
          font-size: 22px;
          font-weight: 800;
          margin: 0 0 14px 0;
          letter-spacing: -0.03em;
        }

        .list-nav {
          display: flex;
          flex-direction: column;
          gap: 10px;
        }

        .list-nav-item {
          width: 100%;
          text-align: left;
          border: 1px solid rgba(255,255,255,0.08);
          background: rgba(255,255,255,0.05);
          color: #e2e8f0;
          border-radius: 18px;
          padding: 12px 14px;
          cursor: pointer;
        }

        .list-nav-item.active {
          background: rgba(59,130,246,0.18);
          border-color: rgba(147,197,253,0.32);
        }

        .list-nav-name {
          font-weight: 800;
          margin-bottom: 4px;
        }

        .list-nav-meta {
          font-size: 13px;
          color: rgba(255,255,255,0.7);
        }

        .editor-header {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 16px;
          margin-bottom: 18px;
        }

        .editor-main {
          flex: 1;
          min-width: 0;
        }

        .editor-actions {
          display: flex;
          gap: 10px;
          flex-wrap: wrap;
          align-items: flex-start;
          justify-content: flex-end;
          flex: 0 0 auto;
        }

        .save-btn-wrap {
          position: relative;
          display: inline-flex;
          flex-direction: column;
          align-items: center;
          min-width: 108px;
        }

        .save-msg {
          position: absolute;
          top: calc(100% + 8px);
          left: 50%;
          transform: translateX(-50%);
          margin-top: 0;
          color: #93c5fd;
          font-size: 14px;
          font-weight: 700;
          white-space: nowrap;
          text-align: center;
        }

        .list-name-input {
          width: 100%;
          padding: 16px 18px;
          border-radius: 16px;
          border: 1px solid rgba(255,255,255,0.12);
          background: rgba(255,255,255,0.06);
          color: #f8fafc;
          outline: none;
          font-size: 22px;
          font-weight: 800;
          margin-bottom: 10px;
        }

        .list-meta {
          color: rgba(255,255,255,0.72);
          font-size: 14px;
          margin-bottom: 18px;
        }

        .movie-list {
          display: flex;
          flex-wrap: wrap;
          gap: 12px;
          min-height: 84px;
          align-items: flex-start;
        }

        .movie-chip {
          display: inline-flex;
          align-items: center;
          gap: 10px;
          background: rgba(255,255,255,0.08);
          color: #e2e8f0;
          padding: 10px 14px;
          border-radius: 999px;
          border: 1px solid rgba(255,255,255,0.08);
          font-size: 14px;
          font-weight: 600;
          max-width: 100%;
          flex: 0 0 auto;
        }

        .movie-chip-text {
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
          max-width: 260px;
        }

        .chip-remove {
          border: none;
          background: transparent;
          color: #93c5fd;
          cursor: pointer;
          font-size: 18px;
          font-weight: 800;
          line-height: 1;
          padding: 0;
        }

        .search-head {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 16px;
          margin-bottom: 14px;
        }

        .search-title-copy {
          color: rgba(255,255,255,0.72);
          font-size: 14px;
        }

        .search-input {
          width: 100%;
          padding: 15px 16px;
          border-radius: 16px;
          border: 1px solid rgba(255,255,255,0.12);
          background: rgba(255,255,255,0.06);
          color: #f8fafc;
          outline: none;
          font-size: 16px;
          margin-bottom: 18px;
        }

        .search-results-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
          gap: 16px;
        }

        .search-card {
          border: 1px solid rgba(255,255,255,0.08);
          background: rgba(255,255,255,0.05);
          color: #e2e8f0;
          border-radius: 18px;
          padding: 12px;
          display: flex;
          flex-direction: column;
          gap: 10px;
          min-width: 0;
        }

        .search-poster {
          width: 100%;
          aspect-ratio: 2 / 3;
          object-fit: cover;
          display: block;
          border-radius: 12px;
          background: rgba(255,255,255,0.05);
        }

        .search-title {
          font-weight: 700;
          line-height: 1.35;
          min-height: 38px;
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

        .danger-btn {
          border: 1px solid rgba(255,255,255,0.12);
          background: rgba(239,68,68,0.12);
          color: #fecaca;
          border-radius: 999px;
          padding: 10px 16px;
          cursor: pointer;
          font-weight: 700;
        }

        .empty-copy {
          color: rgba(255,255,255,0.72);
          line-height: 1.7;
        }

        @media (max-width: 1180px) {
          .layout {
            grid-template-columns: 1fr;
          }
        }

        @media (max-width: 900px) {
          .editor-header {
            flex-direction: column;
          }

          .editor-actions {
            justify-content: flex-start;
          }
        }
      `}</style>

      <div className="edit-wrap">
        <div className="top-bar">
          <div className="top-left">
            <button className="pill-btn" onClick={() => router.push("/")}>
              ← Home
            </button>
          </div>

          <div className="top-right">
            <button
              onClick={() => router.push("/lists")}
              className="pill-btn"
            >
              My Lists
            </button>

            <Show when="signed-out" fallback={<UserButton />}>
              <SignInButton mode="modal">
                <button className="pill-btn">Sign In</button>
              </SignInButton>
            </Show>
          </div>
        </div>

        <h1 className="hero-title">Edit Lists</h1>
        <p className="hero-sub">
          Build the list at the top, then search below with poster results so adding feels more natural.
        </p>

        {lists.length === 0 ? (
          <div className="panel">
            <h2 className="panel-title">No saved lists yet</h2>
            <div className="empty-copy">
              Save a list from the results page first, then come back here to edit it.
            </div>
          </div>
        ) : (
          <div className="layout">
            <aside className="panel">
              <h2 className="panel-title">Your Lists</h2>
              <div className="list-nav">
                {lists.map((list) => (
                  <button
                    key={list.id}
                    className={`list-nav-item ${selectedListId === list.id ? "active" : ""}`}
                    onClick={() => chooseList(list)}
                  >
                    <div className="list-nav-name">{list.name}</div>
                    <div className="list-nav-meta">
                      {list.movies.length} movies • {formatDate(list.createdAt)}
                    </div>
                  </button>
                ))}
              </div>
            </aside>

            <section className="main-stack">
              <div className="panel">
                {selectedList ? (
                  <>
                    <div className="editor-header">
                      <div className="editor-main">
                        <input
                          className="list-name-input"
                          value={draftName}
                          onChange={(e) => setDraftName(e.target.value)}
                          placeholder="List name"
                        />
                        <div className="list-meta">
                          {selectedList.movies.length} movies • Saved {formatDate(selectedList.createdAt)}
                        </div>
                      </div>

                      <div className="editor-actions">
                        <button className="primary-btn" onClick={seeResults}>
                          See Results
                        </button>

                        <div className="save-btn-wrap">
                          <button className="secondary-btn" onClick={saveRename}>
                            Save List
                          </button>
                          {saveMessage && <div className="save-msg">{saveMessage}</div>}
                        </div>

                        <button className="danger-btn" onClick={deleteList}>
                          Delete List
                        </button>
                      </div>
                    </div>

                    <div className="movie-list">
                      {selectedList.movies.map((movie) => (
                        <div key={movie.movie_id} className="movie-chip">
                          <span className="movie-chip-text">{movie.title}</span>
                          <button
                            className="chip-remove"
                            onClick={() => removeMovie(movie.movie_id)}
                            aria-label={`Remove ${movie.title}`}
                            title={`Remove ${movie.title}`}
                          >
                            ×
                          </button>
                        </div>
                      ))}
                    </div>
                  </>
                ) : (
                  <div className="empty-copy">Select a list to edit.</div>
                )}
              </div>

              <div className="panel">
                <div className="search-head">
                  <div>
                    <h2 className="panel-title" style={{ marginBottom: 6 }}>Add Movies</h2>
                    <div className="search-title-copy">
                      Search below and add directly from poster cards.
                    </div>
                  </div>
                </div>

                <input
                  className="search-input"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Search movies to add..."
                />

                {loadingSearch && <div className="empty-copy">Searching...</div>}

                {!loadingSearch && query.trim() && filteredSearchResults.length === 0 && (
                  <div className="empty-copy">No movies found.</div>
                )}

                {!loadingSearch && !query.trim() && (
                  <div className="empty-copy">
                    Start typing to search for movies and add them to the current list.
                  </div>
                )}

                {!loadingSearch && filteredSearchResults.length > 0 && (
                  <div className="search-results-grid">
                    {filteredSearchResults.slice(0, 6).map((movie) => (
                      <div key={movie.movie_id} className="search-card">
                        <img
                          src={movie.poster || FALLBACK_POSTER}
                          alt={movie.title}
                          className="search-poster"
                          onError={(e) => {
                            e.currentTarget.src = FALLBACK_POSTER;
                          }}
                        />
                        <div className="search-title">{movie.title}</div>
                        <button className="primary-btn" onClick={() => addMovie(movie)}>
                          Add
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </section>
          </div>
        )}
      </div>
    </main>
  );
}