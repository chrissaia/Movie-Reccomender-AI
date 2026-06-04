"use client";

import { SignInButton, UserButton, useUser } from "@clerk/nextjs";
import { useEffect, useMemo, useState, kindFromUrl } from "react";
import { useRouter, useSearchParams } from "next/navigation";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

const TMDB_API_KEY = process.env.NEXT_PUBLIC_TMDB_API_KEY ?? "";
const TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w185";
const FALLBACK_POSTER = "/no-poster.png";

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

type SharedMember = {
  user_id: string;
  role: string;
  profile: {
    user_id: string;
    email: string | null;
    name: string | null;
  } | null;
};

type Friendship = {
  id: string;
  other_user_id: string;
  other_user: {
    user_id: string;
    email: string | null;
    name: string | null;
  } | null;
};

type FriendsResponse = {
  friends: Friendship[];
  incoming_requests: Friendship[];
  outgoing_requests: Friendship[];
};

type UnifiedList = SavedList & {
  kind: "personal" | "shared";
  members?: SharedMember[];
  owner_user_id?: string;
};

function formatDate(value: string) {
  try {
    return new Date(value).toLocaleDateString();
  } catch {
    return value;
  }
}

async function getTmdbPoster(title: string): Promise<string> {
  if (!TMDB_API_KEY) return FALLBACK_POSTER;

  try {
    const res = await fetch(
      `https://api.themoviedb.org/3/search/movie?query=${encodeURIComponent(
        title
      )}&api_key=${TMDB_API_KEY}`
    );

    const data = await res.json();

    const match =
      data?.results?.find(
        (movie: { title?: string }) =>
          String(movie.title ?? "").toLowerCase() === title.toLowerCase()
      ) ?? data?.results?.[0];

    if (!match?.poster_path) return FALLBACK_POSTER;
    return `${TMDB_IMAGE_BASE}${match.poster_path}`;
  } catch {
    return FALLBACK_POSTER;
  }
}

export default function EditListsPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const listIdFromUrl = searchParams.get("listId");
  const kindFromUrl = searchParams.get("kind") === "shared" ? "shared" : "personal";

  const { user, isSignedIn, isLoaded } = useUser();
  const userId = user?.id;

  const [lists, setLists] = useState<UnifiedList[]>([]);
  const [selectedListId, setSelectedListId] = useState<string | null>(null);
  const [draftName, setDraftName] = useState("");

  const [query, setQuery] = useState("");
  const [results, setResults] = useState<MovieSearchResult[]>([]);
  const [loadingLists, setLoadingLists] = useState(true);
  const [loadingSearch, setLoadingSearch] = useState(false);
  const [saveMessage, setSaveMessage] = useState("");

  const [friends, setFriends] = useState<Friendship[]>([]);
  const [showSharePanel, setShowSharePanel] = useState(false);

  useEffect(() => {
    const loadLists = async () => {
      if (!isLoaded) return;

      if (!isSignedIn || !userId) {
        setLists([]);
        setSelectedListId(null);
        setDraftName("");
        setLoadingLists(false);
        return;
      }

      setLoadingLists(true);

      try {
        const endpoint =
          kindFromUrl === "shared"
            ? `${API_BASE_URL}/shared-lists`
            : `${API_BASE_URL}/user/lists`;

        const res = await fetch(endpoint, {
          headers: {
            "X-User-Id": userId,
          },
        });

        if (!res.ok) throw new Error("Failed to load lists");

        const data = await res.json();

        const normalizedLists: UnifiedList[] =
          kindFromUrl === "shared"
            ? data.map((list: any) => ({
                id: list.id,
                name: list.name,
                createdAt: list.createdAt,
                movies: list.movies.map((movie: any) => ({
                  movie_id: movie.movie_id,
                  title: movie.title,
                })),
                kind: "shared",
                members: list.members,
                owner_user_id: list.owner_user_id,
              }))
            : data.map((list: SavedList) => ({
                ...list,
                kind: "personal",
              }));

        setLists(normalizedLists);

        const found =
          normalizedLists.find((list) => list.id === listIdFromUrl) ??
          normalizedLists[0] ??
          null;

        setSelectedListId(found?.id ?? null);
        setDraftName(found?.name ?? "");
      } catch (err) {
        console.error(err);
        setLists([]);
      } finally {
        setLoadingLists(false);
      }
    };

    loadLists();
  }, [isLoaded, isSignedIn, userId, listIdFromUrl]);

  useEffect(() => {
      const loadFriends = async () => {
        if (!isSignedIn || !userId) return;

        try {
          const res = await fetch(`${API_BASE_URL}/friends`, {
            headers: {
              "X-User-Id": userId,
            },
          });

          if (!res.ok) throw new Error("Failed to load friends");

          const data: FriendsResponse = await res.json();
          setFriends(data.friends);
        } catch (err) {
          console.error(err);
          setFriends([]);
        }
      };

      loadFriends();
  }, [isSignedIn, userId]);

  useEffect(() => {
    const runSearch = async () => {
      if (!query.trim()) {
        setResults([]);
        return;
      }

      setLoadingSearch(true);

      try {
        const res = await fetch(
          `${API_BASE_URL}/movies/search?q=${encodeURIComponent(query)}`
        );

        if (!res.ok) {
          throw new Error("Search failed");
        }

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

  const selectedList = useMemo(
    () => lists.find((list) => list.id === selectedListId) ?? null,
    [lists, selectedListId]
  );

  const filteredSearchResults = useMemo(() => {
    if (!selectedList) return results;

    const existingIds = new Set(
      selectedList.movies.map((movie) => movie.movie_id)
    );

    return results.filter((movie) => !existingIds.has(movie.movie_id)).slice(0, 5);
  }, [results, selectedList]);

  const chooseList = (list: SavedList) => {
    setSelectedListId(list.id);
    setDraftName(list.name);
    setSaveMessage("");
  };

  const saveListUpdate = async (
      listId: string,
      payload: { name?: string; movies?: SavedMovie[] }
    ) => {
      if (!isSignedIn || !userId) {
        setSaveMessage("Please sign in first.");
        return null;
      }

      if (kindFromUrl === "shared") {
        if (payload.name !== undefined) {
          const res = await fetch(`${API_BASE_URL}/shared-lists/${listId}`, {
            method: "PUT",
            headers: {
              "Content-Type": "application/json",
              "X-User-Id": userId,
            },
            body: JSON.stringify({ name: payload.name }),
          });

          if (!res.ok) throw new Error("Failed to rename shared list");

          const updated = await res.json();

          const normalized: UnifiedList = {
            id: updated.id,
            name: updated.name,
            createdAt: updated.createdAt,
            movies: updated.movies.map((movie: any) => ({
              movie_id: movie.movie_id,
              title: movie.title,
            })),
            kind: "shared",
            members: updated.members,
            owner_user_id: updated.owner_user_id,
          };

          setLists((prev) =>
            prev.map((list) => (list.id === listId ? normalized : list))
          );

          return normalized;
        }

        return selectedList;
      }

      const res = await fetch(`${API_BASE_URL}/user/lists/${listId}`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          "X-User-Id": userId,
        },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error("Failed to update list");

      const updated: SavedList = await res.json();

      const normalized: UnifiedList = {
        ...updated,
        kind: "personal",
      };

      setLists((prev) =>
        prev.map((list) => (list.id === listId ? normalized : list))
      );

      return normalized;
  };

  const saveRename = async () => {
    if (!selectedList) return;

    const trimmed = draftName.trim();
    if (!trimmed) return;

    try {
      await saveListUpdate(selectedList.id, { name: trimmed });
      setSaveMessage(`Saved "${trimmed}"`);
      setTimeout(() => setSaveMessage(""), 2200);
    } catch (err) {
      console.error(err);
      setSaveMessage("Could not save list.");
    }
  };

  const removeMovie = async (movieId: number) => {
      if (!selectedList || !userId) return;

      try {
        if (kindFromUrl === "shared") {
          const res = await fetch(
            `${API_BASE_URL}/shared-lists/${selectedList.id}/movies/${movieId}`,
            {
              method: "DELETE",
              headers: {
                "X-User-Id": userId,
              },
            }
          );

          if (!res.ok) throw new Error("Failed to remove movie");

          const updated = await res.json();

          setLists((prev) =>
            prev.map((list) =>
              list.id === selectedList.id
                ? {
                    ...list,
                    movies: updated.movies.map((item: any) => ({
                      movie_id: item.movie_id,
                      title: item.title,
                    })),
                    members: updated.members,
                  }
                : list
            )
          );
        } else {
          const movies = selectedList.movies.filter(
            (movie) => movie.movie_id !== movieId
          );

          await saveListUpdate(selectedList.id, { movies });
        }
      } catch (err) {
        console.error(err);
        setSaveMessage("Could not remove movie.");
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

  const addMovie = async (movie: SearchMovie) => {
      if (!selectedList || !userId) return;

      try {
        if (kindFromUrl === "shared") {
          const res = await fetch(
            `${API_BASE_URL}/shared-lists/${selectedList.id}/movies`,
            {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                "X-User-Id": userId,
              },
              body: JSON.stringify({
                movie_id: movie.movie_id,
                title: movie.title,
              }),
            }
          );

          if (!res.ok) throw new Error("Failed to add movie");

          const updated = await res.json();

          setLists((prev) =>
            prev.map((list) =>
              list.id === selectedList.id
                ? {
                    ...list,
                    movies: updated.movies.map((item: any) => ({
                      movie_id: item.movie_id,
                      title: item.title,
                    })),
                    members: updated.members,
                  }
                : list
            )
          );
        } else {
          const movies = [
            ...selectedList.movies,
            { movie_id: movie.movie_id, title: movie.title },
          ];

          await saveListUpdate(selectedList.id, { movies });
        }

        setQuery("");
        setResults([]);
      } catch (err) {
        console.error(err);
        setSaveMessage("Could not add movie.");
      }
  };

  const deleteList = async () => {
    if (!selectedList || !isSignedIn || !userId) return;

    try {
      const res = await fetch(`${API_BASE_URL}/user/lists/${selectedList.id}`, {
        method: "DELETE",
        headers: {
          "X-User-Id": userId,
        },
      });

      if (!res.ok) {
        throw new Error("Failed to delete list");
      }

      const nextLists = lists.filter((list) => list.id !== selectedList.id);
      setLists(nextLists);

      if (nextLists.length > 0) {
        setSelectedListId(nextLists[0].id);
        setDraftName(nextLists[0].name);
      } else {
        setSelectedListId(null);
        setDraftName("");
      }

      setSaveMessage("");
    } catch (err) {
      console.error(err);
      setSaveMessage("Could not delete list.");
    }
  };

  const shareWithFriend = async (friend: Friendship) => {
      if (!selectedList || !userId) return;

      try {
        if (kindFromUrl === "personal") {
          const res = await fetch(`${API_BASE_URL}/user/lists/${selectedList.id}/share`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              "X-User-Id": userId,
            },
            body: JSON.stringify({
              member_user_ids: [friend.other_user_id],
            }),
          });

          if (!res.ok) throw new Error("Failed to share list");

          const shared = await res.json();

          router.push(`/edit-list?kind=shared&listId=${shared.id}`);
          return;
        }

        const res = await fetch(
          `${API_BASE_URL}/shared-lists/${selectedList.id}/members`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              "X-User-Id": userId,
            },
            body: JSON.stringify({
              user_id: friend.other_user_id,
              role: "editor",
            }),
          }
        );

        if (!res.ok) throw new Error("Failed to add friend to shared list");

        const updated = await res.json();

        setLists((prev) =>
          prev.map((list) =>
            list.id === selectedList.id
              ? {
                  ...list,
                  members: updated.members,
                  movies: updated.movies.map((item: any) => ({
                    movie_id: item.movie_id,
                    title: item.title,
                  })),
                }
              : list
          )
        );

        setShowSharePanel(false);
        setSaveMessage("Shared list updated.");
      } catch (err) {
        console.error(err);
        setSaveMessage("Could not share list.");
      }
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
          background:
            radial-gradient(circle at 18% 12%, rgba(168, 85, 247, 0.22), transparent 34%),
            radial-gradient(circle at 82% 8%, rgba(99, 102, 241, 0.18), transparent 30%),
            linear-gradient(135deg, #070617 0%, #18102f 38%, #4c1d95 100%);
          color: #f8fafc;
        }

        .wrap {
          max-width: 1220px;
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
          background: linear-gradient(135deg, #ef4444, #991b1b);
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
          border: 1px solid rgba(248,113,113,0.28);
          background: rgba(127,29,29,0.22);
          color: #fecaca;
          border-radius: 999px;
          padding: 10px 16px;
          cursor: pointer;
          font-weight: 800;
        }

        .hero-title {
          font-size: clamp(1.9rem, 3.4vw, 3.3rem);
          line-height: 1.08;
          font-weight: 950;
          letter-spacing: -0.05em;
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
          font-size: 22px;
          font-weight: 900;
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
          background: rgba(185,28,28,0.22);
          border-color: rgba(248,113,113,0.32);
        }

        .list-nav-name {
          font-weight: 900;
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
          color: #fecaca;
          font-size: 14px;
          font-weight: 800;
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
          font-weight: 900;
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
          font-weight: 700;
          max-width: 100%;
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
          color: #fecaca;
          cursor: pointer;
          font-size: 18px;
          font-weight: 900;
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
          font-weight: 800;
          line-height: 1.35;
          min-height: 38px;
        }

        .empty {
          color: rgba(255,255,255,0.72);
          line-height: 1.6;
        }

        @media (max-width: 900px) {
          .layout {
            grid-template-columns: 1fr;
          }

          .editor-header {
            flex-direction: column;
          }

          .editor-actions {
            justify-content: flex-start;
          }

          .save-msg {
            position: static;
            transform: none;
            margin-top: 8px;
          }
        }

        .share-panel {
          border: 1px solid rgba(196, 181, 253, 0.24);
          background: rgba(124, 58, 237, 0.14);
          border-radius: 18px;
          padding: 14px;
          margin-bottom: 18px;
        }

        .share-title {
          font-size: 14px;
          font-weight: 900;
          color: #ddd6fe;
          margin-bottom: 10px;
        }

        .friend-share-row {
          width: 100%;
          display: flex;
          align-items: center;
          gap: 10px;
          border: 1px solid rgba(255,255,255,0.08);
          background: rgba(255,255,255,0.06);
          color: #f8fafc;
          border-radius: 14px;
          padding: 11px 12px;
          cursor: pointer;
          font-weight: 800;
          margin-bottom: 8px;
          text-align: left;
        }

        .add-list-btn {
          width: 100%;
          margin-top: 14px;
        }

      `}</style>

      <div className="wrap">
        <div className="top-bar">
          <button className="pill-btn" onClick={() => router.push("/lists")}>
            ← My Lists
          </button>

          <div className="top-actions">
            <button className="pill-btn" onClick={() => router.push("/")}>
              Search
            </button>

            {!isSignedIn && (
              <SignInButton mode="modal">
                <button className="primary-btn">Sign In</button>
              </SignInButton>
            )}

            {isSignedIn && <UserButton />}
          </div>
        </div>

        <h1 className="hero-title">Edit your movie lists.</h1>
        <div className="hero-sub">
          Rename lists, add better anchors, remove weak picks, and rerun recommendations.
        </div>

        {!isSignedIn && isLoaded && (
          <div className="panel">
            <h2 className="panel-title">Sign in to edit lists.</h2>
            <p className="empty">
              Saved lists are now tied to your account instead of this browser.
            </p>
            <SignInButton mode="modal">
              <button className="primary-btn">Sign In</button>
            </SignInButton>
          </div>
        )}

        {isSignedIn && loadingLists && (
          <div className="panel">Loading your lists...</div>
        )}

        {isSignedIn && !loadingLists && lists.length === 0 && (
          <div className="panel">
            <h2 className="panel-title">No lists to edit yet.</h2>
            <p className="empty">
              Save a list from the results page first, then come back here.
            </p>
            <button className="primary-btn" onClick={() => router.push("/")}>
              Start Searching
            </button>
          </div>
        )}

        {isSignedIn && !loadingLists && lists.length > 0 && (
          <div className="layout">
            <aside className="panel">
              <h2 className="panel-title">Lists</h2>

              <div className="list-nav">
                {lists.map((list) => (
                  <button
                    key={list.id}
                    className={`list-nav-item ${
                      list.id === selectedListId ? "active" : ""
                    }`}
                    onClick={() => chooseList(list)}
                  >
                    <div className="list-nav-name">{list.name}</div>
                    <div className="list-nav-meta">
                      {list.movies.length} movies · {formatDate(list.createdAt)}
                    </div>
                  </button>
                ))}
              </div>
              <button className="primary-btn add-list-btn" onClick={createNewList}>
                  + Add New List
              </button>
            </aside>

            <div className="main-stack">
              <section className="panel">
                {selectedList ? (
                  <>
                    <div className="editor-header">
                      <div className="editor-main">
                        <input
                          className="list-name-input"
                          value={draftName}
                          onChange={(event) => setDraftName(event.target.value)}
                        />

                        <div className="list-meta">
                          {selectedList.movies.length} movies · Saved{" "}
                          {formatDate(selectedList.createdAt)}
                        </div>
                      </div>

                      <div className="editor-actions">
                        <div className="save-btn-wrap">
                          <button className="primary-btn" onClick={saveRename}>
                            Save List
                          </button>
                          {saveMessage && (
                            <div className="save-msg">{saveMessage}</div>
                          )}
                        </div>

                        <button className="secondary-btn" onClick={seeResults}>
                          See Results
                        </button>

                        <button
                          className="danger-btn"
                          onClick={() => {
                            const confirmed = window.confirm(
                              `Are you sure you want to delete "${selectedList?.name}"?`
                            );

                            if (confirmed) {
                              deleteList();
                            }
                          }}
                        >
                          Delete List
                        </button>
                        <button
                          className="secondary-btn"
                          onClick={() => setShowSharePanel((prev) => !prev)}
                        >
                          Share
                        </button>
                      </div>
                    </div>
                    {showSharePanel && (
                      <div className="share-panel">
                        <div className="share-title">
                          {kindFromUrl === "personal"
                            ? "Share this list with a friend"
                            : "Add another friend"}
                        </div>

                        {friends.length === 0 && (
                          <div className="empty">No friends available yet.</div>
                        )}

                        {friends.map((friend) => (
                          <button
                            key={friend.id}
                            className="friend-share-row"
                            onClick={() => shareWithFriend(friend)}
                          >
                            <span>👥</span>
                            <span>
                              {friend.other_user?.name ||
                                friend.other_user?.email ||
                                "Friend"}
                            </span>
                          </button>
                        ))}
                      </div>
                    )}

                    <div className="movie-list">
                      {selectedList.movies.map((movie) => (
                        <span className="movie-chip" key={movie.movie_id}>
                          <span className="movie-chip-text">{movie.title}</span>
                          <button
                            className="chip-remove"
                            onClick={() => removeMovie(movie.movie_id)}
                            aria-label={`Remove ${movie.title}`}
                          >
                            ×
                          </button>
                        </span>
                      ))}
                    </div>
                  </>
                ) : (
                  <div className="empty">Choose a list to edit.</div>
                )}
              </section>

              <section className="panel">
                <div className="search-head">
                  <h2 className="panel-title">Add movies</h2>
                  <div className="search-title-copy">
                    Search and add stronger taste anchors.
                  </div>
                </div>

                <input
                  className="search-input"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Search for a movie..."
                />

                {loadingSearch && <div className="empty">Searching...</div>}

                {!loadingSearch && filteredSearchResults.length > 0 && (
                  <div className="search-results-grid">
                    {filteredSearchResults.map((movie) => (
                      <article className="search-card" key={movie.movie_id}>
                        <img
                          className="search-poster"
                          src={movie.poster || FALLBACK_POSTER}
                          alt={movie.title}
                        />
                        <div className="search-title">{movie.title}</div>
                        <button
                          className="primary-btn"
                          onClick={() => addMovie(movie)}
                          disabled={!selectedList}
                        >
                          Add
                        </button>
                      </article>
                    ))}
                  </div>
                )}

                {!loadingSearch && query.trim() && filteredSearchResults.length === 0 && (
                  <div className="empty">No new matching movies found.</div>
                )}
              </section>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}