"use client";

import { SignInButton, useClerk, useUser } from "@clerk/nextjs";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import AppHeader from "../components/AppHeader";
import MovieDetailModal from "../components/MovieDetailModal";
import { API_BASE_URL, FALLBACK_POSTER } from "../lib/config";
import { logHandledError } from "../lib/log";
import { getTmdbPoster } from "../lib/tmdb";

type ProfilePayload = {
  profile: {
    user_id: string;
    email: string | null;
    name: string | null;
    bio: string | null;
    avatar_url: string | null;
  };
  onboarding_preferences: Record<string, string[]>;
  stats: {
    saved_lists_count: number;
    shared_lists_count: number;
    friends_count: number;
    unique_movies_count: number;
  };
  taste_summary: {
    favorite_genres: string[];
    favorite_directors: string[];
    favorite_actors: string[];
    favorite_keywords: string[];
  };
  lists: { id: string; name: string; createdAt: string }[];
  shared_lists: { id: string; name: string; createdAt: string }[];
  friends: { user_id: string; name: string | null; email: string | null }[];
  watchlist: { movie_id: number; title: string; status: string; createdAt: string }[];
  recent_activity: { type: string; label: string; createdAt: string }[];
};

type MovieSearchResult = {
  movie_id: number;
  title: string;
  poster?: string;
};

const emptyPrefs = {
  favorite_genres: [],
  disliked_genres: [],
  preferred_moods: [],
  preferred_pacing: [],
  preferred_decades: [],
  favorite_movies: [],
  disliked_movies: [],
};

function pretty(value: string) {
  return value.replaceAll("-", " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function valuesFor(values: string[] | undefined) {
  return values && values.length > 0 ? values : [];
}

function emptyCopy(label: string) {
  return <div className="muted">No {label} saved yet.</div>;
}


export default function ProfilePage() {
  const router = useRouter();
  const { openUserProfile, signOut } = useClerk();
  const { user, isSignedIn, isLoaded } = useUser();
  const userId = user?.id;

  const [data, setData] = useState<ProfilePayload | null>(null);
  const [name, setName] = useState("");
  const [bio, setBio] = useState("");
  const [prefs, setPrefs] = useState<Record<string, string[]>>(emptyPrefs);
  const [message, setMessage] = useState("");
  const [loadingProfile, setLoadingProfile] = useState(false);
  const [rankQuery, setRankQuery] = useState("");
  const [rankResults, setRankResults] = useState<MovieSearchResult[]>([]);
  const [rankingMovie, setRankingMovie] = useState<MovieSearchResult | null>(null);
  const [loadingRankSearch, setLoadingRankSearch] = useState(false);
  const [ratingMessage, setRatingMessage] = useState("");
  const [editingBio, setEditingBio] = useState(false);
  const [watchlistOpen, setWatchlistOpen] = useState(true);
  const [editingPrefs, setEditingPrefs] = useState(false);
  const [prefsDraft, setPrefsDraft] = useState<Record<string, string>>(
    Object.fromEntries(Object.keys(emptyPrefs).map((key) => [key, ""]))
  );

  const userEmail = user?.primaryEmailAddress?.emailAddress ?? null;
  const clerkName =
    user?.fullName ?? user?.username ?? user?.firstName ?? userEmail ?? null;

  const loadProfile = async () => {
    if (!userId) return;

    setLoadingProfile(true);

    try {
      const res = await fetch(`${API_BASE_URL}/profile`, {
        headers: { "X-User-Id": userId },
      });

      if (!res.ok) {
        setMessage("Could not load profile.");
        return;
      }

      const payload: ProfilePayload = await res.json();

      setData(payload);
      setName(payload.profile.name ?? clerkName ?? "");
      setBio(payload.profile.bio ?? "");
      const mergedPrefs = {
        ...emptyPrefs,
        ...payload.onboarding_preferences,
      };
      setPrefs(mergedPrefs);
      setPrefsDraft(
        Object.fromEntries(
          Object.entries(mergedPrefs).map(([key, values]) => [key, values.join(", ")])
        )
      );
      setMessage("");
    } catch (err) {
      logHandledError("Profile load failed", err);
      setMessage("Could not reach the profile service.");
    } finally {
      setLoadingProfile(false);
    }
  };

  useEffect(() => {
    const syncAndLoad = async () => {
      if (!isLoaded || !isSignedIn || !userId) return;

      try {
        await fetch(`${API_BASE_URL}/user/profile`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-User-Id": userId,
          },
          body: JSON.stringify({
            email: userEmail,
            name: clerkName,
          }),
        });

        await loadProfile();
      } catch (err) {
        logHandledError("Profile sync failed", err);
        setMessage("Could not reach the profile service.");
        setLoadingProfile(false);
      }
    };

    syncAndLoad();
  }, [isLoaded, isSignedIn, userId, userEmail, clerkName]);

  useEffect(() => {
    const searchMoviesToRank = async () => {
      if (!rankQuery.trim()) {
        setRankResults([]);
        return;
      }

      setLoadingRankSearch(true);

      try {
        const res = await fetch(
          `${API_BASE_URL}/movies/search?q=${encodeURIComponent(rankQuery)}&limit=6`
        );

        if (!res.ok) {
          setRankResults([]);
          return;
        }

        const data: MovieSearchResult[] = await res.json();
        const withPosters = await Promise.all(
          data.map(async (movie) => ({
            ...movie,
            poster: await getTmdbPoster(movie.title),
          }))
        );
        setRankResults(withPosters);
      } catch (err) {
        logHandledError("Movie ranking search failed", err);
        setRankResults([]);
      } finally {
        setLoadingRankSearch(false);
      }
    };

    const timeout = setTimeout(searchMoviesToRank, 250);
    return () => clearTimeout(timeout);
  }, [rankQuery]);

  const saveProfile = async () => {
    if (!userId) return;

    try {
      const res = await fetch(`${API_BASE_URL}/profile`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          "X-User-Id": userId,
        },
        body: JSON.stringify({
          name,
          bio,
          avatar_url: data?.profile.avatar_url ?? null,
        }),
      });

      if (!res.ok) {
        setMessage("Could not save profile.");
        return;
      }

      setMessage("Profile saved.");
      setEditingBio(false);
      await loadProfile();
    } catch (err) {
      logHandledError("Profile save failed", err);
      setMessage("Could not reach the profile service.");
    }
  };

  const profileImage = data?.profile.avatar_url || user?.imageUrl || "";
  const profileInitial =
    (name || user?.firstName || user?.username || userEmail || "M")[0]?.toUpperCase() ??
    "M";

  const createdListActivity =
    data?.recent_activity.filter((item) => item.type === "created_list") ?? [];
  const editedSharedActivity =
    data?.recent_activity.filter((item) => item.type === "edited_shared_list") ?? [];
  const ratedMovieActivity =
    data?.recent_activity.filter((item) => item.type === "rated_movie") ?? [];
  const watchedMovieActivity =
    data?.recent_activity.filter((item) => item.type === "watched_movie") ?? [];

  const chooseRankMovie = (movie: MovieSearchResult) => {
    setRankingMovie(movie);
    setRatingMessage("");
    setRankQuery("");
    setRankResults([]);
  };

  const saveOnboardingPreferences = async () => {
    if (!userId) return;

    const payload = Object.fromEntries(
      Object.entries(prefsDraft).map(([key, value]) => [
        key,
        value
          .split(",")
          .map((item) => item.trim())
          .filter(Boolean),
      ])
    );

    try {
      const res = await fetch(`${API_BASE_URL}/profile/onboarding`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          "X-User-Id": userId,
        },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        setMessage("Could not save preferences.");
        return;
      }

      setMessage("Preferences saved.");
      setEditingPrefs(false);
      await loadProfile();
    } catch (err) {
      logHandledError("Onboarding preferences save failed", err);
      setMessage("Could not reach the profile service.");
    }
  };

  const removeFromWatchlist = async (movieId: number) => {
    if (!userId) return;

    try {
      const res = await fetch(`${API_BASE_URL}/profile/watchlist/${movieId}`, {
        method: "DELETE",
        headers: { "X-User-Id": userId },
      });

      if (!res.ok) return;
      await loadProfile();
    } catch (err) {
      logHandledError("Watchlist remove failed", err);
    }
  };



  return (
    <main className="profile-page">
      <style>{`
        .profile-page {
          min-height: 240vh;
          padding: 28px 24px 64px;
          background:
            radial-gradient(circle at 18% 12%, rgba(168, 85, 247, 0.22), transparent 34%),
            radial-gradient(circle at 82% 8%, rgba(59, 130, 246, 0.18), transparent 30%),
            linear-gradient(135deg, #070617 0%, #10172f 38%, #312e81 100%);
          color: #f8fafc;
        }

        .wrap {
          max-width: 1220px;
          margin: 0 auto;
        }

        .top-bar {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 16px;
          margin-bottom: 30px;
        }

        .top-actions {
          display: flex;
          align-items: center;
          gap: 12px;
          flex-wrap: wrap;
        }

        .pill-btn,
        .secondary-btn {
          border: 1px solid rgba(255,255,255,0.12);
          background: rgba(255,255,255,0.06);
          color: #e2e8f0;
          border-radius: 999px;
          padding: 10px 16px;
          cursor: pointer;
          font-weight: 800;
        }

        .primary-btn {
          border: none;
          background: linear-gradient(135deg, #8b5cf6, #2563eb);
          color: white;
          border-radius: 999px;
          padding: 10px 16px;
          cursor: pointer;
          font-weight: 800;
        }

        .hero {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 22px;
          margin-bottom: 24px;
          border: 1px solid rgba(255,255,255,0.08);
          background: rgba(255,255,255,0.05);
          border-radius: 24px;
          padding: 22px;
          box-shadow: 0 16px 40px rgba(0,0,0,0.28);
        }

        .profile-top {
          display: flex;
          align-items: center;
          gap: 18px;
          min-width: 0;
        }

        .profile-copy {
          padding-top: 12px;
        }

        .profile-avatar,
        .profile-avatar-fallback {
          width: 92px;
          height: 92px;
          border-radius: 999px;
          flex: 0 0 auto;
          border: 1px solid rgba(255,255,255,0.18);
          box-shadow: 0 14px 34px rgba(0,0,0,0.26);
        }

        .profile-avatar {
          object-fit: cover;
        }

        .profile-avatar-fallback {
          display: grid;
          place-items: center;
          background: linear-gradient(135deg, rgba(139,92,246,0.72), rgba(37,99,235,0.68));
          font-size: 34px;
          font-weight: 950;
        }

        .profile-actions {
          display: flex;
          align-items: center;
          justify-content: flex-end;
          gap: 10px;
          flex-wrap: wrap;
        }

        .profile-side {
          display: flex;
          flex-direction: column;
          gap: 12px;
          width: min(420px, 100%);
        }

        .profile-side-actions {
          display: flex;
          align-items: center;
          justify-content: flex-end;
          gap: 10px;
          flex-wrap: wrap;
        }

        .bio-editor {
          border: 1px solid rgba(255,255,255,0.1);
          background: rgba(255,255,255,0.05);
          border-radius: 18px;
          padding: 12px;
        }

        .bio-editor .textarea {
          min-height: 72px;
          margin-bottom: 10px;
        }

        .bio-editor-actions {
          display: flex;
          justify-content: flex-end;
        }

       .profile-card-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 18px;
          align-items: start;
        }

        .card-actions {
          margin-top: 16px;
        }

        .card-section {
          margin-top: 16px;
        }

        .card-section:first-of-type {
          margin-top: 0;
        }

        .section-title {
          color: #c4b5fd;
          font-size: 13px;
          font-weight: 950;
          margin: 0 0 8px;
          text-transform: uppercase;
        }

        .friend-row {
          display: flex;
          align-items: center;
          gap: 10px;
          border-bottom: 1px solid rgba(255,255,255,0.08);
          padding: 10px 0;
        }

        .friend-row:last-child {
          border-bottom: none;
        }

        .friend-avatar {
          display: grid;
          place-items: center;
          width: 34px;
          height: 34px;
          border-radius: 999px;
          background: rgba(255,255,255,0.1);
          border: 1px solid rgba(255,255,255,0.1);
          color: #ddd6fe;
          font-size: 13px;
          font-weight: 950;
          flex: 0 0 auto;
        }

        .friend-name {
          font-weight: 900;
        }

        .friend-sub {
          color: rgba(255,255,255,0.62);
          font-size: 13px;
        }

        .hero-title {
          font-size: clamp(2rem, 4vw, 3.5rem);
          line-height: 1.04;
          font-weight: 950;
          letter-spacing: -0.05em;
          margin: 0 0 10px;
        }

        .hero-sub {
          color: rgba(255,255,255,0.72);
          font-size: 17px;
          line-height: 1.55;
        }

        .profile-bio-text {
          color: rgba(255,255,255,0.68);
          font-size: 15px;
          line-height: 1.55;
          max-width: 720px;
          margin: 10px 0 0;
        }

        .grid {
          display: grid;
          grid-template-columns: 1.1fr 0.9fr;
          gap: 18px;
          align-items: start;
        }

        .panel {
          border: 1px solid rgba(255,255,255,0.08);
          background: rgba(255,255,255,0.05);
          border-radius: 24px;
          padding: 20px;
          box-shadow: 0 16px 40px rgba(0,0,0,0.28);
          margin-bottom: 18px;
        }

        .panel-title {
          font-size: 22px;
          font-weight: 900;
          margin: 0 0 14px;
          letter-spacing: -0.03em;
        }

        .stats {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 12px;
          margin-bottom: 18px;
        }

        .stat-card {
          background: rgba(255,255,255,0.06);
          border: 1px solid rgba(255,255,255,0.08);
          border-radius: 18px;
          padding: 16px;
        }

        .stat-number {
          font-size: 28px;
          font-weight: 950;
        }

        .stat-label {
          color: rgba(255,255,255,0.68);
          font-size: 13px;
          font-weight: 700;
        }

        .input,
        .textarea {
          width: 100%;
          border: 1px solid rgba(255,255,255,0.12);
          background: rgba(255,255,255,0.06);
          color: #f8fafc;
          border-radius: 16px;
          padding: 13px 14px;
          outline: none;
          font-size: 15px;
          margin-bottom: 12px;
        }

        .textarea {
          min-height: 92px;
          resize: vertical;
        }

        .rank-search {
          margin-top: 16px;
        }

        .rank-results-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
          gap: 16px;
          margin-top: 16px;
        }

        .rank-card {
          border: 1px solid rgba(255,255,255,0.08);
          background: rgba(255,255,255,0.05);
          color: #e2e8f0;
          border-radius: 18px;
          padding: 12px;
          display: flex;
          flex-direction: column;
          gap: 10px;
          cursor: pointer;
          text-align: left;
          transition: transform 160ms ease, background 160ms ease, border-color 160ms ease;
        }

        .rank-card:hover,
        .rank-card:focus-visible {
          background: rgba(255,255,255,0.08);
          border-color: rgba(255,255,255,0.16);
          transform: translateY(-2px);
        }

        .rank-poster {
          width: 100%;
          aspect-ratio: 2 / 3;
          object-fit: cover;
          display: block;
          border-radius: 12px;
          background: rgba(255,255,255,0.05);
        }

        .rank-title {
          font-weight: 850;
          line-height: 1.35;
          min-height: 38px;
        }

        .watchlist-toggle {
          width: 100%;
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
          border: 0;
          background: transparent;
          color: #f8fafc;
          cursor: pointer;
          padding: 0;
          text-align: left;
        }

        .watchlist-scroll {
          max-height: 260px;
          overflow-y: auto;
          padding-right: 6px;
          margin-top: 12px;
        }

        .preference-editor {
          display: grid;
          gap: 12px;
        }

        .preference-field label {
          display: block;
          color: #c4b5fd;
          font-size: 12px;
          font-weight: 950;
          text-transform: uppercase;
          margin-bottom: 6px;
        }

        .rating-card {
          width: min(520px, 100%);
          border: 1px solid rgba(255,255,255,0.12);
          background: #0f172a;
          border-radius: 24px;
          padding: 22px;
          box-shadow: 0 24px 80px rgba(0,0,0,0.46);
        }

        .rating-movie-title {
          font-size: 28px;
          font-weight: 950;
          letter-spacing: -0.04em;
          margin: 0 0 6px;
        }

        .rating-movie-id {
          color: rgba(255,255,255,0.62);
          font-size: 13px;
          font-weight: 800;
          margin-bottom: 18px;
        }

        .apple-star-row {
          display: flex;
          flex-wrap: wrap;
          gap: 4px;
          margin-bottom: 14px;
        }

        .apple-star {
          position: relative;
          width: 42px;
          height: 42px;
          border: none;
          background: transparent;
          cursor: pointer;
          padding: 0;
          color: rgba(255,255,255,0.28);
          font-size: 36px;
          line-height: 1;
        }

        .apple-star-empty,
        .apple-star-fill {
          position: absolute;
          inset: 0;
          display: grid;
          place-items: center;
        }

        .apple-star-fill {
          width: var(--star-fill);
          overflow: hidden;
          color: #facc15;
          text-shadow: 0 0 18px rgba(250,204,21,0.34);
        }

        .apple-star-fill span,
        .apple-star-empty span {
          width: 42px;
          text-align: center;
        }

        .apple-star-left,
        .apple-star-right {
          position: absolute;
          top: 0;
          bottom: 0;
          width: 50%;
          z-index: 2;
        }

        .apple-star-left {
          left: 0;
        }

        .apple-star-right {
          right: 0;
        }

        .rating-value {
          color: #fde68a;
          font-weight: 900;
          margin-bottom: 14px;
        }

        .rating-actions {
          display: flex;
          justify-content: flex-end;
          gap: 10px;
          margin-top: 14px;
        }

        .chip-row {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
        }

        .chip {
          background: rgba(255,255,255,0.08);
          border: 1px solid rgba(255,255,255,0.08);
          border-radius: 999px;
          padding: 8px 11px;
          font-size: 13px;
          font-weight: 800;
          color: #e2e8f0;
        }

        .list-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 12px;
          border-bottom: 1px solid rgba(255,255,255,0.08);
          padding: 11px 0;
        }

        .list-row:last-child {
          border-bottom: none;
        }

        .muted {
          color: rgba(255,255,255,0.68);
          line-height: 1.55;
        }

        .message {
          color: #ddd6fe;
          font-weight: 900;
          margin-bottom: 14px;
        }

        .pref-label {
          font-size: 13px;
          font-weight: 900;
          color: #c4b5fd;
          margin: 8px 0 6px;
        }

        @media (max-width: 900px) {
          .profile-card-grid,
          .grid,
          .stats {
            grid-template-columns: 1fr;
          }

          .hero {
            flex-direction: column;
            align-items: flex-start;
          }

          .profile-top {
            align-items: flex-start;
          }

          .profile-actions {
            justify-content: flex-start;
          }

          .profile-side-actions,
          .bio-editor-actions {
            justify-content: flex-start;
          }
        }
      `}</style>

      <div className="wrap">
        <AppHeader
          leading={{ label: "← Search", href: "/" }}
          actions={[
            { label: "My Lists", href: "/lists" },
            { label: "My Friends", href: "/friends" },
          ]}
        />

        {!isSignedIn && isLoaded && (
          <div className="panel">
            <h1 className="panel-title">Sign in to view your profile.</h1>
            <SignInButton mode="modal">
              <button className="primary-btn">Sign In</button>
            </SignInButton>
          </div>
        )}

        {isSignedIn && loadingProfile && !data && (
          <div className="panel">Loading profile...</div>
        )}

        {isSignedIn && isLoaded && !loadingProfile && !data && (
          <div className="panel">
            <h1 className="panel-title">Profile unavailable.</h1>
            <p className="empty">
              {message || "Could not load your profile right now."}
            </p>
          </div>
        )}

        {isSignedIn && data && (
          <>
            <section className="hero">
              <div className="profile-top">
                {profileImage ? (
                  <img className="profile-avatar" src={profileImage} alt="" />
                ) : (
                  <div className="profile-avatar-fallback" aria-hidden="true">
                    {profileInitial}
                  </div>
                )}

                <div className="profile-copy">
                  <h1 className="hero-title">{name || "Your Movie Profile"}</h1>
                  <p className="profile-bio-text">
                    {bio ||
                      "Add a short bio so friends know what kind of movie night you are usually in the mood for."}
                  </p>
                </div>
              </div>

              <div className="profile-side">
                <div className="profile-side-actions">
                  <button
                    className="primary-btn"
                    onClick={() => setEditingBio((prev) => !prev)}
                  >
                    {editingBio ? "Close Editor" : "Edit Bio"}
                  </button>
                  <button className="secondary-btn" onClick={() => openUserProfile()}>
                    Account
                  </button>
                  <button
                    className="secondary-btn"
                    onClick={() => signOut(() => router.push("/"))}
                  >
                    Sign Out
                  </button>
                </div>

                {editingBio && (
                  <div className="bio-editor">
                    <div className="section-title">Bio</div>
                    <textarea
                      className="textarea"
                      value={bio}
                      onChange={(event) => setBio(event.target.value)}
                      placeholder="Add a short movie-night bio..."
                    />
                    <div className="bio-editor-actions">
                      <button className="primary-btn" onClick={saveProfile}>
                        Save Bio
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </section>


            {message && <div className="message">{message}</div>}

            <div className="profile-card-grid">
              <section className="panel">
                <h2 className="panel-title">My Taste Profile</h2>

                <div className="card-section">
                  <div className="section-title">Favorite Genres</div>
                  <div className="chip-row">
                    {valuesFor(data.taste_summary.favorite_genres).map((item) => (
                      <span className="chip" key={item}>{pretty(item)}</span>
                    ))}
                    {data.taste_summary.favorite_genres.length === 0 &&
                      emptyCopy("favorite genres")}
                  </div>
                </div>

                <div className="card-section">
                  <div className="section-title">Favorite Actors</div>
                  <div className="chip-row">
                    {valuesFor(data.taste_summary.favorite_actors).map((item) => (
                      <span className="chip" key={item}>{pretty(item)}</span>
                    ))}
                    {data.taste_summary.favorite_actors.length === 0 &&
                      emptyCopy("favorite actors")}
                  </div>
                </div>

                <div className="card-section">
                  <div className="section-title">Favorite Directors</div>
                  <div className="chip-row">
                    {valuesFor(data.taste_summary.favorite_directors).map((item) => (
                      <span className="chip" key={item}>{pretty(item)}</span>
                    ))}
                    {data.taste_summary.favorite_directors.length === 0 &&
                      emptyCopy("favorite directors")}
                  </div>
                </div>

                <div className="card-section">
                  <div className="section-title">Preferred Moods</div>
                  <div className="chip-row">
                    {valuesFor(prefs.preferred_moods).map((item) => (
                      <span className="chip" key={item}>{pretty(item)}</span>
                    ))}
                    {valuesFor(prefs.preferred_moods).length === 0 &&
                      emptyCopy("preferred moods")}
                  </div>
                </div>

                <div className="card-section">
                  <div className="section-title">Pacing Style</div>
                  <div className="chip-row">
                    {valuesFor(prefs.preferred_pacing).map((item) => (
                      <span className="chip" key={item}>{pretty(item)}</span>
                    ))}
                    {valuesFor(prefs.preferred_pacing).length === 0 &&
                      emptyCopy("pacing styles")}
                  </div>
                </div>
              </section>
              <section className="panel">
                <div className="row-head">
                  <h2 className="panel-title">My Preferences</h2>
                  <button className="pill-btn" onClick={() => setEditingPrefs((editing) => !editing)}>
                    {editingPrefs ? "Close" : "Edit"}
                  </button>
                </div>

                {editingPrefs && (
                  <div className="card-section preference-editor">
                    {Object.keys(emptyPrefs).map((key) => (
                      <div className="preference-field" key={key}>
                        <label>{pretty(key)}</label>
                        <input
                          className="input"
                          value={prefsDraft[key] ?? ""}
                          onChange={(event) =>
                            setPrefsDraft((draft) => ({ ...draft, [key]: event.target.value }))
                          }
                          placeholder="Comma-separated answers"
                        />
                      </div>
                    ))}
                    <button className="primary-btn" onClick={saveOnboardingPreferences}>
                      Save Preferences
                    </button>
                  </div>
                )}

                <div className="card-section">
                  <div className="section-title">Onboarding Answers</div>
                  <div className="chip-row">
                    {Object.entries(prefs).some(([, values]) => values.length > 0) ? (
                      Object.entries(prefs).flatMap(([key, values]) =>
                        values.map((value) => (
                          <span className="chip" key={`${key}-${value}`}>
                            {pretty(value)}
                          </span>
                        ))
                      )
                    ) : (
                      emptyCopy("onboarding answers")
                    )}
                  </div>
                </div>

                <div className="card-section">
                  <div className="section-title">Favorite Genres</div>
                  <div className="chip-row">
                    {valuesFor(prefs.favorite_genres).map((item) => (
                      <span className="chip" key={item}>{pretty(item)}</span>
                    ))}
                    {valuesFor(prefs.favorite_genres).length === 0 &&
                      emptyCopy("favorite genres")}
                  </div>
                </div>

                <div className="card-section">
                  <div className="section-title">Disliked Genres</div>
                  <div className="chip-row">
                    {valuesFor(prefs.disliked_genres).map((item) => (
                      <span className="chip" key={item}>{pretty(item)}</span>
                    ))}
                    {valuesFor(prefs.disliked_genres).length === 0 &&
                      emptyCopy("disliked genres")}
                  </div>
                </div>

                <div className="card-section">
                  <div className="section-title">Movies I Love</div>
                  <div className="chip-row">
                    {valuesFor(prefs.favorite_movies).map((item) => (
                      <span className="chip" key={item}>{pretty(item)}</span>
                    ))}
                    {valuesFor(prefs.favorite_movies).length === 0 &&
                      emptyCopy("favorite movies")}
                  </div>
                </div>

                <div className="card-section">
                  <div className="section-title">Movies I Hate</div>
                  <div className="chip-row">
                    {valuesFor(prefs.disliked_movies).map((item) => (
                      <span className="chip" key={item}>{pretty(item)}</span>
                    ))}
                    {valuesFor(prefs.disliked_movies).length === 0 &&
                      emptyCopy("disliked movies")}
                  </div>
                </div>
              </section>


              <section className="panel">
                <h2 className="panel-title">My Friends</h2>
                <div className="stat-card">
                  <div className="stat-number">{data.stats.friends_count}</div>
                  <div className="stat-label">Friends</div>
                </div>

                <div className="card-section">
                  {data.friends.length === 0 && emptyCopy("friends")}
                  {data.friends.slice(0, 4).map((friend) => {
                    const friendName = friend.name || friend.email || "Friend";

                    return (
                      <div className="friend-row" key={friend.user_id}>
                        <div className="friend-avatar" aria-hidden="true">
                          {friendName[0]?.toUpperCase() ?? "F"}
                        </div>
                        <div>
                          <div className="friend-name">{friendName}</div>
                          {friend.email && <div className="friend-sub">{friend.email}</div>}
                        </div>
                      </div>
                    );
                  })}
                </div>

                <div className="card-actions">
                  <button className="primary-btn" onClick={() => router.push("/friends")}>
                    View Friends
                  </button>
                </div>
              </section>

              <section className="panel">
                <button className="watchlist-toggle" onClick={() => setWatchlistOpen((open) => !open)}>
                  <h2 className="panel-title">My Watchlist</h2>
                  <span>{watchlistOpen ? "Hide" : "Show"}</span>
                </button>

                {watchlistOpen && (
                  <div className="watchlist-scroll">
                    {(data.watchlist ?? []).length === 0 && emptyCopy("watchlist movies")}
                    {(data.watchlist ?? []).map((movie) => (
                      <div className="list-row" key={movie.movie_id}>
                        <span>{movie.title}</span>
                        <button className="pill-btn" onClick={() => removeFromWatchlist(movie.movie_id)}>
                          Remove
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </section>
            </div>
            <section className="panel">
              <h2 className="panel-title">Rank Some Movies</h2>
              <p className="muted">
                Search for a movie, open the poster card, rate it from half a star to five stars, and add a quick note.
              </p>

              <div className="rank-search">
                <input
                  className="input"
                  value={rankQuery}
                  onChange={(event) => setRankQuery(event.target.value)}
                  placeholder="Search movies to rank..."
                />

                {loadingRankSearch && <div className="muted">Searching...</div>}

                {!loadingRankSearch && rankResults.length > 0 && (
                  <div className="rank-results-grid">
                    {rankResults.map((movie) => (
                      <button
                        className="rank-card"
                        key={movie.movie_id}
                        onClick={() => chooseRankMovie(movie)}
                      >
                        <img
                          className="rank-poster"
                          src={movie.poster || FALLBACK_POSTER}
                          alt={movie.title}
                        />
                        <div className="rank-title">{movie.title}</div>
                        <span className="pill-btn">Open Rating Card</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>

              {ratingMessage && <div className="message">{ratingMessage}</div>}
            </section>

            <MovieDetailModal movie={rankingMovie} onClose={() => setRankingMovie(null)} />
          </>
        )}
      </div>
    </main>
  );
}
