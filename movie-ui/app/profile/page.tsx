"use client";

import { SignInButton, useClerk, useUser } from "@clerk/nextjs";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import AppHeader from "../components/AppHeader";
import { API_BASE_URL } from "../lib/config";
import { logHandledError } from "../lib/log";

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
  recent_activity: { type: string; label: string; createdAt: string }[];
};

type MovieSearchResult = {
  movie_id: number;
  title: string;
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

function starFillPercent(rating: number, starNumber: number) {
  if (rating >= starNumber) return 100;
  if (rating >= starNumber - 0.5) return 50;
  return 0;
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
  const [selectedRankMovie, setSelectedRankMovie] =
    useState<MovieSearchResult | null>(null);
  const [movieRating, setMovieRating] = useState(3);
  const [movieDescription, setMovieDescription] = useState("");
  const [ratingMessage, setRatingMessage] = useState("");
  const [savingRating, setSavingRating] = useState(false);

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
      setPrefs({
        ...emptyPrefs,
        ...payload.onboarding_preferences,
      });
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

      try {
        const res = await fetch(
          `${API_BASE_URL}/movies/search?q=${encodeURIComponent(rankQuery)}&limit=6`
        );

        if (!res.ok) {
          setRankResults([]);
          return;
        }

        const data: MovieSearchResult[] = await res.json();
        setRankResults(data);
      } catch (err) {
        logHandledError("Movie ranking search failed", err);
        setRankResults([]);
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
    setSelectedRankMovie(movie);
    setMovieRating(3);
    setMovieDescription("");
    setRatingMessage("");
    setRankQuery("");
    setRankResults([]);
  };

  const saveMovieRating = async () => {
    if (!selectedRankMovie || !userId) return;

    setSavingRating(true);
    setRatingMessage("");

    try {
      const res = await fetch(`${API_BASE_URL}/profile/movie-ratings`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-User-Id": userId,
        },
        body: JSON.stringify({
          movie_id: selectedRankMovie.movie_id,
          title: selectedRankMovie.title,
          rating: movieRating,
          description: movieDescription,
        }),
      });

      if (!res.ok) {
        setRatingMessage("Could not save this rating.");
        return;
      }

      setRatingMessage(`Saved your rating for ${selectedRankMovie.title}.`);
      setSelectedRankMovie(null);
      setMovieDescription("");
    } catch (err) {
      logHandledError("Movie rating save failed", err);
      setRatingMessage("Could not reach the rating service.");
    } finally {
      setSavingRating(false);
    }
  };

  return (
    <main className="profile-page">
      <style>{`
        .profile-page {
          min-height: 100vh;
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
          position: relative;
        }

        .rank-results {
          position: absolute;
          top: calc(100% + 8px);
          left: 0;
          right: 0;
          z-index: 20;
          border: 1px solid rgba(255,255,255,0.1);
          border-radius: 18px;
          overflow: hidden;
          background: #111827;
          box-shadow: 0 18px 48px rgba(0,0,0,0.36);
        }

        .rank-result {
          width: 100%;
          border: none;
          border-bottom: 1px solid rgba(255,255,255,0.08);
          background: transparent;
          color: #f8fafc;
          cursor: pointer;
          padding: 12px 14px;
          text-align: left;
          font-weight: 800;
        }

        .rank-result:hover {
          background: rgba(255,255,255,0.08);
        }

        .rating-card-backdrop {
          position: fixed;
          inset: 0;
          z-index: 60;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 20px;
          background: rgba(0,0,0,0.66);
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
                  <div className="hero-sub">
                    Your personal home base for lists, friends, preferences, and movie taste.
                  </div>
                </div>
              </div>

              <div className="profile-side">
                <div className="profile-side-actions">
                  <button className="primary-btn" onClick={() => openUserProfile()}>
                    Edit Profile
                  </button>
                  <button
                    className="secondary-btn"
                    onClick={() => signOut(() => router.push("/"))}
                  >
                    Sign Out
                  </button>
                </div>

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
                <h2 className="panel-title">My Lists</h2>

                <div className="card-section">
                  <div className="section-title">Personal Saved Lists</div>
                  {data.lists.length === 0 && emptyCopy("personal lists")}
                  {data.lists.map((list) => (
                    <div className="list-row" key={list.id}>
                      <strong>{list.name}</strong>
                      <button
                        className="secondary-btn"
                        onClick={() =>
                          router.push(`/edit-list?kind=personal&listId=${list.id}`)
                        }
                      >
                        Edit
                      </button>
                    </div>
                  ))}
                </div>

                <div className="card-section">
                  <div className="section-title">Shared Lists</div>
                  {data.shared_lists.length === 0 && emptyCopy("shared lists")}
                  {data.shared_lists.map((list) => (
                    <div className="list-row" key={list.id}>
                      <strong>{list.name}</strong>
                      <button
                        className="secondary-btn"
                        onClick={() =>
                          router.push(`/edit-list?kind=shared&listId=${list.id}`)
                        }
                      >
                        Edit
                      </button>
                    </div>
                  ))}
                </div>

                <div className="card-actions">
                  <button className="primary-btn" onClick={() => router.push("/lists")}>
                    View All Lists
                  </button>
                </div>
              </section>



              <section className="panel">
                <h2 className="panel-title">My Activity</h2>

                <div className="card-section">
                  <div className="section-title">Recently Created Lists</div>
                  {createdListActivity.length === 0 && emptyCopy("created lists")}
                  {createdListActivity.map((item) => (
                    <div className="list-row" key={`${item.type}-${item.createdAt}`}>
                      <span>{item.label}</span>
                    </div>
                  ))}
                </div>

                <div className="card-section">
                  <div className="section-title">Recently Edited Shared Lists</div>
                  {editedSharedActivity.length === 0 && emptyCopy("edited shared lists")}
                  {editedSharedActivity.map((item) => (
                    <div className="list-row" key={`${item.type}-${item.createdAt}`}>
                      <span>{item.label}</span>
                    </div>
                  ))}
                </div>

                <div className="card-section">
                  <div className="section-title">Recently Rated Movies</div>
                  {ratedMovieActivity.length === 0 && emptyCopy("rated movies")}
                  {ratedMovieActivity.map((item) => (
                    <div className="list-row" key={`${item.type}-${item.createdAt}`}>
                      <span>{item.label}</span>
                    </div>
                  ))}
                </div>

                <div className="card-section">
                  <div className="section-title">Recently Watched Movies</div>
                  {watchedMovieActivity.length === 0 && emptyCopy("watched movies")}
                  {watchedMovieActivity.map((item) => (
                    <div className="list-row" key={`${item.type}-${item.createdAt}`}>
                      <span>{item.label}</span>
                    </div>
                  ))}
                </div>
              </section>

              <section className="panel">
                <h2 className="panel-title">My Preferences</h2>

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
                <h2 className="panel-title">Rank Some Movies</h2>
                <p className="muted">
                  Search for a movie, rate it from half a star to five stars, and add a quick note.
                </p>

                <div className="rank-search">
                  <input
                    className="input"
                    value={rankQuery}
                    onChange={(event) => setRankQuery(event.target.value)}
                    placeholder="Search movies to rank..."
                  />

                  {rankResults.length > 0 && (
                    <div className="rank-results">
                      {rankResults.map((movie) => (
                        <button
                          className="rank-result"
                          key={movie.movie_id}
                          onClick={() => chooseRankMovie(movie)}
                        >
                          {movie.title}
                        </button>
                      ))}
                    </div>
                  )}
                </div>

                {ratingMessage && <div className="message">{ratingMessage}</div>}
              </section>
            </div>

            {selectedRankMovie && (
              <div className="rating-card-backdrop">
                <div className="rating-card">
                  <h2 className="rating-movie-title">{selectedRankMovie.title}</h2>
                  <div className="rating-movie-id">
                    Movie ID: {selectedRankMovie.movie_id}
                  </div>

                  <div className="section-title">Your Rating</div>
                  <div className="apple-star-row">
                    {[1, 2, 3, 4, 5].map((starNumber) => (
                      <button
                        aria-label={`Rate ${starNumber} stars`}
                        className="apple-star"
                        key={starNumber}
                        style={{
                          "--star-fill": `${starFillPercent(
                            movieRating,
                            starNumber
                          )}%`,
                        } as React.CSSProperties}
                      >
                        <span className="apple-star-empty">
                          <span>★</span>
                        </span>
                        <span className="apple-star-fill" aria-hidden="true">
                          <span>★</span>
                        </span>
                        <span
                          className="apple-star-left"
                          onClick={(event) => {
                            event.stopPropagation();
                            setMovieRating(starNumber - 0.5);
                          }}
                        />
                        <span
                          className="apple-star-right"
                          onClick={(event) => {
                            event.stopPropagation();
                            setMovieRating(starNumber);
                          }}
                        />
                      </button>
                    ))}
                  </div>
                  <div className="rating-value">{movieRating} out of 5</div>

                  <div className="section-title">Description</div>
                  <textarea
                    className="textarea"
                    value={movieDescription}
                    onChange={(event) => setMovieDescription(event.target.value)}
                    placeholder="What did you think? What mood did it fit?"
                  />

                  <div className="rating-actions">
                    <button
                      className="secondary-btn"
                      onClick={() => setSelectedRankMovie(null)}
                    >
                      Cancel
                    </button>
                    <button
                      className="primary-btn"
                      disabled={savingRating}
                      onClick={saveMovieRating}
                    >
                      {savingRating ? "Saving..." : "Save Rating"}
                    </button>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </main>
  );
}
