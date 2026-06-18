"use client";

import { SignInButton, useUser } from "@clerk/nextjs";
import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";

import AppHeader from "../../components/AppHeader";
import { API_BASE_URL } from "../../lib/config";
import { logHandledError } from "../../lib/log";

type FriendProfilePayload = {
  profile: {
    user_id: string;
    name: string | null;
    bio: string | null;
    avatar_url: string | null;
  };
  taste_summary: {
    favorite_genres: string[];
    favorite_directors: string[];
    favorite_actors: string[];
    favorite_keywords: string[];
  };
  taste_match?: {
    similarity_score: number;
    overlap: {
      genres: string[];
      actors: string[];
      directors: string[];
      keywords: string[];
    };
    shared_saved_movies: {
      movie_id: number;
      title: string;
    }[];
    curated_seed_movie_ids: number[];
  };
  stats: {
    shared_lists_count: number;
    friends_count: number;
    unique_movies_count: number;
  };
  shared_lists: {
    id: string;
    name: string;
    owner_user_id: string;
    createdAt: string;
    updatedAt: string;
  }[];
};

function pretty(value: string) {
  return value.replaceAll("-", " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function emptyCopy(label: string) {
  return <div className="muted">No {label} available yet.</div>;
}

const emptyTasteMatch = {
  similarity_score: 0,
  overlap: {
    genres: [],
    actors: [],
    directors: [],
    keywords: [],
  },
  shared_saved_movies: [],
  curated_seed_movie_ids: [],
};

export default function FriendProfilePage() {
  const params = useParams<{ userId: string }>();
  const router = useRouter();
  const { user, isLoaded, isSignedIn } = useUser();
  const viewerUserId = user?.id;

  const [data, setData] = useState<FriendProfilePayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [creatingCuratedProfile, setCreatingCuratedProfile] = useState(false);
  const [curatedMessage, setCuratedMessage] = useState("");
  const [friendRecommendations, setFriendRecommendations] = useState<{ movie_id: number; title: string }[]>([]);

  useEffect(() => {
    const loadProfile = async () => {
      if (!isLoaded || !isSignedIn || !viewerUserId || !params.userId) return;

      setLoading(true);
      setMessage("");

      try {
        const res = await fetch(`${API_BASE_URL}/profiles/${params.userId}`, {
          headers: {
            "X-User-Id": viewerUserId,
          },
        });

        if (!res.ok) {
          setMessage("This profile is not available.");
          setData(null);
          return;
        }

        const payload: FriendProfilePayload = await res.json();
        setData(payload);
      } catch (err) {
        logHandledError("Friend profile load failed", err);
        setMessage("Could not reach the profile service.");
      } finally {
        setLoading(false);
      }
    };

    loadProfile();
  }, [isLoaded, isSignedIn, viewerUserId, params.userId]);

  const displayName = data?.profile.name || "Movie Friend";
  const profileInitial = displayName[0]?.toUpperCase() ?? "M";
  const tasteMatch = data?.taste_match ?? emptyTasteMatch;
  const similarityPercent = Math.round(tasteMatch.similarity_score * 100);

  const copySharedList = async (listId: string, name: string) => {
    if (!viewerUserId) return;

    try {
      const res = await fetch(`${API_BASE_URL}/shared-lists/${listId}/copy-to-my-lists`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-User-Id": viewerUserId,
        },
        body: JSON.stringify({ name: `Copy of ${name}` }),
      });

      setCuratedMessage(res.ok ? "List copied to My Lists." : "Could not copy that list.");
    } catch (err) {
      logHandledError("Shared list copy failed", err);
      setCuratedMessage("Could not reach the list service.");
    }
  };

  const loadFriendRecommendations = async () => {
    if (!viewerUserId || !params.userId) return;

    try {
      const res = await fetch(`${API_BASE_URL}/profiles/${params.userId}/recommend`, {
        method: "POST",
        headers: { "X-User-Id": viewerUserId },
      });

      if (!res.ok) {
        setCuratedMessage("Could not load friend recommendations yet.");
        return;
      }

      const payload = await res.json();
      setFriendRecommendations(payload.recommendations ?? []);
      setCuratedMessage("Recommendations loaded from your shared taste.");
    } catch (err) {
      logHandledError("Friend recommendations load failed", err);
      setCuratedMessage("Could not reach the recommendation service.");
    }
  };

  const createCuratedProfile = async () => {
    if (!viewerUserId || !params.userId || !data) return;

    setCreatingCuratedProfile(true);
    setCuratedMessage("");

    try {
      const res = await fetch(
        `${API_BASE_URL}/profiles/${params.userId}/curated-taste-profile`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-User-Id": viewerUserId,
          },
          body: JSON.stringify({
            name: `${displayName} + Me Taste Profile`,
          }),
        }
      );

      if (!res.ok) {
        setCuratedMessage("Could not create a curated profile yet.");
        return;
      }

      setCuratedMessage("Curated taste profile created as a shared list.");
    } catch (err) {
      logHandledError("Curated taste profile create failed", err);
      setCuratedMessage("Could not reach the profile service.");
    } finally {
      setCreatingCuratedProfile(false);
    }
  };

  return (
    <main className="friend-profile-page">
      <style>{`
        .friend-profile-page {
          min-height: 100vh;
          padding: 28px 24px 64px;
          background:
            radial-gradient(circle at 18% 12%, rgba(168, 85, 247, 0.22), transparent 34%),
            radial-gradient(circle at 82% 8%, rgba(59, 130, 246, 0.18), transparent 30%),
            linear-gradient(135deg, #070617 0%, #10172f 38%, #312e81 100%);
          color: #f8fafc;
        }

        .wrap {
          max-width: 1100px;
          margin: 0 auto;
        }

        .hero {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 20px;
          border: 1px solid rgba(255,255,255,0.08);
          background: rgba(255,255,255,0.05);
          border-radius: 24px;
          padding: 22px;
          box-shadow: 0 16px 40px rgba(0,0,0,0.28);
          margin: 28px 0 18px;
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
          width: 88px;
          height: 88px;
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
          font-size: 32px;
          font-weight: 950;
        }

        .hero-title {
          font-size: clamp(2rem, 4vw, 3.3rem);
          line-height: 1.04;
          font-weight: 950;
          letter-spacing: -0.05em;
          margin: 0 0 10px;
        }

        .hero-sub,
        .muted {
          color: rgba(255,255,255,0.72);
          line-height: 1.55;
        }

        .profile-card-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 18px;
          align-items: start;
        }

        .panel {
          border: 1px solid rgba(255,255,255,0.08);
          background: rgba(255,255,255,0.05);
          border-radius: 24px;
          padding: 20px;
          box-shadow: 0 16px 40px rgba(0,0,0,0.28);
        }

        .panel-title {
          font-size: 22px;
          font-weight: 900;
          margin: 0 0 14px;
          letter-spacing: -0.03em;
        }

        .stat-row {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
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

        .section-title {
          color: #c4b5fd;
          font-size: 13px;
          font-weight: 950;
          margin: 16px 0 8px;
          text-transform: uppercase;
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

        .taste-match-panel {
          margin-bottom: 18px;
        }

        .match-head {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 18px;
          margin-bottom: 16px;
        }

        .match-score {
          display: grid;
          place-items: center;
          width: 86px;
          height: 86px;
          border-radius: 999px;
          border: 1px solid rgba(255,255,255,0.14);
          background: linear-gradient(135deg, rgba(139,92,246,0.38), rgba(59,130,246,0.28));
          box-shadow: 0 14px 34px rgba(0,0,0,0.22);
          flex: 0 0 auto;
        }

        .match-score-number {
          font-size: 25px;
          font-weight: 950;
          line-height: 1;
        }

        .match-score-label {
          color: rgba(255,255,255,0.68);
          font-size: 11px;
          font-weight: 850;
          margin-top: 5px;
        }

        .match-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 16px;
        }

        .match-actions {
          display: flex;
          align-items: center;
          gap: 12px;
          margin-top: 18px;
          flex-wrap: wrap;
        }

        .match-empty {
          color: rgba(255,255,255,0.58);
          font-size: 13px;
        }

        @media (max-width: 850px) {
          .profile-card-grid,
          .match-grid,
          .stat-row {
            grid-template-columns: 1fr;
          }

          .hero {
            flex-direction: column;
            align-items: flex-start;
          }
        }
      `}</style>

      <div className="wrap">
        <AppHeader
          leading={{ label: "← Friends", href: "/friends" }}
          actions={[{ label: "My Profile", href: "/profile" }]}
        />

        {!isSignedIn && isLoaded && (
          <div className="panel">
            <h1 className="panel-title">Sign in to view profiles.</h1>
            <SignInButton mode="modal">
              <button className="primary-btn">Sign In</button>
            </SignInButton>
          </div>
        )}

        {isSignedIn && loading && <div className="panel">Loading profile...</div>}

        {isSignedIn && !loading && message && !data && (
          <div className="panel">
            <h1 className="panel-title">Profile unavailable.</h1>
            <p className="muted">{message}</p>
          </div>
        )}

        {isSignedIn && data && (
          <>
            <section className="hero">
              <div className="profile-top">
                {data.profile.avatar_url ? (
                  <img
                    className="profile-avatar"
                    src={data.profile.avatar_url}
                    alt=""
                  />
                ) : (
                  <div className="profile-avatar-fallback" aria-hidden="true">
                    {profileInitial}
                  </div>
                )}

                <div className="profile-copy">
                  <h1 className="hero-title">{displayName}</h1>
                  <div className="hero-sub">
                    {data.profile.bio ||
                      "This friend has not added a bio yet."}
                  </div>
                </div>
              </div>

              <button
                className="primary-btn"
                onClick={() => router.push(`/lists/${data.profile.user_id}`)}
              >
                View Their Lists
              </button>
            </section>

            <section className="stat-row">
              <div className="stat-card">
                <div className="stat-number">{data.stats.shared_lists_count}</div>
                <div className="stat-label">Shared Lists</div>
              </div>
              <div className="stat-card">
                <div className="stat-number">{data.stats.friends_count}</div>
                <div className="stat-label">Friends</div>
              </div>
              <div className="stat-card">
                <div className="stat-number">{data.stats.unique_movies_count}</div>
                <div className="stat-label">Movies Saved</div>
              </div>
            </section>

            <section className="panel taste-match-panel">
              <div className="match-head">
                <div>
                  <h2 className="panel-title">Taste Match</h2>
                  <div className="muted">
                    Compare your saved movie taste with {displayName} and turn the overlap into a shared list.
                  </div>
                </div>

                <div className="match-score" aria-label={`${similarityPercent}% taste match`}>
                  <div>
                    <div className="match-score-number">{similarityPercent}%</div>
                    <div className="match-score-label">Match</div>
                  </div>
                </div>
              </div>

              <div className="match-grid">
                {[
                  ["Shared Genres", tasteMatch.overlap.genres],
                  ["Shared Actors", tasteMatch.overlap.actors],
                  ["Shared Directors", tasteMatch.overlap.directors],
                  ["Shared Keywords", tasteMatch.overlap.keywords],
                ].map(([label, values]) => (
                  <div key={label as string}>
                    <div className="section-title">{label as string}</div>
                    <div className="chip-row">
                      {(values as string[]).map((item) => (
                        <span className="chip" key={`${label}-${item}`}>
                          {pretty(item)}
                        </span>
                      ))}
                      {(values as string[]).length === 0 && (
                        <div className="match-empty">No overlap yet.</div>
                      )}
                    </div>
                  </div>
                ))}
              </div>

              <div className="section-title">Shared Saved Movies</div>
              <div className="chip-row">
                {tasteMatch.shared_saved_movies.map((movie) => (
                  <span className="chip" key={movie.movie_id}>
                    {movie.title}
                  </span>
                ))}
                {tasteMatch.shared_saved_movies.length === 0 && (
                  <div className="match-empty">No shared saved movies yet.</div>
                )}
              </div>

              {friendRecommendations.length > 0 && (
                <>
                  <div className="section-title">Recommendations From This Profile</div>
                  <div className="chip-row">
                    {friendRecommendations.slice(0, 12).map((movie) => (
                      <span className="chip" key={movie.movie_id}>{movie.title}</span>
                    ))}
                  </div>
                </>
              )}
            </section>
              <section className="panel">
                <h2 className="panel-title">Taste Profile</h2>

                <div className="section-title">Favorite Genres</div>
                <div className="chip-row">
                  {data.taste_summary.favorite_genres.map((item) => (
                    <span className="chip" key={item}>{pretty(item)}</span>
                  ))}
                  {data.taste_summary.favorite_genres.length === 0 &&
                    emptyCopy("favorite genres")}
                </div>

                <div className="section-title">Favorite Actors</div>
                <div className="chip-row">
                  {data.taste_summary.favorite_actors.map((item) => (
                    <span className="chip" key={item}>{pretty(item)}</span>
                  ))}
                  {data.taste_summary.favorite_actors.length === 0 &&
                    emptyCopy("favorite actors")}
                </div>

                <div className="section-title">Favorite Directors</div>
                <div className="chip-row">
                  {data.taste_summary.favorite_directors.map((item) => (
                    <span className="chip" key={item}>{pretty(item)}</span>
                  ))}
                  {data.taste_summary.favorite_directors.length === 0 &&
                    emptyCopy("favorite directors")}
                </div>
              </section>
          </>
        )}
      </div>
    </main>
  );
}
