"use client";

import { SignInButton, useUser } from "@clerk/nextjs";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import AppHeader from "../components/AppHeader";
import { API_BASE_URL } from "../lib/config";

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

function listToText(values: string[] | undefined) {
  return (values ?? []).join(", ");
}

function textToList(value: string) {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

export default function ProfilePage() {
  const router = useRouter();
  const { user, isSignedIn, isLoaded } = useUser();
  const userId = user?.id;

  const [data, setData] = useState<ProfilePayload | null>(null);
  const [name, setName] = useState("");
  const [bio, setBio] = useState("");
  const [prefs, setPrefs] = useState<Record<string, string[]>>(emptyPrefs);
  const [message, setMessage] = useState("");

  const userEmail = user?.primaryEmailAddress?.emailAddress ?? null;
  const clerkName =
    user?.fullName ?? user?.username ?? user?.firstName ?? userEmail ?? null;

  const loadProfile = async () => {
    if (!userId) return;

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
      console.error(err);
      setMessage("Could not reach the profile service.");
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
        console.error(err);
        setMessage("Could not reach the profile service.");
      }
    };

    syncAndLoad();
  }, [isLoaded, isSignedIn, userId, userEmail, clerkName]);

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
      console.error(err);
      setMessage("Could not reach the profile service.");
    }
  };

  const savePreferences = async () => {
    if (!userId) return;

    try {
      const res = await fetch(`${API_BASE_URL}/profile/onboarding`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          "X-User-Id": userId,
        },
        body: JSON.stringify(prefs),
      });

      if (!res.ok) {
        setMessage("Could not save preferences.");
        return;
      }

      setMessage("Preferences saved.");
      await loadProfile();
    } catch (err) {
      console.error(err);
      setMessage("Could not reach the profile service.");
    }
  };

  const updatePref = (key: string, value: string) => {
    setPrefs((prev) => ({
      ...prev,
      [key]: textToList(value),
    }));
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
          gap: 20px;
          margin-bottom: 24px;
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
          .grid,
          .stats {
            grid-template-columns: 1fr;
          }

          .hero {
            flex-direction: column;
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

        {isSignedIn && !data && <div className="panel">Loading profile...</div>}

        {isSignedIn && data && (
          <>
            <section className="hero">
              <div>
                <h1 className="hero-title">{name || "Your Movie Profile"}</h1>
                <div className="hero-sub">
                  Your personal home base for lists, friends, preferences, and movie taste.
                </div>
              </div>
            </section>

            {message && <div className="message">{message}</div>}

            <section className="stats">
              <div className="stat-card">
                <div className="stat-number">{data.stats.saved_lists_count}</div>
                <div className="stat-label">Saved Lists</div>
              </div>
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

            <div className="grid">
              <div>
                <section className="panel">
                  <h2 className="panel-title">Edit Profile</h2>
                  <input
                    className="input"
                    value={name}
                    onChange={(event) => setName(event.target.value)}
                    placeholder="Display name"
                  />
                  <textarea
                    className="textarea"
                    value={bio}
                    onChange={(event) => setBio(event.target.value)}
                    placeholder="Short bio..."
                  />
                  <button className="primary-btn" onClick={saveProfile}>
                    Save Profile
                  </button>
                </section>

                <section className="panel">
                  <h2 className="panel-title">Onboarding Preferences</h2>

                  {Object.keys(emptyPrefs).map((key) => (
                    <div key={key}>
                      <div className="pref-label">{pretty(key.replaceAll("_", " "))}</div>
                      <input
                        className="input"
                        value={listToText(prefs[key])}
                        onChange={(event) => updatePref(key, event.target.value)}
                        placeholder="Comma-separated values"
                      />
                    </div>
                  ))}

                  <button className="primary-btn" onClick={savePreferences}>
                    Save Preferences
                  </button>
                </section>

                <section className="panel">
                  <h2 className="panel-title">My Taste Profile</h2>

                  <div className="pref-label">Favorite Genres</div>
                  <div className="chip-row">
                    {data.taste_summary.favorite_genres.map((item) => (
                      <span className="chip" key={item}>{pretty(item)}</span>
                    ))}
                  </div>

                  <div className="pref-label">Favorite Directors</div>
                  <div className="chip-row">
                    {data.taste_summary.favorite_directors.map((item) => (
                      <span className="chip" key={item}>{pretty(item)}</span>
                    ))}
                  </div>

                  <div className="pref-label">Favorite Actors</div>
                  <div className="chip-row">
                    {data.taste_summary.favorite_actors.map((item) => (
                      <span className="chip" key={item}>{pretty(item)}</span>
                    ))}
                  </div>

                  <div className="pref-label">Signals / Keywords</div>
                  <div className="chip-row">
                    {data.taste_summary.favorite_keywords.map((item) => (
                      <span className="chip" key={item}>{pretty(item)}</span>
                    ))}
                  </div>
                </section>
              </div>

              <div>
                <section className="panel">
                  <h2 className="panel-title">My Lists</h2>
                  {data.lists.length === 0 && <div className="muted">No saved lists yet.</div>}
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
                </section>

                <section className="panel">
                  <h2 className="panel-title">Shared Lists</h2>
                  {data.shared_lists.length === 0 && (
                    <div className="muted">No shared lists yet.</div>
                  )}
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
                </section>

                <section className="panel">
                  <h2 className="panel-title">My Friends</h2>
                  {data.friends.length === 0 && <div className="muted">No friends yet.</div>}
                  {data.friends.map((friend) => (
                    <div className="list-row" key={friend.user_id}>
                      <strong>{friend.name || friend.email || "Friend"}</strong>
                    </div>
                  ))}
                </section>

                <section className="panel">
                  <h2 className="panel-title">Recent Activity</h2>
                  {data.recent_activity.length === 0 && (
                    <div className="muted">No recent activity yet.</div>
                  )}
                  {data.recent_activity.map((item) => (
                    <div className="list-row" key={`${item.type}-${item.createdAt}`}>
                      <span>{item.label}</span>
                    </div>
                  ))}
                </section>
              </div>
            </div>
          </>
        )}
      </div>
    </main>
  );
}
