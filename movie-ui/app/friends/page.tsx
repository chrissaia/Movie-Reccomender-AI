"use client";

import { SignInButton, useUser } from "@clerk/nextjs";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import AppHeader from "../components/AppHeader";
import { API_BASE_URL } from "../lib/config";
import type { FriendsResponse, UserProfile } from "../types";

export default function FriendsPage() {
  const router = useRouter();
  const { user, isSignedIn, isLoaded } = useUser();
  const userId = user?.id;

  const [query, setQuery] = useState("");
  const [searchResults, setSearchResults] = useState<UserProfile[]>([]);
  const [friendsData, setFriendsData] = useState<FriendsResponse>({
    friends: [],
    incoming_requests: [],
    outgoing_requests: [],
  });
  const [message, setMessage] = useState("");

  const userEmail = user?.primaryEmailAddress?.emailAddress ?? null;
  const userName =
    user?.fullName ?? user?.username ?? user?.firstName ?? userEmail ?? null;

  const authHeaders: Record<string, string> = userId
    ? {
        "X-User-Id": userId,
      }
    : {};

  const loadFriends = async () => {
    if (!userId) return;

    const res = await fetch(`${API_BASE_URL}/friends`, {
      headers: authHeaders,
    });

    if (!res.ok) throw new Error("Failed to load friends");

    const data: FriendsResponse = await res.json();
    setFriendsData(data);
  };

  useEffect(() => {
    const syncProfile = async () => {
      if (!isLoaded || !isSignedIn || !userId) return;

      await fetch(`${API_BASE_URL}/user/profile`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-User-Id": userId,
        },
        body: JSON.stringify({
          email: userEmail,
          name: userName,
        }),
      });

      await loadFriends();
    };

    syncProfile();
  }, [isLoaded, isSignedIn, userId, userEmail, userName]);

  useEffect(() => {
    const searchUsers = async () => {
      if (!query.trim() || !userId) {
        setSearchResults([]);
        return;
      }

      const res = await fetch(
        `${API_BASE_URL}/users/search?q=${encodeURIComponent(query)}`,
        {
          headers: authHeaders,
        }
      );

      if (!res.ok) {
        setSearchResults([]);
        return;
      }

      const data: UserProfile[] = await res.json();
      setSearchResults(data);
    };

    const timeout = setTimeout(searchUsers, 250);
    return () => clearTimeout(timeout);
  }, [query, userId]);

  const sendRequest = async (receiverUserId: string) => {
    if (!userId) return;

    const res = await fetch(`${API_BASE_URL}/friends/request`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-User-Id": userId,
      },
      body: JSON.stringify({
        receiver_user_id: receiverUserId,
      }),
    });

    if (!res.ok) {
      setMessage("Could not send friend request.");
      return;
    }

    setMessage("Friend request sent.");
    setQuery("");
    setSearchResults([]);
    await loadFriends();
  };

  const acceptRequest = async (friendshipId: string) => {
    if (!userId) return;

    const res = await fetch(`${API_BASE_URL}/friends/accept/${friendshipId}`, {
      method: "POST",
      headers: authHeaders,
    });

    if (!res.ok) {
      setMessage("Could not accept request.");
      return;
    }

    setMessage("Friend request accepted.");
    await loadFriends();
  };

  const removeFriend = async (friendshipId: string) => {
    if (!userId) return;

    const res = await fetch(`${API_BASE_URL}/friends/${friendshipId}`, {
      method: "DELETE",
      headers: authHeaders,
    });

    if (!res.ok) {
      setMessage("Could not remove friend.");
      return;
    }

    setMessage("Friend removed.");
    await loadFriends();
  };

  const displayName = (profile: UserProfile | null) => {
    if (!profile) return "Unknown user";
    return profile.name || profile.email || profile.user_id;
  };

  return (
    <main className="friends-page">
      <style>{`
        .friends-page {
          min-height: 100vh;
          padding: 28px 24px 56px;
          background:
            radial-gradient(circle at 18% 12%, rgba(168, 85, 247, 0.22), transparent 34%),
            radial-gradient(circle at 82% 8%, rgba(99, 102, 241, 0.18), transparent 30%),
            linear-gradient(135deg, #070617 0%, #18102f 38%, #4c1d95 100%);
          color: #f8fafc;
        }

        .wrap {
          max-width: 1100px;
          margin: 0 auto;
        }

        .top-bar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 16px;
          margin-bottom: 32px;
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

        .primary-btn {
          border: none;
          background: linear-gradient(135deg, #8b5cf6, #6d28d9);
          color: white;
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
          font-size: clamp(2rem, 4vw, 3.5rem);
          line-height: 1.05;
          font-weight: 950;
          letter-spacing: -0.05em;
          margin: 0 0 10px;
        }

        .hero-sub {
          color: rgba(255,255,255,0.72);
          font-size: 17px;
          margin-bottom: 28px;
        }

        .grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
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

        .search-input {
          width: 100%;
          padding: 15px 16px;
          border-radius: 16px;
          border: 1px solid rgba(255,255,255,0.12);
          background: rgba(255,255,255,0.06);
          color: #f8fafc;
          outline: none;
          font-size: 16px;
          margin-bottom: 16px;
        }

        .stack {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .user-card {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
          border: 1px solid rgba(255,255,255,0.08);
          background: rgba(255,255,255,0.05);
          border-radius: 18px;
          padding: 14px;
        }

        .user-name {
          font-weight: 900;
          margin-bottom: 4px;
        }

        .user-email {
          color: rgba(255,255,255,0.66);
          font-size: 14px;
        }

        .user-actions {
          display: flex;
          align-items: center;
          gap: 10px;
          flex-wrap: wrap;
          justify-content: flex-end;
        }

        .section {
          margin-bottom: 22px;
        }

        .empty {
          color: rgba(255,255,255,0.68);
          line-height: 1.6;
        }

        .message {
          margin-bottom: 18px;
          color: #ddd6fe;
          font-weight: 800;
        }

        @media (max-width: 850px) {
          .grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>

      <div className="wrap">
        <AppHeader
          leading={{ label: "← Search", href: "/" }}
          actions={[{ label: "My Lists", href: "/lists" }]}
        />

        <h1 className="hero-title">Friends</h1>
        <div className="hero-sub">
          Add friends now. Shared movie recommendations come next.
        </div>

        {message && <div className="message">{message}</div>}

        {!isSignedIn && isLoaded && (
          <div className="panel">
            <h2 className="panel-title">Sign in to add friends.</h2>
            <p className="empty">
              Friends are tied to your account and saved in the backend.
            </p>
            <SignInButton mode="modal">
              <button className="primary-btn">Sign In</button>
            </SignInButton>
          </div>
        )}

        {isSignedIn && (
          <div className="grid">
            <section className="panel">
              <h2 className="panel-title">Find people</h2>

              <input
                className="search-input"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search by name or email..."
              />

              <div className="stack">
                {searchResults.map((profile) => (
                  <div className="user-card" key={profile.user_id}>
                    <div>
                      <div className="user-name">{displayName(profile)}</div>
                      <div className="user-email">{profile.email}</div>
                    </div>

                    <div className="user-actions">
                      <button
                        className="pill-btn"
                        onClick={() => router.push(`/profile/${profile.user_id}`)}
                      >
                        View
                      </button>
                      <button
                        className="primary-btn"
                        onClick={() => sendRequest(profile.user_id)}
                      >
                        Add
                      </button>
                    </div>
                  </div>
                ))}

                {query.trim() && searchResults.length === 0 && (
                  <div className="empty">No matching users found.</div>
                )}
              </div>
            </section>

            <section className="panel">
              <div className="section">
                <h2 className="panel-title">Your friends</h2>

                <div className="stack">
                  {friendsData.friends.map((friend) => (
                    <div className="user-card" key={friend.id}>
                      <div>
                        <div className="user-name">
                          {displayName(friend.other_user)}
                        </div>
                        <div className="user-email">
                          {friend.other_user?.email}
                        </div>
                      </div>

                      <div className="user-actions">
                        <button
                          className="pill-btn"
                          onClick={() =>
                            router.push(`/profile/${friend.other_user_id}`)
                          }
                        >
                          View
                        </button>
                        <button
                          className="danger-btn"
                          onClick={() => removeFriend(friend.id)}
                        >
                          Remove
                        </button>
                      </div>
                    </div>
                  ))}

                  {friendsData.friends.length === 0 && (
                    <div className="empty">No friends yet.</div>
                  )}
                </div>
              </div>

              <div className="section">
                <h2 className="panel-title">Incoming requests</h2>

                <div className="stack">
                  {friendsData.incoming_requests.map((request) => (
                    <div className="user-card" key={request.id}>
                      <div>
                        <div className="user-name">
                          {displayName(request.other_user)}
                        </div>
                        <div className="user-email">
                          {request.other_user?.email}
                        </div>
                      </div>

                      <button
                        className="primary-btn"
                        onClick={() => acceptRequest(request.id)}
                      >
                        Accept
                      </button>
                    </div>
                  ))}

                  {friendsData.incoming_requests.length === 0 && (
                    <div className="empty">No incoming requests.</div>
                  )}
                </div>
              </div>

              <div className="section">
                <h2 className="panel-title">Sent requests</h2>

                <div className="stack">
                  {friendsData.outgoing_requests.map((request) => (
                    <div className="user-card" key={request.id}>
                      <div>
                        <div className="user-name">
                          {displayName(request.other_user)}
                        </div>
                        <div className="user-email">
                          {request.other_user?.email}
                        </div>
                      </div>

                      <button
                        className="danger-btn"
                        onClick={() => removeFriend(request.id)}
                      >
                        Cancel
                      </button>
                    </div>
                  ))}

                  {friendsData.outgoing_requests.length === 0 && (
                    <div className="empty">No sent requests.</div>
                  )}
                </div>
              </div>
            </section>
          </div>
        )}
      </div>
    </main>
  );
}
