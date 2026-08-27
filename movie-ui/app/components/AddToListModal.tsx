"use client";

import { useEffect, useMemo, useState } from "react";
import { useUser } from "@clerk/nextjs";

import { API_BASE_URL } from "../lib/config";
import { logHandledError } from "../lib/log";

type MovieForList = {
  movie_id: number;
  title: string;
};

type ListMovie = {
  movie_id: number;
  title: string;
};

type PersonalList = {
  id: string;
  name: string;
  movies: ListMovie[];
};

type SharedMember = {
  user_id: string;
  role: string;
};

type SharedList = {
  id: string;
  name: string;
  movies: ListMovie[];
  members: SharedMember[];
};

type ListChoice = {
  key: string;
  id: string;
  name: string;
  kind: "personal" | "shared";
  movies: ListMovie[];
};

type Props = {
  open: boolean;
  movie: MovieForList | null;
  onClose: () => void;
  onAdded?: (message: string) => void;
};

export default function AddToListModal({
  open,
  movie,
  onClose,
  onAdded,
}: Props) {
  const { user, isSignedIn } = useUser();
  const userId = user?.id;

  const [choices, setChoices] = useState<ListChoice[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!open || !movie) return;

    setSelected(new Set());
    setError("");

    if (!isSignedIn || !userId) {
      setChoices([]);
      setError("Sign in to add movies to a list.");
      return;
    }

    const controller = new AbortController();

    const loadLists = async () => {
      setLoading(true);

      try {
        const headers = { "X-User-Id": userId };
        const [personalRes, sharedRes] = await Promise.all([
          fetch(`${API_BASE_URL}/user/lists`, {
            headers,
            signal: controller.signal,
          }),
          fetch(`${API_BASE_URL}/shared-lists`, {
            headers,
            signal: controller.signal,
          }),
        ]);

        if (!personalRes.ok || !sharedRes.ok) {
          throw new Error("Could not load lists");
        }

        const personal: PersonalList[] = await personalRes.json();
        const shared: SharedList[] = await sharedRes.json();

        const editableShared = shared.filter((list) =>
          (list.members ?? []).some(
            (member) =>
              member.user_id === userId &&
              (member.role === "owner" || member.role === "editor"),
          ),
        );

        setChoices([
          ...personal.map((list) => ({
            key: `personal:${list.id}`,
            id: list.id,
            name: list.name,
            kind: "personal" as const,
            movies: list.movies ?? [],
          })),
          ...editableShared.map((list) => ({
            key: `shared:${list.id}`,
            id: list.id,
            name: list.name,
            kind: "shared" as const,
            movies: list.movies ?? [],
          })),
        ]);
      } catch (err) {
        if ((err as Error).name === "AbortError") return;
        logHandledError("List picker load failed", err);
        setError("Could not load your lists.");
      } finally {
        setLoading(false);
      }
    };

    loadLists();

    return () => controller.abort();
  }, [open, movie, isSignedIn, userId]);

  const existingKeys = useMemo(() => {
    if (!movie) return new Set<string>();

    return new Set(
      choices
        .filter((choice) =>
          choice.movies.some((item) => item.movie_id === movie.movie_id),
        )
        .map((choice) => choice.key),
    );
  }, [choices, movie]);

  if (!open || !movie) return null;

  const toggle = (key: string) => {
    if (existingKeys.has(key) || saving) return;

    setSelected((current) => {
      const next = new Set(current);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  };

  const save = async () => {
    if (!userId || selected.size === 0) return;

    setSaving(true);
    setError("");

    try {
      const selectedChoices = choices.filter((choice) =>
        selected.has(choice.key),
      );

      await Promise.all(
        selectedChoices.map(async (choice) => {
          const headers = {
            "Content-Type": "application/json",
            "X-User-Id": userId,
          };

          if (choice.kind === "personal") {
            const res = await fetch(`${API_BASE_URL}/user/lists/${choice.id}`, {
              method: "PUT",
              headers,
              body: JSON.stringify({
                movies: [
                  ...choice.movies,
                  {
                    movie_id: movie.movie_id,
                    title: movie.title,
                  },
                ],
              }),
            });

            if (!res.ok) {
              throw new Error(`Could not update ${choice.name}`);
            }
            return;
          }

          const res = await fetch(
            `${API_BASE_URL}/shared-lists/${choice.id}/movies`,
            {
              method: "POST",
              headers,
              body: JSON.stringify({
                movie_id: movie.movie_id,
                title: movie.title,
              }),
            },
          );

          if (!res.ok) {
            throw new Error(`Could not update ${choice.name}`);
          }
        }),
      );

      const count = selectedChoices.length;
      onAdded?.(
        `Added ${movie.title} to ${count} ${count === 1 ? "list" : "lists"}.`,
      );
      onClose();
    } catch (err) {
      logHandledError("Add to lists failed", err);
      setError("Could not add that movie to every selected list.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div
      className="add-list-backdrop"
      onClick={() => {
        if (!saving) onClose();
      }}
    >
      <style>{`
        .add-list-backdrop {
          position: fixed;
          inset: 0;
          z-index: 120;
          display: grid;
          place-items: center;
          padding: 20px;
          background: rgba(0, 0, 0, 0.72);
        }
        .add-list-modal {
          width: min(460px, 100%);
          max-height: min(620px, 88vh);
          overflow: hidden;
          border: 1px solid rgba(255,255,255,0.12);
          border-radius: 24px;
          background: #0f172a;
          color: #f8fafc;
          box-shadow: 0 30px 90px rgba(0,0,0,0.55);
        }
        .add-list-head {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 16px;
          padding: 20px 20px 12px;
        }
        .add-list-title {
          margin: 0;
          font-size: 1.35rem;
          font-weight: 950;
        }
        .add-list-subtitle {
          margin-top: 5px;
          color: rgba(255,255,255,0.62);
          font-size: 0.9rem;
        }
        .add-list-close {
          width: 34px;
          height: 34px;
          flex: 0 0 auto;
          border-radius: 999px;
          border: 1px solid rgba(255,255,255,0.14);
          background: rgba(255,255,255,0.06);
          color: #f8fafc;
          cursor: pointer;
          font-size: 20px;
        }
        .add-list-options {
          max-height: 390px;
          overflow-y: auto;
          padding: 8px 14px;
        }
        .add-list-option {
          display: flex;
          align-items: center;
          gap: 12px;
          width: 100%;
          padding: 12px;
          margin-bottom: 8px;
          border: 1px solid rgba(255,255,255,0.08);
          border-radius: 14px;
          background: rgba(255,255,255,0.04);
          cursor: pointer;
        }
        .add-list-option:hover {
          background: rgba(255,255,255,0.08);
        }
        .add-list-option.disabled {
          cursor: default;
          opacity: 0.66;
        }
        .add-list-option input {
          width: 18px;
          height: 18px;
          accent-color: #8b5cf6;
        }
        .add-list-name {
          min-width: 0;
          flex: 1;
          font-weight: 850;
        }
        .add-list-badge {
          border: 1px solid rgba(196,181,253,0.24);
          border-radius: 999px;
          padding: 3px 7px;
          color: #ddd6fe;
          background: rgba(139,92,246,0.12);
          font-size: 10px;
          font-weight: 900;
          text-transform: uppercase;
        }
        .add-list-existing {
          color: rgba(255,255,255,0.5);
          font-size: 11px;
          font-weight: 800;
        }
        .add-list-empty,
        .add-list-error {
          padding: 18px 12px;
          color: rgba(255,255,255,0.68);
          line-height: 1.45;
        }
        .add-list-error {
          color: #fecaca;
        }
        .add-list-actions {
          display: flex;
          justify-content: flex-end;
          gap: 10px;
          padding: 14px 20px 20px;
          border-top: 1px solid rgba(255,255,255,0.08);
        }
        .add-list-cancel,
        .add-list-done {
          border-radius: 999px;
          padding: 9px 16px;
          font-weight: 900;
          cursor: pointer;
        }
        .add-list-cancel {
          border: 1px solid rgba(255,255,255,0.12);
          background: rgba(255,255,255,0.05);
          color: #f8fafc;
        }
        .add-list-done {
          border: 1px solid rgba(167,139,250,0.32);
          background: #7c3aed;
          color: white;
        }
        .add-list-done:disabled,
        .add-list-cancel:disabled {
          cursor: not-allowed;
          opacity: 0.45;
        }
      `}</style>

      <div
        className="add-list-modal"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="add-list-head">
          <div>
            <h2 className="add-list-title">Add to List</h2>
            <div className="add-list-subtitle">{movie.title}</div>
          </div>
          <button
            className="add-list-close"
            type="button"
            onClick={onClose}
            disabled={saving}
            aria-label="Close"
          >
            ×
          </button>
        </div>

        <div className="add-list-options">
          {loading && <div className="add-list-empty">Loading your lists...</div>}

          {!loading && error && <div className="add-list-error">{error}</div>}

          {!loading && !error && choices.length === 0 && (
            <div className="add-list-empty">
              You do not have any personal or editable shared lists yet.
            </div>
          )}

          {!loading &&
            !error &&
            choices.map((choice) => {
              const alreadyAdded = existingKeys.has(choice.key);
              const checked = alreadyAdded || selected.has(choice.key);

              return (
                <label
                  className={`add-list-option ${alreadyAdded ? "disabled" : ""}`}
                  key={choice.key}
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    disabled={alreadyAdded || saving}
                    onChange={() => toggle(choice.key)}
                  />
                  <span className="add-list-name">{choice.name}</span>
                  {choice.kind === "shared" && (
                    <span className="add-list-badge">Shared</span>
                  )}
                  {alreadyAdded && (
                    <span className="add-list-existing">Already added</span>
                  )}
                </label>
              );
            })}
        </div>

        <div className="add-list-actions">
          <button
            className="add-list-cancel"
            type="button"
            onClick={onClose}
            disabled={saving}
          >
            Cancel
          </button>
          <button
            className="add-list-done"
            type="button"
            onClick={save}
            disabled={saving || loading || selected.size === 0}
          >
            {saving ? "Adding..." : "Done"}
          </button>
        </div>
      </div>
    </div>
  );
}
