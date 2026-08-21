"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import AuthProfileButton from "./components/AuthProfileButton";
import { API_BASE_URL, FALLBACK_POSTER } from "./lib/config";
import { logHandledError } from "./lib/log";
import { getTmdbOptionalPoster } from "./lib/tmdb";
import type { Movie } from "./types";

export default function Home() {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Movie[]>([]);
  const [selected, setSelected] = useState<Movie[]>([]);
  const [showDropdown, setShowDropdown] = useState(false);
  const boxRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (!boxRef.current?.contains(e.target as Node)) setShowDropdown(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  useEffect(() => {
    const searchMovies = async () => {
      if (!query.trim()) {
        setResults([]);
        return;
      }

      try {
        const res = await fetch(
          `${API_BASE_URL}/movies/search?q=${encodeURIComponent(query)}`,
        );
        const data = await res.json();

        const withPosters = await Promise.all(
          data.map(async (movie: Movie) => {
            const poster = await getTmdbOptionalPoster(movie.title);
            return {
              ...movie,
              poster,
            };
          }),
        );

        setResults(withPosters);
        setShowDropdown(true);
      } catch (err) {
        logHandledError("Movie search failed", err);
      }
    };

    const timeoutId = setTimeout(searchMovies, 250);
    return () => clearTimeout(timeoutId);
  }, [query]);

  const filteredResults = useMemo(() => {
    return results.filter(
      (m) => !selected.some((s) => s.movie_id === m.movie_id),
    );
  }, [results, selected]);

  const addMovie = (movie: Movie) => {
    setSelected((prev) => [...prev, movie]);
    setQuery("");
    setResults([]);
    setShowDropdown(false);
  };

  const removeMovie = (movieId: number) => {
    setSelected((prev) => prev.filter((m) => m.movie_id !== movieId));
  };

  const isSentenceSearch = query.trim().split(/\s+/).length >= 4;

  const discoverFromSentence = () => {
    const trimmed = query.trim();
    if (!trimmed) return;
    router.push(`/discover?q=${encodeURIComponent(trimmed)}`);
  };

  const seeResults = () => {
    if (!selected.length) {
      discoverFromSentence();
      return;
    }
    localStorage.setItem("selectedMovies", JSON.stringify(selected));
    router.push("/results");
  };

  return (
    <main
      style={{
        minHeight: "100vh",
        padding: 24,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background:
          "linear-gradient(135deg, #020617 0%, #0f172a 35%, #1e3a8a 100%)",
      }}
    >
      <div className="absolute top-6 right-6 z-20 flex items-center gap-3">
        <button
          className="border border-white/12 bg-white/4 text-slate-200 rounded-full px-4 py-1.5 cursor-pointer font-bold"
          onClick={() => router.push("/friends")}
        >
          Friends
        </button>
        <button
          onClick={() => router.push("/lists")}
          className="border border-white/12 bg-white/4 text-slate-200 rounded-full px-4 py-1.5 cursor-pointer font-bold"
        >
          My Lists
        </button>
        <AuthProfileButton />
      </div>
      <div style={{ width: "100%", maxWidth: 900, textAlign: "center" }}>
        <h1
          style={{
            fontSize: "clamp(2.6rem, 5vw, 4.4rem)",
            fontWeight: 800,
            letterSpacing: "-0.05em",
            color: "#f8fafc",
            marginBottom: 14,
          }}
        >
          Curate your next movie night
        </h1>

        <p
          style={{
            fontSize: 20,
            color: "rgba(255,255,255,0.72)",
            marginBottom: 30,
          }}
        >
          Search a Few Favorites, Get <strong>Instant</strong> Recommendations
        </p>

        <div ref={boxRef} style={{ position: "relative", marginBottom: 24 }}>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onFocus={() => {
              if (results.length > 0) setShowDropdown(true);
            }}
            onKeyDown={(event) => {
              if (event.key === "Enter" && isSentenceSearch)
                discoverFromSentence();
            }}
            placeholder="Search movies or describe a vibe..."
            style={{
              width: "100%",
              padding: "22px 24px",
              borderRadius: 999,
              border: "1px solid rgba(255,255,255,0.14)",
              fontSize: 20,
              outline: "none",
              color: "#f8fafc",
              background: "rgba(255,255,255,0.06)",
              boxShadow: "0 18px 40px rgba(0,0,0,0.28)",
              backdropFilter: "blur(8px)",
              WebkitBackdropFilter: "blur(8px)",
            }}
          />

          {showDropdown && filteredResults.length > 0 && (
            <div
              style={{
                position: "absolute",
                top: "calc(100% + 12px)",
                left: 0,
                right: 0,
                background: "#0f172a",
                borderRadius: 24,
                boxShadow: "0 24px 60px rgba(0,0,0,0.45)",
                overflowY: "auto",
                maxHeight: 420,
                textAlign: "left",
                zIndex: 20,
                border: "1px solid rgba(255,255,255,0.08)",
              }}
            >
              {filteredResults.map((movie) => (
                <button
                  key={movie.movie_id}
                  onClick={() => addMovie(movie)}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 14,
                    width: "100%",
                    padding: 14,
                    border: "none",
                    background: "transparent",
                    cursor: "pointer",
                    borderBottom: "1px solid rgba(255,255,255,0.06)",
                    color: "#f8fafc",
                  }}
                >
                  <img
                    src={movie.poster || FALLBACK_POSTER}
                    alt={movie.title}
                    onError={(e) => {
                      e.currentTarget.src = FALLBACK_POSTER;
                    }}
                    style={{
                      width: 54,
                      height: 80,
                      objectFit: "cover",
                      borderRadius: 8,
                      flexShrink: 0,
                    }}
                  />
                  <span style={{ fontWeight: 600, fontSize: 17 }}>
                    {movie.title}
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>

        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: 12,
            justifyContent: "center",
            marginBottom: 28,
            minHeight: 48,
          }}
        >
          {selected.map((movie) => (
            <div
              key={movie.movie_id}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                background: "rgba(255,255,255,0.08)",
                color: "#e2e8f0",
                padding: "12px 16px",
                borderRadius: 999,
                border: "1px solid rgba(255,255,255,0.08)",
              }}
            >
              <span>{movie.title}</span>
              <button
                onClick={() => removeMovie(movie.movie_id)}
                style={{
                  border: "none",
                  background: "transparent",
                  cursor: "pointer",
                  color: "#93c5fd",
                  fontWeight: 700,
                  fontSize: 18,
                }}
              >
                ×
              </button>
            </div>
          ))}
        </div>

        <button
          onClick={seeResults}
          style={{
            padding: "18px 30px",
            borderRadius: 999,
            border: "none",
            background: "linear-gradient(135deg, #3b82f6, #2563eb)",
            color: "white",
            fontSize: 18,
            fontWeight: 700,
            cursor: "pointer",
            boxShadow: "0 18px 35px rgba(37, 99, 235, 0.28)",
          }}
        >
          See Results
        </button>
      </div>
    </main>
  );
}
