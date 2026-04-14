"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";

type Movie = {
  movie_id: number;
  title: string;
  poster?: string;
};

const TMDB_API_KEY = "5d1421370bac86b6d373cf6be6f0e942";
const TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w185";
const FALLBACK_POSTER = "/no-poster.png";

async function getTmdbPoster(title: string): Promise<string | undefined> {
  try {
    const res = await fetch(
      `https://api.themoviedb.org/3/search/movie?query=${encodeURIComponent(title)}&api_key=${TMDB_API_KEY}`
    );
    const data = await res.json();

    const match =
      data?.results?.find(
        (m: any) => String(m.title).toLowerCase() === title.toLowerCase()
      ) || data?.results?.[0];

    if (!match?.poster_path) return undefined;
    return `${TMDB_IMAGE_BASE}${match.poster_path}`;
  } catch {
    return undefined;
  }
}

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
          `http://127.0.0.1:8000/movies/search?q=${encodeURIComponent(query)}`
        );
        const data = await res.json();

        const withPosters = await Promise.all(
          data.map(async (movie: Movie) => {
            const poster = await getTmdbPoster(movie.title);
            return {
              ...movie,
              poster,
            };
          })
        );

        setResults(withPosters);
        setShowDropdown(true);
      } catch (err) {
        console.error(err);
      }
    };

    const timeoutId = setTimeout(searchMovies, 250);
    return () => clearTimeout(timeoutId);
  }, [query]);

  const filteredResults = useMemo(() => {
    return results.filter(
      (m) => !selected.some((s) => s.movie_id === m.movie_id)
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

  const seeResults = () => {
    if (!selected.length) return;
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
            placeholder="Search movies..."
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
                  <span style={{ fontWeight: 600, fontSize: 17 }}>{movie.title}</span>
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