"use client";

import { Suspense, useEffect, useState } from "react";
import { useUser } from "@clerk/nextjs";
import { useSearchParams, useRouter } from "next/navigation";

import AppHeader from "../components/AppHeader";
import AuthProfileButton from "../components/AuthProfileButton";

import AddToListModal from "../components/AddToListModal";
import MovieDetailModal from "../components/MovieDetailModal";
import { API_BASE_URL, FALLBACK_POSTER } from "../lib/config";
import { logHandledError } from "../lib/log";
import { getTmdbDetails } from "../lib/tmdb";

type DiscoveryMovie = {
  movie_id: number;
  title: string;
  year?: string;
  score?: number;
  poster?: string;
  overview?: string;
  tmdb_overview?: string;
  tmdb_genres?: string;
  match_reasons?: string[];
  semantic_score?: number;
};

function DiscoverContent() {
  const router = useRouter();
  const { user, isSignedIn } = useUser();
  const searchParams = useSearchParams();
  const query = searchParams.get("q") ?? "";
  const [movies, setMovies] = useState<DiscoveryMovie[]>([]);
  const [activeMovie, setActiveMovie] = useState<DiscoveryMovie | null>(null);
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState("");
  const [listMessage, setListMessage] = useState("");
  const [listMovie, setListMovie] = useState<DiscoveryMovie | null>(null);

  useEffect(() => {
    const load = async () => {
      if (!query.trim()) {
        setLoading(false);
        return;
      }

      setLoading(true);
      setMessage("");

      try {
        const res = await fetch(
          `${API_BASE_URL}/movies/discover?q=${encodeURIComponent(query)}&limit=30`,
        );
        if (!res.ok) throw new Error("Discovery failed");
        const data: DiscoveryMovie[] = await res.json();
        const enriched = await Promise.all(
          data.map(async (movie) => ({
            ...movie,
            ...(await getTmdbDetails(movie.title)),
          })),
        );
        setMovies(enriched);
      } catch (err) {
        logHandledError("Discovery load failed", err);
        setMessage("Could not search that movie idea.");
      } finally {
        setLoading(false);
      }
    };

    load();
  }, [query]);

  return (
    <main className="min-h-screen px-6 py-7 text-slate-50 bg-[linear-gradient(135deg,#070617_0%,#10172f_38%,#312e81_100%)]">
      <div className="max-w-6xl mx-auto">
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

        <section className="my-7">
          <h1 className="text-4xl md:text-6xl font-black tracking-tight mb-3">
            {query}
          </h1>
          <p className="text-white/70 text-lg">
            A single-list discovery search based on your sentence.
          </p>
        </section>

        {loading && (
          <div className="rounded-3xl border border-white/10 bg-white/5 p-6">
            Searching...
          </div>
        )}
        {message && (
          <div className="rounded-3xl border border-white/10 bg-white/5 p-6">
            {message}
          </div>
        )}
        {listMessage && (
          <div className="rounded-3xl border border-white/10 bg-white/5 p-4 mb-4">
            {listMessage}
          </div>
        )}

        <div className="grid gap-4">
          {movies.map((movie, index) => (
            <article
              key={movie.movie_id}
              className="grid grid-cols-[72px_1fr_auto] items-center gap-4 rounded-2xl border border-white/10 bg-white/5 p-3 text-left hover:bg-white/8"
            >
              <button
                className="text-left"
                onClick={() => setActiveMovie(movie)}
                aria-label={`Open ${movie.title}`}
              >
                <img
                  className="h-28 w-18 rounded-xl object-cover bg-white/5"
                  src={movie.poster || FALLBACK_POSTER}
                  alt={movie.title}
                />
              </button>
              <button
                className="text-left"
                onClick={() => setActiveMovie(movie)}
              >
                <div className="text-xs font-black text-violet-200 mb-1">
                  #{index + 1}
                </div>
                <div className="text-xl font-black">{movie.title}</div>
                <div className="text-white/62 text-sm mt-1">
                  {[movie.year, movie.tmdb_genres].filter(Boolean).join(" · ")}
                </div>
                <p className="text-white/70 line-clamp-2 mt-2">
                  {movie.overview ||
                    movie.tmdb_overview ||
                    "Open for more details."}
                </p>
                {(movie.match_reasons ?? []).length > 0 && (
                  <div className="flex flex-wrap gap-2 mt-3">
                    {(movie.match_reasons ?? []).map((reason) => (
                      <span
                        className="rounded-full border border-violet-300/20 bg-violet-300/10 px-2.5 py-1 text-xs font-black text-violet-100"
                        key={reason}
                      >
                        {reason}
                      </span>
                    ))}
                  </div>
                )}
              </button>
              {isSignedIn && (
                <button
                  className="rounded-full border border-white/12 bg-white/8 px-4 py-2 text-sm font-black text-slate-100 hover:bg-white/14"
                  onClick={() => setListMovie(movie)}
                >
                  Add to List
                </button>
              )}
            </article>
          ))}
        </div>
      </div>

      <MovieDetailModal
        movie={activeMovie}
        onClose={() => setActiveMovie(null)}
      />

      <AddToListModal
        open={Boolean(listMovie)}
        movie={listMovie}
        onClose={() => setListMovie(null)}
        onAdded={setListMessage}
      />
    </main>
  );
}

export default function DiscoverPage() {
  return (
    <Suspense
      fallback={
        <main className="min-h-screen px-6 py-7 text-slate-50 bg-[linear-gradient(135deg,#070617_0%,#10172f_38%,#312e81_100%)]">
          Loading discovery...
        </main>
      }
    >
      <DiscoverContent />
    </Suspense>
  );
}
