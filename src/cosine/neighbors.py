from __future__ import annotations

import numpy as np
import pandas as pd


def build_topk_neighbors(
    similarity_matrix: pd.DataFrame | np.ndarray,
    top_k: int = 50,
) -> pd.DataFrame:
    sim = np.asarray(similarity_matrix)

    if sim.ndim != 2:
        raise ValueError("similarity_matrix must be 2D")

    n_rows, n_cols = sim.shape
    if n_rows != n_cols:
        raise ValueError("similarity_matrix must be square")

    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    top_k = min(top_k, n_rows - 1)
    rows = []

    for i in range(n_rows):
        sims = sim[i]
        candidate_idx = np.argsort(sims)[::-1]

        rank = 0
        for j in candidate_idx:
            if i == j:
                continue

            rank += 1
            rows.append(
                {
                    "source_movie_id": int(i),
                    "neighbor_movie_id": int(j),
                    "rank": int(rank),
                    "cosine_score": float(sims[j]),
                }
            )

            if rank >= top_k:
                break

    return pd.DataFrame(rows)