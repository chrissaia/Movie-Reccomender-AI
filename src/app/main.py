from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.serving.recommend import (
    get_recommendations,
    search_movies,
    recommendation_to_dict,
    search_result_to_dict,
)

from src.serving.inference import (
    predict_combined,
    predict_single,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    movie_id: int
    top_k: int = 5


class CombinedRecommendRequest(BaseModel):
    movie_ids: list[int]
    top_k: int = 10

class TasteRowItem(BaseModel):
    movie_id: int
    title: str
    score: float | None = None

class TasteRow(BaseModel):
    title: str
    items: list[TasteRowItem]

class CombinedRecommendationResponse(BaseModel):
    movie_ids: list[int]
    top_k: int
    headline: str | None = None
    taste_summary: dict | None = None
    taste_rows: list[TasteRow] = []
    recommendations: list[dict]



@app.get("/movies/search")
def search(q: str, limit: int = 10):
    try:
        results = search_movies(q, limit)
        return [search_result_to_dict(result) for result in results]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}") from e



@app.get("/recommend/{movie_id}")
def recommend(movie_id: int, top_k: int = 5):
    try:
        return predict_single(movie_id=movie_id, top_k=top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {e}") from e



@app.post("/recommend/by-movie")
def recommend_by_movie(req: RecommendRequest):
    try:
        return predict_single(movie_id=req.movie_id, top_k=req.top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {e}") from e



@app.post("/recommend/combined", response_model=CombinedRecommendationResponse)
def recommend_combined(req: CombinedRecommendRequest):
    try:
        return predict_combined(
            movie_ids=req.movie_ids,
            top_k=req.top_k,
            min_support=1,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Combined recommendation failed: {e}") from e