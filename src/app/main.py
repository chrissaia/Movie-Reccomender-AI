from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.serving.recommend import (
    get_recommendations,
    search_movies,
    get_combined_recommendations,
    recommendation_to_dict,
    search_result_to_dict,
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


@app.get("/recommend/{movie_id}")
def recommend(movie_id: int, top_k: int = 5):
    results = get_recommendations(movie_id, top_k)
    return {
        "movie_id": movie_id,
        "top_k": top_k,
        "recommendations": [recommendation_to_dict(rec) for rec in results],
    }


@app.get("/movies/search")
def search(q: str, limit: int = 10):
    results = search_movies(q, limit)
    return [search_result_to_dict(result) for result in results]


@app.post("/recommend/by-movie")
def recommend_by_movie(req: RecommendRequest):
    results = get_recommendations(req.movie_id, req.top_k)
    return {
        "movie_id": req.movie_id,
        "top_k": req.top_k,
        "recommendations": [recommendation_to_dict(rec) for rec in results],
    }


@app.post("/recommend/combined")
def recommend_combined(req: CombinedRecommendRequest):
    results = get_combined_recommendations(
        movie_ids=req.movie_ids,
        top_k=req.top_k,
        min_support=1,
    )
    return {
        "movie_ids": req.movie_ids,
        "top_k": req.top_k,
        "recommendations": [recommendation_to_dict(rec) for rec in results],
    }