from fastapi import FastAPI
from src.serving.recommend import get_recommendations, search_movies, get_combined_recommendations
from pydantic import BaseModel


app = FastAPI()

@app.get("/recommend/{movie_id}")
def recommend(movie_id: int, top_k: int = 5):
    results = get_recommendations(movie_id, top_k)
    return {
        "movie_id": movie_id,
        "top_k": top_k,
        "recommendations": [
            {"title": title, "score": score}
            for title, score in results
        ]
    }


@app.get("/movies/search")
def search(query: str):
    results = search_movies(query)
    return [{"movie_id": mid, "title": name} for mid, name in results]




class RecommendRequest(BaseModel):
    movie_id: int
    top_k: int = 5

@app.post("/recommend/by-movie")
def recommend_by_movie(req: RecommendRequest):
    results = get_recommendations(req.movie_id, req.top_k)
    return {
        "movie_id": req.movie_id,
        "top_k": req.top_k,
        "recommendations": [
            {"title": title, "score": score}
            for title, score in results
        ]
    }

class CombinedRecommendRequest(BaseModel):
    movie_ids: list[int]
    top_k: int = 10

@app.post("/recommend/combined")
def recommend_combined(req: CombinedRecommendRequest):
    results = get_combined_recommendations(req.movie_ids, req.top_k)
    return {
        "movie_ids": req.movie_ids,
        "top_k": req.top_k,
        "recommendations": [
            {"movie_id": movie_id, "title": title, "score": score}
            for movie_id, title, score in results
        ]
    }