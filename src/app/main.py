from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel

from src.db.profile import (
    ensure_profile_tables,
    get_profile_home,
    update_profile,
    update_onboarding_preferences,
)

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

from src.db.user_lists import (
    ensure_user_list_tables,
    list_user_lists,
    create_user_list,
    update_user_list,
    delete_user_list,
)

from src.db.social import (
    ensure_social_tables,
    upsert_user_profile,
    search_user_profiles,
    create_friend_request,
    list_friends_and_requests,
    accept_friend_request,
    delete_friendship,
)

from src.db.shared_lists import (
    ensure_shared_list_tables,
    list_shared_lists,
    get_shared_list,
    create_shared_list,
    update_shared_list_name,
    add_shared_list_member,
    remove_shared_list_member,
    add_shared_list_movie,
    remove_shared_list_movie,
    delete_shared_list,
    convert_personal_list_to_shared,
)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_user_list_tables()
    ensure_social_tables()
    ensure_shared_list_tables()
    ensure_profile_tables()
    yield


class RecommendRequest(BaseModel):
    movie_id: int
    top_k: int = 5

class CombinedRecommendRequest(BaseModel):
    movie_ids: list[int]
    top_k: int = 10

class OrganizedRow(BaseModel):
    type: str
    title: str
    pinned: bool = False
    row_score: float | None = None
    items: list[dict]


class CombinedRecommendationResponse(BaseModel):
    movie_ids: list[int]
    top_k: int
    headline: str | None = None
    taste_summary: dict | None = None
    organized_rows: list[OrganizedRow] = []
    taste_rows: list[dict] = []
    recommendations: list[dict]


# --------------------------
#   TASTE PROFILE
# --------------------------

class TasteRowItem(BaseModel):
    movie_id: int
    title: str
    score: float | None = None

class TasteRow(BaseModel):
    title: str
    items: list[TasteRowItem]


# --------------------------
#   CREATE LISTS
# --------------------------

class SavedMovieRequest(BaseModel):
    movie_id: int
    title: str

class CreateListRequest(BaseModel):
    name: str
    movies: list[SavedMovieRequest]

class UpdateListRequest(BaseModel):
    name: str | None = None
    movies: list[SavedMovieRequest] | None = None



# --------------------------
#   FRIEND INTERACTION
# --------------------------

class UserProfileRequest(BaseModel):
    email: str | None = None
    name: str | None = None

class FriendRequestCreate(BaseModel):
    receiver_user_id: str


# --------------------------
#   SHARED LISTS
# --------------------------

class CreateSharedListRequest(BaseModel):
    name: str
    member_user_ids: list[str] = []
    movies: list[SavedMovieRequest] = []

class UpdateSharedListRequest(BaseModel):
    name: str

class AddSharedMemberRequest(BaseModel):
    user_id: str
    role: str = "editor"

class AddSharedMovieRequest(BaseModel):
    movie_id: int
    title: str

class SharePersonalListRequest(BaseModel):
    member_user_ids: list[str]


# --------------------------
#   PROFILE
# --------------------------

class UpdateProfileRequest(BaseModel):
    name: str | None = None
    bio: str | None = None
    avatar_url: str | None = None


class OnboardingPreferencesRequest(BaseModel):
    favorite_genres: list[str] = []
    disliked_genres: list[str] = []
    preferred_moods: list[str] = []
    preferred_pacing: list[str] = []
    preferred_decades: list[str] = []
    favorite_movies: list[str] = []
    disliked_movies: list[str] = []



def require_user_id(x_user_id: str | None) -> str:
    if not x_user_id:
        raise HTTPException(status_code=401, detail="Missing signed-in user")
    return x_user_id


# --------------------------
#   MOVIE RECOMMENDATION
# --------------------------

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



# --------------------------
#   USER PROFILE
# --------------------------


@app.get("/user/lists")
def get_lists(x_user_id: str | None = Header(default=None)):
    user_id = require_user_id(x_user_id)
    return list_user_lists(user_id)


@app.post("/user/lists")
def create_list(req: CreateListRequest, x_user_id: str | None = Header(default=None)):
    user_id = require_user_id(x_user_id)
    movies = [movie.model_dump() for movie in req.movies]
    return create_user_list(user_id, req.name, movies)


@app.put("/user/lists/{list_id}")
def update_list(
    list_id: str,
    req: UpdateListRequest,
    x_user_id: str | None = Header(default=None),
):
    user_id = require_user_id(x_user_id)
    movies = [movie.model_dump() for movie in req.movies] if req.movies is not None else None

    try:
        return update_user_list(user_id, list_id, req.name, movies)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.delete("/user/lists/{list_id}")
def delete_list(list_id: str, x_user_id: str | None = Header(default=None)):
    user_id = require_user_id(x_user_id)
    delete_user_list(user_id, list_id)
    return {"ok": True}



@app.post("/user/profile")
def sync_user_profile(
    req: UserProfileRequest,
    x_user_id: str | None = Header(default=None),
):
    user_id = require_user_id(x_user_id)

    return upsert_user_profile(
        user_id=user_id,
        email=req.email,
        name=req.name,
    )


@app.get("/users/search")
def search_users(
    q: str,
    limit: int = 10,
    x_user_id: str | None = Header(default=None),
):
    user_id = require_user_id(x_user_id)

    return search_user_profiles(
        current_user_id=user_id,
        query=q,
        limit=limit,
    )

@app.post("/user/lists/{list_id}/share")
def share_personal_list(
    list_id: str,
    req: SharePersonalListRequest,
    x_user_id: str | None = Header(default=None),
):
    user_id = require_user_id(x_user_id)

    try:
        return convert_personal_list_to_shared(
            owner_user_id=user_id,
            personal_list_id=list_id,
            member_user_ids=req.member_user_ids,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e



# --------------------------
#   FRIEND REQUESTING
# --------------------------

@app.post("/friends/request")
def send_friend_request(
    req: FriendRequestCreate,
    x_user_id: str | None = Header(default=None),
):
    user_id = require_user_id(x_user_id)

    try:
        return create_friend_request(
            requester_user_id=user_id,
            receiver_user_id=req.receiver_user_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/friends")
def get_friends(x_user_id: str | None = Header(default=None)):
    user_id = require_user_id(x_user_id)
    return list_friends_and_requests(user_id)


@app.post("/friends/accept/{friendship_id}")
def accept_friend(
    friendship_id: str,
    x_user_id: str | None = Header(default=None),
):
    user_id = require_user_id(x_user_id)

    try:
        return accept_friend_request(
            user_id=user_id,
            friendship_id=friendship_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.delete("/friends/{friendship_id}")
def remove_friend(
    friendship_id: str,
    x_user_id: str | None = Header(default=None),
):
    user_id = require_user_id(x_user_id)

    try:
        delete_friendship(
            user_id=user_id,
            friendship_id=friendship_id,
        )
        return {"ok": True}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


# --------------------------
#   SHARED LISTS
# --------------------------

@app.get("/shared-lists")
def get_all_shared_lists(x_user_id: str | None = Header(default=None)):
    user_id = require_user_id(x_user_id)
    return list_shared_lists(user_id)


@app.get("/shared-lists/{shared_list_id}")
def get_one_shared_list(
    shared_list_id: str,
    x_user_id: str | None = Header(default=None),
):
    user_id = require_user_id(x_user_id)

    try:
        return get_shared_list(user_id, shared_list_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e



@app.post("/shared-lists")
def create_new_shared_list(
    req: CreateSharedListRequest,
    x_user_id: str | None = Header(default=None),
):
    user_id = require_user_id(x_user_id)

    movies = [movie.model_dump() for movie in req.movies]

    try:
        return create_shared_list(
            owner_user_id=user_id,
            name=req.name,
            member_user_ids=req.member_user_ids,
            movies=movies,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.put("/shared-lists/{shared_list_id}")
def rename_shared_list(
    shared_list_id: str,
    req: UpdateSharedListRequest,
    x_user_id: str | None = Header(default=None),
):
    user_id = require_user_id(x_user_id)

    try:
        return update_shared_list_name(
            user_id=user_id,
            shared_list_id=shared_list_id,
            name=req.name,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/shared-lists/{shared_list_id}/members")
def add_member_to_shared_list(
    shared_list_id: str,
    req: AddSharedMemberRequest,
    x_user_id: str | None = Header(default=None),
):
    user_id = require_user_id(x_user_id)

    try:
        return add_shared_list_member(
            owner_user_id=user_id,
            shared_list_id=shared_list_id,
            member_user_id=req.user_id,
            role=req.role,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.delete("/shared-lists/{shared_list_id}/members/{member_user_id}")
def remove_member_from_shared_list(
    shared_list_id: str,
    member_user_id: str,
    x_user_id: str | None = Header(default=None),
):
    user_id = require_user_id(x_user_id)

    try:
        return remove_shared_list_member(
            current_user_id=user_id,
            shared_list_id=shared_list_id,
            member_user_id=member_user_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/shared-lists/{shared_list_id}/movies")
def add_movie_to_shared_list(
    shared_list_id: str,
    req: AddSharedMovieRequest,
    x_user_id: str | None = Header(default=None),
):
    user_id = require_user_id(x_user_id)

    try:
        return add_shared_list_movie(
            user_id=user_id,
            shared_list_id=shared_list_id,
            movie_id=req.movie_id,
            title=req.title,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.delete("/shared-lists/{shared_list_id}/movies/{movie_id}")
def delete_movie_from_shared_list(
    shared_list_id: str,
    movie_id: int,
    x_user_id: str | None = Header(default=None),
):
    user_id = require_user_id(x_user_id)

    try:
        return remove_shared_list_movie(
            user_id=user_id,
            shared_list_id=shared_list_id,
            movie_id=movie_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.delete("/shared-lists/{shared_list_id}")
def delete_one_shared_list(
    shared_list_id: str,
    x_user_id: str | None = Header(default=None),
):
    user_id = require_user_id(x_user_id)

    try:
        delete_shared_list(user_id, shared_list_id)
        return {"ok": True}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/shared-lists/{shared_list_id}/recommend")
def recommend_from_shared_list(
    shared_list_id: str,
    top_k: int = 10,
    x_user_id: str | None = Header(default=None),
):
    user_id = require_user_id(x_user_id)

    try:
        shared_list = get_shared_list(user_id, shared_list_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    movie_ids = [movie["movie_id"] for movie in shared_list["movies"]]

    if not movie_ids:
        raise HTTPException(status_code=400, detail="Shared list has no movies")

    return predict_combined(
        movie_ids=movie_ids,
        top_k=top_k,
        min_support=1,
    )



# --------------------------
#   PROFILE
# --------------------------


@app.get("/profile")
def get_profile(x_user_id: str | None = Header(default=None)):
    user_id = require_user_id(x_user_id)
    return get_profile_home(user_id)


@app.put("/profile")
def update_my_profile(
    req: UpdateProfileRequest,
    x_user_id: str | None = Header(default=None),
):
    user_id = require_user_id(x_user_id)

    return update_profile(
        user_id=user_id,
        name=req.name,
        bio=req.bio,
        avatar_url=req.avatar_url,
    )


@app.put("/profile/onboarding")
def update_my_onboarding(
    req: OnboardingPreferencesRequest,
    x_user_id: str | None = Header(default=None),
):
    user_id = require_user_id(x_user_id)
    return update_onboarding_preferences(user_id, req.model_dump())

