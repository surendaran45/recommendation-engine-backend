from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
import pandas as pd
import numpy as np
import pickle
import os
import requests

app = FastAPI(title="Recommendation Engine API")
logger = logging.getLogger(__name__)

cors_origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Load all models and data
print("Loading models...")
with open('models/movie_user_matrix.pkl', 'rb') as f:
    movie_user_matrix = pickle.load(f)
with open('models/movie_similarity.pkl', 'rb') as f:
    movie_similarity = pickle.load(f)
with open('models/movie_id_to_idx.pkl', 'rb') as f:
    movie_id_to_idx = pickle.load(f)
with open('models/movie_user_idx.pkl', 'rb') as f:
    movie_user_idx = pickle.load(f)

with open('models/anime_similarity.pkl', 'rb') as f:
    anime_similarity = pickle.load(f)
with open('models/anime_id_to_idx.pkl', 'rb') as f:
    anime_id_to_idx = pickle.load(f)

movies_df = pd.read_csv('models/movies_metadata.csv')
anime_df = pd.read_csv('models/anime_metadata.csv')
anime_ratings_df = pd.read_csv('models/anime_ratings_active.csv')

# Build reverse mappings
idx_to_movie_id = {v: k for k, v in movie_id_to_idx.items()}
idx_to_anime_id = {v: k for k, v in anime_id_to_idx.items()}

print("✓ All models loaded")

# ========================
# REQUEST MODELS
# ========================

class RatingRequest(BaseModel):
    user_id: int = Field(ge=1)
    item_id: int = Field(ge=1)
    rating: float = Field(ge=0.5, le=5.0)

class RecommendationRequest(BaseModel):
    user_id: int = Field(ge=1)
    top_n: int = Field(default=10, ge=1, le=50)

class GenreFilterRequest(BaseModel):
    genre: str = Field(min_length=1, max_length=80)
    top_n: int = Field(default=10, ge=1, le=50)

# ========================
# MOVIE ENDPOINTS
# ========================

@app.get("/")
def root():
    return {"message": "Recommendation Engine API running!"}

@app.get("/movies")
def get_movies(limit: int = 20):
    """Get list of movies"""
    movies_list = movies_df.head(limit)[['movieId', 'title', 'genres']].to_dict(orient='records')
    # Convert numpy types to Python native types
    for m in movies_list:
        m['movieId'] = int(m['movieId'])
    return {
        "count": len(movies_df),
        "movies": movies_list
    }

@app.get("/movie/{movie_id}")
def get_movie(movie_id: int):
    """Get movie details"""
    movie = movies_df[movies_df['movieId'] == movie_id]
    if movie.empty:
        raise HTTPException(status_code=404, detail="Movie not found")
    result = movie.iloc[0].to_dict()
    result['movieId'] = int(result['movieId'])
    return result

@app.post("/recommend-movies")
def recommend_movies(req: RecommendationRequest):
    """Get movie recommendations for a user"""
    try:
        if req.user_id not in movie_user_idx:
            return {"error": "User not found", "recommendations": []}
        
        user_idx = movie_user_idx[req.user_id]
        user_ratings = movie_user_matrix.iloc[user_idx].values
        
        # Get movies user has rated
        rated_indices = np.where(user_ratings > 0)[0]
        
        if len(rated_indices) == 0:
            return {"error": "User has no ratings", "recommendations": []}
        
        # Find similar movies
        similarities = np.zeros(len(movie_similarity))
        for rated_idx in rated_indices:
            similarities += movie_similarity[rated_idx] * user_ratings[rated_idx]
        
        # Exclude already-rated movies
        similarities[rated_indices] = -1
        
        # Get top N
        top_indices = np.argsort(similarities)[-req.top_n:][::-1]
        
        recommendations = []
        for idx in top_indices:
            if similarities[idx] > 0:
                movie_id = int(idx_to_movie_id[idx])
                movie = movies_df[movies_df['movieId'] == movie_id].iloc[0]
                recommendations.append({
                    'movieId': movie_id,
                    'title': str(movie['title']),
                    'genres': str(movie['genres']),
                    'score': float(similarities[idx])
                })
        
        return {"user_id": req.user_id, "recommendations": recommendations}
    except Exception as e:
        return {"error": str(e), "recommendations": []}

@app.get("/movies/genre/{genre}")
def movies_by_genre(genre: str, limit: int = 20):
    """Get movies by genre"""
    filtered = movies_df[movies_df['genres'].str.contains(genre, case=False, na=False, regex=False)]
    if filtered.empty:
        return {"genre": genre, "count": 0, "movies": []}
    movies_list = filtered[['movieId', 'title', 'genres']].head(limit).to_dict(orient='records')
    for m in movies_list:
        m['movieId'] = int(m['movieId'])
    return {
        "genre": genre,
        "count": len(filtered),
        "movies": movies_list
    }

# ========================
# ANIME ENDPOINTS
# ========================

@app.get("/anime")
def get_anime(limit: int = 20):
    """Get list of anime"""
    anime_list = anime_df.head(limit)[['anime_id', 'name', 'genre']].to_dict(orient='records')
    for a in anime_list:
        a['anime_id'] = int(a['anime_id'])
    return {
        "count": len(anime_df),
        "anime": anime_list
    }

@app.get("/anime/{anime_id}")
def get_anime_details(anime_id: int):
    """Get anime details"""
    anime = anime_df[anime_df['anime_id'] == anime_id]
    if anime.empty:
        raise HTTPException(status_code=404, detail="Anime not found")
    result = anime.iloc[0].to_dict()
    result['anime_id'] = int(result['anime_id'])
    return result

@app.post("/recommend-anime")
def recommend_anime(req: RecommendationRequest):
    """Get anime recommendations for a user"""
    try:
        # Get user's rated anime
        user_ratings = anime_ratings_df[anime_ratings_df['user_id'] == req.user_id]
        
        if user_ratings.empty:
            return {"error": "User has no anime ratings", "recommendations": []}
        
        # Find anime they rated highly
        good_anime = user_ratings[user_ratings['rating'] >= 3]['anime_id'].values
        
        if len(good_anime) == 0:
            return {"error": "User has no good ratings", "recommendations": []}
        
        # Find similar anime
        similarities = np.zeros(len(anime_similarity))
        for aid in good_anime:
            if aid in anime_id_to_idx:
                aid_idx = anime_id_to_idx[aid]
                similarities += anime_similarity[aid_idx]
        
        # Exclude already-rated anime
        rated_indices = [anime_id_to_idx[aid] for aid in user_ratings['anime_id'].values if aid in anime_id_to_idx]
        similarities[rated_indices] = -1
        
        # Get top N
        top_indices = np.argsort(similarities)[-req.top_n:][::-1]
        
        recommendations = []
        for idx in top_indices:
            if similarities[idx] > 0:
                anime_id = int(idx_to_anime_id[idx])
                anime = anime_df[anime_df['anime_id'] == anime_id].iloc[0]
                recommendations.append({
                    'anime_id': anime_id,
                    'name': str(anime['name']),
                    'genre': str(anime['genre']),
                    'score': float(similarities[idx])
                })
        
        return {"user_id": req.user_id, "recommendations": recommendations}
    except Exception as e:
        return {"error": str(e), "recommendations": []}

@app.get("/anime/genre/{genre}")
def anime_by_genre(genre: str, limit: int = 20):
    """Get anime by genre"""
    filtered = anime_df[anime_df['genre'].str.contains(genre, case=False, na=False, regex=False)]
    if filtered.empty:
        return {"genre": genre, "count": 0, "anime": []}
    anime_list = filtered[['anime_id', 'name', 'genre']].head(limit).to_dict(orient='records')
    for a in anime_list:
        a['anime_id'] = int(a['anime_id'])
    return {
        "genre": genre,
        "count": len(filtered),
        "anime": anime_list
    }

@app.post("/rate")
def rate_item(req: RatingRequest):
    """Save a user rating (would go to database in production)"""
    return {
        "status": "saved",
        "user_id": req.user_id,
        "item_id": req.item_id,
        "rating": req.rating
    }

@app.get("/anime/search")
def search_anime(q: str):
    """Search anime by name"""
    filtered = anime_df[anime_df['name'].str.contains(q, case=False, na=False)]
    results = filtered[['anime_id', 'name', 'genre']].head(20).to_dict(orient='records')
    for a in results:
        a['anime_id'] = int(a['anime_id'])
    return {"query": q, "count": len(results), "results": results}

@app.get("/trending/movies")
def trending_movies():
    """Get trending movies (most rated)"""
    top_movies = movies_df.head(15)[['movieId', 'title', 'genres']].to_dict(orient='records')
    for m in top_movies:
        m['movieId'] = int(m['movieId'])
    return {"trending": top_movies}

@app.get("/trending/anime")
def trending_anime():
    """Get trending anime (highest rated)"""
    top_anime = anime_df.nlargest(15, 'rating')[['anime_id', 'name', 'genre', 'rating']].to_dict(orient='records')
    for a in top_anime:
        a['anime_id'] = int(a['anime_id'])
    return {"trending": top_anime}

@app.get("/user/{user_id}/stats")
def user_stats(user_id: str):
    """Get user statistics"""
    try:
        user_ratings = anime_ratings_df[anime_ratings_df['user_id'] == user_id]
        total_ratings = len(user_ratings)
        avg_rating = float(user_ratings['rating'].mean()) if total_ratings > 0 else 0
        
        return {
            "user_id": user_id,
            "total_ratings": total_ratings,
            "average_rating": round(avg_rating, 2),
            "favorite_genre": "Action" if total_ratings > 0 else "N/A"
        }
    except:
        return {"user_id": user_id, "total_ratings": 0, "average_rating": 0, "favorite_genre": "N/A"}

@app.get("/user/{user_id}/ratings")
def user_ratings(user_id: str):
    """Get user's rating history"""
    try:
        user_ratings = anime_ratings_df[anime_ratings_df['user_id'] == user_id].sort_values('rating', ascending=False)
        ratings_list = user_ratings[['anime_id', 'rating']].head(10).to_dict(orient='records')
        for r in ratings_list:
            r['anime_id'] = int(r['anime_id'])
        return {"ratings": ratings_list}
    except:
        return {"ratings": []}

# Keep provider credentials in server-side environment settings.
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE = "https://api.themoviedb.org/3"

@app.get("/movie/{movie_id}/details")
def get_movie_details(movie_id: int):
    """Get movie details with poster from TMDB"""
    try:
        matching_movies = movies_df[movies_df["movieId"] == movie_id]
        if matching_movies.empty:
            raise HTTPException(status_code=404, detail="Movie not found")
        movie = matching_movies.iloc[0]

        # Poster lookup is optional; the endpoint works without a TMDB key.
        poster_path = None
        if TMDB_API_KEY:
            response = requests.get(
                f"{TMDB_BASE}/search/movie",
                params={"api_key": TMDB_API_KEY, "query": str(movie["title"])},
                timeout=5,
            )
            if response.status_code == 200:
                results = response.json().get("results", [])
                if results:
                    poster_path = results[0].get("poster_path")
        
        return {
            'movieId': int(movie_id),
            'title': str(movie['title']),
            'genres': str(movie['genres']),
            'poster': f"https://image.tmdb.org/t/p/w342{poster_path}" if poster_path else None,
            'year': '2020'  # Add year if available
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("TMDB movie details lookup failed")
        raise HTTPException(status_code=502, detail="Movie details are temporarily unavailable.") from None

@app.get("/movies/search")
def search_movies(q: str, sort: str = "title", order: str = "asc"):
    """Search and sort movies."""
    if not q.strip() or len(q) > 100:
        raise HTTPException(status_code=422, detail="q must contain 1 to 100 characters")
    if sort != "title":
        raise HTTPException(status_code=422, detail="sort must be 'title'")
    if order not in {"asc", "desc"}:
        raise HTTPException(status_code=422, detail="order must be 'asc' or 'desc'")
    filtered = movies_df[movies_df["title"].str.contains(q, case=False, na=False, regex=False)]
    filtered = filtered.sort_values("title", ascending=(order == "asc"))
    results = filtered[["movieId", "title", "genres"]].head(20).to_dict(orient="records")
    for movie in results:
        movie["movieId"] = int(movie["movieId"])
    return {"query": q, "count": len(results), "results": results}

@app.get("/movies/filter")
def filter_movies(genre: str = None, limit: int = 20):
    """Filter movies by genre"""
    filtered = movies_df
    if genre:
        filtered = filtered[filtered['genres'].str.contains(genre, case=False, na=False)]
    
    results = filtered[['movieId', 'title', 'genres']].head(limit).to_dict(orient='records')
    for m in results:
        m['movieId'] = int(m['movieId'])
    return {"genre": genre, "count": len(results), "results": results}

@app.get("/anime/filter")
def filter_anime(genre: str = None, min_rating: float = 0, limit: int = 20):
    """Filter anime by genre and rating"""
    filtered = anime_df
    if genre:
        filtered = filtered[filtered['genre'].str.contains(genre, case=False, na=False)]
    filtered = filtered[filtered['rating'] >= min_rating]
    
    results = filtered[['anime_id', 'name', 'genre', 'rating']].head(limit).to_dict(orient='records')
    for a in results:
        a['anime_id'] = int(a['anime_id'])
    return {"genre": genre, "count": len(results), "results": results}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "movies": len(movies_df),
        "anime": len(anime_df),
        "movie_users": len(movie_user_idx)
    }