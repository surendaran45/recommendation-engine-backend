import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load MovieLens data
print("Loading MovieLens data...")
movies = pd.read_csv('data/movielens/movies.csv')
ratings = pd.read_csv('data/movielens/ratings.csv')

print(f"Movies: {len(movies)}, Ratings: {len(ratings)}")
print("\nMovies sample:")
print(movies.head())
print("\nRatings sample:")
print(ratings.head())

# Load Anime data
print("\n\nLoading MyAnimeList data...")
anime = pd.read_csv('data/anime/anime.csv')
anime_ratings = pd.read_csv('data/anime/rating.csv')

print(f"Anime: {len(anime)}, Ratings: {len(anime_ratings)}")
print("\nAnime sample:")
print(anime.head())
print("\nAnime ratings sample:")
print(anime_ratings.head())

# Basic cleaning
movies.dropna(subset=['title'], inplace=True)
ratings.dropna(subset=['userId', 'movieId', 'rating'], inplace=True)
anime.dropna(subset=['name'], inplace=True)
anime_ratings.dropna(subset=['user_id', 'anime_id', 'rating'], inplace=True)

# Save cleaned versions
movies.to_csv('data/movies_clean.csv', index=False)
ratings.to_csv('data/ratings_clean.csv', index=False)
anime.to_csv('data/anime_clean.csv', index=False)
anime_ratings.to_csv('data/anime_ratings_clean.csv', index=False)

print("\n✅ Data cleaning complete!")
print(f"Movies cleaned: {len(movies)}")
print(f"Ratings cleaned: {len(ratings)}")
print(f"Anime cleaned: {len(anime)}")
print(f"Anime ratings cleaned: {len(anime_ratings)}")