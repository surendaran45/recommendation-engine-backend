# Recommendation Engine Backend

A FastAPI service for movie and anime discovery, similarity-based recommendations, genre filtering, rating history, and optional poster lookup through TMDB.

## Features

- Movie/anime listing and detail endpoints
- Personalized movie and anime recommendations
- Genre and rating filters, search, trending lists, and user rating history
- Optional TMDB poster enrichment
- Request validation for rating and recommendation limits

## Stack

Python, FastAPI, pandas, NumPy, scikit-learn artifacts, requests, and Uvicorn.

## Repository layout

- `src/backend/main.py` — API routes and request validation
- `models/` — serialized model artifacts and metadata CSV files expected by the backend
- `requirements.txt` — runtime dependencies

## Run locally

The backend currently loads model artifacts using relative paths under `models/`. Start it from the directory that contains that folder.

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.backend.main:app --reload
```

Open `/docs` for the interactive API reference. If model artifacts are stored in a different directory, adjust the working directory or configure absolute paths before starting the service.

## Environment variables

- `TMDB_API_KEY` — optional server-side key used to fetch movie posters.
- `CORS_ORIGINS` — comma-separated list of trusted frontend origins. The wildcard default is for development; use explicit origins in production.

Never commit provider credentials. A TMDB API key was previously committed in source history; treat it as exposed and rotate/revoke it in the provider dashboard. Moving the key to an environment variable does not invalidate the old credential.

## Main endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/` | Basic service response |
| GET | `/health` | Dataset/model counts |
| GET | `/movies`, `/anime` | List items |
| GET | `/movie/{movie_id}`, `/anime/{anime_id}` | Item details |
| POST | `/recommend-movies`, `/recommend-anime` | Recommendations |
| GET | `/movies/search?q=...` | Search movies |
| GET | `/anime/search?q=...` | Search anime |
| GET | `/movies/genre/{genre}`, `/anime/genre/{genre}` | Genre lookup |
| GET | `/trending/movies`, `/trending/anime` | Trending lists |
| GET | `/movie/{movie_id}/details` | Movie data plus optional TMDB poster |
| POST | `/rate` | Demo response; currently does not persist ratings |

## Tests and CI

The source-contract tests check for duplicate HTTP method/path registrations and verify that the TMDB key is read from environment configuration. GitHub Actions also compiles the backend source. These checks are not a substitute for integration tests against real model artifacts and representative data.

```bash
pip install -r requirements-dev.txt
pytest -q
python -m compileall -q src/backend
```

## Known limitations

- `/rate` is a demo endpoint and does not persist user ratings.
- Validate recommendation quality and data assumptions with representative inputs.
- Configure CORS explicitly for production.
- Only load trusted pickle artifacts; pickle files can execute code when deserialized.
