# Furnisher Advisor

Upload a photo of a room, understand its colours and furniture, and get recommendations that fit your style and budget.

**Status:** v2 rebuild in progress. See the roadmap below.

## What's here

| Path | What |
|---|---|
| `backend/` | FastAPI service (auth, contact, saved favourites), tests, Dockerfile |
| `render.yaml` | Render blueprint for the API |
| `.github/workflows/ci.yml` | Runs the test suite on SQLite and Postgres |

The v2 web frontend and the vision/recommendation services are added in later phases.

## Run the backend locally

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env        # then set SECRET_KEY (see the comment in the file)
uvicorn app.main:app --reload
```

Interactive API docs: http://localhost:8000/docs

## Tests

```bash
cd backend
pytest -q                                   # SQLite, in memory
TEST_DATABASE_URL=postgresql://user:pass@localhost:5432/db pytest -q   # real Postgres
```

## Deploy (Render)

1. Create a Postgres database on Render and copy its **Internal Database URL**.
2. New → Blueprint → select this repo (`render.yaml`).
3. Set `DATABASE_URL`, `CORS_ORIGINS` (your frontend's URL) and `FRONTEND_URL`. `SECRET_KEY` is generated for you.
4. Check `https://<service>.onrender.com/health`.

## Roadmap

- [x] **Phase 0** — clean FastAPI backend, tests, CI, deploy config
- [ ] **Phase 1** — evaluation harness and audit of the v1 pipeline
- [ ] **Phase 2** — perception: open-vocabulary detection, wall/floor segmentation, colour in CIELAB
- [ ] **Phase 3** — live product catalog, recommender and budget optimizer
- [ ] **Phase 4** — group decisions, chat assistant, new frontend
