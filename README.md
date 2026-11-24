# Furnisher Advisor

Take a photo of a room and get furnishing suggestions grounded in what is actually in the
photo: the real wall colour, the furniture already present, and what fits your budget.

## What it does today

- **Wall colour from real wall pixels.** A segmentation model finds the wall and the colour
  is estimated in CIELAB (perceptual colour space), then scored with CIEDE2000.
  On 300 independent NYUv2 photos the median colour error is 1.9, against 17.2 for the
  K-Means baseline (see `docs/evaluation.md`).
- **Accounts, favourites and contact** via a FastAPI backend with JWT auth, tests on SQLite
  and Postgres, and a Render blueprint.
- **A reproducible evaluation harness** (`eval/`) that scores every method on labelled public
  photos, with unit tests.

## In progress

Furniture detection beyond the 80 COCO classes, live product catalogue and prices, a
budget optimiser, shared room boards for group decisions, a tool-using chat assistant,
and a mobile-first frontend. These are not built yet.

## Layout

| Path | Contents |
|---|---|
| `backend/` | FastAPI service, models, auth, tests |
| `eval/` | Evaluation code and Kaggle runners |
| `docs/evaluation.md` | Methods, results and caveats |

## Run the backend

```bash
cd backend
pip install -r requirements-dev.txt
cp .env.example .env   # set SECRET_KEY (16+ chars) and DATABASE_URL
pytest
uvicorn app.main:app --reload
```

## Run the evaluation

```bash
python -m pytest eval/tests
# on a Kaggle notebook with internet enabled:
python -m eval.kaggle.phase2_wall --per-room 60
python -m eval.kaggle.phase2_nyu --limit 300
```

## Origin

Furnisher Advisor builds on the Personalized Furnishing Advisor, a 2025 BTech major project.
