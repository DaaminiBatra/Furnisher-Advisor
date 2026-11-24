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
- **Furniture detection beyond COCO.** A YOLOv8s detector distilled from an open-vocabulary
  model covers 18 furniture types (held-out F1 0.70 against 0.54 for COCO).
- **A product catalogue and recommender.** About 5,000 furniture products from the public
  Amazon Berkeley Objects dataset, ranked for the wall colour by explicit colour-harmony
  rules (CIEDE2000, hue relations, contrast), and a budget planner that picks the best set of
  missing items for a budget (`POST /api/recommend`). Prices are rough planning ranges,
  because the open data has no prices (`docs/catalog.md`).
- **Shared room boards for group decisions.** Invite people with a code, propose products,
  vote love / fine / no, and get one proposal per item type that favours agreement over a
  split vote and flags contested choices and budget overruns (`/api/boards`).
- **A reproducible evaluation harness** (`eval/`) that scores every method on labelled public
  photos, with unit tests.

## In progress

Live shop prices, a tool-using chat assistant, and a mobile-first frontend. These are not built yet.

## Layout

| Path | Contents |
|---|---|
| `backend/` | FastAPI service, models, auth, tests |
| `catalog/` | Builds the product catalogue from ABO listings |
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
