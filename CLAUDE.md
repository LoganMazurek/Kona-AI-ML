# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Kona-AI-ML predicts event revenue for Kona Ice franchises and serves the predictions through a Flask web app. Franchises log in, enter event details (date, time, duration, industry, equipment, ZIP), and receive a revenue prediction with a confidence interval. Predictions are enriched with weather, demographic, historical, and competition data, then fed to a 3-model ensemble (XGBoost + CatBoost + LightGBM). The app also tracks predicted-vs-actual outcomes and supports bulk Excel upload.

## Running and developing

There is no build step — it's a Python/Flask app. Install dependencies and run from the repo root so `source/` resolves on `sys.path` (handled by `conftest.py`, `wsgi.py`, and per-test `conftest.py`).

```bash
pip install -r requirements.txt                          # runtime/inference deps (what the prod image installs)
pip install -r requirements.txt -r requirements-dev.txt  # add training + test tooling (optuna, matplotlib, pytest)
# Python 3.11+ (Dockerfile uses 3.14-slim)

# Run the prediction web app locally (binds 127.0.0.1:5000, debug off by default)
python source/web_app.py
FLASK_DEBUG=1 python source/web_app.py    # enable Flask debug/reload

# Run the production WSGI entrypoints (what gunicorn loads)
gunicorn wsgi:app          # the main prediction app  (source/web_app.py)
gunicorn wsgi_router:app   # the landing/router page   (source/router_app.py)
```

### Tests

`pytest.ini` sets `testpaths = tests`. Tests cover auth and the login/password-reset web flows; they use Flask's test client and SQLite.

```bash
pytest                                   # full suite
pytest tests/test_auth.py                # one file
pytest tests/test_web_login_reset.py::test_name   # one test
```

The bulk-upload flow has a separate end-to-end readiness script (also the only CI check, see below):

```bash
python scripts/bulk_upload_readiness_check.py
```

### Load testing

`scripts/k6_*.js` are [k6](https://k6.io) scripts for load testing the live app (production traffic simulation and large bulk uploads).

## Architecture

### Two Flask apps, one Docker image

The single image (`Dockerfile`) is run as two containers via `docker-compose.yml`:
- **`kona-ml`** (port 8001 → `wsgi:app`): the prediction app from `source/web_app.py`. This is the bulk of the system.
- **`router`** (port 8002 → `wsgi_router:app`): a tiny landing page (`source/router_app.py`) serving `templates/router.html` for subdomain routing.

`nginx` config lives in `deploy/nginx/`. Gunicorn runs with `--workers 1 --threads 2` (the in-process model objects and SQLite connection assume a single worker).

### Prediction request flow

`source/web_app.py` (~3400 lines) is the heart of the app. The `/predict` route:
1. Takes form data (event date/time, duration, industry, equipment, ZIP/city).
2. Enriches with **demographics** (`source/demographics.py`, US Census API), **weather** (`source/weather_data.py` + `weather_providers.py`, Open-Meteo archive API), and franchise **history/competition** features.
3. Assembles exactly **46 features** via `clean_layers_feature_preparation.py` (`prepare_features_for_clean_layers_model`). The canonical feature list lives in `production/models/clean_layers_feature_names.json` — feature count/order/dtypes must match the trained model exactly.
4. Predicts via `ProductionModelManager` (`production_model_manager.py`), which inverse-MAE-weights the three models. Confidence intervals come from conformal prediction (`source/conformal_prediction.py`).
5. Optionally explains the prediction via SHAP (`source/prediction_explainer.py`).

### Model strategy (staged rollout)

`ProductionModelManager` implements a two-phase strategy:
- **Phase 1 (default):** every franchise uses the global "merged ensemble" loaded from `production/models/clean_layers_*.joblib`.
- **Phase 2:** once a franchise accumulates ≥100 completed events (`FRANCHISE_DATA_THRESHOLD`) and a franchise-specific model trains with R² ≥ 0.25 (`FRANCHISE_MODEL_CONFIDENCE_R2`), that franchise switches to its own model. `source/franchise_model_training.py` (`FranchiseModelTrainer`) trains these from feature snapshots stored in the DB.

`production/models/` holds the committed production artifacts: the three `.joblib` models, `clean_layers_ensemble_weights.json`, `clean_layers_feature_names.json`, `categorical_mappings.json`, training metrics, and `deployment_metadata.json`. Note `.gitignore` excludes `models/` and `*.joblib` globally, but `production/models/` files are tracked — keep them committed.

### Training pipeline

`source/Kona_AI_ML.py` defines `EnsembleRegressor` (the class joblib unpickles when loading models — it must stay importable) and a training CLI:

```bash
python source/Kona_AI_ML.py --train --trials 60          # full Optuna-optimized ensemble
python source/Kona_AI_ML.py --train --quick              # skip optimization, pre-tuned params
python source/Kona_AI_ML.py --train --xgboost-only       # XGBoost only
```

Training depends on a legacy `feature_engineering.py` module that is **not present** in this workspace; training/legacy preprocessing paths raise `ImportError` by design. Inference does not need it.

### Database

`source/franchise_db.py` (`FranchiseDatabase`) is a hand-rolled SQLite layer (no ORM). The DB file is `source/franchise_data.db`, created/migrated on startup via `init_schema()`. Core tables: `franchises`, `models`, `sessions`, `predictions`, plus password-reset tokens. Auth (`source/auth.py`) uses `werkzeug` pbkdf2:sha256 hashing and hex session tokens; sessions are cookie-based (`franchise_session`, 24h).

The DB is **not** in version control (`*.db` is gitignored) and is the production state of record. Before any production deploy, snapshot it (see below).

## Imports: dual-style fallback

Modules across `source/` use a try/except import pattern: package-style (`from Kona_AI_ML.X import ...`) falling back to flat (`from X import ...`). This lets the code run both as an installed package and as loose scripts with `source/` on the path. When adding cross-module imports, follow the same pattern rather than picking one style.

## Configuration (environment variables)

- `SECRET_KEY` — Flask session signing (defaults to a dev key; must be set in prod).
- `SESSION_COOKIE_SECURE` — `"true"` in production.
- `FLASK_DEBUG` — `"1"` enables debug/reload (local only).
- `ENABLE_USZIPCODE`, `ENABLE_PREDICTION_EXPLAINER` — feature toggles (default on).
- `SMTP_HOST` / `SMTP_PORT` / `SMTP_USER` / `SMTP_PASSWORD` / `SMTP_FROM` / `SMTP_USE_TLS` — email for password-reset/username-reminder (`source/email_utils.py`).

The Census API key is currently hardcoded in `source/demographics.py`.

## Deploy and DB safety

Production runs on a droplet via `docker compose` (plugin, not legacy `docker-compose`). Always snapshot the DB before deploying:

```bash
./deploy/db_snapshot.sh                  # timestamped snapshot + SHA256 into backups/
git pull
docker compose down && docker compose up -d --build

# rollback if needed (creates a pre-rollback safety snapshot first)
./deploy/db_rollback.sh source/franchise_data.db backups/<snapshot-file>.db --yes
```

Full runbook: `deploy/DB_BACKUP_ROLLBACK.md`. Reset a franchise's data with `scripts/reset_franchise_data.py` (makes a backup first).

## CI

Two GitHub Actions workflows run on PRs and pushes to `main` (Python 3.11):
- `.github/workflows/bulk-upload-readiness.yml` — runs `scripts/bulk_upload_readiness_check.py`.
- `.github/workflows/tests.yml` — runs the `pytest` suite (installs `requirements.txt` plus `pytest`).
