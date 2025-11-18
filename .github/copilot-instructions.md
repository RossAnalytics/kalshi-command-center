<!-- Copied/merged guidance for AI coding agents in this repo -->
# Copilot Instructions for Kalshi Market Command Center

Purpose: Give an AI coding agent just-enough, repository-specific context to be productive when editing, extending, or debugging this Streamlit app.

- **Repo entrypoint**: `app.py` — a single-file Streamlit app that fetches public market data from Kalshi and renders dashboards.
- **Main dependencies**: listed in `requirements.txt` (requests, pandas, altair, streamlit).

Architecture & key patterns
- Single-file structure: UI, data-fetching, and helpers all live in `app.py`. Expect changes to touch the same file.
- Data layer: `fetch_markets(status: str | None, limit: int)` queries `BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"` and returns a `pandas.DataFrame`.
  - Normalizes JSON with `pd.json_normalize`, coerces time fields to datetimes, and computes `yes_mid` and `market_implied_prob_yes` (prices are in cents).
  - Caching: `@st.cache_data(ttl=30)` is used to reduce requests; clearing cache is supported via UI and `st.cache_data.clear()`.
- UI layout: Streamlit `sidebar` for filters, three tabs (`Overview`, `Market Explorer`, `Market Details`). Charts use `altair` and tables use `st.dataframe`.

Developer workflows (explicit, reproducible)
- Install dependencies (PowerShell):
  - `pip install -r requirements.txt`
- Run locally (PowerShell):
  - `streamlit run app.py`
  - Or with explicit Python: `& C:\path\to\python.exe -m streamlit run app.py`
- Debugging notes:
  - Use the "Manual refresh now" button or set "Clear cache & refresh on run" in the sidebar to bypass `st.cache_data` during development.
  - HTTP errors surface via `resp.raise_for_status()` in `fetch_markets`; wrap or mock `requests.get` in tests.

Project-specific conventions and gotchas
- Prices are in cents. The code converts mid price to probability with `yes_mid / 100.0`. When adding any price math, preserve this cents→prob convention.
- Column names are relied upon across UI and filtering: `ticker`, `title`, `status`, `yes_bid`, `yes_ask`, `yes_mid`, `market_implied_prob_yes`, `volume`, `close_time`. Ensure transformations keep these names or update UI references.
- Time handling: datetimes are converted with `pd.to_datetime(..., errors='coerce')` and `time_to_close` is computed in UTC. Keep UTC assumptions when adding time logic.
- Light styling is embedded via raw HTML/CSS in `st.markdown(..., unsafe_allow_html=True)`. Small visual tweaks can be done inline; larger style refactors should remain minimal.

Integration points & extension hints
- External API: `BASE_URL` points to Kalshi public API. Any authenticated endpoints will require adding secure key handling (do not hardcode keys). For local dev, use environment variables and the `os` module.
- To add a model overlay (the app leaves `model_prob_yes` and `edge_pct` as placeholders): compute and assign these inside `fetch_markets` before returning the DataFrame. Example:
  - `df['model_prob_yes'] = df['market_implied_prob_yes'] * 0.95  # placeholder`
  - `df['edge_pct'] = 100 * (df['model_prob_yes'] - df['market_implied_prob_yes'])`
- If adding more endpoints or background jobs, prefer creating new modules instead of bloating `app.py` — but mirror the same data->UI flow.

Testing & CI guidance
- There are no tests in the repo. For small additions, write unit tests that mock `requests.get` responses (use `requests-mock` or `unittest.mock`) and validate `fetch_markets` transforms.
- Keep tests focused on `fetch_markets` and DataFrame shape/column expectations.

PR & code-change guidance for AI agents
- Make minimal, focused edits. This repo is intentionally simple: prefer changing `app.py` directly but isolate new logic into functions.
- Preserve user-facing behavior unless the change is explicitly an UX improvement. Keep caching and default `limit` behavior consistent.

Files to reference when making changes
- `app.py` — main source of truth for behavior and UI
- `requirements.txt` — update for new dependencies

If any part of the codebase is unclear, ask for sample API responses (the `markets` JSON) or permission to add a small fixture under `tests/fixtures/` to lock in expected payloads.

-- End of file
