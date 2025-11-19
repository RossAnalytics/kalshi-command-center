# build_history.py
#
# Pull settled markets from Kalshi PUBLIC API and save to CSV for modeling.
# No auth, no SDK, no enum drama.

from datetime import datetime, timezone
import time
import requests
import pandas as pd

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2/markets"

DAYS_BACK = 365
PAGE_LIMIT = 1000          # max per docs
MAX_MARKETS = 30000
OUT_CSV = "kalshi_markets_history.csv"


def fetch_settled_markets(
    days_back: int = DAYS_BACK,
    page_limit: int = PAGE_LIMIT,
    max_markets: int = MAX_MARKETS,
) -> pd.DataFrame:
    """
    Uses the public /markets endpoint with ?status=settled and timestamp filters.
    Avoids kalshi_python SDK so we don't hit the 'finalized' enum bug.
    """
    now_ts = int(datetime.now(timezone.utc).timestamp())
    min_ts = now_ts - days_back * 24 * 3600

    all_markets: list[dict] = []
    cursor: str | None = None

    while True:
        params = {
            "status": "settled",
            "limit": page_limit,
            "min_close_ts": min_ts,
            "max_close_ts": now_ts,
        }
        if cursor:
            params["cursor"] = cursor

        resp = requests.get(BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        markets = data.get("markets", [])
        if not markets:
            break

        all_markets.extend(markets)
        cursor = data.get("cursor")

        print(
            f"Fetched {len(all_markets)} markets so far "
            f"(cursor={'<end>' if not cursor else cursor[:10] + '...'})"
        )

        if not cursor or len(all_markets) >= max_markets:
            break

        time.sleep(0.2)  # be polite

    if not all_markets:
        return pd.DataFrame()

    df = pd.json_normalize(all_markets)

    keep_cols = [
        "ticker",
        "event_ticker",
        "title",
        "category",
        "status",
        "result",
        "yes_bid",
        "yes_ask",
        "no_bid",
        "no_ask",
        "last_price",
        "volume",
        "open_interest",
        "close_time",
        "expiration_time",
        "settlement_value",
        "expiration_value",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    # 1 = YES, 0 = NO, NaN = other resolution types
    df["label_yes"] = df["result"].map({"yes": 1, "no": 0})

    for col in ["close_time", "expiration_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def main():
    print(f"Fetching up to {MAX_MARKETS} settled markets over last {DAYS_BACK} days…")
    df = fetch_settled_markets()
    if df.empty:
        print("No markets fetched.")
        return

    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df)} rows to {OUT_CSV}")
    print(df.head())


if __name__ == "__main__":
    main()
