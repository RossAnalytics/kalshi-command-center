from datetime import datetime, timezone
import pandas as pd
from kalshi_client import get_kalshi_client

DAYS_BACK = 365
PAGE_LIMIT = 1000
MAX_MARKETS = 30000
OUT_CSV = "kalshi_markets_history.csv"

def fetch_settled_markets(days_back=DAYS_BACK,
                          page_limit=PAGE_LIMIT,
                          max_markets=MAX_MARKETS) -> pd.DataFrame:
    client = get_kalshi_client()

    now_ts = int(datetime.now(timezone.utc).timestamp())
    min_ts = now_ts - days_back * 24 * 3600

    all_markets = []
    cursor = None

    while True:
        resp = client.get_markets(
            limit=page_limit,
            cursor=cursor,
            status="settled",
            min_close_ts=min_ts,
            max_close_ts=now_ts,
        )

        markets = resp.markets or []
        if not markets:
            break

        for m in markets:
            all_markets.append(m.to_dict())

        cursor = resp.cursor
        print(f"Fetched {len(all_markets)} markets so far "
              f"(cursor={'<end>' if not cursor else cursor[:10] + '...'})")

        if not cursor or len(all_markets) >= max_markets:
            break

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
