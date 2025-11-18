# app.py
import requests
import pandas as pd
import altair as alt
import streamlit as st
from datetime import datetime, timezone

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

# ------------------------- DATA LAYER ------------------------- #

@st.cache_data(ttl=30)
def fetch_markets(status: str | None, limit: int):
    """
    Fetch markets from Kalshi public GetMarkets endpoint.
    Uses only unauthenticated, public market data.
    """
    params = {
        "limit": limit,
    }
    if status and status != "all":
        params["status"] = status

    url = f"{BASE_URL}/markets"
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    markets = data.get("markets", [])

    if not markets:
        return pd.DataFrame()

    df = pd.json_normalize(markets)

    # Time fields
    for col in [
        "created_time",
        "open_time",
        "close_time",
        "expected_expiration_time",
        "expiration_time",
    ]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # YES mid & implied probability
    def mid(row, bid_col, ask_col):
        bid = row.get(bid_col)
        ask = row.get(ask_col)
        if pd.isna(bid) and pd.isna(ask):
            return None
        if pd.isna(bid):
            return ask
        if pd.isna(ask):
            return bid
        return (bid + ask) / 2.0

    if "yes_bid" in df.columns and "yes_ask" in df.columns:
        df["yes_mid"] = df.apply(
            lambda r: mid(r, "yes_bid", "yes_ask"), axis=1
        )
        # prices are in cents → convert to prob between 0 and 1
        df["market_implied_prob_yes"] = df["yes_mid"] / 100.0
    else:
        df["yes_mid"] = None
        df["market_implied_prob_yes"] = None

    # Basic fallbacks
    if "volume" not in df.columns:
        df["volume"] = 0

    # Placeholder for your own model overlay (future)
    df["model_prob_yes"] = None
    df["edge_pct"] = None

    # Time to close
    now = datetime.now(timezone.utc)
    if "close_time" in df.columns:
        df["time_to_close"] = df["close_time"] - now
    else:
        df["time_to_close"] = None

    return df


# ------------------------- STYLING ------------------------- #

st.set_page_config(
    page_title="Kalshi Market Command Center",
    layout="wide",
    page_icon="📊",
)

# Light custom CSS to not look like a 2009 intranet tool
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    .kpi-card {
        padding: 0.9rem 1.1rem;
        border-radius: 0.8rem;
        background: #0f172a;
        color: #e5e7eb;
        border: 1px solid #1f2937;
    }
    .kpi-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9ca3af;
        margin-bottom: 0.15rem;
    }
    .kpi-value {
        font-size: 1.35rem;
        font-weight: 600;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #9ca3af;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Kalshi Market Command Center")
st.caption("Live probabilities, structure, and volume across active markets.")

# ------------------------- SIDEBAR FILTERS ------------------------- #

st.sidebar.header("Controls")

status_map = {
    "Open only": "open",
    "Closed only": "closed",
    "Settled only": "settled",
    "All statuses": "all",
}
status_label = st.sidebar.selectbox(
    "Market status filter",
    options=list(status_map.keys()),
    index=0,
)
status = status_map[status_label]

limit = st.sidebar.slider(
    "Max markets to load",
    min_value=100,
    max_value=1000,
    step=100,
    value=500,
)

min_volume = st.sidebar.number_input(
    "Minimum volume",
    min_value=0,
    value=0,
    step=10,
)

prob_range = st.sidebar.slider(
    "Implied YES probability range (%)",
    min_value=0,
    max_value=100,
    value=(0, 100),
)

auto_refresh = st.sidebar.checkbox(
    "Clear cache & refresh on run",
    value=False,
    help="Useful while watching live markets.",
)

if auto_refresh:
    st.cache_data.clear()

with st.sidebar:
    st.markdown("---")
    if st.button("Manual refresh now"):
        st.cache_data.clear()
        st.rerun()

# ------------------------- FETCH DATA ------------------------- #

try:
    df = fetch_markets(status=status, limit=limit)
except Exception as e:
    st.error(f"Error fetching markets: {e}")
    st.stop()

if df.empty:
    st.warning("No markets returned. Try loosening the filters.")
    st.stop()

# ------------------------- FILTERING ------------------------- #

# Volume filter
df = df[df["volume"].fillna(0) >= min_volume]

# Probability filter
if "market_implied_prob_yes" in df.columns:
    low, high = prob_range
    df = df[
        (df["market_implied_prob_yes"].notna())
        & (df["market_implied_prob_yes"] >= low / 100.0)
        & (df["market_implied_prob_yes"] <= high / 100.0)
    ]

if df.empty:
    st.warning("No markets left after applying filters.")
    st.stop()

# ------------------------- LAYOUT: TABS ------------------------- #

tab_overview, tab_table, tab_details = st.tabs(
    ["📈 Overview", "📋 Market Explorer", "🔍 Market Details"]
)

# ========================= OVERVIEW TAB ========================= #

with tab_overview:
    total_markets = len(df)
    open_count = int((df["status"] == "open").sum()) if "status" in df.columns else None
    avg_prob = (
        df["market_implied_prob_yes"].mean()
        if df["market_implied_prob_yes"].notna().any()
        else None
    )
    total_volume = int(df["volume"].fillna(0).sum())

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">Markets (filtered)</div>
              <div class="kpi-value">{total_markets}</div>
              <div class="kpi-sub">Across all categories</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        open_val = open_count if open_count is not None else "–"
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">Open markets</div>
              <div class="kpi-value">{open_val}</div>
              <div class="kpi-sub">Status reported by exchange</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        avg_text = f"{avg_prob*100:.1f}%" if avg_prob is not None else "–"
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">Avg implied YES probability</div>
              <div class="kpi-value">{avg_text}</div>
              <div class="kpi-sub">Mid-price converted to probability</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">Total volume</div>
              <div class="kpi-value">{total_volume:,}</div>
              <div class="kpi-sub">Sum of contract volume</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Probability distribution")

    if df["market_implied_prob_yes"].notna().any():
        chart_df = df[["ticker", "market_implied_prob_yes"]].dropna()
        chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X(
                    "market_implied_prob_yes:Q",
                    bin=alt.Bin(maxbins=20),
                    title="Implied YES probability",
                ),
                y=alt.Y("count()", title="Number of markets"),
                tooltip=[
                    alt.Tooltip("count()", title="Markets"),
                ],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No probability data available for charting.")

# ========================= TABLE TAB ========================= #

with tab_table:
    st.markdown("### Market Explorer")

    # Columns to show
    columns = [
        "ticker",
        "title",
        "status",
        "yes_bid",
        "yes_ask",
        "yes_mid",
        "market_implied_prob_yes",
        "volume",
        "close_time",
    ]
    display_cols = [c for c in columns if c in df.columns]

    df_table = df[display_cols].copy()

    if "market_implied_prob_yes" in df_table.columns:
        df_table["Implied YES %"] = (df_table["market_implied_prob_yes"] * 100).round(1)
        df_table.drop(columns=["market_implied_prob_yes"], inplace=True)

    if "yes_mid" in df_table.columns:
        df_table["yes_mid"] = df_table["yes_mid"].round(1)

    # Sort by volume then prob
    sort_cols = []
    if "volume" in df_table.columns:
        sort_cols.append("volume")
    if "Implied YES % " in df_table.columns:
        sort_cols.append("Implied YES %")

    if sort_cols:
        df_table = df_table.sort_values(sort_cols, ascending=[False] * len(sort_cols))

    st.dataframe(
        df_table,
        use_container_width=True,
        hide_index=True,
    )

# ========================= DETAILS TAB ========================= #

with tab_details:
    st.markdown("### Inspect a single market")

    ticker_options = df["ticker"].tolist()
    selected_ticker = st.selectbox(
        "Select market ticker",
        options=ticker_options,
    )

    row = df[df["ticker"] == selected_ticker].iloc[0]

    st.subheader(row.get("title", selected_ticker))
    st.caption(f"Ticker: {row.get('ticker')} • Event: {row.get('event_ticker')}")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Pricing & probabilities")
        yes_bid = row.get("yes_bid")
        yes_ask = row.get("yes_ask")
        yes_mid = row.get("yes_mid")
        imp_prob = row.get("market_implied_prob_yes")

        st.write(f"YES bid: **{yes_bid}¢**" if pd.notna(yes_bid) else "YES bid: –")
        st.write(f"YES ask: **{yes_ask}¢**" if pd.notna(yes_ask) else "YES ask: –")
        st.write(
            f"YES mid: **{yes_mid:.1f}¢**"
            if yes_mid is not None and pd.notna(yes_mid)
            else "YES mid: –"
        )

        if imp_prob is not None and pd.notna(imp_prob):
            st.write(f"Market-implied YES probability: **{imp_prob:.1%}**")
        else:
            st.write("Market-implied YES probability: –")

        model_prob = row.get("model_prob_yes")
        edge_pct = row.get("edge_pct")

        if model_prob is not None and pd.notna(model_prob):
            st.write(f"Model YES probability: **{model_prob:.1%}**")
        if edge_pct is not None and pd.notna(edge_pct):
            st.write(f"Model edge vs market: **{edge_pct:.2f} percentage points**")

    with c2:
        st.markdown("#### Timing")
        close_time = row.get("close_time")
        exp_time = row.get("expected_expiration_time")
        ttc = row.get("time_to_close")

        st.write(f"Status: **{row.get('status', '–')}**")
        st.write(f"Close time (UTC): {close_time}")
        st.write(f"Expected expiration (UTC): {exp_time}")

        if ttc is not None and not pd.isna(ttc):
            hours = ttc.total_seconds() / 3600.0
            st.write(f"Time to close: ~**{hours:.2f} hours**")

    st.markdown("#### Raw payload (debug)")
    st.json(row.to_dict())
