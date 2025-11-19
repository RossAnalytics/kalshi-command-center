# app.py  — Kalshi Market Command Center (model-driven)

import os
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import requests
import streamlit as st
import altair as alt
import joblib
from streamlit_autorefresh import st_autorefresh  # only used if AUTOREFRESH_SECONDS > 0

# ------------------------- CONFIG ------------------------- #

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
DEFAULT_LIMIT = 500        # default number of markets to fetch
AUTOREFRESH_SECONDS = 0    # set >0 to enable auto-refresh e.g., 60

# --------------------- DATA FUNCTIONS --------------------- #

@st.cache_data(ttl=60)
def fetch_markets(limit: int = DEFAULT_LIMIT, status: str | None = None) -> pd.DataFrame:
    params = {"limit": limit}
    if status and status != "all":
        params["status"] = status

    resp = requests.get(f"{BASE_URL}/markets", params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    markets = data.get("markets", [])
    if not markets:
        return pd.DataFrame()

    df = pd.json_normalize(markets)

    for col in ["open_time", "close_time", "expected_expiration_time", "expiration_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    numeric_cols = [
        "yes_bid", "yes_ask", "no_bid", "no_ask",
        "last_price", "volume", "volume_24h",
        "open_interest", "liquidity"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if {"yes_bid", "yes_ask"}.issubset(df.columns):
        df["yes_mid"] = (df["yes_bid"] + df["yes_ask"]) / 2.0
    else:
        df["yes_mid"] = df.get("last_price", np.nan)

    df["market_implied_prob_yes"] = df["yes_mid"] / 100.0

    now = datetime.now(timezone.utc)
    if "close_time" in df.columns:
        df["hours_to_close"] = (df["close_time"] - now).dt.total_seconds() / 3600.0
    else:
        df["hours_to_close"] = np.nan

    if "category" not in df.columns:
        df["category"] = "Unknown"

    return df

@st.cache_resource
def load_edge_model():
    try:
        bundle = joblib.load("kalshi_edge_model.joblib")
        return bundle["model"], bundle["feature_cols"]
    except Exception as e:
        st.warning(f"Could not load edge model: {e}")
        return None, None

def compute_model_probs(df: pd.DataFrame) -> pd.DataFrame:
    model, feature_cols = load_edge_model()
    df = df.copy()

    if model is None or feature_cols is None:
        df["model_prob_yes"] = df["market_implied_prob_yes"]
        df["edge_pct"] = 0.0
        df["confidence_score"] = 0.0
        return df

    if "market_implied_prob_yes" not in df.columns:
        df["market_implied_prob_yes"] = df["yes_mid"] / 100.0

    df["log_volume"] = np.log1p(df["volume"].fillna(0))

    for col in ["expected_expiration_time", "expiration_time", "close_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    exp = df.get("expected_expiration_time")
    exp2 = df.get("expiration_time")
    close = df.get("close_time")

    ref = exp if exp is not None else (exp2 if exp2 is not None else close)

    now = datetime.now(timezone.utc)
    days = (ref - now).dt.total_seconds() / (3600 * 24)
    df["days_to_expiration"] = days.clip(lower=-30, upper=365).fillna(0.0)

    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0

    X_live = df[feature_cols].fillna(0.0).values
    probs = model.predict_proba(X_live)[:, 1]
    df["model_prob_yes"] = probs

    df["edge_pct"] = (df["model_prob_yes"] - df["market_implied_prob_yes"]) * 100.0

    abs_edge = df["edge_pct"].abs()
    edge_norm = abs_edge / (abs_edge.max() + 1e-9)
    vol_norm = df["log_volume"] / (df["log_volume"].max() + 1e-9)

    conf_raw = 0.6 * edge_norm + 0.4 * vol_norm
    df["confidence_score"] = (conf_raw * 100).round(1)

    return df

def attach_movers(df: pd.DataFrame, session_col: str = "market_implied_prob_yes") -> pd.DataFrame:
    df = df.copy()
    base = df[session_col]
    df["session_move"] = base - base.median()
    df["session_move_abs"] = df["session_move"].abs()
    return df

def kpi_card(label: str, value: str, sub: str = ""):
    st.metric(label, value, sub)

def prob_hist_chart(df: pd.DataFrame):
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("market_implied_prob_yes:Q", bin=alt.Bin(step=0.05),
                    title="Implied YES probability"),
            y=alt.Y("count():Q", title="Number of markets"),
            tooltip=["count()"]
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)

def volume_vs_prob_chart(df: pd.DataFrame):
    chart = (
        alt.Chart(df)
        .mark_circle(size=40)
        .encode(
            x=alt.X("market_implied_prob_yes:Q", title="Implied YES probability"),
            y=alt.Y("volume:Q", title="Volume"),
            color=alt.Color("category:N", legend=None),
            tooltip=["ticker", "title", "category", "market_implied_prob_yes", "volume"],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)

def main():
    st.set_page_config(
        page_title="Kalshi Market Command Center",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Kalshi Market Command Center")
    st.caption("Live probabilities, structure, and model-driven signals across active markets.")

    if AUTOREFRESH_SECONDS > 0:
        st_autorefresh(interval=AUTOREFRESH_SECONDS * 1000, key="auto_refresh")

    st.sidebar.header("Filters")
    status_filter = st.sidebar.selectbox(
        "Market status", ["all", "open", "active", "closed", "settled"], index=0
    )
    limit = st.sidebar.slider("Max markets", 100, 1000, DEFAULT_LIMIT, step=100)
    # Preload a small sample to build category list
    sample = fetch_markets(limit=200, status=status_filter)
    categories = sorted(sample["category"].dropna().unique().tolist())
    category_filter = st.sidebar.selectbox("Category", ["All"] + categories, index=0)
    search_text = st.sidebar.text_input("Search title / ticker")

    df = fetch_markets(limit=limit, status=status_filter)
    if category_filter != "All":
        df = df[df["category"] == category_filter]
    if search_text:
        df = df[
            df["title"].str.contains(search_text, case=False, na=False)
            | df["ticker"].str.contains(search_text, case=False, na=False)
        ]

    if df.empty:
        st.error("No markets found with current filters.")
        return

    df = compute_model_probs(df)
    df = attach_movers(df)

    tab_overview, tab_signals, tab_explorer, tab_details, tab_top3 = st.tabs(
        ["📊 Overview", "📈 Signals", "🧭 Market Explorer", "🔍 Market Details", "🎯 Signal of the Day"]
    )

    # Overview tab
    with tab_overview:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Markets (filtered)", f"{len(df):,}")
        open_count = df["status"].eq("open").sum() if "status" in df.columns else 0
        col2.metric("Open markets", f"{open_count:,}")
        avg_prob = df["market_implied_prob_yes"].mean() * 100
        col3.metric("Avg implied YES", f"{avg_prob:0.1f}%")
        total_vol = df["volume"].sum()
        col4.metric("Total volume", f"{total_vol:,.0f}")

        st.subheader("Probability distribution")
        prob_hist_chart(df)
        st.subheader("Volume vs probability")
        volume_vs_prob_chart(df)

    # Signals tab
    with tab_signals:
        st.subheader("Model vs Market Signals")
        min_conf = st.slider("Min confidence", 0.0, 100.0, 60.0, step=5.0)
        min_edge = st.slider("Min |edge| (pp)", 0.0, 20.0, 5.0, step=1.0)

        sig_df = df[
            (df["confidence_score"] >= min_conf)
            & (df["edge_pct"].abs() >= min_edge)
        ].copy()

        ca, cb, cc = st.columns(3)
        ca.metric("Signals (filtered)", f"{len(sig_df):,}")
        big_disagreements = df["edge_pct"].abs().ge(10).sum()
        cb.metric("Large disagreements (|edge|≥10)", f"{big_disagreements:,}")
        high_conf = df["confidence_score"].ge(70).sum()
        cc.metric("High confidence (≥70)", f"{high_conf:,}")

        st.markdown("#### Top positive edges")
        top_pos = (
            sig_df.sort_values("edge_pct", ascending=False)
            .head(15)[
                ["ticker", "title", "category",
                 "market_implied_prob_yes", "model_prob_yes",
                 "edge_pct", "confidence_score", "volume"]
            ]
        )
        st.dataframe(top_pos, use_container_width=True)

        st.markdown("#### Top negative edges")
        top_neg = (
            sig_df.sort_values("edge_pct", ascending=True)
            .head(15)[
                ["ticker", "title", "category",
                 "market_implied_prob_yes", "model_prob_yes",
                 "edge_pct", "confidence_score", "volume"]
            ]
        )
        st.dataframe(top_neg, use_container_width=True)

    # Explorer tab
    with tab_explorer:
        st.subheader("Market Explorer")
        display_cols = [
            "ticker", "title", "category",
            "market_implied_prob_yes", "model_prob_yes",
            "edge_pct", "confidence_score",
            "volume", "open_interest", "hours_to_close"
        ]
        display_cols = [c for c in display_cols if c in df.columns]
        sorted_df = df.sort_values("edge_pct", ascending=False)
        st.dataframe(sorted_df[display_cols], use_container_width=True, height=600)

    # Details tab
    with tab_details:
        st.subheader("Per-market details")

        search_input = st.text_input("Search ticker or title", "")
        if search_input:
            mask = df["ticker"].str.contains(search_input, case=False, na=False) \
                   | df["title"].str.contains(search_input, case=False, na=False)
            filtered = df[mask]
        else:
            filtered = df.copy()

        st.write(f"{len(filtered):,} matching markets")

        choice = st.selectbox(
            "Choose a market",
            filtered["ticker"].tolist(),
            format_func=lambda x: filtered[filtered["ticker"] == x]["title"].iloc[0]
        )

        row = df[df["ticker"] == choice].iloc[0]

        st.markdown(f"## {row['title']}")
        st.write(f"**Ticker:** `{row['ticker']}`   |   **Category:** {row.get('category','N/A')}")
        if "status" in row:
            st.write(f"**Status:** {row['status']}   |   **Close time:** {row.get('close_time','N/A')}")

        d1, d2, d3 = st.columns(3)
        d1.metric("Market implied YES", f"{row['market_implied_prob_yes']*100:.1f}%")
        d2.metric("Model YES", f"{row['model_prob_yes']*100:.1f}%", f"{row['edge_pct']:0.1f} pp edge")
        d3.metric("Confidence", f"{row['confidence_score']:0.1f}")

        st.markdown("---")

        with st.expander("Trading & volume details", expanded=True):
            st.write(f"Volume: {row.get('volume', 'N/A')}")
            st.write(f"Open Interest: {row.get('open_interest', 'N/A')}")
            st.write(f"Hours to Close: {row.get('hours_to_close', 'N/A'):.1f}")

        with st.expander("Raw data"):
            st.json(row.to_dict())

    # Signal of the Day tab
    with tab_top3:
        st.header("🎯 Top 3 Signals of the Day")

        now = datetime.now(timezone.utc)
        pool = df.copy()
        if "close_time" in pool.columns:
            pool = pool[
                (pool["close_time"] >= now) &
                (pool["close_time"] <= now + pd.Timedelta(hours=24))
            ]

        pool["abs_edge"] = pool["edge_pct"].abs()
        vol_cut = pool["volume"].quantile(0.50)
        pool = pool[pool["volume"] >= vol_cut]
        pool["score"] = pool["confidence_score"] * pool["abs_edge"]
        top3 = pool.sort_values("score", ascending=False).head(3)

        if top3.empty:
            st.write("No strong signals closing in next 24 hours.")
        else:
            for idx, (_, row) in enumerate(top3.iterrows(), start=1):
                with st.container():
                    st.subheader(f"Signal #{idx}")
                    st.markdown(f"**{row['title']}**")
                    st.markdown(f"*Ticker:* `{row['ticker']}`  |  *Category:* {row.get('category','N/A')}")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Market YES", f"{row['market_implied_prob_yes']*100:.1f}%")
                    c2.metric("Model YES", f"{row['model_prob_yes']*100:.1f}%", f"{row['edge_pct']:0.1f} pp edge")
                    c3.metric("Confidence", f"{row['confidence_score']:0.1f}")
                    st.write(f"**Volume:** {row['volume']:,}  |  **Edge Magnitude:** {row['abs_edge']:.1f} pp")
                    if "close_time" in row:
                        hours_left = (row["close_time"] - now).total_seconds() / 3600.0
                        st.write(f"**Closing in:** {hours_left:.1f} hours")
                    st.markdown("---")

if __name__ == "__main__":
    main()
