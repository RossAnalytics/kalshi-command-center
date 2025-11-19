# app.py  — Kalshi Market Command Center (model-driven)

import os
from datetime import datetime, timezone

import requests
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import joblib

# ------------------------- CONFIG ------------------------- #

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

DEFAULT_LIMIT = 500        # max markets to fetch for dashboard
AUTOREFRESH_SECONDS = 60   # set to 0 to disable auto-refresh

# --------------------- DATA FUNCTIONS --------------------- #

@st.cache_data(ttl=60)
def fetch_markets(limit: int = DEFAULT_LIMIT, status: str | None = None) -> pd.DataFrame:
    """
    Fetch markets from Kalshi public /markets endpoint.
    Only uses unauthenticated public data.
    """
    params = {"limit": limit}
    if status:
        params["status"] = status

    resp = requests.get(f"{BASE_URL}/markets", params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    markets = data.get("markets", [])
    if not markets:
        return pd.DataFrame()

    df = pd.json_normalize(markets)

    # Convert timestamps
    for col in ["open_time", "close_time", "expected_expiration_time", "expiration_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Numeric columns
    num_cols = [
        "yes_bid", "yes_ask", "no_bid", "no_ask",
        "last_price", "volume", "volume_24h",
        "open_interest", "liquidity"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Mid prices / implied probs
    if {"yes_bid", "yes_ask"}.issubset(df.columns):
        mid = (df["yes_bid"] + df["yes_ask"]) / 2.0
        df["yes_mid"] = mid
    else:
        df["yes_mid"] = df.get("last_price", np.nan)

    df["market_implied_prob_yes"] = df["yes_mid"] / 100.0

    # Hours to close
    now = datetime.now(timezone.utc)
    if "close_time" in df.columns:
        df["hours_to_close"] = (df["close_time"] - now).dt.total_seconds() / 3600.0
    else:
        df["hours_to_close"] = np.nan

    # Basic category cleanup
    if "category" not in df.columns:
        df["category"] = "Unknown"

    return df


@st.cache_resource
def load_edge_model():
    """
    Load trained logistic regression model from kalshi_edge_model.joblib.

    File comes from train_model.py:
      - model: sklearn LogisticRegression
      - feature_cols: list of feature names used in training
    """
    try:
        bundle = joblib.load("kalshi_edge_model.joblib")
        return bundle["model"], bundle["feature_cols"]
    except Exception as e:
        st.warning(f"Could not load edge model: {e}")
        return None, None


def compute_model_probs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply trained model to live markets to get:
      - model_prob_yes
      - edge_pct (model vs market)
      - confidence_score
    """
    model, feature_cols = load_edge_model()
    df = df.copy()

    # If model not available, fall back to identity
    if model is None or feature_cols is None:
        df["model_prob_yes"] = df["market_implied_prob_yes"]
        df["edge_pct"] = 0.0
        df["confidence_score"] = 0.0
        return df

    # Ensure basic features exist
    if "market_implied_prob_yes" not in df.columns:
        df["market_implied_prob_yes"] = df["yes_mid"] / 100.0

    vol = df["volume"].fillna(0)
    df["log_volume"] = np.log1p(vol)

    # Days to expiration
    for col in ["expected_expiration_time", "expiration_time", "close_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    exp = df.get("expected_expiration_time")
    exp2 = df.get("expiration_time")
    close = df.get("close_time")

    ref = None
    if exp is not None:
        ref = exp
    elif exp2 is not None:
        ref = exp2
    else:
        ref = close

    now = datetime.now(timezone.utc)
    days = (ref - now).dt.total_seconds() / (3600 * 24)
    df["days_to_expiration"] = days.clip(lower=-30, upper=365).fillna(0)

    # Build feature matrix in same order as training
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    X_live = df[feature_cols].fillna(0.0).values

    probs = model.predict_proba(X_live)[:, 1]
    df["model_prob_yes"] = probs

    # Edge vs market (percentage points)
    df["edge_pct"] = (df["model_prob_yes"] - df["market_implied_prob_yes"]) * 100.0

    # Confidence: based on edge magnitude + volume
    abs_edge = df["edge_pct"].abs()
    edge_norm = abs_edge / (abs_edge.max() + 1e-9)

    vol_norm = np.log1p(df["volume"].fillna(0)) / (
        np.log1p(df["volume"].fillna(0)).max() + 1e-9
    )

    conf_raw = 0.6 * edge_norm + 0.4 * vol_norm
    df["confidence_score"] = (conf_raw * 100).round(1)

    return df


def attach_movers(df: pd.DataFrame, session_col: str = "market_implied_prob_yes") -> pd.DataFrame:
    """
    For now, session movers = just relative deviation from median.
    (Later you can persist prior snapshots.)
    """
    df = df.copy()
    base = df[session_col]
    df["session_move"] = base - base.median()
    df["session_move_abs"] = df["session_move"].abs()
    return df


# ------------------------- UI HELPERS ------------------------- #

def kpi_card(label: str, value: str, sub: str):
    st.metric(label, value, sub)


def prob_hist_chart(df: pd.DataFrame):
    hist = (
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
    st.altair_chart(hist, use_container_width=True)


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


# --------------------------- UI APP --------------------------- #

def main():
    st.set_page_config(
        page_title="Kalshi Market Command Center",
        layout="wide",
    )

    st.title("Kalshi Market Command Center")
    st.caption("Live probabilities, structure, and model-driven signals across active markets.")

    # Auto-refresh
    if AUTOREFRESH_SECONDS > 0:
        st_autorefresh = st.experimental_rerun  # placeholder if you later add streamlit_autorefresh

    # Sidebar filters
    st.sidebar.header("Filters")

    status_filter = st.sidebar.selectbox(
        "Market status",
        options=["open", "active", "closed", "all"],
        index=0,
    )
    status_param = None if status_filter == "all" else status_filter

    limit = st.sidebar.slider("Max markets", 100, 1000, DEFAULT_LIMIT, step=100)

    # Fetch & model
    df = fetch_markets(limit=limit, status=status_param)
    if df.empty:
        st.error("No markets found from Kalshi API.")
        return

    df = compute_model_probs(df)
    df = attach_movers(df)

    # Tabs
    tab_overview, tab_signals, tab_explorer, tab_details = st.tabs(
        ["📊 Overview", "📈 Signals", "🧭 Market Explorer", "🔍 Market Details"]
    )

    # ------------- OVERVIEW TAB ------------- #
    with tab_overview:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            kpi_card("MARKETS (FILTERED)", f"{len(df):,}", "Across all categories")

        with col2:
            open_count = (df["status"] == "open").sum() if "status" in df.columns else 0
            kpi_card("OPEN MARKETS", f"{open_count:,}", "As reported by exchange")

        with col3:
            avg_prob = df["market_implied_prob_yes"].mean() * 100
            kpi_card("AVG IMPLIED YES PROB", f"{avg_prob:0.1f}%", "Mid-price → probability")

        with col4:
            total_vol = df["volume"].sum()
            kpi_card("TOTAL VOLUME", f"{total_vol:,.0f}", "Sum of contract volume")

        st.subheader("Probability distribution")
        prob_hist_chart(df)

        st.subheader("Volume vs probability")
        volume_vs_prob_chart(df)

    # ------------- SIGNALS TAB ------------- #
    with tab_signals:
        st.subheader("Model vs market signals")

        min_conf = st.slider("Minimum confidence score", 0.0, 100.0, 60.0, step=5.0)
        min_edge = st.slider("Minimum |edge| (percentage points)", 0.0, 20.0, 5.0, step=1.0)

        sig_df = df[
            (df["confidence_score"] >= min_conf)
            & (df["edge_pct"].abs() >= min_edge)
        ].copy()

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            kpi_card("Signals (filtered)", f"{len(sig_df):,}", "Model vs market")

        with col_b:
            big_disagreements = (df["edge_pct"].abs() >= 10).sum()
            kpi_card("Large disagreements (|edge| ≥ 10)", f"{big_disagreements:,}", "")

        with col_c:
            high_conf = (df["confidence_score"] >= 70).sum()
            kpi_card("High confidence (≥ 70)", f"{high_conf:,}", "")

        st.markdown("#### Top positive edges (model > market)")
        top_pos = (
            sig_df.sort_values("edge_pct", ascending=False)
            .head(15)
            .loc[:, ["ticker", "title", "category",
                     "market_implied_prob_yes", "model_prob_yes",
                     "edge_pct", "confidence_score", "volume"]]
        )
        st.dataframe(top_pos, use_container_width=True)

        st.markdown("#### Top negative edges (model < market)")
        top_neg = (
            sig_df.sort_values("edge_pct", ascending=True)
            .head(15)
            .loc[:, ["ticker", "title", "category",
                     "market_implied_prob_yes", "model_prob_yes",
                     "edge_pct", "confidence_score", "volume"]]
        )
        st.dataframe(top_neg, use_container_width=True)

    # ------------- MARKET EXPLORER TAB ------------- #
    with tab_explorer:
        st.subheader("Market Explorer")

        cat_options = sorted(df["category"].dropna().unique().tolist())
        selected_cat = st.selectbox("Category", options=["All"] + cat_options)

        search = st.text_input("Search title / ticker", "")

        filt = df.copy()
        if selected_cat != "All":
            filt = filt[filt["category"] == selected_cat]

        if search:
            s = search.lower()
            filt = filt[
                filt["title"].str.lower().str.contains(s, na=False)
                | filt["ticker"].str.lower().str.contains(s, na=False)
            ]

        cols = [
            "ticker", "title", "category",
            "market_implied_prob_yes", "model_prob_yes",
            "edge_pct", "confidence_score",
            "volume", "open_interest",
            "hours_to_close",
        ]
        cols = [c for c in cols if c in filt.columns]

        st.dataframe(
            filt[cols].sort_values("edge_pct", ascending=False),
            use_container_width=True,
            height=600,
        )

    # ------------- DETAILS TAB ------------- #
    with tab_details:
        st.subheader("Per-market details")

        tickers = df["ticker"].tolist()
        choice = st.selectbox("Choose a market", options=tickers)

        row = df[df["ticker"] == choice].iloc[0]

        st.markdown(f"### {row['title']}")
        st.caption(f"Ticker: `{row['ticker']}` | Category: {row.get('category', 'N/A')}")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Market implied YES", f"{row['market_implied_prob_yes']*100:0.1f}%")
        with c2:
            st.metric("Model YES", f"{row['model_prob_yes']*100:0.1f}%",
                      f"{row['edge_pct']:0.1f} pp edge")
        with c3:
            st.metric("Confidence", f"{row['confidence_score']:0.1f}")

        st.write("**Raw fields**")
        st.json(row.to_dict())


if __name__ == "__main__":
    main()
