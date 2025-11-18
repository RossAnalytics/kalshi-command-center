import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, log_loss
import joblib

HISTORY_CSV = "kalshi_markets_history.csv"
MODEL_PATH = "kalshi_edge_model.joblib"

FEATURE_COLS = [
    "market_implied_prob_yes",
    "log_volume",
    "days_to_expiration",
]

def prepare_data(path=HISTORY_CSV):
    df = pd.read_csv(path)

    df = df.dropna(subset=["label_yes", "last_price"])
    df["label_yes"] = df["label_yes"].astype(int)

    df["market_implied_prob_yes"] = df["last_price"] / 100.0

    vol = df["volume"].fillna(0)
    df["log_volume"] = np.log1p(vol)

    for col in ["close_time", "expiration_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    exp = df.get("expiration_time")
    close = df.get("close_time")
    ref = exp.fillna(close) if exp is not None else close

    now = datetime.now(timezone.utc)
    days = (ref - now).dt.total_seconds() / (3600 * 24)
    df["days_to_expiration"] = days.clip(lower=-30, upper=365).fillna(0)

    X = df[FEATURE_COLS].fillna(0.0).values
    y = df["label_yes"].values

    return df, X, y

def main():
    df, X, y = prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    p_test = model.predict_proba(X_test)[:, 1]
    brier = brier_score_loss(y_test, p_test)
    ll = log_loss(y_test, p_test)

    print(f"Test Brier score: {brier:.4f}")
    print(f"Test log loss   : {ll:.4f}")

    joblib.dump(
        {
            "model": model,
            "feature_cols": FEATURE_COLS,
        },
        MODEL_PATH,
    )
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
