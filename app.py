"""
Gold (XAUUSD) Macro Intelligence Dashboard
-------------------------------------------
Run with:  streamlit run app.py --theme.base dark

Requires: streamlit>=1.37, yfinance, pandas
"""

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import yfinance as yf

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
REFRESH_SECONDS = 300          # 5 minutes
STALE_AFTER_MINUTES = 30       # flag a feed whose last 5m bar is older than this
VIX_RISK_OFF_LEVEL = 20.0
MAX_SCORE = 4

GOLD = {"symbol": "GC=F", "name": "Gold Futures"}

# key -> (symbol, display name, decimals)
ASSETS = {
    "GOLD": ("GC=F", "Gold Futures", 2),
    "DXY": ("DX-Y.NYB", "US Dollar Index", 3),
    "TNX": ("^TNX", "10Y Treasury Yield", 3),
    "VIX": ("^VIX", "VIX Volatility", 2),
    "ES": ("ES=F", "S&P 500 Futures", 2),
}

ET = ZoneInfo("America/New_York")

st.set_page_config(
    page_title="Gold Macro Intelligence",
    page_icon="🥇",
    layout="wide",
)

# Dark-mode styling (also launch with --theme.base dark for full effect)
st.markdown(
    """
    <style>
        .stApp { background-color: #0e1117; color: #e6e6e6; }
        div[data-testid="stMetric"] {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 18px 20px;
        }
        div[data-testid="stMetricValue"] { font-size: 2.4rem; }
        .status-bar {
            background: #161b22; border: 1px solid #30363d; border-radius: 10px;
            padding: 10px 16px; font-size: 0.95rem; color: #c9d1d9;
        }
        h1, h2, h3 { color: #f5c542; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Market session logic (CME Globex metals/equity futures, approximate)
# Open Sun 18:00 ET -> Fri 17:00 ET, with a daily break 17:00-18:00 ET.
# Exchange holidays are not modelled.
# ----------------------------------------------------------------------------
def is_market_open(now_et: datetime) -> bool:
    wd = now_et.weekday()  # Mon=0 ... Sun=6
    t = now_et.hour + now_et.minute / 60
    if wd == 5:                       # Saturday
        return False
    if wd == 4 and t >= 17:           # Friday after close
        return False
    if wd == 6 and t < 18:            # Sunday before open
        return False
    if wd in (0, 1, 2, 3) and 17 <= t < 18:  # daily maintenance break
        return False
    return True


# ----------------------------------------------------------------------------
# Data fetching
# ----------------------------------------------------------------------------
def fetch_asset(symbol: str):
    """Return last price, change vs previous 5m bar close, and last bar time."""
    try:
        hist = yf.Ticker(symbol).history(period="5d", interval="5m")
        closes = hist["Close"].dropna()
        if len(closes) < 2:
            return None
        last, prev = float(closes.iloc[-1]), float(closes.iloc[-2])
        ts = closes.index[-1]
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        change = last - prev
        pct = (change / prev * 100) if prev else 0.0
        return {"last": last, "prev": prev, "change": change, "pct": pct, "bar_time": ts}
    except Exception:
        return None


def fetch_all() -> dict:
    keys = list(ASSETS.keys())
    with ThreadPoolExecutor(max_workers=len(keys)) as pool:
        results = list(pool.map(lambda k: fetch_asset(ASSETS[k][0]), keys))
    return dict(zip(keys, results))


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def build_scorecard(data: dict):
    """Return (breakdown DataFrame, total score)."""
    rules = [
        ("DXY", "Dollar falling", lambda d: d["change"] < 0),
        ("TNX", "Yields falling", lambda d: d["change"] < 0),
        ("VIX", f"VIX > {VIX_RISK_OFF_LEVEL:.0f} (risk-off)", lambda d: d["last"] > VIX_RISK_OFF_LEVEL),
        ("ES", "S&P 500 falling (safe-haven bid)", lambda d: d["change"] < 0),
    ]
    now_utc = datetime.now(ZoneInfo("UTC"))
    rows, total = [], 0

    for key, condition, test in rules:
        symbol, name, dec = ASSETS[key]
        d = data.get(key)
        if d is None:
            rows.append({
                "Driver": name, "Ticker": symbol, "Live Value": "N/A",
                "5m Move": "N/A", "5m %": "N/A", "Bullish Condition": condition,
                "Data": "❌ No data", "Score": 0,
            })
            continue

        points = 1 if test(d) else 0
        total += points
        age_min = (now_utc - d["bar_time"].astimezone(ZoneInfo("UTC"))).total_seconds() / 60
        freshness = "🟢 Live" if age_min <= STALE_AFTER_MINUTES else f"🟡 Stale ({age_min / 60:.1f}h)"
        arrow = "▲" if d["change"] > 0 else ("▼" if d["change"] < 0 else "■")

        rows.append({
            "Driver": name,
            "Ticker": symbol,
            "Live Value": f"{d['last']:,.{dec}f}",
            "5m Move": f"{arrow} {d['change']:+.{dec}f}",
            "5m %": f"{d['pct']:+.3f}%",
            "Bullish Condition": condition,
            "Data": freshness,
            "Score": points,
        })

    return pd.DataFrame(rows), total


def classify(score: int):
    confidence = score / MAX_SCORE * 100
    if confidence >= 75:
        tier = "High Conviction"
    elif confidence >= 50:
        tier = "Moderate Conviction"
    else:
        tier = "Low Conviction / Chop"

    if score >= 3:
        bias = "🟢 Bullish"
    elif score == 2:
        bias = "🟡 Neutral"
    else:
        bias = "🔴 Bearish"
    return bias, confidence, tier


# ----------------------------------------------------------------------------
# Dashboard fragment: re-runs every second (for the countdown), but only hits
# the network every REFRESH_SECONDS while the market is open.
# ----------------------------------------------------------------------------
@st.fragment(run_every="1s")
def dashboard():
    now_et = datetime.now(ET)
    market_open = is_market_open(now_et)
    now_ts = time.time()

    ss = st.session_state
    ss.setdefault("data", None)
    ss.setdefault("last_fetch_ts", 0.0)
    ss.setdefault("last_update", None)

    # Controls
    top_l, top_r = st.columns([5, 1])
    with top_r:
        manual = st.button("🔄 Refresh now", use_container_width=True)

    due = (now_ts - ss.last_fetch_ts) >= REFRESH_SECONDS
    first_load = ss.data is None
    if first_load or manual or (market_open and due):
        with st.spinner("Fetching market data..."):
            ss.data = fetch_all()
        ss.last_fetch_ts = time.time()
        ss.last_update = datetime.now().astimezone()
        now_ts = ss.last_fetch_ts

    # Status bar: last update + countdown
    remaining = max(0, int(REFRESH_SECONDS - (now_ts - ss.last_fetch_ts)))
    mins, secs = divmod(remaining, 60)
    if market_open:
        countdown = f"⏱ Next refresh in <b>{mins:02d}:{secs:02d}</b>"
        session = "🟢 Futures session OPEN"
    else:
        countdown = "⏸ Auto-refresh paused (market closed)"
        session = "🔴 Futures session CLOSED"
    last_str = ss.last_update.strftime("%Y-%m-%d %H:%M:%S %Z")
    with top_l:
        st.markdown(
            f'<div class="status-bar">🕒 Last update: <b>{last_str}</b> &nbsp;|&nbsp; '
            f'{countdown} &nbsp;|&nbsp; {session} '
            f'&nbsp;|&nbsp; ET: {now_et.strftime("%a %H:%M")}</div>',
            unsafe_allow_html=True,
        )

    data = ss.data
    df, score = build_scorecard(data)
    bias, confidence, tier = classify(score)

    # Headline metrics
    st.write("")
    c1, c2, c3 = st.columns(3)
    c1.metric("Macro Bias", bias, f"{score}/{MAX_SCORE} points", delta_color="off")
    c2.metric("Confidence", f"{confidence:.0f}%")
    c3.metric("Rating Tier", tier)
    st.progress(int(confidence), text=f"Bullish confidence: {confidence:.0f}%")

    # Live asset strip
    st.subheader("Live Prices")
    cols = st.columns(len(ASSETS))
    for col, (key, (symbol, name, dec)) in zip(cols, ASSETS.items()):
        d = data.get(key)
        if d is None:
            col.metric(name, "N/A")
        else:
            col.metric(name, f"{d['last']:,.{dec}f}", f"{d['change']:+.{dec}f} ({d['pct']:+.2f}%)")

    # Breakdown table
    st.subheader("Driver Breakdown")
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Score": st.column_config.NumberColumn("Score", format="%d / 1"),
        },
    )

    if any(v is None for v in data.values()):
        st.warning("Some feeds returned no data (Yahoo may be rate-limiting or the symbol is inactive). "
                   "Missing drivers score 0.")
    if any(d is not None and (datetime.now(ZoneInfo("UTC")) - d["bar_time"].astimezone(ZoneInfo("UTC"))).total_seconds()
           > STALE_AFTER_MINUTES * 60 for d in data.values()):
        st.info("Some feeds are stale: index data (e.g. ^TNX, ^VIX) only updates during its own trading hours "
                "and Yahoo may delay it ~15 minutes.")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
st.title("🥇 Gold (XAUUSD) Macro Intelligence")
st.caption("Real-time macro driver scoring: DXY • US10Y • VIX • S&P 500 futures. "
           "Data via Yahoo Finance (yfinance). Not financial advice.")
dashboard()
