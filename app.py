import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Elliott Wave 0 & 5 — Pure A/B/C Rules", layout="wide")
st.title("📈 PURE Elliott Wave 0 & 5 Detection (A OR B OR C)")
st.caption("Strict Elliott + Structural Swing + Fibonacci + Trend Filters")


# ---------------- DATA LOADER ----------------
@st.cache_data(ttl=3600)
def load_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna().copy()
    df.index = pd.to_datetime(df.index)
    return df


# ---------------- INDICATORS ----------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # SMAs
    for win in [20, 50, 200]:
        df[f"SMA_{win}"] = df["Close"].rolling(win).mean()

    # RSI(14)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    roll_up = gain.rolling(14).mean()
    roll_down = loss.rolling(14).mean()

    rs = roll_up / roll_down.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    return df


# ---------------- TIMEFRAME PARAMETERS ----------------
def get_params(tf: str) -> dict:
    if tf == "1h":
        return dict(pivot_k=3, swing_window=25, fib_window=80, future_n=7)
    elif tf == "1wk":
        return dict(pivot_k=2, swing_window=10, fib_window=30, future_n=3)
    else:  # daily default
        return dict(pivot_k=5, swing_window=20, fib_window=60, future_n=5)


# ---------------- PIVOTS ----------------
def pivot_lows(df: pd.DataFrame, k: int) -> np.ndarray:
    lows = df["Low"].values
    n = len(df)
    piv = np.zeros(n, dtype=bool)
    for i in range(k, n - k):
        if lows[i] == lows[i - k:i + k + 1].min():
            piv[i] = True
    return piv


def pivot_highs(df: pd.DataFrame, k: int) -> np.ndarray:
    highs = df["High"].values
    n = len(df)
    piv = np.zeros(n, dtype=bool)
    for i in range(k, n - k):
        if highs[i] == highs[i - k:i + k + 1].max():
            piv[i] = True
    return piv


# ----------------------------------------------------------
# ----------------- WAVE 0 RULES (A, B, C) -----------------
# ----------------------------------------------------------
def rule_A0(df: pd.DataFrame, piv: np.ndarray, params: dict) -> np.ndarray:
    """Strict-ish Elliott style: RSI bottom + reversal + future up."""
    n = len(df)
    A0 = np.zeros(n, dtype=bool)

    rsi = df["RSI_14"].values
    close = df["Close"].values
    future_n = params["future_n"]

    for i in range(2, n - future_n - 5):
        if not piv[i]:
            continue

        if np.isnan(rsi[i]) or np.isnan(rsi[i - 1]):
            continue

        # RSI oversold-ish AND turning up
        if rsi[i] > 40 or rsi[i] <= rsi[i - 1]:
            continue

        # Price turning up
        if close[i] <= close[i - 1]:
            continue

        # Future confirmation: after N bars, price higher than 0
        if close[i + future_n] <= close[i]:
            continue

        A0[i] = True

    return A0


def rule_B0(df: pd.DataFrame, piv: np.ndarray, params: dict) -> np.ndarray:
    """Pure swing low: lowest in recent window + future price up."""
    n = len(df)
    B0 = np.zeros(n, dtype=bool)

    low = df["Low"].values
    close = df["Close"].values
    swing = params["swing_window"]
    future_n = params["future_n"]

    for i in range(n - future_n):
        if not piv[i]:
            continue

        start = max(0, i - swing)
        if low[i] != low[start:i + 1].min():
            continue

        if close[i + future_n] <= close[i]:
            continue

        B0[i] = True

    return B0


def rule_C0(df: pd.DataFrame, piv: np.ndarray, params: dict) -> np.ndarray:
    """Fibonacci retracement: 0 near 61.8–100% retracement of prior move."""
    n = len(df)
    C0 = np.zeros(n, dtype=bool)

    low = df["Low"].values
    fib_win = params["fib_window"]

    for i in range(n):
        if not piv[i]:
            continue

        start = max(0, i - fib_win)
        if start >= i:
            continue

        prev_high = df["High"].iloc[start:i].max()
        if prev_high <= 0:
            continue

        retr = (prev_high - low[i]) / prev_high  # % drop from prev high

        if 0.58 <= retr <= 1.05:  # 58%–105% zone
            C0[i] = True

    return C0


# ----------------------------------------------------------
# ----------------- FINAL WAVE 0 (A OR B OR C) --------------
# ----------------------------------------------------------
def add_wave0(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    df = df.copy()
    params = get_params(timeframe)

    piv = pivot_lows(df, params["pivot_k"])

    A0 = rule_A0(df, piv, params)
    B0 = rule_B0(df, piv, params)
    C0 = rule_C0(df, piv, params)

    combined = A0 | B0 | C0

    low = df["Low"].values
    close = df["Close"].values
    n = len(df)

    # 1) Cluster filter: keep LOWEST low in each neighborhood
    cluster_mask = np.zeros(n, dtype=bool)
    last_idx = None
    last_low = None

    for idx in np.where(combined)[0]:
        if last_idx is None:
            cluster_mask[idx] = True
            last_idx = idx
            last_low = low[idx]
        else:
            if idx - last_idx < params["swing_window"]:
                # same cluster — keep lower low
                if low[idx] < last_low:
                    cluster_mask[last_idx] = False
                    cluster_mask[idx] = True
                    last_idx = idx
                    last_low = low[idx]
            else:
                cluster_mask[idx] = True
                last_idx = idx
                last_low = low[idx]

    # 2) Strong trend filter:
    # After 0 → no lower low AND final close higher than 0's close
    protect_n = 2 * params["swing_window"]
    final = np.zeros(n, dtype=bool)
    candidate_idxs = np.where(cluster_mask)[0]

    for idx in candidate_idxs:
        start_f = idx + 1
        end_f = min(n, idx + 1 + protect_n)

        if start_f >= end_f:
            # no enough future bars, we keep it
            final[idx] = True
            continue

        future_lows = low[start_f:end_f]
        future_closes = close[start_f:end_f]

        # If any lower low occurs → discard
        if future_lows.min() < low[idx]:
            continue

        # Require net up move: last close in window > close at 0
        if future_closes[-1] <= close[idx]:
            continue

        final[idx] = True

    df["Wave0"] = final
    return df


# ----------------------------------------------------------
# ----------------- WAVE 5 RULES (A, B, C) -----------------
# ----------------------------------------------------------
def rule_A5(df: pd.DataFrame, piv: np.ndarray, params: dict) -> np.ndarray:
    """Strict-ish Elliott style top: RSI top + reversal + future down."""
    n = len(df)
    A5 = np.zeros(n, dtype=bool)

    rsi = df["RSI_14"].values
    close = df["Close"].values
    future_n = params["future_n"]

    for i in range(2, n - future_n - 5):
        if not piv[i]:
            continue

        if np.isnan(rsi[i]) or np.isnan(rsi[i - 1]):
            continue

        # RSI overbought-ish AND turning down
        if rsi[i] < 60 or rsi[i] >= rsi[i - 1]:
            continue

        # Price turning down
        if close[i] >= close[i - 1]:
            continue

        # Future confirmation: after N bars, price below 5
        if close[i + future_n] >= close[i]:
            continue

        A5[i] = True

    return A5


def rule_B5(df: pd.DataFrame, piv: np.ndarray, params: dict) -> np.ndarray:
    """Pure swing high: highest in recent window + future price down."""
    n = len(df)
    B5 = np.zeros(n, dtype=bool)

    high = df["High"].values
    close = df["Close"].values
    swing = params["swing_window"]
    future_n = params["future_n"]

    for i in range(n - future_n):
        if not piv[i]:
            continue

        start = max(0, i - swing)
        if high[i] != high[start:i + 1].max():
            continue

        if close[i + future_n] >= close[i]:
            continue

        B5[i] = True

    return B5


def rule_C5(df: pd.DataFrame, piv: np.ndarray, params: dict) -> np.ndarray:
    """Fibonacci extension: 5 near 1.0–2.0 extension of prior swing."""
    n = len(df)
    C5 = np.zeros(n, dtype=bool)

    high = df["High"].values
    low = df["Low"].values
    fib_win = params["fib_window"]

    for i in range(n):
        if not piv[i]:
            continue

        start = max(0, i - fib_win)
        if start >= i:
            continue

        prev_low = low[start:i].min()
        if prev_low <= 0:
            continue

        ext = (high[i] - prev_low) / prev_low  # % move from swing low

        # rough terminal extension band
        if 0.95 <= ext <= 2.05:
            C5[i] = True

    return C5


# ----------------------------------------------------------
# ----------------- FINAL WAVE 5 (A OR B OR C) --------------
# ----------------------------------------------------------
def add_wave5(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    df = df.copy()
    params = get_params(timeframe)

    piv = pivot_highs(df, params["pivot_k"])

    A5 = rule_A5(df, piv, params)
    B5 = rule_B5(df, piv, params)
    C5 = rule_C5(df, piv, params)

    combined = A5 | B5 | C5

    high = df["High"].values
    close = df["Close"].values
    n = len(df)

    # 1) Cluster filter: keep HIGHEST high in each neighborhood
    cluster_mask = np.zeros(n, dtype=bool)
    last_idx = None
    last_high = None

    for idx in np.where(combined)[0]:
        if last_idx is None:
            cluster_mask[idx] = True
            last_idx = idx
            last_high = high[idx]
        else:
            if idx - last_idx < params["swing_window"]:
                # same cluster — keep higher high
                if high[idx] > last_high:
                    cluster_mask[last_idx] = False
                    cluster_mask[idx] = True
                    last_idx = idx
                    last_high = high[idx]
            else:
                cluster_mask[idx] = True
                last_idx = idx
                last_high = high[idx]

    # 2) Strong trend filter:
    # After 5 → no higher high AND final close lower than 5's close
    protect_n = 2 * params["swing_window"]
    final = np.zeros(n, dtype=bool)
    candidate_idxs = np.where(cluster_mask)[0]

    for idx in candidate_idxs:
        start_f = idx + 1
        end_f = min(n, idx + 1 + protect_n)

        if start_f >= end_f:
            # not enough future bars, we keep it
            final[idx] = True
            continue

        future_highs = high[start_f:end_f]
        future_closes = close[start_f:end_f]

        # If any higher high occurs → discard
        if future_highs.max() > high[idx]:
            continue

        # Require net down move: last close in window < close at 5
        if future_closes[-1] >= close[idx]:
            continue

        final[idx] = True

    df["Wave5"] = final
    return df


# ----------------------------------------------------------
# ----------------- CHART PLOTTING -------------------------
# ----------------------------------------------------------
def plot_chart(df: pd.DataFrame, title: str):
    if df is None or df.empty:
        return go.Figure()

    x = df.index

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.15, 0.15],
        vertical_spacing=0.03,
        specs=[
            [{"type": "candlestick"}],
            [{"type": "bar"}],
            [{"type": "scatter"}],
        ],
    )

    # Price
    fig.add_trace(
        go.Candlestick(
            x=x,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    # SMAs
    for s in [20, 50, 200]:
        col = f"SMA_{s}"
        if col in df.columns:
            fig.add_trace(
                go.Scatter(x=x, y=df[col], mode="lines", name=f"SMA {s}"),
                row=1,
                col=1,
            )

    # Wave 0
    if "Wave0" in df.columns:
        w0 = df[df["Wave0"]]
        if not w0.empty:
            fig.add_trace(
                go.Scatter(
                    x=w0.index,
                    y=w0["Low"] * 0.995,
                    mode="text",
                    text=["0"] * len(w0),
                    textposition="middle center",
                    name="Wave 0",
                ),
                row=1,
                col=1,
            )

    # Wave 5
    if "Wave5" in df.columns:
        w5 = df[df["Wave5"]]
        if not w5.empty:
            fig.add_trace(
                go.Scatter(
                    x=w5.index,
                    y=w5["High"] * 1.005,
                    mode="text",
                    text=["5"] * len(w5),
                    textposition="middle center",
                    name="Wave 5",
                ),
                row=1,
                col=1,
            )

    # Volume
    if "Volume" in df.columns:
        fig.add_trace(
            go.Bar(x=x, y=df["Volume"], name="Volume"),
            row=2,
            col=1,
        )

    # RSI
    if "RSI_14" in df.columns:
        fig.add_trace(
            go.Scatter(x=x, y=df["RSI_14"], mode="lines", name="RSI 14"),
            row=3,
            col=1,
        )
        fig.add_hrect(
            y0=30,
            y1=70,
            fillcolor="lightgray",
            opacity=0.2,
            line_width=0,
            row=3,
            col=1,
        )

    fig.update_layout(
        title=title,
        height=900,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=40, b=10),
    )

    return fig


# ----------------------------------------------------------
# --------------------------- UI ---------------------------
# ----------------------------------------------------------
st.sidebar.header("Settings")

default_tickers = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "AXISBANK.NS",
    "TATAMOTORS.NS",
    "ADANIENT.NS",
]

ticker = st.sidebar.selectbox("Select Symbol", default_tickers)
custom = st.sidebar.text_input("Or Custom Symbol (Yahoo code)")

if custom.strip():
    ticker = custom.strip()

tabs = st.tabs(["⏱ 1H", "📅 Daily", "📆 Weekly"])


# 1H
with tabs[0]:
    df_h = load_data(ticker, "60d", "1h")
    if df_h.empty:
        st.warning("No hourly data.")
    else:
        df_h = add_indicators(df_h)
        df_h = add_wave0(df_h, "1h")
        df_h = add_wave5(df_h, "1h")
        fig_h = plot_chart(df_h, f"{ticker} — 1H")
        st.plotly_chart(fig_h, use_container_width=True)

# Daily
with tabs[1]:
    df_d = load_data(ticker, "3y", "1d")
    if df_d.empty:
        st.warning("No daily data.")
    else:
        df_d = add_indicators(df_d)
        df_d = add_wave0(df_d, "1d")
        df_d = add_wave5(df_d, "1d")
        fig_d = plot_chart(df_d, f"{ticker} — Daily (3Y)")
        st.plotly_chart(fig_d, use_container_width=True)

# Weekly
with tabs[2]:
    df_w = load_data(ticker, "10y", "1wk")
    if df_w.empty:
        st.warning("No weekly data.")
    else:
        df_w = add_indicators(df_w)
        df_w = add_wave0(df_w, "1wk")
        df_w = add_wave5(df_w, "1wk")
        fig_w = plot_chart(df_w, f"{ticker} — Weekly (10Y)")
        st.plotly_chart(fig_w, use_container_width=True)
