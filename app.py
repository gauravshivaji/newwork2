import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Elliott Pure 0 & 5", layout="wide")

st.title("📈 PURE Elliott Wave 0 & 5 Detection (A OR B OR C)")
st.caption("Strict Elliott + Pure Swing Structure + Fibonacci Terminal Logic")


# ---------------- DATA LOADER ----------------
@st.cache_data(ttl=3600)
def load_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna().copy()
    df.index = pd.to_datetime(df.index)
    return df


# ---------------- INDICATORS ----------------
def add_indicators(df):
    df = df.copy()
    for win in [20, 50, 200]:
        df[f"SMA_{win}"] = df["Close"].rolling(win).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    roll_up = gain.rolling(14).mean()
    roll_down = loss.rolling(14).mean()

    rs = roll_up / (roll_down.replace(0, np.nan))
    df["RSI_14"] = 100 - (100 / (1 + rs))

    return df


# ---------------- TIMEFRAME PARAMS ----------------
def get_params(tf):
    if tf == "1h":
        return dict(pivot_k=3, swing_window=25, fib_window=70, future_n=7)
    if tf == "1wk":
        return dict(pivot_k=2, swing_window=10, fib_window=30, future_n=3)
    return dict(pivot_k=5, swing_window=20, fib_window=50, future_n=5)


# ---------------- PIVOT FINDER ----------------
def pivot_highs(df, k):
    highs = df["High"].values
    n = len(df)
    piv = np.zeros(n, dtype=bool)
    for i in range(k, n - k):
        if highs[i] == highs[i - k:i + k + 1].max():
            piv[i] = True
    return piv

def pivot_lows(df, k):
    lows = df["Low"].values
    n = len(df)
    piv = np.zeros(n, dtype=bool)
    for i in range(k, n - k):
        if lows[i] == lows[i - k:i + k + 1].min():
            piv[i] = True
    return piv


# --------------------------------------------------
# ---------------- RULE A for Wave-5 ----------------
# STRICT ELLIOTT BOOK
# --------------------------------------------------
def rule_A5(df, piv5, params):
    n = len(df)
    A5 = np.zeros(n, dtype=bool)

    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    rsi = df["RSI_14"].values
    future_n = params["future_n"]

    for i in range(n - future_n - 10):
        if not piv5[i]:
            continue

        # RSI > 70 and falling (overbought + reversal)
        if rsi[i] < 70 or rsi[i] >= rsi[i - 1]:
            continue

        # Price falling
        if close[i] >= close[i - 1]:
            continue

        # Terminal impulse check
        # Lookback strong upward move
        hist_high = high[max(0, i - 20):i].max()
        if high[i] < hist_high:
            continue

        # Post-5 decline confirmation
        if close[i + future_n] >= close[i]:
            continue

        A5[i] = True
    return A5


# --------------------------------------------------
# ---------------- RULE B for Wave-5 ----------------
# PURE SWING STRUCTURAL HIGH
# --------------------------------------------------
def rule_B5(df, piv5, params):
    n = len(df)
    B5 = np.zeros(n, dtype=bool)

    high = df["High"].values
    close = df["Close"].values
    future_n = params["future_n"]
    swing_window = params["swing_window"]

    for i in range(n - future_n):
        if not piv5[i]:
            continue

        # Highest high in the recent swing window
        if high[i] != high[i - swing_window:i + 1].max():
            continue

        # Price should fall after
        if close[i + future_n] >= close[i]:
            continue

        B5[i] = True
    return B5


# --------------------------------------------------
# ---------------- RULE C for Wave-5 ----------------
# FIBONACCI TERMINAL EXTENSION
# --------------------------------------------------
def rule_C5(df, piv5, params):
    n = len(df)
    C5 = np.zeros(n, dtype=bool)

    high = df["High"].values
    low = df["Low"].values
    fib_window = params["fib_window"]

    for i in range(n):
        if not piv5[i]:
            continue

        # Lookback a swing low to measure extension
        start = max(0, i - fib_window)
        prev_low = low[start:i].min()
        move = high[i] - prev_low
        if prev_low <= 0:
            continue

        ext = move / prev_low

        # Fibonacci terminal zones (1.0, 1.272, 1.618, 2.0)
        if 0.95 <= ext <= 2.05:
            C5[i] = True

    return C5


# --------------------------------------------------
# ---------------- COMBINE 0 AND 5 ----------------
# --------------------------------------------------
def add_wave0(df, timeframe):
    df = df.copy()
    params = get_params(timeframe)

    piv0 = pivot_lows(df, params["pivot_k"])

    # ---- RULE A0, B0, C0 reused from earlier code ----
    # For simplicity, reusing pure-swing rule for 0
    lows = df["Low"].values
    close = df["Close"].values

    A0 = np.zeros(len(df), dtype=bool)
    B0 = np.zeros(len(df), dtype=bool)
    C0 = np.zeros(len(df), dtype=bool)

    future_n = params["future_n"]

    # Simple pure-0 logic from previous version
    for i in range(len(df) - future_n):
        if not piv0[i]:
            continue

        # B0: swing lowest
        if lows[i] == lows[i - params["swing_window"]:i + 1].min():
            if close[i + future_n] > close[i]:
                B0[i] = True

        # A0: RSI bottom + reversal
        if df["RSI_14"].iloc[i] < 40 and df["RSI_14"].iloc[i] > df["RSI_14"].iloc[i - 1]:
            if close[i] > close[i - 1]:
                A0[i] = True

        # C0: Fibonacci retracement bottom
        start = max(0, i - params["fib_window"])
        prev_high = df["High"].iloc[start:i].max()
        if prev_high > 0:
            retr = (prev_high - lows[i]) / prev_high
            if 0.58 <= retr <= 1.05:
                C0[i] = True

    combined = A0 | B0 | C0

    # Cluster filter
    final = np.zeros(len(df), dtype=bool)
    last = None
    last_low = None

    for idx in np.where(combined)[0]:
        if last is None:
            final[idx] = True
            last = idx
            last_low = lows[idx]
        else:
            if idx - last < params["swing_window"]:
                if lows[idx] < last_low:
                    final[last] = False
                    final[idx] = True
                    last = idx
                    last_low = lows[idx]
            else:
                final[idx] = True
                last = idx
                last_low = lows[idx]

    df["Wave0"] = final
    return df


def add_wave5(df, timeframe):
    df = df.copy()
    params = get_params(timeframe)

    piv5 = pivot_highs(df, params["pivot_k"])

    A5 = rule_A5(df, piv5, params)
    B5 = rule_B5(df, piv5, params)
    C5 = rule_C5(df, piv5, params)

    combined = A5 | B5 | C5

    highs = df["High"].values
    final = np.zeros(len(df), dtype=bool)

    last = None
    last_high = None

    for idx in np.where(combined)[0]:
        if last is None:
            final[idx] = True
            last = idx
            last_high = highs[idx]
        else:
            if idx - last < params["swing_window"]:
                if highs[idx] > last_high:
                    final[last] = False
                    final[idx] = True
                    last = idx
                    last_high = highs[idx]
            else:
                final[idx] = True
                last = idx
                last_high = highs[idx]

    df["Wave5"] = final
    return df


# --------------------------------------------------
# ---------------- CHART ----------------
# --------------------------------------------------
def plot_chart(df, title):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.15, 0.15],
        vertical_spacing=0.03,
        specs=[
            [{"type": "candlestick"}],
            [{"type": "bar"}],
            [{"type": "scatter"}]
        ]
    )

    x = df.index

    fig.add_trace(go.Candlestick(
        x=x,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price"
    ), row=1, col=1)

    # SMAs
    for s in [20, 50, 200]:
        fig.add_trace(go.Scatter(
            x=x, y=df[f"SMA_{s}"], mode="lines", name=f"SMA{s}"
        ), row=1, col=1)

    # Wave 0 markers
    w0 = df[df["Wave0"]]
    if not w0.empty:
        fig.add_trace(go.Scatter(
            x=w0.index,
            y=w0["Low"] * 0.995,
            text=["0"] * len(w0),
            mode="text",
            name="Wave0",
            textposition="middle center"
        ), row=1, col=1)

    # Wave 5 markers
    w5 = df[df["Wave5"]]
    if not w5.empty:
        fig.add_trace(go.Scatter(
            x=w5.index,
            y=w5["High"] * 1.005,
            text=["5"] * len(w5),
            mode="text",
            name="Wave5",
            textposition="middle center"
        ), row=1, col=1)

    # Volume
    fig.add_trace(go.Bar(x=x, y=df["Volume"]), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=x, y=df["RSI_14"], mode="lines"), row=3, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="gray", opacity=0.1, row=3, col=1)

    fig.update_layout(height=900, title=title)
    return fig


# --------------------------------------------------
# ---------------- SIDEBAR ----------------
# --------------------------------------------------
st.sidebar.header("Settings")

tickers = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "ICICIBANK.NS",
    "HDFCBANK.NS", "SBIN.NS", "AXISBANK.NS", "MARUTI.NS"
]

ticker = st.sidebar.selectbox("Select Symbol", tickers)
custom = st.sidebar.text_input("Custom Symbol (optional)")
if custom.strip():
    ticker = custom.strip()

tabs = st.tabs(["1H", "Daily", "Weekly"])

# 1H
with tabs[0]:
    df = load_data(ticker, "60d", "1h")
    df = add_indicators(df)
    df = add_wave0(df, "1h")
    df = add_wave5(df, "1h")
    st.plotly_chart(plot_chart(df, f"{ticker} — 1H"), use_container_width=True)

# DAILY
with tabs[1]:
    df = load_data(ticker, "3y", "1d")
    df = add_indicators(df)
    df = add_wave0(df, "1d")
    df = add_wave5(df, "1d")
    st.plotly_chart(plot_chart(df, f"{ticker} — Daily"), use_container_width=True)

# WEEKLY
with tabs[2]:
    df = load_data(ticker, "10y", "1wk")
    df = add_indicators(df)
    df = add_wave0(df, "1wk")
    df = add_wave5(df, "1wk")
    st.plotly_chart(plot_chart(df, f"{ticker} — Weekly"), use_container_width=True)
