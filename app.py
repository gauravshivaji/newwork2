import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="TradingView Style Multi-Timeframe Dashboard",
    layout="wide",
)

st.title("📈 TradingView-Style Stock Dashboard")
st.caption("Hourly / Daily / Weekly — Candles + SMA + Volume + RSI + PURE Wave 0 (A OR B OR C)")


# ---------------- DATA LOADER ----------------
@st.cache_data(ttl=3600)
def load_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

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
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    roll_up = gain.rolling(14).mean()
    roll_down = loss.rolling(14).mean()

    rs = roll_up / (roll_down.replace(0, np.nan))
    df["RSI_14"] = 100 - (100 / (1 + rs))

    return df


# ---------------- TIMEFRAME PARAMS ----------------
def get_params(tf):
    if tf == "1h":
        return dict(pivot_k=3, fib_window=80, swing_window=30, future_n=7)
    elif tf == "1wk":
        return dict(pivot_k=2, fib_window=30, swing_window=10, future_n=4)
    else:
        return dict(pivot_k=5, fib_window=60, swing_window=25, future_n=7)


# ---------------- PIVOT LOW FUNCTION ----------------
def get_pivot_lows(df, k):
    lows = df["Low"].values
    n = len(df)
    pivot_low = np.zeros(n, dtype=bool)
    for i in range(k, n - k):
        if lows[i] == lows[i - k:i + k + 1].min():
            pivot_low[i] = True
    return pivot_low


# ---------------- RULE A (STRICT ELLIOTT 0) ----------------
def rule_A(df, pivot_low, params):
    n = len(df)
    A = np.zeros(n, dtype=bool)

    low = df["Low"].values
    high = df["High"].values
    close = df["Close"].values
    rsi = df["RSI_14"].values
    k = params["pivot_k"]
    future_n = params["future_n"]

    for i in range(k, n - future_n - 10):
        if not pivot_low[i]:
            continue

        # RSI Oversold + rising
        if rsi[i] > 40 or rsi[i] <= rsi[i - 1]:
            continue

        # Price rising
        if close[i] <= close[i - 1]:
            continue

        # Wave-1 (impulse)
        future_high = high[i + 1:i + 1 + 20].max()
        if future_high <= high[i - 1]:
            continue

        # confirm long-term trend shift
        if not (close[i + future_n] > close[i]):
            continue

        A[i] = True

    return A


# ---------------- RULE B (PURE SWING LOW STRUCTURE) ----------------
def rule_B(df, pivot_low, params):
    n = len(df)
    B = np.zeros(n, dtype=bool)
    low = df["Low"].values
    close = df["Close"].values
    future_n = params["future_n"]

    for i in range(n - future_n):
        if not pivot_low[i]:
            continue
        # Pure swing low = low is lowest in last swing_window bars
        if low[i] != low[max(0, i - params["swing_window"]):i + 1].min():
            continue
        if close[i + future_n] <= close[i]:
            continue
        B[i] = True

    return B


# ---------------- RULE C (FIBONACCI END CORRECTION) ----------------
def rule_C(df, pivot_low, params):
    n = len(df)
    C = np.zeros(n, dtype=bool)
    low = df["Low"].values
    fib_win = params["fib_window"]

    for i in range(n):
        if not pivot_low[i]:
            continue

        start = max(0, i - fib_win)
        prev_high = df["High"].iloc[start:i].max()

        if prev_high <= 0:
            continue

        retr = (prev_high - low[i]) / prev_high

        # Fibonacci zones: 61.8%, 78.6%, 100%
        if 0.58 <= retr <= 1.05:
            C[i] = True

    return C


# ---------------- COMBINED PURE 0 = A OR B OR C ----------------
def add_wave0(df, timeframe):
    df = df.copy()
    params = get_params(timeframe)

    pivot_low = get_pivot_lows(df, params["pivot_k"])

    A = rule_A(df, pivot_low, params)
    B = rule_B(df, pivot_low, params)
    C = rule_C(df, pivot_low, params)

    combined = A | B | C

    # Clean cluster: keep lowest in cluster
    lows = df["Low"].values
    idxs = np.where(combined)[0]

    final = np.zeros(len(df), dtype=bool)

    last = None
    last_low = None
    min_gap = params["swing_window"]

    for idx in idxs:
        if last is None:
            final[idx] = True
            last = idx
            last_low = lows[idx]
        else:
            if idx - last < min_gap:
                # keep lower low
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


# ---------------- DRAW CHART ----------------
def plot_chart(df, title):
    x = df.index

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.15, 0.15],
        vertical_spacing=0.02,
        specs=[[{"type": "candlestick"}],
               [{"type": "bar"}],
               [{"type": "scatter"}]]
    )

    fig.add_trace(go.Candlestick(
        x=x, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"]
    ), row=1, col=1)

    # SMA
    for s in [20, 50, 200]:
        fig.add_trace(go.Scatter(
            x=x, y=df[f"SMA_{s}"], mode="lines", name=f"SMA{s}"
        ), row=1, col=1)

    # Wave 0
    w0 = df[df["Wave0"]]
    if not w0.empty:
        fig.add_trace(go.Scatter(
            x=w0.index,
            y=w0["Low"] * 0.995,
            mode="text",
            text=["<b>0</b>"] * len(w0),
            name="Wave 0",
            textposition="middle center"
        ), row=1, col=1)

    # Volume
    fig.add_trace(go.Bar(
        x=x, y=df["Volume"]
    ), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=x, y=df["RSI_14"]
    ), row=3, col=1)

    fig.add_hrect(
        y0=30, y1=70, fillcolor="lightgray",
        opacity=0.2, row=3, col=1
    )

    fig.update_layout(height=900, title=title)

    return fig


# ---------------- SIDEBAR ----------------
st.sidebar.header("Settings")

tickers = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "ADANIENT.NS","TATAMOTORS.NS","TATASTEEL.NS","SBIN.NS","AXISBANK.NS"
]

ticker = st.sidebar.selectbox("Ticker", tickers)

custom = st.sidebar.text_input("Custom symbol (optional)")
if custom.strip():
    ticker = custom.strip()

tabs = st.tabs(["1H", "Daily", "Weekly"])

# ---------------- HOURLY ----------------
with tabs[0]:
    df = load_data(ticker, "60d", "1h")
    df = add_indicators(df)
    df = add_wave0(df, "1h")
    st.plotly_chart(plot_chart(df, f"{ticker} — 1H"), use_container_width=True)

# ---------------- DAILY ----------------
with tabs[1]:
    df = load_data(ticker, "3y", "1d")
    df = add_indicators(df)
    df = add_wave0(df, "1d")
    st.plotly_chart(plot_chart(df, f"{ticker} — Daily"), use_container_width=True)

# ---------------- WEEKLY ----------------
with tabs[2]:
    df = load_data(ticker, "10y", "1wk")
    df = add_indicators(df)
    df = add_wave0(df, "1wk")
    st.plotly_chart(plot_chart(df, f"{ticker} — Weekly"), use_container_width=True)
