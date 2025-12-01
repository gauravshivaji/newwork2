import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Elliott Wave 0 & 5 — Pure A/B/C Rules", layout="wide")
st.title("📈 PURE Elliott Wave 0 & 5 Detection (A OR B OR C)")
st.caption("Strict Elliott + Structural Swing + Fibonacci Terminal Logic")


# ---------------- DATA LOADER ----------------
@st.cache_data(ttl=3600)
def load_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    return df


# ---------------- INDICATORS ----------------
def add_indicators(df):
    df = df.copy()

    # SMAs
    for win in [20, 50, 200]:
        df[f"SMA_{win}"] = df["Close"].rolling(win).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    roll_up = gain.rolling(14).mean()
    roll_down = loss.rolling(14).mean()

    rs = roll_up / roll_down.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    return df


# ---------------- TIMEFRAME PARAMETERS ----------------
def get_params(tf):
    if tf == "1h":
        return dict(pivot_k=3, swing_window=25, fib_window=80, future_n=7)
    if tf == "1wk":
        return dict(pivot_k=2, swing_window=10, fib_window=30, future_n=3)
    return dict(pivot_k=5, swing_window=20, fib_window=60, future_n=5)


# ---------------- PIVOT FUNCTIONS ----------------
def pivot_lows(df, k):
    lows = df["Low"].values
    n = len(df)
    piv = np.zeros(n, dtype=bool)
    for i in range(k, n - k):
        if lows[i] == lows[i - k:i + k + 1].min():
            piv[i] = True
    return piv


def pivot_highs(df, k):
    highs = df["High"].values
    n = len(df)
    piv = np.zeros(n, dtype=bool)
    for i in range(k, n - k):
        if highs[i] == highs[i - k:i + k + 1].max():
            piv[i] = True
    return piv


# ----------------------------------------------------------
# ----------------- RULE A (Strict Elliott Bottom) ----------
# ----------------------------------------------------------
def rule_A0(df, piv, params):
    A0 = np.zeros(len(df), dtype=bool)
    rsi = df["RSI_14"].values
    close = df["Close"].values
    future_n = params["future_n"]

    for i in range(2, len(df) - future_n - 5):

        if not piv[i]:
            continue

        # RSI oversold + rising
        if rsi[i] > 40 or rsi[i] <= rsi[i - 1]:
            continue

        # price rising
        if close[i] <= close[i - 1]:
            continue

        # future confirmation
        if close[i + future_n] <= close[i]:
            continue

        A0[i] = True

    return A0


# ----------------------------------------------------------
# ----------------- RULE B (Pure Swing Bottom) -------------
# ----------------------------------------------------------
def rule_B0(df, piv, params):
    B0 = np.zeros(len(df), dtype=bool)
    low = df["Low"].values
    close = df["Close"].values
    swing = params["swing_window"]
    future_n = params["future_n"]

    for i in range(len(df) - future_n):
        if not piv[i]:
            continue

        start = max(0, i - swing)

        if low[i] != low[start:i + 1].min():
            continue

        if close[i + future_n] <= close[i]:
            continue

        B0[i] = True

    return B0


# ----------------------------------------------------------
# ----------------- RULE C (Fibonacci Retracement) ----------
# ----------------------------------------------------------
def rule_C0(df, piv, params):
    C0 = np.zeros(len(df), dtype=bool)
    low = df["Low"].values
    fib_win = params["fib_window"]

    for i in range(len(df)):
        if not piv[i]:
            continue

        start = max(0, i - fib_win)
        prev_high = df["High"].iloc[start:i].max()

        if prev_high <= 0:
            continue

        retr = (prev_high - low[i]) / prev_high

        if 0.58 <= retr <= 1.05:
            C0[i] = True

    return C0


# ----------------------------------------------------------
# ----------------- FINAL WAVE 0 (A OR B OR C) -------------
# ----------------------------------------------------------
def add_wave0(df, timeframe):
    df = df.copy()
    params = get_params(timeframe)

    piv = pivot_lows(df, params["pivot_k"])

    # --- A/B/C rules as before ---
    A0 = rule_A0(df, piv, params)
    B0 = rule_B0(df, piv, params)
    C0 = rule_C0(df, piv, params)

    combined = A0 | B0 | C0

    lows = df["Low"].values
    close = df["Close"].values
    n = len(df)

    # ---------- 1) Cluster filter: keep lowest in each neighborhood ----------
    cluster_mask = np.zeros(n, dtype=bool)

    last = None
    last_low = None

    for idx in np.where(combined)[0]:
        if last is None:
            cluster_mask[idx] = True
            last = idx
            last_low = lows[idx]
        else:
            if idx - last < params["swing_window"]:
                # same neighborhood → keep LOWER low
                if lows[idx] < last_low:
                    cluster_mask[last] = False
                    cluster_mask[idx] = True
                    last = idx
                    last_low = lows[idx]
            else:
                cluster_mask[idx] = True
                last = idx
                last_low = lows[idx]

    # ---------- 2) Strong trend filter: AFTER 0, no downtrend ----------
    # - In next protect_n bars:
    #   • no lower low than 0
    #   • final close > close at 0  (net up, not down)
    protect_n = 2 * params["swing_window"]  # stricter window

    final = np.zeros(n, dtype=bool)
    candidate_idxs = np.where(cluster_mask)[0]

    for idx in candidate_idxs:
        start_f = idx + 1
        end_f = min(n, idx + 1 + protect_n)

        # Not enough future data → keep it (can't disprove)
        if start_f >= end_f:
            final[idx] = True
            continue

        future_lows = lows[start_f:end_f]
        future_closes = close[start_f:end_f]

        # If any lower low occurs → this wasn't the final bottom
        if future_lows.min() < lows[idx]:
            continue

        # If final close is not above 0's close → treat as not real uptrend
        if future_closes[-1] <= close[idx]:
            continue

        final[idx] = True

    df["Wave0"] = final
    return df


# ----------------------------------------------------------
# ----------------- RULE A (Strict Elliott Top) -------------
# ----------------------------------------------------------
def add_wave5(df, timeframe):
    df = df.copy()
    params = get_params(timeframe)

    piv = pivot_highs(df, params["pivot_k"])

    A5 = rule_A5(df, piv, params)
    B5 = rule_B5(df, piv, params)
    C5 = rule_C5(df, piv, params)

    combined = A5 | B5 | C5

    highs = df["High"].values
    close = df["Close"].values
    n = len(df)

    # ---------- 1) Cluster filter: keep HIGHEST in each neighborhood ----------
    cluster_mask = np.zeros(n, dtype=bool)

    last = None
    last_high = None

    for idx in np.where(combined)[0]:
        if last is None:
            cluster_mask[idx] = True
            last = idx
            last_high = highs[idx]
        else:
            if idx - last < params["swing_window"]:
                # same neighborhood → keep HIGHER high
                if highs[idx] > last_high:
                    cluster_mask[last] = False
                    cluster_mask[idx] = True
                    last = idx
                    last_high = highs[idx]
            else:
                cluster_mask[idx] = True
                last = idx
                last_high = highs[idx]

    # ---------- 2) Strong trend filter: AFTER 5, no uptrend ----------
    # - In next protect_n bars:
    #   • no higher high than 5
    #   • final close < close at 5  (net down, not up)
    protect_n = 2 * params["swing_window"]

    final = np.zeros(n, dtype=bool)
    candidate_idxs = np.where(cluster_mask)[0]

    for idx in candidate_idxs:
        start_f = idx + 1
        end_f = min(n, idx + 1 + protect_n)

        # Not enough future data → keep it (can't disprove)
        if start_f >= end_f:
            final[idx] = True
            continue

        future_highs = highs[start_f:end_f]
        future_closes = close[start_f:end_f]

        # If any higher high occurs → this wasn't the real top
        if future_highs.max() > highs[idx]:
            continue

        # If final close is not BELOW 5's close → still up / sideways
        if future_closes[-1] >= close[idx]:
            continue

        final[idx] = True

    df["Wave5"] = final
    return df



# ----------------------------------------------------------
# ----------------- RULE B (Pure Swing Top) ----------------
# ----------------------------------------------------------
def rule_B5(df, piv, params):
    B5 = np.zeros(len(df), dtype=bool)
    high = df["High"].values
    close = df["Close"].values
    swing = params["swing_window"]
    future_n = params["future_n"]

    for i in range(len(df) - future_n):

        if not piv[i]:
            continue

        start = max(0, i - swing)

        if high[i] != high[start:i + 1].max():
            continue

        if close[i + future_n] >= close[i]:
            continue

        B5[i] = True

    return B5


# ----------------------------------------------------------
# ----------------- RULE C (Fibonacci Extensions) ----------
# ----------------------------------------------------------
def rule_C5(df, piv, params):
    C5 = np.zeros(len(df), dtype=bool)
    high = df["High"].values
    low = df["Low"].values
    fib_win = params["fib_window"]

    for i in range(len(df)):

        if not piv[i]:
            continue

        start = max(0, i - fib_win)
        prev_low = low[start:i].min()

        if prev_low <= 0:
            continue

        ext = (high[i] - prev_low) / prev_low

        # terminal fib zones
        if 0.95 <= ext <= 2.05:
            C5[i] = True

    return C5


# ----------------------------------------------------------
# ----------------- FINAL WAVE 5 (A OR B OR C) --------------
# ----------------------------------------------------------
def add_wave5(df, timeframe):
    df = df.copy()
    params = get_params(timeframe)

    piv = pivot_highs(df, params["pivot_k"])

    A5 = rule_A5(df, piv, params)
    B5 = rule_B5(df, piv, params)
    C5 = rule_C5(df, piv, params)

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


# ----------------------------------------------------------
# --------------------------- CHART -------------------------
# ----------------------------------------------------------
def plot_chart(df, title):
    x = df.index

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.15, 0.15],
        vertical_spacing=0.03,
        specs=[
            [{"type": "candlestick"}],
            [{"type": "bar"}],
            [{"type": "scatter"}]
        ]
    )

    # Price
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

    # Wave 0
    w0 = df[df["Wave0"]]
    if not w0.empty:
        fig.add_trace(go.Scatter(
            x=w0.index,
            y=w0["Low"] * 0.995,
            text=["0"] * len(w0),
            mode="text",
            name="Wave 0",
            textposition="middle center"
        ), row=1, col=1)

    # Wave 5
    w5 = df[df["Wave5"]]
    if not w5.empty:
        fig.add_trace(go.Scatter(
            x=w5.index,
            y=w5["High"] * 1.005,
            text=["5"] * len(w5),
            mode="text",
            name="Wave 5",
            textposition="middle center"
        ), row=1, col=1)

    # Volume
    fig.add_trace(go.Bar(x=x, y=df["Volume"]), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=x, y=df["RSI_14"], mode="lines"), row=3, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="gray", opacity=0.1, row=3, col=1)

    fig.update_layout(height=900, title=title)
    return fig


# ----------------------------------------------------------
# --------------------------- SIDEBAR ------------------------
# ----------------------------------------------------------
st.sidebar.header("Settings")

tickers = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "ICICIBANK.NS", "HDFCBANK.NS",
    "SBIN.NS", "AXISBANK.NS", "TATAMOTORS.NS", "ADANIENT.NS"
]

ticker = st.sidebar.selectbox("Select Symbol", tickers)
custom = st.sidebar.text_input("Custom Symbol")
if custom.strip():
    ticker = custom.strip()

tabs = st.tabs(["1H", "Daily", "Weekly"])


# -------------------- HOURLY --------------------
with tabs[0]:
    df = load_data(ticker, "60d", "1h")
    df = add_indicators(df)
    df = add_wave0(df, "1h")
    df = add_wave5(df, "1h")
    st.plotly_chart(plot_chart(df, f"{ticker} — 1H"), use_container_width=True)


# -------------------- DAILY ---------------------
with tabs[1]:
    df = load_data(ticker, "3y", "1d")
    df = add_indicators(df)
    df = add_wave0(df, "1d")
    df = add_wave5(df, "1d")
    st.plotly_chart(plot_chart(df, f"{ticker} — Daily"), use_container_width=True)


# -------------------- WEEKLY --------------------
with tabs[2]:
    df = load_data(ticker, "10y", "1wk")
    df = add_indicators(df)
    df = add_wave0(df, "1wk")
    df = add_wave5(df, "1wk")
    st.plotly_chart(plot_chart(df, f"{ticker} — Weekly"), use_container_width=True)
