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
st.caption("Hourly / Daily / Weekly — Candles + SMA + Volume + RSI + Wave 0 (Looser Book Rules)")


# ---------------- DATA LOADER ----------------
@st.cache_data(ttl=3600)
def load_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance.
    period: '60d', '3y', '10y', etc.
    interval: '1h', '1d', '1wk'
    """
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # Flatten MultiIndex columns if present (some yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna().copy()

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    return df


# ---------------- INDICATORS ----------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add SMA(20/50/200) + RSI(14).
    """
    df = df.copy()
    if df.empty or "Close" not in df.columns:
        return df

    # Simple Moving Averages
    for win in [20, 50, 200]:
        df[f"SMA_{win}"] = df["Close"].rolling(window=win).mean()

    # RSI(14)
    window = 14
    delta = df["Close"].diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    roll_up = gain.rolling(window=window, min_periods=window).mean()
    roll_down = loss.rolling(window=window, min_periods=window).mean()

    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    df["RSI_14"] = rsi

    return df


# ---------------- TIMEFRAME PARAMS FOR WAVE 0 ----------------
def get_wave0_params(timeframe: str) -> dict:
    """
    Slightly looser parameters for Wave 0 detection depending on timeframe.
    """
    if timeframe == "1h":
        return dict(
            pivot_k=3,
            min_impulse_pct=0.008,   # 0.8% move from 0 to Wave1
            min_w1_bars=6,
            max_w1_bars=80,
            min_zero_gap=20,
            future_n=7,              # next 7 candles close > close at 0
        )
    elif timeframe == "1wk":
        return dict(
            pivot_k=2,
            min_impulse_pct=0.10,    # 10% move from 0 to Wave1
            min_w1_bars=3,
            max_w1_bars=40,
            min_zero_gap=8,
            future_n=4,
        )
    else:  # "1d" default
        return dict(
            pivot_k=5,
            min_impulse_pct=0.03,    # 3% move from 0 to Wave1
            min_w1_bars=4,
            max_w1_bars=60,
            min_zero_gap=25,
            future_n=7,
        )


# ---------------- WAVE 0 DETECTION ONLY (LOOSER BOOK-INSPIRED) ----------------
def add_wave0_labels(df: pd.DataFrame, timeframe: str = "1d") -> pd.DataFrame:
    """
    Detect Wave 0 only, using a LOOSER but still "book-inspired" rule:

    1) Wave 0 is a pivot low (local minimum in a window).
    2) RSI is oversold-ish (< 40) and turning up vs previous bar.
    3) Price also turning up (close[i] > close[i-1]).
    4) After 0, price makes a decent impulse up (min %).
    5) That impulse breaks the previous pivot high (even slightly).
    6) Next N bars close above close at 0 (your old rule: "after few candles price > 0").
    7) If multiple 0s near each other, keep the lowest low in that cluster.
    """
    df = df.copy()
    df["Wave0"] = False

    needed_cols = {"Open", "High", "Low", "Close", "RSI_14"}
    if df.empty or not needed_cols.issubset(df.columns):
        return df

    params = get_wave0_params(timeframe)
    pivot_k = params["pivot_k"]
    min_impulse_pct = params["min_impulse_pct"]
    min_w1_bars = params["min_w1_bars"]
    max_w1_bars = params["max_w1_bars"]
    min_zero_gap = params["min_zero_gap"]
    future_n = params["future_n"]

    open_ = df["Open"].values
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    rsi = df["RSI_14"].values

    n = len(df)
    if n < 2 * pivot_k + 5:
        return df

    # ---------- 1) Identify pivot lows & pivot highs ----------
    pivot_low = np.zeros(n, dtype=bool)
    pivot_high = np.zeros(n, dtype=bool)

    for i in range(pivot_k, n - pivot_k):
        window_l = low[i - pivot_k: i + pivot_k + 1]
        if low[i] == window_l.min():
            pivot_low[i] = True

        window_h = high[i - pivot_k: i + pivot_k + 1]
        if high[i] == window_h.max():
            pivot_high[i] = True

    # Previous pivot high index for each bar
    prev_pivot_high_idx = -np.ones(n, dtype=int)
    last_ph = -1
    for i in range(n):
        prev_pivot_high_idx[i] = last_ph
        if pivot_high[i]:
            last_ph = i

    candidate_0 = np.zeros(n, dtype=bool)

    # ---------- 2) Check each pivot low as potential Wave 0 ----------
    for i in range(pivot_k, n - pivot_k):

        if not pivot_low[i]:
            continue

        L0 = low[i]

        # --- RSI + reversal check (looser) ---
        if np.isnan(rsi[i]):
            continue

        # oversold-ish
        if not (rsi[i] < 40):
            continue

        # RSI rising and price rising vs previous bar
        if i > 0:
            if not (rsi[i] > rsi[i - 1] and close[i] > close[i - 1]):
                continue
        else:
            continue

        # Future N bars: price should be higher than price at 0 (your rule)
        if i + future_n < n:
            if not (close[i + future_n] > close[i]):
                continue
        else:
            # not enough future data, skip
            continue

        # Need a previous pivot high to break
        ph = prev_pivot_high_idx[i]
        if ph == -1:
            continue

        prev_high = high[ph]

        # --- Impulse after 0 (Wave-1 proxy) ---
        start_f = i + 1
        end_f = min(i + max_w1_bars, n - 1)
        if end_f - start_f + 1 < min_w1_bars:
            continue

        fut_highs = high[start_f: end_f + 1]
        max_future_high = fut_highs.max()
        idx_loc = fut_highs.argmax()
        idx_wave1 = start_f + idx_loc

        # Time rule: at least min_w1_bars candles from 0
        if (idx_wave1 - i) < min_w1_bars:
            continue

        # Impulse strength %
        pct_up = (max_future_high - L0) / L0
        if pct_up < min_impulse_pct:
            continue

        # Break of previous swing high (even slightly)
        if max_future_high <= prev_high:
            continue

        # Passed all conditions → candidate Wave 0
        candidate_0[i] = True

    # ---------- 3) Cluster cleanup: keep lowest 0 in each neighborhood ----------
    idx_candidates = np.where(candidate_0)[0]
    final_wave0 = np.zeros(n, dtype=bool)

    last_kept = None
    last_low_val = None

    for idx in idx_candidates:
        this_low = low[idx]

        if last_kept is None:
            final_wave0[idx] = True
            last_kept = idx
            last_low_val = this_low
        else:
            if idx - last_kept < min_zero_gap:
                # same cluster, keep lower low
                if this_low < last_low_val:
                    final_wave0[last_kept] = False
                    final_wave0[idx] = True
                    last_kept = idx
                    last_low_val = this_low
                # else ignore
            else:
                final_wave0[idx] = True
                last_kept = idx
                last_low_val = this_low

    df["Wave0"] = final_wave0
    return df


# ---------------- BIG CHART ----------------
def make_tv_style_chart(df: pd.DataFrame, title: str):
    """
    TradingView-style layout:
    Row 1: Candlestick (larger) + SMA + Wave 0 labels
    Row 2: Volume
    Row 3: RSI
    """
    if df is None or df.empty:
        return go.Figure()

    x = df.index

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.15, 0.13],
        vertical_spacing=0.02,
        specs=[
            [{"type": "candlestick"}],
            [{"type": "bar"}],
            [{"type": "scatter"}],
        ],
    )

    # --- Candles ---
    fig.add_trace(
        go.Candlestick(
            x=x,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # --- SMAs ---
    for win, name in zip([20, 50, 200], ["SMA 20", "SMA 50", "SMA 200"]):
        col_name = f"SMA_{win}"
        if col_name in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=df[col_name],
                    mode="lines",
                    name=name,
                ),
                row=1,
                col=1,
            )

    # --- Wave 0 labels (below lows) ---
    if "Wave0" in df.columns:
        wave0_df = df[df["Wave0"]]
        if not wave0_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=wave0_df.index,
                    y=wave0_df["Low"] * 0.995,
                    mode="text",
                    text=["<b>0</b>"] * len(wave0_df),
                    textposition="middle center",
                    name="Wave 0",
                ),
                row=1,
                col=1,
            )

    # --- Volume ---
    if "Volume" in df.columns:
        fig.add_trace(
            go.Bar(
                x=x,
                y=df["Volume"],
                name="Volume",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # --- RSI ---
    if "RSI_14" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df["RSI_14"],
                mode="lines",
                name="RSI 14",
            ),
            row=3,
            col=1,
        )
        fig.add_hrect(
            y0=30,
            y1=70,
            line_width=0,
            fillcolor="LightGray",
            opacity=0.2,
            row=3,
            col=1,
        )

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        height=900,
        margin=dict(l=10, r=10, t=40, b=10),
    )

    fig.update_xaxes(showspikes=True)
    fig.update_yaxes(showspikes=True)

    return fig


# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Settings")

default_tickers = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS", "BHARTIARTL.NS", "BPCL.NS",
    "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS",
    "INFY.NS", "ITC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS",
    "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS",
    "SBILIFE.NS", "SBIN.NS", "SUNPHARMA.NS", "TATACONSUM.NS", "TATAMOTORS.NS",
    "TATASTEEL.NS", "TCS.NS", "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS",
]

ticker = st.sidebar.selectbox(
    "Select Symbol",
    options=default_tickers,
    index=0,
)

custom = st.sidebar.text_input(
    "Or type custom symbol (Yahoo format, e.g. COFORGE.NS, AAPL, TSLA)",
    value="",
)

if custom.strip():
    ticker = custom.strip()

st.sidebar.write("---")
st.sidebar.markdown(
    """
**Note:**  
- Data from Yahoo Finance  
- Only trading days/hours are returned  
  (no Saturdays, Sundays, or exchange holidays)
"""
)

# ---------------- MAIN CONTENT ----------------
tabs = st.tabs(["⏱ Hourly", "📅 Daily", "📆 Weekly"])

# Hourly
with tabs[0]:
    st.subheader(f"⏱ Hourly — last 60 days — {ticker}")
    df_h = load_data(ticker, period="60d", interval="1h")
    df_h = add_indicators(df_h)
    df_h = add_wave0_labels(df_h, timeframe="1h")

    if df_h.empty:
        st.warning("No hourly data found for this symbol.")
    else:
        fig_h = make_tv_style_chart(df_h, f"{ticker} — Hourly (60D)")
        st.plotly_chart(fig_h, use_container_width=True)

# Daily
with tabs[1]:
    st.subheader(f"📅 Daily — last 3 years — {ticker}")
    df_d = load_data(ticker, period="3y", interval="1d")
    df_d = add_indicators(df_d)
    df_d = add_wave0_labels(df_d, timeframe="1d")

    if df_d.empty:
        st.warning("No daily data found for this symbol.")
    else:
        fig_d = make_tv_style_chart(df_d, f"{ticker} — Daily (3Y)")
        st.plotly_chart(fig_d, use_container_width=True)

# Weekly
with tabs[2]:
    st.subheader(f"📆 Weekly — last 10 years — {ticker}")
    df_w = load_data(ticker, period="10y", interval="1wk")
    df_w = add_indicators(df_w)
    df_w = add_wave0_labels(df_w, timeframe="1wk")

    if df_w.empty:
        st.warning("No weekly data found for this symbol.")
    else:
        fig_w = make_tv_style_chart(df_w, f"{ticker} — Weekly (10Y)")
        st.plotly_chart(fig_w, use_container_width=True)
