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
st.caption("Hourly / Daily / Weekly — Candles + SMA + Volume + RSI + Wave 0 (Book Rules)")


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
    Parameters for Wave 0 detection depending on timeframe.
    - pivot_k: window for local pivot lows/highs
    - min_impulse_pct: minimum rally % from 0 to future high (Wave 1 strength)
    - min_break_pct: minimum breakout % above previous swing high
    - min_w1_bars: minimum bars in the post-0 rally
    - max_w1_bars: maximum bars to search for that rally
    - min_zero_gap: minimum distance between 0s; if closer, keep lower low
    """
    if timeframe == "1h":
        return dict(
            pivot_k=3,
            min_impulse_pct=0.015,   # 1.5% move from 0 to Wave1
            min_break_pct=0.003,     # 0.3% above previous swing high
            min_w1_bars=8,
            max_w1_bars=80,
            min_zero_gap=30,
        )
    elif timeframe == "1wk":
        return dict(
            pivot_k=2,
            min_impulse_pct=0.20,    # 20% move from 0 to Wave1
            min_break_pct=0.02,      # 2% above previous swing high
            min_w1_bars=3,
            max_w1_bars=40,
            min_zero_gap=10,
        )
    else:  # "1d" default
        return dict(
            pivot_k=5,
            min_impulse_pct=0.05,    # 5% move from 0 to Wave1
            min_break_pct=0.01,      # 1% above previous swing high
            min_w1_bars=5,
            max_w1_bars=60,
            min_zero_gap=30,
        )


# ---------------- WAVE 0 DETECTION ONLY (BOOK RULE STYLE) ----------------
def add_wave0_labels(df: pd.DataFrame, timeframe: str = "1d") -> pd.DataFrame:
    """
    Detect Wave 0 only, using "book style" rules:

    1) Wave 0 is a major pivot low.
    2) It is the lowest point before a strong new impulsive rally (proxy for Wave 1).
    3) That rally:
       - Lasts at least min_w1_bars candles,
       - Makes a % move (min_impulse_pct) above the Wave 0 low,
       - Breaks above the last pivot high by at least min_break_pct.
    4) Momentum reversal at Wave 0:
       - RSI oversold (< 30), and
       - Either bullish divergence vs previous pivot low OR hammer-ish candle.
    5) If multiple 0s are within min_zero_gap candles, keep only the lowest low.
    """
    df = df.copy()
    df["Wave0"] = False

    needed_cols = {"Open", "High", "Low", "Close", "RSI_14"}
    if df.empty or not needed_cols.issubset(df.columns):
        return df

    params = get_wave0_params(timeframe)
    pivot_k = params["pivot_k"]
    min_impulse_pct = params["min_impulse_pct"]
    min_break_pct = params["min_break_pct"]
    min_w1_bars = params["min_w1_bars"]
    max_w1_bars = params["max_w1_bars"]
    min_zero_gap = params["min_zero_gap"]

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
        window_l = low[i - pivot_k : i + pivot_k + 1]
        if low[i] == window_l.min():
            pivot_low[i] = True

        window_h = high[i - pivot_k : i + pivot_k + 1]
        if high[i] == window_h.max():
            pivot_high[i] = True

    # Precompute previous pivot low & previous pivot high indices
    prev_pivot_low_idx = -np.ones(n, dtype=int)
    prev_pivot_high_idx = -np.ones(n, dtype=int)

    last_pl = -1
    last_ph = -1
    for i in range(n):
        prev_pivot_low_idx[i] = last_pl
        prev_pivot_high_idx[i] = last_ph
        if pivot_low[i]:
            last_pl = i
        if pivot_high[i]:
            last_ph = i

    candidate_0 = np.zeros(n, dtype=bool)

    # ---------- 2) Check each pivot low as potential Wave 0 ----------
    for i in range(pivot_k, n - pivot_k):

        if not pivot_low[i]:
            continue

        L0 = low[i]

        # --- Rule 4: Reversal / RSI confirmation ---
        rsi_i = rsi[i]
        if np.isnan(rsi_i) or rsi_i >= 35:  # want oversold-ish
            continue

        # Bullish divergence vs previous pivot low (if exists)
        div_ok = False
        pl = prev_pivot_low_idx[i]
        if pl != -1 and not np.isnan(rsi[pl]):
            if low[i] < low[pl] and rsi[i] > rsi[pl]:
                div_ok = True

        # Simple hammer / bullish candle pattern
        candle_ok = False
        if not np.isnan(open_[i]) and not np.isnan(close[i]):
            body = abs(close[i] - open_[i])
            lower_wick = min(open_[i], close[i]) - L0
            # Hammer-style: long lower wick vs body + bullish close
            if close[i] > open_[i] and lower_wick > 1.5 * body:
                candle_ok = True

        # Need RSI oversold AND (divergence OR candle)
        if not (rsi_i < 30 and (div_ok or candle_ok)):
            continue

        # --- Rule 3, 6, 7, 8: Strong impulsive rally after 0 ("Wave 1") ---
        # Get previous pivot high (resistance to break)
        ph = prev_pivot_high_idx[i]
        if ph == -1:
            # no structure to break, skip
            continue

        prev_high = high[ph]

        # Future window where Wave 1 can develop
        start_f = i + 1
        end_f = min(i + max_w1_bars, n - 1)
        if end_f - start_f + 1 < min_w1_bars:
            continue

        fut_highs = high[start_f : end_f + 1]
        max_future_high = fut_highs.max()
        idx_loc = fut_highs.argmax()
        idx_wave1 = start_f + idx_loc  # index of highest point in that window

        # Time rule: at least min_w1_bars candles from 0
        if (idx_wave1 - i) < min_w1_bars:
            continue

        # Impulse size in %
        pct_up = (max_future_high - L0) / L0
        if pct_up < min_impulse_pct:
            continue

        # Break of previous swing high
        if max_future_high < prev_high * (1.0 + min_break_pct):
            continue

        # Optional: check that the move into 0 was downward (previous few closes)
        if i >= 3:
            if not (close[i] < close[i - 1] <= close[i - 2]):
                # we can relax this, but it's nice to have
                pass

        # If all checks pass, mark candidate as Wave 0
        candidate_0[i] = True

    # ---------- 3) If multiple 0s close together, keep the lowest low ----------
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
                # same cluster, keep the lower low
                if this_low < last_low_val:
                    final_wave0[last_kept] = False
                    final_wave0[idx] = True
                    last_kept = idx
                    last_low_val = this_low
                # else ignore this one
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
