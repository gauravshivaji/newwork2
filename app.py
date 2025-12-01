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
st.caption("Hourly / Daily / Weekly — Candles + SMA + Volume + RSI + Wave 0–5 + B")


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


# ---------------- TIMEFRAME-SPECIFIC FIB RULES ----------------
def get_wave_params(timeframe: str) -> dict:
    """
    Return Fibonacci + structural parameters depending on timeframe.
    timeframe: '1h', '1d', '1wk'
    """
    # Defaults for daily if something weird comes
    params_daily = dict(
        retr2_min=0.50,
        retr2_max=0.786,
        retr2_ideal=0.618,
        ext3_min=1.2,
        ext3_max=2.6,
        ext3_ideal=1.618,
        retr4_min=0.236,
        retr4_max=0.5,
        retr4_ideal=0.382,
        min_05_bars=10,
    )

    if timeframe == "1h":
        return dict(
            retr2_min=0.45,
            retr2_max=0.82,
            retr2_ideal=0.618,
            ext3_min=1.0,
            ext3_max=2.8,
            ext3_ideal=1.618,
            retr4_min=0.20,
            retr4_max=0.42,
            retr4_ideal=0.382,
            min_05_bars=15,
        )
    elif timeframe == "1wk":
        return dict(
            retr2_min=0.50,
            retr2_max=0.786,
            retr2_ideal=0.618,
            ext3_min=1.2,
            ext3_max=2.8,
            ext3_ideal=1.618,
            retr4_min=0.236,
            retr4_max=0.382,
            retr4_ideal=0.382,
            min_05_bars=5,
        )
    else:
        return params_daily


# ---------------- WAVE 0–5 + B DETECTION ----------------
def add_wave_labels(df: pd.DataFrame, timeframe: str = "1d") -> pd.DataFrame:
    """
    Detect Wave 0, 1, 2, 3, 4, 5 and B using:
      - Your custom rules for 0 and 5 (Option C full)
      - Book Fibonacci rules for 1-2-3-4 (timeframe-specific)

    Wave 0:
      - RSI < 20, RSI rising, Close rising  OR  structural low
      - Future (7 bars) price higher than at 0
      - If two 0s within 30 bars, keep LOWER low
      - If 0,0 appear consecutively in time, drop FIRST
      - If 0 appears within 30 bars AFTER a 5 → drop that 0

    Wave 5:
      - RSI > 80, RSI falling, Close falling  OR  structural high
      - Future (7 bars) price lower than at 5
      - If two 5s within 30 bars, keep HIGHER high
      - If 5,5 appear consecutively in time, drop FIRST

    B wave:
      - Between two 0s within 30 bars, where 2nd 0 < 1st 0 (price falls)
      - Do NOT mark 5 between them
      - Mark B at highest high between those two 0s

    Inside each 0→5:
      Wave 1:
        - Swing high after 0, before 5 (pivot high)
      Wave 2 (Fib retracement of Wave 1):
        - Swing low after 1, before 5
        - Retraces Fib-band (tf-specific) of Wave 1
        - L2 > L0
        - Choose retracement closest to ideal (e.g., 0.618)
      Wave 3 (Fib extension of Wave 1 from 2):
        - Pivot high after 2, before 5
        - Extension between ext3_min & ext3_max
        - Choose closest to ext3_ideal
      Wave 4 (Fib retracement of Wave 3):
        - Pivot low after 3, before 5
        - Retracement between retr4_min & retr4_max
        - L4 > H1 (no overlap with wave 1)
        - Choose closest to retr4_ideal
    """
    df = df.copy()

    # init columns
    for col in ["Wave0", "Wave1", "Wave2", "Wave3", "Wave4", "Wave5", "WaveB"]:
        df[col] = False

    needed_cols = {"RSI_14", "Low", "High", "Close"}
    if df.empty or not needed_cols.issubset(df.columns):
        return df

    rsi = df["RSI_14"]
    close = df["Close"]
    low = df["Low"]
    high = df["High"]

    n = len(df)
    eps = 1e-8
    min_gap_same_type = 30   # candles between same-type (for 0/5)
    pivot_k = 2              # window for swing high/low
    future_n = 7             # your "after 7 candles price should be higher/lower"
    params = get_wave_params(timeframe)
    min_05_bars = params["min_05_bars"]

    # ---------------- WAVE 0 (BOTTOM) ----------------
    # RSI-based condition: oversold + RSI & price starting to rise
    cond_rsi_price_0 = (rsi < 20) & (rsi > rsi.shift(1)) & (close > close.shift(1))

    # Structural extreme low (100-bar lookback/forward)
    rolling_min_low = low.rolling(window=101, center=True, min_periods=1).min()
    cond_extreme_low = low <= (rolling_min_low + eps)

    base_zero = cond_rsi_price_0 | cond_extreme_low

    # Future 7 bars: price should be higher than price at 0
    future_higher = np.zeros(n, dtype=bool)
    for i in range(n - future_n):
        if close.iloc[i + future_n] > close.iloc[i]:
            future_higher[i] = True

    wave0_mask = base_zero & future_higher

    idx_candidates_0 = np.where(wave0_mask)[0]
    final_wave0_mask = np.zeros(n, dtype=bool)

    last_kept_0 = None
    last_low_0 = None

    for i in idx_candidates_0:
        this_low = low.iloc[i]

        if last_kept_0 is None:
            final_wave0_mask[i] = True
            last_kept_0 = i
            last_low_0 = this_low
        else:
            if i - last_kept_0 < min_gap_same_type:
                # keep the LOWER low
                if this_low < last_low_0:
                    final_wave0_mask[last_kept_0] = False
                    final_wave0_mask[i] = True
                    last_kept_0 = i
                    last_low_0 = this_low
                else:
                    continue
            else:
                final_wave0_mask[i] = True
                last_kept_0 = i
                last_low_0 = this_low

    # ---------------- WAVE 5 (TOP) ----------------
    # RSI-based condition: overbought + RSI & price starting to fall
    cond_rsi_price_5 = (rsi > 80) & (rsi < rsi.shift(1)) & (close < close.shift(1))

    # Structural extreme high (100-bar lookback/forward)
    rolling_max_high = high.rolling(window=101, center=True, min_periods=1).max()
    cond_extreme_high = high >= (rolling_max_high - eps)

    base_five = cond_rsi_price_5 | cond_extreme_high

    # Future 7 bars: price should be lower than price at 5
    future_lower = np.zeros(n, dtype=bool)
    for i in range(n - future_n):
        if close.iloc[i + future_n] < close.iloc[i]:
            future_lower[i] = True

    wave5_mask = base_five & future_lower

    idx_candidates_5 = np.where(wave5_mask)[0]
    final_wave5_mask = np.zeros(n, dtype=bool)

    last_kept_5 = None
    last_high_5 = None

    for i in idx_candidates_5:
        this_high = high.iloc[i]

        if last_kept_5 is None:
            final_wave5_mask[i] = True
            last_kept_5 = i
            last_high_5 = this_high
        else:
            if i - last_kept_5 < min_gap_same_type:
                # keep the HIGHER high
                if this_high > last_high_5:
                    final_wave5_mask[last_kept_5] = False
                    final_wave5_mask[i] = True
                    last_kept_5 = i
                    last_high_5 = this_high
                else:
                    continue
            else:
                final_wave5_mask[i] = True
                last_kept_5 = i
                last_high_5 = this_high

    # ---------------- DROP FIRST IF 0,0 or 5,5 SEQUENTIAL ----------------
    events = []
    idx0 = np.where(final_wave0_mask)[0]
    idx5 = np.where(final_wave5_mask)[0]

    for idx in idx0:
        events.append((idx, "0"))
    for idx in idx5:
        events.append((idx, "5"))

    events.sort(key=lambda x: x[0])

    last_type = None
    last_index = None

    for idx, typ in events:
        if last_type is None:
            last_type = typ
            last_index = idx
            continue

        if typ == last_type:
            # same type → DROP FIRST, keep current
            if last_type == "0":
                final_wave0_mask[last_index] = False
            else:
                final_wave5_mask[last_index] = False
            last_index = idx
            last_type = typ
        else:
            last_type = typ
            last_index = idx

    # ---------------- 0 AFTER 5 WITHIN 30 BARS → DROP THAT 0 ----------------
    idx0 = np.where(final_wave0_mask)[0]
    idx5 = np.where(final_wave5_mask)[0]

    for i5 in idx5:
        # zeros that appear after this 5 within 30 bars
        close_zeros = idx0[(idx0 > i5) & (idx0 - i5 <= 30)]
        for iz in close_zeros:
            final_wave0_mask[iz] = False

    # Recompute idx0 after this filtering
    idx0 = np.where(final_wave0_mask)[0]

    # ---------------- B WAVE BETWEEN TWO 0s (PRICE FALLS) ----------------
    waveB_mask = np.zeros(n, dtype=bool)

    # Between two 0s within 30 bars, if 2nd 0 < 1st 0 → no 5, mark B at highest high between them
    for i in range(len(idx0) - 1):
        i0 = idx0[i]
        i1 = idx0[i + 1]

        if i1 - i0 <= 30 and low.iloc[i1] < low.iloc[i0]:
            # price has fallen between the two 0s → find B
            segment = df.iloc[i0:i1 + 1]
            if not segment.empty:
                idx_high = segment["High"].idxmax()
                pos = df.index.get_loc(idx_high)
                waveB_mask[pos] = True

                # remove any 5 that falls between these 0s (we "replace" 5 with B)
                pos_range = np.arange(i0, i1 + 1)
                final_wave5_mask[pos_range] = False

    # set columns now so waves 1–4 use the updated masks
    df["Wave0"] = final_wave0_mask
    df["Wave5"] = final_wave5_mask
    df["WaveB"] = waveB_mask

    # ---------------- WAVE 1 & 2 (FIB-BASED, TIMEFRAME-SPECIFIC) ----------------
    wave1_mask = np.zeros(n, dtype=bool)
    wave2_mask = np.zeros(n, dtype=bool)

    wave0_idx = np.where(df["Wave0"].values)[0]
    wave5_idx = np.where(df["Wave5"].values)[0]

    retr2_min = params["retr2_min"]
    retr2_max = params["retr2_max"]
    retr2_ideal = params["retr2_ideal"]

    for w0 in wave0_idx:
        later_5 = wave5_idx[wave5_idx > w0]
        if len(later_5) == 0:
            continue
        w5 = later_5[0]

        if w5 - w0 < min_05_bars:
            continue

        L0 = low.iloc[w0]

        best_score_12 = np.inf
        best_w1 = None
        best_w2 = None

        start_w1 = w0 + pivot_k
        end_w1 = w5 - pivot_k - 2
        if end_w1 <= start_w1:
            continue

        for i in range(start_w1, end_w1):
            window_h = high.iloc[i - pivot_k: i + pivot_k + 1]
            if high.iloc[i] < window_h.max():
                continue  # not a pivot high

            H1 = high.iloc[i]
            len1 = H1 - L0
            if len1 <= 0:
                continue

            start_w2 = i + pivot_k
            end_w2 = w5 - pivot_k
            if end_w2 <= start_w2:
                continue

            for j in range(start_w2, end_w2):
                window_l = low.iloc[j - pivot_k: j + pivot_k + 1]
                if low.iloc[j] > window_l.min():
                    continue  # not a pivot low

                L2 = low.iloc[j]
                if L2 <= L0:
                    continue  # must not retrace 100%

                retr = (H1 - L2) / len1
                if retr < retr2_min or retr > retr2_max:
                    continue

                score = abs(retr - retr2_ideal)
                if score < best_score_12:
                    best_score_12 = score
                    best_w1 = i
                    best_w2 = j

        if best_w1 is not None and best_w2 is not None:
            wave1_mask[best_w1] = True
            wave2_mask[best_w2] = True

    df["Wave1"] = wave1_mask
    df["Wave2"] = wave2_mask

    # ---------------- WAVE 3 & 4 (FIB-BASED, TIMEFRAME-SPECIFIC) ----------------
    wave3_mask = np.zeros(n, dtype=bool)
    wave4_mask = np.zeros(n, dtype=bool)

    wave1_idx = np.where(df["Wave1"].values)[0]
    wave2_idx = np.where(df["Wave2"].values)[0]

    ext3_min = params["ext3_min"]
    ext3_max = params["ext3_max"]
    ext3_ideal = params["ext3_ideal"]
    retr4_min = params["retr4_min"]
    retr4_max = params["retr4_max"]
    retr4_ideal = params["retr4_ideal"]

    for w0 in wave0_idx:
        later_5 = wave5_idx[wave5_idx > w0]
        if len(later_5) == 0:
            continue
        w5 = later_5[0]

        # 1 and 2 inside this 0→5 segment
        seg_w1 = wave1_idx[(wave1_idx > w0) & (wave1_idx < w5)]
        seg_w2 = wave2_idx[(wave2_idx > w0) & (wave2_idx < w5)]
        if len(seg_w1) == 0 or len(seg_w2) == 0:
            continue

        w1 = seg_w1[0]
        seg_w2_after = seg_w2[seg_w2 > w1]
        if len(seg_w2_after) == 0:
            continue
        w2 = seg_w2_after[0]

        L0 = low.iloc[w0]
        H1 = high.iloc[w1]
        L2 = low.iloc[w2]

        len1 = H1 - L0
        if len1 <= 0:
            continue

        # --- Wave 3 candidates ---
        best_score_3 = np.inf
        best_w3 = None

        start_w3 = w2 + pivot_k
        end_w3 = w5 - pivot_k - 2
        if end_w3 <= start_w3:
            continue

        for i in range(start_w3, end_w3):
            window_h = high.iloc[i - pivot_k: i + pivot_k + 1]
            if high.iloc[i] < window_h.max():
                continue  # not pivot high

            H3 = high.iloc[i]
            if H3 <= H1:
                continue

            ext3 = (H3 - L2) / len1
            if ext3 < ext3_min or ext3 > ext3_max:
                continue

            score3 = abs(ext3 - ext3_ideal)
            if score3 < best_score_3:
                best_score_3 = score3
                best_w3 = i

        if best_w3 is None:
            continue

        wave3_mask[best_w3] = True
        H3 = high.iloc[best_w3]

        # --- Wave 4 candidates ---
        best_score_4 = np.inf
        best_w4 = None

        start_w4 = best_w3 + pivot_k
        end_w4 = w5 - pivot_k
        if end_w4 <= start_w4:
            continue

        len3 = H3 - L2
        if len3 <= 0:
            continue

        for j in range(start_w4, end_w4):
            window_l = low.iloc[j - pivot_k: j + pivot_k + 1]
            if low.iloc[j] > window_l.min():
                continue  # not pivot low

            L4 = low.iloc[j]

            # no overlap with wave1 territory (loose Elliott rule)
            if L4 <= H1:
                continue

            retr4 = (H3 - L4) / len3
            if retr4 < retr4_min or retr4 > retr4_max:
                continue

            score4 = abs(retr4 - retr4_ideal)
            if score4 < best_score_4:
                best_score_4 = score4
                best_w4 = j

        if best_w4 is not None:
            wave4_mask[best_w4] = True

    df["Wave3"] = wave3_mask
    df["Wave4"] = wave4_mask

    return df


# ---------------- BIG CHART ----------------
def make_tv_style_chart(df: pd.DataFrame, title: str):
    """
    Bigger TradingView-style layout:
    Row 1: Candlestick (larger) + SMA + Wave labels
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

    # --- Wave 5 labels (above highs) ---
    if "Wave5" in df.columns:
        wave5_df = df[df["Wave5"]]
        if not wave5_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=wave5_df.index,
                    y=wave5_df["High"] * 1.005,
                    mode="text",
                    text=["<b>5</b>"] * len(wave5_df),
                    textposition="middle center",
                    name="Wave 5",
                ),
                row=1,
                col=1,
            )

    # --- Wave B labels (above highs, a bit more) ---
    if "WaveB" in df.columns:
        waveB_df = df[df["WaveB"]]
        if not waveB_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=waveB_df.index,
                    y=waveB_df["High"] * 1.02,
                    mode="text",
                    text=["<b>B</b>"] * len(waveB_df),
                    textposition="middle center",
                    name="Wave B",
                ),
                row=1,
                col=1,
            )

    # --- Wave 1 labels ---
    if "Wave1" in df.columns:
        wave1_df = df[df["Wave1"]]
        if not wave1_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=wave1_df.index,
                    y=wave1_df["High"] * 1.01,
                    mode="text",
                    text=["<b>1</b>"] * len(wave1_df),
                    textposition="middle center",
                    name="Wave 1",
                ),
                row=1,
                col=1,
            )

    # --- Wave 2 labels (below lows) ---
    if "Wave2" in df.columns:
        wave2_df = df[df["Wave2"]]
        if not wave2_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=wave2_df.index,
                    y=wave2_df["Low"] * 0.99,
                    mode="text",
                    text=["<b>2</b>"] * len(wave2_df),
                    textposition="middle center",
                    name="Wave 2",
                ),
                row=1,
                col=1,
            )

    # --- Wave 3 labels (above highs) ---
    if "Wave3" in df.columns:
        wave3_df = df[df["Wave3"]]
        if not wave3_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=wave3_df.index,
                    y=wave3_df["High"] * 1.015,
                    mode="text",
                    text=["<b>3</b>"] * len(wave3_df),
                    textposition="middle center",
                    name="Wave 3",
                ),
                row=1,
                col=1,
            )

    # --- Wave 4 labels (below lows) ---
    if "Wave4" in df.columns:
        wave4_df = df[df["Wave4"]]
        if not wave4_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=wave4_df.index,
                    y=wave4_df["Low"] * 0.985,
                    mode="text",
                    text=["<b>4</b>"] * len(wave4_df),
                    textposition="middle center",
                    name="Wave 4",
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
    df_h = add_wave_labels(df_h, timeframe="1h")

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
    df_d = add_wave_labels(df_d, timeframe="1d")

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
    df_w = add_wave_labels(df_w, timeframe="1wk")

    if df_w.empty:
        st.warning("No weekly data found for this symbol.")
    else:
        fig_w = make_tv_style_chart(df_w, f"{ticker} — Weekly (10Y)")
        st.plotly_chart(fig_w, use_container_width=True)
