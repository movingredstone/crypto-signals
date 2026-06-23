import numpy as np
import pandas as pd


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _wilder(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def add_adx(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    up = df["high"].diff()
    down = -df["low"].diff()

    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr = _true_range(df)
    atr = _wilder(tr, n)

    plus_di = 100 * _wilder(pd.Series(plus_dm, index=df.index), n) / atr
    minus_di = 100 * _wilder(pd.Series(minus_dm, index=df.index), n) / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx14"] = _wilder(dx, n)
    df["plus_di14"] = plus_di
    df["minus_di14"] = minus_di
    return df


def add_supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> pd.DataFrame:
    tr = _true_range(df)
    atr = _wilder(tr, period)

    hl2 = (df["high"] + df["low"]) / 2
    upper = (hl2 + mult * atr).values
    lower = (hl2 - mult * atr).values
    close = df["close"].values
    n = len(df)

    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    direction = np.ones(n, dtype=int)  # +1 uptrend, -1 downtrend

    for i in range(n):
        if i == 0 or np.isnan(upper[i]):
            final_upper[i] = upper[i]
            final_lower[i] = lower[i]
            direction[i] = 1
            continue

        final_upper[i] = (
            upper[i]
            if (upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1])
            else final_upper[i - 1]
        )
        final_lower[i] = (
            lower[i]
            if (lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1])
            else final_lower[i - 1]
        )

        if close[i] > final_upper[i - 1]:
            direction[i] = 1
        elif close[i] < final_lower[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

    df["supertrend_dir"] = direction
    return df


def add_keltner(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 14, mult: float = 1.5) -> pd.DataFrame:
    mid = df["close"].ewm(span=ema_period, adjust=False, min_periods=ema_period).mean()
    tr = _true_range(df)
    atr = _wilder(tr, atr_period)
    df["kc_mid"] = mid
    df["kc_upper"] = mid + mult * atr
    df["kc_lower"] = mid - mult * atr
    return df


def add_realized_vol(df: pd.DataFrame, window: int = 24, rank_window: int = 300) -> pd.DataFrame:
    ret = df["close"].pct_change()
    rv = ret.rolling(window, min_periods=window).std()
    df["rv"] = rv
    df["rv_pct"] = rv.rolling(rank_window, min_periods=100).rank(pct=True) * 100
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # EMA
    df["ema20"] = df["close"].ewm(span=20, adjust=False, min_periods=20).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False, min_periods=200).mean()

    # RSI 14
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))

    # ATR 14
    tr = _true_range(df)
    df["atr14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    # ATR percentile
    df["atr_pct"] = df["atr14"].rolling(200, min_periods=50).rank(pct=True) * 100

    # Rolling VWAP approximation
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    price_volume = typical_price * df["volume"]

    df["vwap"] = (
        price_volume.rolling(96, min_periods=20).sum()
        / df["volume"].rolling(96, min_periods=20).sum()
    )

    # Volume ratio
    df["vol_ma20"] = df["volume"].rolling(20, min_periods=20).mean()
    df["volume_ratio"] = df["volume"] / df["vol_ma20"]

    # Candle structure
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    body = (df["close"] - df["open"]).abs()

    df["body_ratio"] = body / candle_range
    df["upper_wick_ratio"] = (
        df["high"] - df[["open", "close"]].max(axis=1)
    ) / candle_range
    df["lower_wick_ratio"] = (
        df[["open", "close"]].min(axis=1) - df["low"]
    ) / candle_range

    # Recent swing levels: shifted to avoid lookahead
    df["recent_swing_high"] = df["high"].rolling(20, min_periods=20).max().shift(1)
    df["recent_swing_low"] = df["low"].rolling(20, min_periods=20).min().shift(1)

    # Simple trend labels
    df["ema_bull"] = (df["close"] > df["ema200"]) & (df["ema20"] > df["ema50"])
    df["ema_bear"] = (df["close"] < df["ema200"]) & (df["ema20"] < df["ema50"])

    # --- Extra factor families (OHLCV-only) ---
    df = add_adx(df, n=14)
    df = add_supertrend(df, period=10, mult=3.0)
    df = add_keltner(df, ema_period=20, atr_period=14, mult=1.5)
    df = add_realized_vol(df, window=24, rank_window=300)

    return df
