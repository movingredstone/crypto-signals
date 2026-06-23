from pathlib import Path
import time
import requests
import pandas as pd


BASE_URL = "https://fapi.binance.com"

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]


def interval_to_millis(interval: str) -> int:
    unit = interval[-1]
    value = int(interval[:-1])

    table = {
        "m": 60_000,
        "h": 60 * 60_000,
        "d": 24 * 60 * 60_000,
    }

    if unit not in table:
        raise ValueError(f"Unsupported interval: {interval}")

    return value * table[unit]


def to_millis(date_text: str) -> int:
    return int(pd.Timestamp(date_text, tz="UTC").timestamp() * 1000)


def clean_klines(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["number_of_trades"] = pd.to_numeric(df["number_of_trades"], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.as_unit("ns")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True).dt.as_unit("ns")

    df = df.drop(columns=["ignore"], errors="ignore")
    df = df.drop_duplicates(subset=["open_time"])
    df = df.sort_values("open_time").reset_index(drop=True)

    return df


def download_klines(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    url = f"{BASE_URL}/fapi/v1/klines"

    start_ms = to_millis(start_date)
    end_ms = to_millis(end_date)
    step_ms = interval_to_millis(interval)

    all_rows = []
    current = start_ms

    while current < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current,
            "endTime": end_ms,
            "limit": 1500,
        }

        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()

        rows = response.json()

        if not rows:
            break

        all_rows.extend(rows)

        last_open = int(rows[-1][0])
        next_start = last_open + step_ms

        if next_start <= current:
            break

        current = next_start
        time.sleep(0.12)

    df = pd.DataFrame(all_rows, columns=KLINE_COLUMNS)
    return clean_klines(df)


def load_or_download_klines(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    safe_start = start_date.replace("-", "")
    safe_end = end_date.replace("-", "")

    file_path = raw_dir / f"{symbol}_{interval}_{safe_start}_{safe_end}.csv"

    if file_path.exists():
        df = pd.read_csv(file_path)
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True).dt.as_unit("ns")
        df["close_time"] = pd.to_datetime(df["close_time"], utc=True).dt.as_unit("ns")
        return df.sort_values("open_time").reset_index(drop=True)

    print(f"Downloading Binance futures klines: {symbol} {interval} {start_date} ~ {end_date}")
    df = download_klines(symbol, interval, start_date, end_date)
    df.to_csv(file_path, index=False)
    print(f"Saved: {file_path}")

    return df


# ---------------------------------------------------------------------------
# Funding rate (full history available on Binance futures).
# /fapi/v1/fundingRate returns 8h funding events with full history.
# ---------------------------------------------------------------------------
def download_funding_rate(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    url = f"{BASE_URL}/fapi/v1/fundingRate"

    start_ms = to_millis(start_date)
    end_ms = to_millis(end_date)

    rows = []
    current = start_ms

    while current < end_ms:
        params = {
            "symbol": symbol,
            "startTime": current,
            "endTime": end_ms,
            "limit": 1000,
        }
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        batch = response.json()

        if not batch:
            break

        rows.extend(batch)
        last_time = int(batch[-1]["fundingTime"])
        next_start = last_time + 1

        if next_start <= current:
            break

        current = next_start
        time.sleep(0.12)

        if len(batch) < 1000:
            break

    if not rows:
        return pd.DataFrame(columns=["funding_time", "funding_rate"])

    df = pd.DataFrame(rows)
    df["funding_time"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True).dt.as_unit("ns")
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df = df[["funding_time", "funding_rate"]].drop_duplicates("funding_time")
    return df.sort_values("funding_time").reset_index(drop=True)


def load_or_download_funding(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    safe_start = start_date.replace("-", "")
    safe_end = end_date.replace("-", "")
    file_path = raw_dir / f"{symbol}_funding_{safe_start}_{safe_end}.csv"

    if file_path.exists():
        df = pd.read_csv(file_path)
        df["funding_time"] = pd.to_datetime(df["funding_time"], utc=True).dt.as_unit("ns")
        return df.sort_values("funding_time").reset_index(drop=True)

    print(f"Downloading Binance funding rate: {symbol} {start_date} ~ {end_date}")
    df = download_funding_rate(symbol, start_date, end_date)
    df.to_csv(file_path, index=False)
    print(f"Saved: {file_path}")
    return df


# ---------------------------------------------------------------------------
# Open interest / long-short ratio.
# NOTE: Binance only serves ~last 30 days for these endpoints, so they are
# NOT usable for a multi-year backtest. Kept as optional helpers only.
# Liquidation history has no reliable public endpoint and is intentionally omitted.
# ---------------------------------------------------------------------------
def download_open_interest_hist(symbol: str, period: str = "1h", limit: int = 500) -> pd.DataFrame:
    url = f"{BASE_URL}/futures/data/openInterestHist"
    params = {"symbol": symbol, "period": period, "limit": min(limit, 500)}
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    rows = response.json()
    if not rows:
        return pd.DataFrame(columns=["time", "open_interest"])
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["open_interest"] = pd.to_numeric(df["sumOpenInterest"], errors="coerce")
    return df[["time", "open_interest"]].sort_values("time").reset_index(drop=True)


def download_long_short_ratio(symbol: str, period: str = "1h", limit: int = 500) -> pd.DataFrame:
    url = f"{BASE_URL}/futures/data/globalLongShortAccountRatio"
    params = {"symbol": symbol, "period": period, "limit": min(limit, 500)}
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    rows = response.json()
    if not rows:
        return pd.DataFrame(columns=["time", "long_short_ratio"])
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["long_short_ratio"] = pd.to_numeric(df["longShortRatio"], errors="coerce")
    return df[["time", "long_short_ratio"]].sort_values("time").reset_index(drop=True)
