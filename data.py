from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class MarketData:
    close: pd.DataFrame
    returns: pd.DataFrame
    vol: pd.DataFrame
    momentum: pd.DataFrame


def flatten_symbols(universe: Dict[str, Iterable[str]]) -> list[str]:
    seen = set()
    out = []
    for symbols in universe.values():
        for sym in symbols:
            s = sym.upper().strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
    return out


def load_ohlcv(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    if not symbols:
        raise ValueError("No symbols provided")

    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    for sym in symbols:
        one: pd.DataFrame | None = None
        for attempt in range(3):
            try:
                one = yf.download(
                    tickers=sym,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
                if one is not None and not one.empty:
                    break
            except Exception:
                pass
            time.sleep(0.4 * (attempt + 1))

        if one is None or one.empty:
            failed.append(sym)
            continue

        one = one.copy()
        if isinstance(one.columns, pd.MultiIndex):
            if sym in one.columns.get_level_values(0):
                one = one.xs(sym, axis=1, level=0, drop_level=True)
            elif sym in one.columns.get_level_values(1):
                one = one.xs(sym, axis=1, level=1, drop_level=True)
            else:
                one.columns = one.columns.get_level_values(-1)
        one.columns = [str(c) for c in one.columns]
        one.columns = pd.MultiIndex.from_product([[sym], one.columns])
        frames.append(one)

    if not frames:
        raise RuntimeError("No data downloaded. Check symbols/date range/network.")

    raw = pd.concat(frames, axis=1).sort_index(axis=1)
    if failed:
        print(f"Warning: failed downloads for symbols: {', '.join(failed)}")
    return raw


def build_market_data(
    close: pd.DataFrame, momentum_lookback_days: int, vol_lookback_days: int
) -> MarketData:
    returns = close.pct_change()
    vol = returns.rolling(vol_lookback_days).std() * np.sqrt(252)
    momentum = close / close.shift(momentum_lookback_days) - 1.0
    return MarketData(close=close, returns=returns, vol=vol, momentum=momentum)
