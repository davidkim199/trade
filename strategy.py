from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class StrategyParams:
    top_n_per_segment: int
    min_price: float
    max_weight_per_symbol: float
    target_gross_exposure: float
    cash_buffer_pct: float


def score_row(momentum_row: pd.Series, vol_row: pd.Series) -> pd.Series:
    # Higher momentum and lower volatility gets a better score.
    return momentum_row - 0.5 * vol_row


def score_for_date(date: pd.Timestamp, momentum: pd.DataFrame, vol: pd.DataFrame) -> pd.Series:
    return score_row(momentum.loc[date], vol.loc[date])


def compute_target_weights_for_date(
    date: pd.Timestamp,
    close: pd.DataFrame,
    momentum: pd.DataFrame,
    vol: pd.DataFrame,
    universe: Dict[str, list[str]],
    params: StrategyParams,
    score_adjustments: dict[str, float] | None = None,
    blocked_symbols: set[str] | None = None,
) -> pd.Series:
    px = close.loc[date]
    v = vol.loc[date]
    score = score_for_date(date=date, momentum=momentum, vol=vol)
    blocked = blocked_symbols or set()
    if score_adjustments:
        for sym, adj in score_adjustments.items():
            if sym in score.index and pd.notna(score[sym]):
                score.loc[sym] = float(score[sym]) + float(adj)

    selected: list[str] = []
    for _, segment_symbols in universe.items():
        candidates = [s for s in segment_symbols if s in score.index]
        candidates = [s for s in candidates if s not in blocked]
        candidates = [s for s in candidates if pd.notna(px.get(s)) and px[s] >= params.min_price]
        candidates = [s for s in candidates if pd.notna(score.get(s)) and pd.notna(v.get(s)) and v[s] > 0]
        if not candidates:
            continue
        ranked = sorted(candidates, key=lambda s: score[s], reverse=True)
        selected.extend(ranked[: params.top_n_per_segment])

    if not selected:
        return pd.Series(0.0, index=close.columns)

    selected = list(dict.fromkeys(selected))
    inv_vol = pd.Series({s: 1.0 / max(v[s], 1e-8) for s in selected})
    w = inv_vol / inv_vol.sum()

    w = w.clip(upper=params.max_weight_per_symbol)
    if w.sum() > 0:
        w = w / w.sum()

    deployable = max(0.0, params.target_gross_exposure - params.cash_buffer_pct)
    w = w * deployable

    out = pd.Series(0.0, index=close.columns)
    out.loc[w.index] = w.values
    return out
