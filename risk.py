from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class RiskLimits:
    min_equity: float
    max_daily_loss_pct: float
    max_order_notional_pct_equity: float
    max_total_turnover_pct_equity: float
    min_order_notional_usd: float


class DailyEquityState:
    def __init__(self, path: str = "state/equity_state.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict:
        if not self.path.exists():
            return {}
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, data: dict) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_day_open_equity(self, day_key: str) -> float | None:
        data = self.load()
        if data.get("day") == day_key:
            val = data.get("open_equity")
            return float(val) if val is not None else None
        return None

    def update_day_open_equity(self, day_key: str, equity: float) -> None:
        data = {"day": day_key, "open_equity": float(equity)}
        self.save(data)


def current_day_key_utc() -> str:
    return pd.Timestamp.utcnow().strftime("%Y-%m-%d")


def check_equity_guards(equity: float, day_open_equity: float | None, limits: RiskLimits) -> tuple[bool, str]:
    if equity < limits.min_equity:
        return False, f"Equity guard failed: equity ${equity:,.2f} < min_equity ${limits.min_equity:,.2f}"
    if day_open_equity is not None and day_open_equity > 0:
        pnl_pct = equity / day_open_equity - 1.0
        if pnl_pct <= -abs(limits.max_daily_loss_pct):
            return False, (
                f"Daily loss guard failed: day PnL {pnl_pct:.2%} <= "
                f"-{abs(limits.max_daily_loss_pct):.2%}"
            )
    return True, "ok"


def cap_order_deltas(deltas: pd.Series, equity: float, limits: RiskLimits) -> pd.Series:
    if deltas.empty:
        return deltas

    capped = deltas.copy()
    per_order_cap = abs(limits.max_order_notional_pct_equity) * equity
    if per_order_cap > 0:
        capped = capped.clip(lower=-per_order_cap, upper=per_order_cap)

    total_turnover = float(capped.abs().sum())
    turnover_cap = abs(limits.max_total_turnover_pct_equity) * equity
    if turnover_cap > 0 and total_turnover > turnover_cap:
        scale = turnover_cap / total_turnover
        capped = capped * scale

    min_notional = max(0.0, limits.min_order_notional_usd)
    if min_notional > 0:
        capped = capped[capped.abs() >= min_notional]
    return capped
