from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest import BacktestResult


def _monthly_returns(equity_curve: pd.Series) -> pd.Series:
    month_end = equity_curve.resample("ME").last()
    return month_end.pct_change().fillna(0.0)


def _yearly_returns(equity_curve: pd.Series) -> pd.Series:
    year_end = equity_curve.resample("YE").last()
    return year_end.pct_change().fillna(0.0)


def export_backtest_report(
    result: BacktestResult,
    stats: dict[str, float],
    output_dir: str = "reports",
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    equity_path = out / "equity_curve.csv"
    daily_path = out / "daily_returns.csv"
    turnover_path = out / "turnover.csv"
    monthly_path = out / "monthly_returns.csv"
    yearly_path = out / "yearly_returns.csv"
    stats_path = out / "stats.json"

    result.equity_curve.rename("equity").to_csv(equity_path, index_label="date")
    result.daily_returns.rename("daily_return").to_csv(daily_path, index_label="date")
    result.turnover.rename("turnover").to_csv(turnover_path, index_label="date")
    _monthly_returns(result.equity_curve).rename("monthly_return").to_csv(monthly_path, index_label="date")
    _yearly_returns(result.equity_curve).rename("yearly_return").to_csv(yearly_path, index_label="date")

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return {
        "equity_curve": str(equity_path.resolve()),
        "daily_returns": str(daily_path.resolve()),
        "turnover": str(turnover_path.resolve()),
        "monthly_returns": str(monthly_path.resolve()),
        "yearly_returns": str(yearly_path.resolve()),
        "stats": str(stats_path.resolve()),
    }
