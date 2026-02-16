from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from strategy import StrategyParams, compute_target_weights_for_date


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    weights: pd.DataFrame
    daily_returns: pd.Series
    turnover: pd.Series


def run_backtest(
    close: pd.DataFrame,
    momentum: pd.DataFrame,
    vol: pd.DataFrame,
    universe: Dict[str, list[str]],
    strategy_params: StrategyParams,
    rebalance_every_days: int,
    initial_capital: float,
    fee_bps: float,
    slippage_bps: float,
) -> BacktestResult:
    dates = close.index
    n = len(dates)
    if n < 3:
        raise ValueError("Not enough data for backtest.")

    weights = pd.DataFrame(0.0, index=dates, columns=close.columns)
    current_w = pd.Series(0.0, index=close.columns)
    asset_returns = close.pct_change().fillna(0.0)
    portfolio_returns = pd.Series(0.0, index=dates)
    turnover_series = pd.Series(0.0, index=dates)
    cost_per_turnover = (fee_bps + slippage_bps) / 10000.0

    for i, dt in enumerate(dates):
        if i == 0:
            weights.loc[dt] = current_w
            continue

        should_rebalance = i % rebalance_every_days == 0
        if should_rebalance:
            target_w = compute_target_weights_for_date(
                date=dt,
                close=close,
                momentum=momentum,
                vol=vol,
                universe=universe,
                params=strategy_params,
            )
            turnover = (target_w - current_w).abs().sum()
            current_w = target_w
        else:
            turnover = 0.0
        turnover_series.loc[dt] = turnover

        gross_ret = float((current_w * asset_returns.loc[dt]).sum())
        net_ret = gross_ret - turnover * cost_per_turnover
        current_w = current_w * (1.0 + asset_returns.loc[dt])
        sw = current_w.sum()
        if sw > 0:
            current_w = current_w / sw * strategy_params.target_gross_exposure
        weights.loc[dt] = current_w
        portfolio_returns.loc[dt] = net_ret

    equity_curve = (1.0 + portfolio_returns).cumprod() * initial_capital
    return BacktestResult(
        equity_curve=equity_curve,
        weights=weights,
        daily_returns=portfolio_returns,
        turnover=turnover_series,
    )


def summarize_performance(equity_curve: pd.Series, daily_returns: pd.Series) -> dict:
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0
    ann_return = (1.0 + total_return) ** (252.0 / max(len(daily_returns), 1)) - 1.0
    ann_vol = daily_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    rolling_max = equity_curve.cummax()
    drawdown = equity_curve / rolling_max - 1.0
    max_dd = drawdown.min()
    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }
