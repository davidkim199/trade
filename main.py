from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml

from ai_agent import AgentDecision, NullResearchAgent, OpenAIResearchAgent, ResearchAgent
from backtest import run_backtest, summarize_performance
from broker import AlpacaBroker, PaperPrintBroker
from data import build_market_data, flatten_symbols, load_ohlcv
from reporting import export_backtest_report
from risk import DailyEquityState, RiskLimits, cap_order_deltas, check_equity_guards, current_day_key_utc
from strategy import StrategyParams, compute_target_weights_for_date, score_for_date


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_close(raw: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    close_names = {"close", "adj close", "adjclose"}

    if isinstance(raw.columns, pd.MultiIndex):
        for lvl in range(raw.columns.nlevels):
            labels = raw.columns.get_level_values(lvl)
            lowered = [str(x).lower() for x in labels]
            if any(x in close_names for x in lowered):
                picked = [labels[i] for i, x in enumerate(lowered) if x in close_names]
                # Pick first matching label on that level and collapse it.
                close = raw.xs(picked[0], axis=1, level=lvl, drop_level=True)
                if isinstance(close, pd.Series):
                    close = close.to_frame(name=symbols[0])
                close.columns = [str(c) for c in close.columns]
                close = close.reindex(columns=symbols)
                close = close.apply(pd.to_numeric, errors="coerce")
                close = close.dropna(how="all")
                if not close.empty:
                    return close

    # Single-level fallback, case-insensitive.
    cols = {str(c).lower(): c for c in raw.columns}
    for name in ["close", "adj close", "adjclose"]:
        if name in cols:
            c = cols[name]
            out = raw[[c]].rename(columns={c: symbols[0]})
            out = out.apply(pd.to_numeric, errors="coerce").dropna(how="all")
            if not out.empty:
                return out

    raise RuntimeError(
        f"Could not extract close prices from downloaded data. Columns={list(raw.columns)[:10]}"
    )


def build_strategy_params(cfg: dict) -> StrategyParams:
    scfg = cfg["strategy"]
    rcfg = cfg["risk"]
    return StrategyParams(
        top_n_per_segment=int(scfg["top_n_per_segment"]),
        min_price=float(scfg["min_price"]),
        max_weight_per_symbol=float(scfg["max_weight_per_symbol"]),
        target_gross_exposure=float(rcfg["target_gross_exposure"]),
        cash_buffer_pct=float(rcfg["cash_buffer_pct"]),
    )


def build_risk_limits(cfg: dict) -> RiskLimits:
    rcfg = cfg["risk"]
    return RiskLimits(
        min_equity=float(rcfg.get("min_equity", 1000.0)),
        max_daily_loss_pct=float(rcfg.get("max_daily_loss_pct", 0.03)),
        max_order_notional_pct_equity=float(rcfg.get("max_order_notional_pct_equity", 0.2)),
        max_total_turnover_pct_equity=float(rcfg.get("max_total_turnover_pct_equity", 0.5)),
        min_order_notional_usd=float(rcfg.get("min_order_notional_usd", 100.0)),
    )


@dataclass
class EconomicsParams:
    enabled: bool
    expected_edge_bps: float
    extra_safety_bps: float
    fee_bps: float
    slippage_bps: float
    max_api_cost_per_run_usd: float


def build_economics_params(cfg: dict) -> EconomicsParams:
    ecfg = cfg.get("economics", {})
    bcfg = cfg.get("backtest", {})
    return EconomicsParams(
        enabled=bool(ecfg.get("enabled", True)),
        expected_edge_bps=float(ecfg.get("expected_edge_bps", 25.0)),
        extra_safety_bps=float(ecfg.get("extra_safety_bps", 5.0)),
        fee_bps=float(ecfg.get("fee_bps", bcfg.get("fee_bps", 2.0))),
        slippage_bps=float(ecfg.get("slippage_bps", bcfg.get("slippage_bps", 3.0))),
        max_api_cost_per_run_usd=float(ecfg.get("max_api_cost_per_run_usd", 0.05)),
    )


def make_broker(cfg: dict):
    broker_cfg = cfg.get("broker", {})
    mode = cfg.get("mode", "paper")
    if mode == "live":
        return AlpacaBroker(
            base_url=broker_cfg["base_url"],
            api_key=broker_cfg["api_key"],
            api_secret=broker_cfg["api_secret"],
        )
    if (
        broker_cfg.get("api_key")
        and broker_cfg.get("api_secret")
        and broker_cfg.get("api_key") != "YOUR_ALPACA_KEY"
        and broker_cfg.get("api_secret") != "YOUR_ALPACA_SECRET"
    ):
        return AlpacaBroker(
            base_url=broker_cfg.get("base_url", "https://paper-api.alpaca.markets"),
            api_key=broker_cfg["api_key"],
            api_secret=broker_cfg["api_secret"],
        )
    return PaperPrintBroker()


def make_research_agent(cfg: dict) -> ResearchAgent:
    acfg = cfg.get("ai_agent", {})
    if not bool(acfg.get("enabled", False)):
        return NullResearchAgent()

    provider = str(acfg.get("provider", "openai")).lower()
    if provider != "openai":
        print(f"Unknown ai_agent provider '{provider}', disabling agent.")
        return NullResearchAgent()

    api_key = str(acfg.get("api_key", "")).strip()
    if not api_key or api_key == "YOUR_OPENAI_API_KEY":
        api_key = str(os.getenv("OPENAI_API_KEY", "")).strip()
    if not api_key or api_key == "YOUR_OPENAI_API_KEY":
        print("ai_agent enabled but api_key missing; disabling agent.")
        return NullResearchAgent()

    return OpenAIResearchAgent(
        api_key=api_key,
        model=str(acfg.get("model", "gpt-4o-mini")),
        base_url=str(acfg.get("base_url", "https://api.openai.com/v1")),
        timeout_seconds=int(acfg.get("timeout_seconds", 30)),
        max_abs_adjustment=float(acfg.get("max_abs_adjustment", 0.15)),
        input_cost_per_1k_tokens_usd=float(acfg.get("input_cost_per_1k_tokens_usd", 0.00015)),
        output_cost_per_1k_tokens_usd=float(acfg.get("output_cost_per_1k_tokens_usd", 0.0006)),
    )


def _runtime_paths(cfg: dict) -> dict[str, Path]:
    rcfg = cfg.get("runtime", {})
    status_file = Path(rcfg.get("status_file", "state/status.json"))
    status_history_file = Path(rcfg.get("status_history_file", "state/status_history.csv"))
    orders_file = Path(rcfg.get("orders_file", "state/latest_orders.csv"))
    targets_file = Path(rcfg.get("targets_file", "state/latest_targets.csv"))
    api_costs_file = Path(rcfg.get("api_costs_file", "state/api_costs.csv"))
    for p in [status_file, status_history_file, orders_file, targets_file, api_costs_file]:
        p.parent.mkdir(parents=True, exist_ok=True)
    return {
        "status": status_file,
        "status_history": status_history_file,
        "orders": orders_file,
        "targets": targets_file,
        "api_costs": api_costs_file,
    }


def _write_status(cfg: dict, payload: dict) -> None:
    path = _runtime_paths(cfg)["status"]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False)


def _append_api_cost_row(cfg: dict, row: dict) -> None:
    path = _runtime_paths(cfg)["api_costs"]
    frame = pd.DataFrame([row])
    if path.exists():
        prior = pd.read_csv(path)
        frame = pd.concat([prior, frame], ignore_index=True)
    frame.to_csv(path, index=False)


def _append_status_row(cfg: dict, row: dict) -> None:
    path = _runtime_paths(cfg)["status_history"]
    frame = pd.DataFrame([row])
    if path.exists():
        prior = pd.read_csv(path)
        frame = pd.concat([prior, frame], ignore_index=True)
    frame.to_csv(path, index=False)


def run_backtest_cmd(cfg: dict) -> None:
    universe: Dict[str, list[str]] = cfg["universe"]
    symbols = flatten_symbols(universe)
    bcfg = cfg["backtest"]
    raw = load_ohlcv(symbols=symbols, start=bcfg["start"], end=bcfg["end"])
    close = extract_close(raw, symbols)
    mcfg = cfg["strategy"]
    mkt = build_market_data(
        close=close,
        momentum_lookback_days=int(mcfg["momentum_lookback_days"]),
        vol_lookback_days=int(mcfg["vol_lookback_days"]),
    )
    res = run_backtest(
        close=mkt.close,
        momentum=mkt.momentum,
        vol=mkt.vol,
        universe=universe,
        strategy_params=build_strategy_params(cfg),
        rebalance_every_days=int(mcfg["rebalance_every_days"]),
        initial_capital=float(bcfg["initial_capital"]),
        fee_bps=float(bcfg["fee_bps"]),
        slippage_bps=float(bcfg["slippage_bps"]),
    )
    stats = summarize_performance(res.equity_curve, res.daily_returns)
    outputs = export_backtest_report(res, stats, output_dir=bcfg.get("report_dir", "reports"))

    print("Backtest complete")
    for k, v in stats.items():
        print(f"{k:>18}: {v: .4f}")
    for name, path in outputs.items():
        print(f"{name:>18}: {path}")


def compute_order_deltas(
    cfg: dict, equity: float, current_positions: dict[str, float]
) -> tuple[pd.Series, pd.Series, pd.Timestamp, AgentDecision]:
    universe: Dict[str, list[str]] = cfg["universe"]
    symbols = flatten_symbols(universe)
    mcfg = cfg["strategy"]
    raw = load_ohlcv(symbols=symbols, start="2020-01-01", end=pd.Timestamp.now("UTC").strftime("%Y-%m-%d"))
    close = extract_close(raw, symbols)
    mkt = build_market_data(
        close=close,
        momentum_lookback_days=int(mcfg["momentum_lookback_days"]),
        vol_lookback_days=int(mcfg["vol_lookback_days"]),
    )
    dt = close.index[-1]
    base_scores = score_for_date(date=dt, momentum=mkt.momentum, vol=mkt.vol)
    agent = make_research_agent(cfg)
    decision = agent.decide(
        date=dt,
        universe=universe,
        prices=close.loc[dt],
        momentum=mkt.momentum.loc[dt],
        vol=mkt.vol.loc[dt],
        base_scores=base_scores,
    )
    target_w = compute_target_weights_for_date(
        date=dt,
        close=close,
        momentum=mkt.momentum,
        vol=mkt.vol,
        universe=universe,
        params=build_strategy_params(cfg),
        score_adjustments=decision.score_adjustments,
        blocked_symbols=set(decision.blocked_symbols),
    )
    print(
        "Agent decision | "
        f"adjustments={len(decision.score_adjustments)} blocked={len(decision.blocked_symbols)} "
        f"rationale={decision.rationale[:120]}"
    )
    current_notional = pd.Series(current_positions, dtype=float)
    target_notional = target_w * equity
    deltas = target_notional.subtract(current_notional, fill_value=0.0)
    return deltas, target_notional, dt, decision


def run_once_cmd(cfg: dict) -> None:
    broker = make_broker(cfg)
    now_utc = pd.Timestamp.now("UTC")
    equity = broker.get_equity()
    positions = broker.get_positions()
    risk_limits = build_risk_limits(cfg)
    econ = build_economics_params(cfg)
    state = DailyEquityState(cfg.get("state_file", "state/equity_state.json"))
    paths = _runtime_paths(cfg)
    day_key = current_day_key_utc()
    day_open = state.get_day_open_equity(day_key)
    if day_open is None:
        state.update_day_open_equity(day_key, equity)
        day_open = equity

    status = {
        "timestamp_utc": now_utc.isoformat(),
        "mode": cfg.get("mode", "paper"),
        "equity": float(equity),
        "day_open_equity": float(day_open),
        "day_pnl_usd": float(equity - day_open),
        "day_pnl_pct": float(equity / day_open - 1.0) if day_open > 0 else 0.0,
        "risk_guard_ok": True,
        "risk_guard_reason": "ok",
        "orders_submitted": 0,
        "signal_date": None,
        "agent_rationale": "",
        "agent_api_cost_usd": 0.0,
        "expected_edge_bps": float(econ.expected_edge_bps),
        "estimated_total_cost_bps": 0.0,
        "estimated_net_edge_bps": 0.0,
        "economics_ok": True,
        "economics_reason": "ok",
    }

    ok, reason = check_equity_guards(equity=equity, day_open_equity=day_open, limits=risk_limits)
    if not ok:
        status["risk_guard_ok"] = False
        status["risk_guard_reason"] = reason
        _write_status(cfg, status)
        _append_status_row(
            cfg,
            {
                "timestamp_utc": status["timestamp_utc"],
                "mode": status["mode"],
                "equity": status["equity"],
                "day_open_equity": status["day_open_equity"],
                "day_pnl_usd": status["day_pnl_usd"],
                "day_pnl_pct": status["day_pnl_pct"],
                "risk_guard_ok": status["risk_guard_ok"],
                "orders_submitted": status["orders_submitted"],
                "agent_api_cost_usd": status["agent_api_cost_usd"],
                "estimated_net_edge_bps": status["estimated_net_edge_bps"],
                "economics_ok": status["economics_ok"],
            },
        )
        _write_csv(paths["orders"], pd.DataFrame(columns=["symbol", "raw_delta_usd", "capped_order_usd"]))
        print(f"Risk guard blocked trading: {reason}")
        return

    deltas, target_notional, dt, decision = compute_order_deltas(cfg, equity=equity, current_positions=positions)
    capped = cap_order_deltas(deltas, equity=equity, limits=risk_limits)
    status["signal_date"] = str(dt.date())
    status["agent_rationale"] = decision.rationale
    status["agent_api_cost_usd"] = float(decision.api_cost_usd)

    targets_df = (
        target_notional.rename("target_notional_usd")
        .reset_index()
        .rename(columns={"index": "symbol"})
        .sort_values("target_notional_usd", ascending=False)
    )
    _write_csv(paths["targets"], targets_df)

    orders_df = pd.DataFrame(
        {
            "symbol": deltas.index.tolist(),
            "raw_delta_usd": deltas.values,
            "capped_order_usd": [float(capped.get(sym, 0.0)) for sym in deltas.index],
        }
    )
    orders_df = orders_df.sort_values("capped_order_usd", key=lambda s: s.abs(), ascending=False)
    _write_csv(paths["orders"], orders_df)

    turnover_usd = float(capped.abs().sum())
    api_cost_bps = (float(decision.api_cost_usd) / equity * 10000.0) if equity > 0 else 0.0
    trade_cost_bps = econ.fee_bps + econ.slippage_bps
    total_cost_bps = trade_cost_bps + api_cost_bps + econ.extra_safety_bps
    net_expected_bps = econ.expected_edge_bps - total_cost_bps
    status["estimated_total_cost_bps"] = float(total_cost_bps)
    status["estimated_net_edge_bps"] = float(net_expected_bps)

    _append_api_cost_row(
        cfg,
        {
            "timestamp_utc": now_utc.isoformat(),
            "signal_date": str(dt.date()),
            "equity_usd": float(equity),
            "turnover_usd": turnover_usd,
            "prompt_tokens": int(decision.prompt_tokens),
            "completion_tokens": int(decision.completion_tokens),
            "api_cost_usd": float(decision.api_cost_usd),
            "estimated_total_cost_bps": float(total_cost_bps),
            "expected_edge_bps": float(econ.expected_edge_bps),
            "estimated_net_edge_bps": float(net_expected_bps),
        },
    )

    print(f"Signal date: {dt.date()} | equity=${equity:,.2f}")
    if econ.enabled:
        if decision.api_cost_usd > econ.max_api_cost_per_run_usd:
            reason = (
                f"Econ gate blocked trading: API cost ${decision.api_cost_usd:.4f} > "
                f"max ${econ.max_api_cost_per_run_usd:.4f}"
            )
            status["economics_ok"] = False
            status["economics_reason"] = reason
            _write_status(cfg, status)
            _append_status_row(
                cfg,
                {
                    "timestamp_utc": status["timestamp_utc"],
                    "mode": status["mode"],
                    "equity": status["equity"],
                    "day_open_equity": status["day_open_equity"],
                    "day_pnl_usd": status["day_pnl_usd"],
                    "day_pnl_pct": status["day_pnl_pct"],
                    "risk_guard_ok": status["risk_guard_ok"],
                    "orders_submitted": status["orders_submitted"],
                    "agent_api_cost_usd": status["agent_api_cost_usd"],
                    "estimated_net_edge_bps": status["estimated_net_edge_bps"],
                    "economics_ok": status["economics_ok"],
                },
            )
            print(reason)
            return
        if net_expected_bps <= 0:
            reason = (
                f"Econ gate blocked trading: expected_edge_bps {econ.expected_edge_bps:.2f} <= "
                f"cost_bps {total_cost_bps:.2f}"
            )
            status["economics_ok"] = False
            status["economics_reason"] = reason
            _write_status(cfg, status)
            _append_status_row(
                cfg,
                {
                    "timestamp_utc": status["timestamp_utc"],
                    "mode": status["mode"],
                    "equity": status["equity"],
                    "day_open_equity": status["day_open_equity"],
                    "day_pnl_usd": status["day_pnl_usd"],
                    "day_pnl_pct": status["day_pnl_pct"],
                    "risk_guard_ok": status["risk_guard_ok"],
                    "orders_submitted": status["orders_submitted"],
                    "agent_api_cost_usd": status["agent_api_cost_usd"],
                    "estimated_net_edge_bps": status["estimated_net_edge_bps"],
                    "economics_ok": status["economics_ok"],
                },
            )
            print(reason)
            return
    if capped.empty:
        print("No orders after risk caps/thresholds.")
        _write_status(cfg, status)
        _append_status_row(
            cfg,
            {
                "timestamp_utc": status["timestamp_utc"],
                "mode": status["mode"],
                "equity": status["equity"],
                "day_open_equity": status["day_open_equity"],
                "day_pnl_usd": status["day_pnl_usd"],
                "day_pnl_pct": status["day_pnl_pct"],
                "risk_guard_ok": status["risk_guard_ok"],
                "orders_submitted": status["orders_submitted"],
                "agent_api_cost_usd": status["agent_api_cost_usd"],
                "estimated_net_edge_bps": status["estimated_net_edge_bps"],
                "economics_ok": status["economics_ok"],
            },
        )
        return

    submitted = 0
    for sym, notional in capped.items():
        broker.submit_market_order(sym, float(notional))
        submitted += 1
    status["orders_submitted"] = submitted
    _write_status(cfg, status)
    _append_status_row(
        cfg,
        {
            "timestamp_utc": status["timestamp_utc"],
            "mode": status["mode"],
            "equity": status["equity"],
            "day_open_equity": status["day_open_equity"],
            "day_pnl_usd": status["day_pnl_usd"],
            "day_pnl_pct": status["day_pnl_pct"],
            "risk_guard_ok": status["risk_guard_ok"],
            "orders_submitted": status["orders_submitted"],
            "agent_api_cost_usd": status["agent_api_cost_usd"],
            "estimated_net_edge_bps": status["estimated_net_edge_bps"],
            "economics_ok": status["economics_ok"],
        },
    )


def should_run_cycle(now_utc: pd.Timestamp, run_at_utc: str, last_run_day: str | None) -> bool:
    hh, mm = run_at_utc.split(":")
    target = now_utc.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
    today = now_utc.strftime("%Y-%m-%d")
    return now_utc >= target and last_run_day != today


def run_loop_cmd(cfg: dict) -> None:
    scfg = cfg.get("schedule", {})
    run_at_utc = str(scfg.get("run_at_utc", "14:35"))
    poll_seconds = int(scfg.get("poll_seconds", 30))
    state_file = Path(scfg.get("loop_state_file", "state/loop_state.json"))
    state_file.parent.mkdir(parents=True, exist_ok=True)

    if state_file.exists():
        with open(state_file, "r", encoding="utf-8") as f:
            loop_state = yaml.safe_load(f) or {}
    else:
        loop_state = {}
    last_run_day = loop_state.get("last_run_day")

    print(f"Starting loop. UTC schedule={run_at_utc}, poll_seconds={poll_seconds}")
    while True:
        now_utc = pd.Timestamp.now("UTC")
        if should_run_cycle(now_utc, run_at_utc=run_at_utc, last_run_day=last_run_day):
            try:
                run_once_cmd(cfg)
            except Exception as e:
                _write_status(
                    cfg,
                    {
                        "timestamp_utc": now_utc.isoformat(),
                        "mode": cfg.get("mode", "paper"),
                        "risk_guard_ok": False,
                        "risk_guard_reason": f"runtime_error: {e}",
                        "orders_submitted": 0,
                        "signal_date": None,
                        "agent_rationale": "",
                    },
                )
                print(f"run-once failed: {e}")
            last_run_day = now_utc.strftime("%Y-%m-%d")
            with open(state_file, "w", encoding="utf-8") as f:
                yaml.safe_dump({"last_run_day": last_run_day}, f)
        time.sleep(poll_seconds)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LV/MV/HV electrical equipment trading bot")
    p.add_argument("command", choices=["backtest", "run-once", "run-loop"])
    p.add_argument("--config", default="config.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.command == "backtest":
        run_backtest_cmd(cfg)
    elif args.command == "run-once":
        run_once_cmd(cfg)
    elif args.command == "run-loop":
        run_loop_cmd(cfg)


if __name__ == "__main__":
    main()
