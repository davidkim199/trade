from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml


st.set_page_config(page_title="LV/MV/HV Bot Dashboard", layout="wide")


def load_config(path: str = "config.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def runtime_paths(cfg: dict) -> dict[str, Path]:
    rcfg = cfg.get("runtime", {})
    return {
        "status": Path(rcfg.get("status_file", "state/status.json")),
        "orders": Path(rcfg.get("orders_file", "state/latest_orders.csv")),
        "targets": Path(rcfg.get("targets_file", "state/latest_targets.csv")),
        "equity_state": Path(cfg.get("state_file", "state/equity_state.json")),
        "equity_curve": Path(cfg.get("backtest", {}).get("report_dir", "reports")) / "equity_curve.csv",
        "daily_returns": Path(cfg.get("backtest", {}).get("report_dir", "reports")) / "daily_returns.csv",
        "stats": Path(cfg.get("backtest", {}).get("report_dir", "reports")) / "stats.json",
    }


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


cfg = load_config()
paths = runtime_paths(cfg)

st.title("LV/MV/HV Electrical Equipment Bot")
st.caption("Local status UI for signals, risk guards, orders, targets, and reports.")
st.button("Refresh")

status = read_json(paths["status"])
equity_state = read_json(paths["equity_state"])
orders = read_csv(paths["orders"])
targets = read_csv(paths["targets"])
stats = read_json(paths["stats"])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Mode", str(status.get("mode", cfg.get("mode", "unknown"))).upper())
col2.metric("Equity", f"${float(status.get('equity', 0.0)):,.2f}")
col3.metric("Orders Submitted", int(status.get("orders_submitted", 0)))
guard_ok = bool(status.get("risk_guard_ok", False))
col4.metric("Risk Guard", "OK" if guard_ok else "BLOCKED")

st.write("Last Run (UTC):", status.get("timestamp_utc", "n/a"))
st.write("Signal Date:", status.get("signal_date", "n/a"))
if status.get("risk_guard_reason"):
    st.write("Risk Reason:", status.get("risk_guard_reason"))
if status.get("agent_rationale"):
    st.write("Agent Rationale:", status.get("agent_rationale"))

st.subheader("Day State")
st.json(equity_state if equity_state else {"info": "No day state yet."})

st.subheader("Latest Orders")
if orders.empty:
    st.info("No order snapshot available yet. Run `python main.py run-once --config config.yaml` first.")
else:
    view = orders.copy()
    if "capped_order_usd" in view.columns:
        view = view[view["capped_order_usd"].abs() > 0].copy()
    st.dataframe(view, width="stretch")

st.subheader("Latest Target Notional")
if targets.empty:
    st.info("No targets snapshot available yet.")
else:
    view = targets.copy()
    if "target_notional_usd" in view.columns:
        view = view.sort_values("target_notional_usd", ascending=False).head(20)
    st.dataframe(view, width="stretch")

st.subheader("Backtest Stats")
if stats:
    st.json(stats)
else:
    st.info("No backtest stats yet. Run `python main.py backtest --config config.yaml` first.")

st.subheader("Equity Curve")
equity_curve = read_csv(paths["equity_curve"])
if equity_curve.empty:
    st.info("No equity curve found.")
else:
    if "date" in equity_curve.columns:
        equity_curve["date"] = pd.to_datetime(equity_curve["date"], errors="coerce")
        equity_curve = equity_curve.dropna(subset=["date"]).set_index("date")
    if "equity" in equity_curve.columns:
        st.line_chart(equity_curve["equity"])

st.subheader("Daily Returns")
daily_returns = read_csv(paths["daily_returns"])
if daily_returns.empty:
    st.info("No daily returns found.")
else:
    if "date" in daily_returns.columns:
        daily_returns["date"] = pd.to_datetime(daily_returns["date"], errors="coerce")
        daily_returns = daily_returns.dropna(subset=["date"]).set_index("date")
    if "daily_return" in daily_returns.columns:
        st.bar_chart(daily_returns["daily_return"].tail(120))
