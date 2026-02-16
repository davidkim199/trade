from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd
import streamlit as st
import yaml


st.set_page_config(page_title="LV/MV/HV Bot Dashboard", layout="wide")


def load_config() -> tuple[dict, str]:
    p = Path("config.yaml")
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}, str(p)
    p = Path("config.example.yaml")
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}, str(p)
    return {}, "config.yaml"


def runtime_paths(cfg: dict) -> dict[str, Path]:
    rcfg = cfg.get("runtime", {})
    return {
        "status": Path(rcfg.get("status_file", "state/status.json")),
        "orders": Path(rcfg.get("orders_file", "state/latest_orders.csv")),
        "targets": Path(rcfg.get("targets_file", "state/latest_targets.csv")),
        "api_costs": Path(rcfg.get("api_costs_file", "state/api_costs.csv")),
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


cfg, cfg_path = load_config()
paths = runtime_paths(cfg)

st.title("LV/MV/HV Electrical Equipment Bot")
st.caption("Local status UI for signals, risk guards, orders, targets, and reports.")
st.button("Refresh")
st.caption(f"Config: `{cfg_path}`")

st.sidebar.header("Auto Refresh")
auto_refresh = st.sidebar.checkbox("Enable Auto Refresh", value=False)
refresh_seconds = int(st.sidebar.number_input("Refresh Interval (sec)", min_value=5, max_value=300, value=10))

col_a, col_b = st.columns(2)
run_once_clicked = col_a.button("Run Signal Now")
backtest_clicked = col_b.button("Run Backtest Now")


def run_bot_command(args: list[str]) -> tuple[int, str]:
    cmd = [sys.executable, "main.py"] + args + ["--config", cfg_path]
    try:
        env = dict(os.environ)
        try:
            if "OPENAI_API_KEY" not in env:
                secret_key = st.secrets.get("OPENAI_API_KEY", "")
                if secret_key:
                    env["OPENAI_API_KEY"] = str(secret_key)
        except Exception:
            pass
        out = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
        msg = (out.stdout or "") + ("\n" + out.stderr if out.stderr else "")
        return out.returncode, msg.strip()
    except Exception as e:
        return 1, str(e)


if run_once_clicked:
    rc, msg = run_bot_command(["run-once"])
    if rc == 0:
        st.success("Signal run complete.")
    else:
        st.error("Signal run failed.")
    if msg:
        st.code(msg)

if backtest_clicked:
    rc, msg = run_bot_command(["backtest"])
    if rc == 0:
        st.success("Backtest complete.")
    else:
        st.error("Backtest failed.")
    if msg:
        st.code(msg)

status = read_json(paths["status"])
equity_state = read_json(paths["equity_state"])
orders = read_csv(paths["orders"])
targets = read_csv(paths["targets"])
stats = read_json(paths["stats"])
api_costs = read_csv(paths["api_costs"])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Mode", str(status.get("mode", cfg.get("mode", "unknown"))).upper())
if "equity" in status:
    col2.metric("Equity", f"${float(status.get('equity', 0.0)):,.2f}")
else:
    col2.metric("Equity", "n/a")
col3.metric("Orders Submitted", int(status.get("orders_submitted", 0)))
if "risk_guard_ok" not in status:
    col4.metric("Risk Guard", "N/A")
else:
    guard_ok = bool(status.get("risk_guard_ok", False))
    col4.metric("Risk Guard", "OK" if guard_ok else "BLOCKED")

st.write("Last Run (UTC):", status.get("timestamp_utc", "n/a"))
st.write("Signal Date:", status.get("signal_date", "n/a"))
if status.get("risk_guard_reason"):
    st.write("Risk Reason:", status.get("risk_guard_reason"))
if status.get("agent_rationale"):
    st.write("Agent Rationale:", status.get("agent_rationale"))
if "agent_api_cost_usd" in status:
    st.write("Agent API Cost (last run):", f"${float(status.get('agent_api_cost_usd', 0.0)):.6f}")
if "estimated_total_cost_bps" in status:
    st.write("Estimated Total Cost (bps):", f"{float(status.get('estimated_total_cost_bps', 0.0)):.2f}")
if "estimated_net_edge_bps" in status:
    st.write("Estimated Net Edge (bps):", f"{float(status.get('estimated_net_edge_bps', 0.0)):.2f}")
if status.get("economics_reason"):
    st.write("Economics Gate:", status.get("economics_reason"))

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

st.subheader("API Cost Tracker")
if api_costs.empty:
    st.info("No API cost records yet.")
else:
    total_api_cost = float(api_costs.get("api_cost_usd", pd.Series(dtype=float)).fillna(0.0).sum())
    st.metric("Cumulative API Cost", f"${total_api_cost:,.4f}")
    show = api_costs.copy()
    keep = [
        "timestamp_utc",
        "signal_date",
        "api_cost_usd",
        "prompt_tokens",
        "completion_tokens",
        "estimated_total_cost_bps",
        "expected_edge_bps",
        "estimated_net_edge_bps",
    ]
    keep = [c for c in keep if c in show.columns]
    st.dataframe(show[keep].tail(50), width="stretch")

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

if auto_refresh:
    st.caption(f"Auto-refreshing every {refresh_seconds} seconds...")
    time.sleep(refresh_seconds)
    st.rerun()
