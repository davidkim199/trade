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
import yfinance as yf


st.set_page_config(page_title="US + Korea Bot Dashboard", layout="wide")


MARKET_CONFIGS = {
    "US": Path("config.us.yaml"),
    "Korea": Path("config.kr.yaml"),
}


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def runtime_paths(cfg: dict) -> dict[str, Path]:
    rcfg = cfg.get("runtime", {})
    report_dir = Path(cfg.get("backtest", {}).get("report_dir", "reports"))
    return {
        "status": Path(rcfg.get("status_file", "state/status.json")),
        "orders": Path(rcfg.get("orders_file", "state/latest_orders.csv")),
        "targets": Path(rcfg.get("targets_file", "state/latest_targets.csv")),
        "api_costs": Path(rcfg.get("api_costs_file", "state/api_costs.csv")),
        "equity_curve": report_dir / "equity_curve.csv",
        "stats": report_dir / "stats.json",
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


def run_bot_command(config_path: Path, command: str) -> tuple[int, str]:
    cmd = [sys.executable, "main.py", command, "--config", str(config_path)]
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


def display_symbol(sym: str, names: dict[str, str]) -> str:
    name = names.get(sym, "")
    return f"{name} ({sym})" if name else sym


@st.cache_data(ttl=120)
def fetch_latest_prices(symbols: tuple[str, ...]) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame(columns=["symbol", "last_price"])

    def extract_last_close(raw: pd.DataFrame, sym: str) -> float | None:
        if raw is None or raw.empty:
            return None
        close_names = {"close", "adj close", "adjclose"}

        frame = raw.copy()
        if isinstance(frame.columns, pd.MultiIndex):
            for lvl in range(frame.columns.nlevels):
                labels = frame.columns.get_level_values(lvl)
                lowered = [str(x).lower() for x in labels]
                if any(x in close_names for x in lowered):
                    picked = [labels[i] for i, x in enumerate(lowered) if x in close_names][0]
                    out = frame.xs(picked, axis=1, level=lvl, drop_level=True)
                    if isinstance(out, pd.Series):
                        vals = pd.to_numeric(out, errors="coerce").dropna()
                        return float(vals.iloc[-1]) if not vals.empty else None
                    # If still multi-col, prefer symbol col if present, else first numeric col.
                    out.columns = [str(c) for c in out.columns]
                    if sym in out.columns:
                        vals = pd.to_numeric(out[sym], errors="coerce").dropna()
                        return float(vals.iloc[-1]) if not vals.empty else None
                    for c in out.columns:
                        vals = pd.to_numeric(out[c], errors="coerce").dropna()
                        if not vals.empty:
                            return float(vals.iloc[-1])
                    return None

        cols = {str(c).lower(): c for c in frame.columns}
        for name in ["close", "adj close", "adjclose"]:
            if name in cols:
                vals = pd.to_numeric(frame[cols[name]], errors="coerce").dropna()
                return float(vals.iloc[-1]) if not vals.empty else None
        return None

    rows: list[dict] = []
    for sym in symbols:
        try:
            raw = yf.download(
                tickers=sym,
                period="7d",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            px = extract_last_close(raw, sym)
            if px is None:
                continue
            rows.append({"symbol": sym, "last_price": float(px)})
        except Exception:
            continue
    return pd.DataFrame(rows)


def render_market_panel(name: str, cfg_path: Path) -> None:
    cfg = load_yaml(cfg_path)
    symbol_names = cfg.get("symbol_names", {})
    paths = runtime_paths(cfg)
    status = read_json(paths["status"])
    orders = read_csv(paths["orders"])
    targets = read_csv(paths["targets"])
    stats = read_json(paths["stats"])
    api_costs = read_csv(paths["api_costs"])
    eq = read_csv(paths["equity_curve"])

    st.subheader(f"{name} Market")
    st.caption(f"Config: `{cfg_path}`")

    universe = cfg.get("universe", {})
    symbols = []
    for syms in universe.values():
        symbols.extend(syms)
    symbols = list(dict.fromkeys(symbols))
    symbol_labels = [display_symbol(s, symbol_names) for s in symbols]
    st.caption("Universe: " + (", ".join(symbol_labels) if symbol_labels else "n/a"))

    prices = fetch_latest_prices(tuple(symbols))
    st.markdown("**Current Prices**")
    if prices.empty:
        st.info("No price snapshot available right now.")
    else:
        p = prices.copy()
        p["company"] = p["symbol"].map(lambda s: symbol_names.get(s, ""))
        p = p[["company", "symbol", "last_price"]].sort_values("symbol")
        st.dataframe(p, width="stretch")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mode", str(status.get("mode", cfg.get("mode", "unknown"))).upper())
    c2.metric("Equity", f"${float(status.get('equity', 0.0)):,.2f}" if "equity" in status else "n/a")
    c3.metric("Orders", int(status.get("orders_submitted", 0)))
    if "risk_guard_ok" in status:
        c4.metric("Risk", "OK" if bool(status.get("risk_guard_ok")) else "BLOCKED")
    else:
        c4.metric("Risk", "N/A")

    st.write("Last Run (UTC):", status.get("timestamp_utc", "n/a"))
    st.write("Signal Date:", status.get("signal_date", "n/a"))
    st.write("Agent Rationale:", status.get("agent_rationale", "n/a"))
    if "economics_reason" in status:
        st.write("Economics Gate:", status.get("economics_reason", "n/a"))

    btn1, btn2 = st.columns(2)
    if btn1.button(f"Run Signal ({name})"):
        rc, msg = run_bot_command(cfg_path, "run-once")
        if rc == 0:
            st.success("Signal run complete.")
        else:
            st.error("Signal run failed.")
        if msg:
            st.code(msg)

    if btn2.button(f"Run Backtest ({name})"):
        rc, msg = run_bot_command(cfg_path, "backtest")
        if rc == 0:
            st.success("Backtest complete.")
        else:
            st.error("Backtest failed.")
        if msg:
            st.code(msg)

    st.markdown("**Latest Orders**")
    if orders.empty:
        st.info("No orders yet.")
    else:
        view = orders.copy()
        if "symbol" in view.columns:
            view["company"] = view["symbol"].map(lambda s: symbol_names.get(s, ""))
        if "capped_order_usd" in view.columns:
            view = view[view["capped_order_usd"].abs() > 0]
        st.dataframe(view, width="stretch")

    st.markdown("**Latest Target Notional**")
    if targets.empty:
        st.info("No targets yet.")
    else:
        view = targets.copy()
        if "symbol" in view.columns:
            view["company"] = view["symbol"].map(lambda s: symbol_names.get(s, ""))
        if "target_notional_usd" in view.columns:
            view = view.sort_values("target_notional_usd", ascending=False)
        st.dataframe(view.head(20), width="stretch")

    st.markdown("**Position Plan (This Run)**")
    if targets.empty:
        st.info("No position plan available yet.")
    else:
        tgt = targets.copy()
        if "symbol" not in tgt.columns or "target_notional_usd" not in tgt.columns:
            st.info("Target data format missing required columns.")
        else:
            ords = orders.copy() if not orders.empty else pd.DataFrame(columns=["symbol", "capped_order_usd"])
            if "symbol" not in ords.columns or "capped_order_usd" not in ords.columns:
                ords = pd.DataFrame(columns=["symbol", "capped_order_usd"])

            merged = tgt.merge(
                ords[["symbol", "capped_order_usd"]],
                on="symbol",
                how="left",
            )
            merged["capped_order_usd"] = pd.to_numeric(merged["capped_order_usd"], errors="coerce").fillna(0.0)
            merged["target_notional_usd"] = pd.to_numeric(merged["target_notional_usd"], errors="coerce").fillna(0.0)
            # For paper mode with no persisted positions, treat this run's capped orders as current change.
            merged["estimated_current_notional_usd"] = merged["capped_order_usd"]
            merged["remaining_to_target_usd"] = merged["target_notional_usd"] - merged["estimated_current_notional_usd"]
            merged["company"] = merged["symbol"].map(lambda s: symbol_names.get(s, ""))
            cols = [
                "company",
                "symbol",
                "estimated_current_notional_usd",
                "target_notional_usd",
                "remaining_to_target_usd",
            ]
            merged = merged[cols].sort_values("target_notional_usd", ascending=False)
            st.dataframe(merged, width="stretch")
            st.caption(
                "Estimated current notional is based on this run's capped orders. "
                "It is not a broker-confirmed fill ledger."
            )

    st.markdown("**Backtest Stats**")
    if stats:
        st.json(stats)
    else:
        st.info("No backtest stats yet.")

    st.markdown("**API Cost**")
    if api_costs.empty:
        st.info("No API cost records yet.")
    else:
        total = float(api_costs.get("api_cost_usd", pd.Series(dtype=float)).fillna(0.0).sum())
        st.metric("Cumulative API Cost", f"${total:,.4f}")
        cols = [c for c in ["timestamp_utc", "signal_date", "api_cost_usd", "estimated_net_edge_bps"] if c in api_costs.columns]
        st.dataframe(api_costs[cols].tail(20), width="stretch")

    st.markdown("**Equity Curve**")
    if eq.empty:
        st.info("No equity curve yet.")
    else:
        if "date" in eq.columns:
            eq["date"] = pd.to_datetime(eq["date"], errors="coerce")
            eq = eq.dropna(subset=["date"]).set_index("date")
        if "equity" in eq.columns:
            st.line_chart(eq["equity"])


st.title("US + Korea Trading Dashboard")
st.caption("Run and monitor US and Korean stock universes side by side.")
st.button("Refresh")

st.sidebar.header("Auto Refresh")
auto_refresh = st.sidebar.checkbox("Enable Auto Refresh", value=False)
refresh_seconds = int(st.sidebar.number_input("Refresh Interval (sec)", min_value=5, max_value=300, value=10))

left, right = st.columns(2)
with left:
    render_market_panel("US", MARKET_CONFIGS["US"])
with right:
    render_market_panel("Korea", MARKET_CONFIGS["Korea"])

if auto_refresh:
    st.caption(f"Auto-refreshing every {refresh_seconds} seconds...")
    time.sleep(refresh_seconds)
    st.rerun()
