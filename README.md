# LV/MV/HV Electrical Equipment Trading Bot

This project is a Python trading bot focused on electrical equipment stocks across low-voltage (LV), medium-voltage (MV), and high-voltage (HV) segments.

It includes:
- Segment-aware stock universe
- Momentum + volatility ranking strategy
- Optional AI research agent for score tilts/block lists
- Position sizing with risk caps
- Daily loop for signal generation
- Hard risk guards (equity floor, daily loss brake, turnover/order caps)
- Paper/live broker interface (Alpaca implementation stub)
- Simple backtester on daily OHLCV data
- Backtest report export (stats + daily/monthly/yearly CSVs)
- Streamlit dashboard for current status and latest signals

## Important
- This is not financial advice.
- Use paper trading first.
- Validate symbols, slippage, and transaction costs before live use.

## Quick Start

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Copy config:

```bash
cp config.example.yaml config.yaml
```

4. Run a backtest:

```bash
python main.py backtest --config config.yaml
```

5. Run one live/paper cycle:

```bash
python3 main.py run-once --config config.yaml
```

6. Run daily scheduler loop:

```bash
python3 main.py run-loop --config config.yaml
```

7. Run the local dashboard:

```bash
streamlit run dashboard.py
```

## Strategy Summary

At each rebalance:
- Compute lookback momentum (default 90d) per symbol.
- Penalize high realized volatility (default 20d).
- Rank symbols within each segment (LV/MV/HV).
- Select top N per segment.
- Size positions by inverse volatility with max weight constraints.
- Apply optional stop-loss checks in live loop.
- If enabled, AI agent can add small score adjustments and block symbols before sizing.

## AI Agent

The bot includes an optional AI research agent layer in `ai_agent.py`.

- Default is `disabled` (no-op).
- When enabled, it calls OpenAI to return:
  - `score_adjustments` (small additive tilts per symbol)
  - `blocked_symbols` (symbols to exclude this cycle)
  - `rationale` (logged text)
- Hard risk guards still run after agent output and can block/cap orders.

Enable in `config.yaml`:

```yaml
ai_agent:
  enabled: true
  provider: openai
  model: "gpt-4o-mini"
  base_url: "https://api.openai.com/v1"
  api_key: "YOUR_OPENAI_API_KEY"
  timeout_seconds: 30
  max_abs_adjustment: 0.15
```

## Files

- `main.py`: CLI entrypoint
- `dashboard.py`: local Streamlit status dashboard
- `strategy.py`: ranking and allocation logic
- `backtest.py`: portfolio simulation
- `data.py`: yfinance data loading and feature prep
- `broker.py`: broker interface + Alpaca adapter
- `risk.py`: risk guardrails and state store
- `reporting.py`: report export helpers
- `config.example.yaml`: configurable parameters

## Commands

- `python3 main.py backtest --config config.yaml`
- `python3 main.py run-once --config config.yaml`
- `python3 main.py run-loop --config config.yaml`
- `streamlit run dashboard.py`

## Report Outputs

Backtest exports to `reports/` by default:
- `equity_curve.csv`
- `daily_returns.csv`
- `turnover.csv`
- `monthly_returns.csv`
- `yearly_returns.csv`
- `stats.json`

Runtime status exports to `state/`:
- `status.json`
- `latest_orders.csv`
- `latest_targets.csv`
