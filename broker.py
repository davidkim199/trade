from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol

import requests


class Broker(Protocol):
    def get_positions(self) -> Dict[str, float]:
        ...

    def get_equity(self) -> float:
        ...

    def submit_market_order(self, symbol: str, notional_usd: float) -> None:
        ...


@dataclass
class PaperPrintBroker:
    equity: float = 100000.0

    def get_positions(self) -> Dict[str, float]:
        return {}

    def get_equity(self) -> float:
        return self.equity

    def submit_market_order(self, symbol: str, notional_usd: float) -> None:
        print(f"[PAPER-PRINT] ORDER {symbol}: ${notional_usd:,.2f}")


@dataclass
class AlpacaBroker:
    base_url: str
    api_key: str
    api_secret: str

    def _headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json",
        }

    def get_positions(self) -> Dict[str, float]:
        url = f"{self.base_url}/v2/positions"
        resp = requests.get(url, headers=self._headers(), timeout=20)
        resp.raise_for_status()
        data = resp.json()
        return {p["symbol"]: float(p["market_value"]) for p in data}

    def get_equity(self) -> float:
        url = f"{self.base_url}/v2/account"
        resp = requests.get(url, headers=self._headers(), timeout=20)
        resp.raise_for_status()
        return float(resp.json()["equity"])

    def submit_market_order(self, symbol: str, notional_usd: float) -> None:
        side = "buy" if notional_usd > 0 else "sell"
        url = f"{self.base_url}/v2/orders"
        payload = {
            "symbol": symbol,
            "notional": round(abs(notional_usd), 2),
            "side": side,
            "type": "market",
            "time_in_force": "day",
        }
        resp = requests.post(url, json=payload, headers=self._headers(), timeout=20)
        resp.raise_for_status()
