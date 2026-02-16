from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

import pandas as pd
import requests


@dataclass
class AgentDecision:
    score_adjustments: dict[str, float]
    blocked_symbols: list[str]
    rationale: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    api_cost_usd: float = 0.0


class ResearchAgent(Protocol):
    def decide(
        self,
        date: pd.Timestamp,
        universe: dict[str, list[str]],
        prices: pd.Series,
        momentum: pd.Series,
        vol: pd.Series,
        base_scores: pd.Series,
    ) -> AgentDecision:
        ...


@dataclass
class NullResearchAgent:
    def decide(
        self,
        date: pd.Timestamp,
        universe: dict[str, list[str]],
        prices: pd.Series,
        momentum: pd.Series,
        vol: pd.Series,
        base_scores: pd.Series,
    ) -> AgentDecision:
        return AgentDecision(score_adjustments={}, blocked_symbols=[], rationale="agent_disabled")


@dataclass
class OpenAIResearchAgent:
    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: int = 30
    max_abs_adjustment: float = 0.15
    input_cost_per_1k_tokens_usd: float = 0.00015
    output_cost_per_1k_tokens_usd: float = 0.0006

    def _build_payload(
        self,
        date: pd.Timestamp,
        universe: dict[str, list[str]],
        prices: pd.Series,
        momentum: pd.Series,
        vol: pd.Series,
        base_scores: pd.Series,
    ) -> dict:
        symbols = sorted(set(base_scores.index.tolist()))
        snapshot = []
        for sym in symbols:
            snapshot.append(
                {
                    "symbol": sym,
                    "price": float(prices.get(sym)) if pd.notna(prices.get(sym)) else None,
                    "momentum_lookback": float(momentum.get(sym)) if pd.notna(momentum.get(sym)) else None,
                    "realized_vol": float(vol.get(sym)) if pd.notna(vol.get(sym)) else None,
                    "base_score": float(base_scores.get(sym)) if pd.notna(base_scores.get(sym)) else None,
                }
            )

        system_msg = (
            "You are a cautious equity-research assistant for an execution bot. "
            "You may only suggest small score tilts and optional symbol blocks. "
            "Do not provide leverage, derivatives, or concentration suggestions."
        )
        user_msg = {
            "date": date.strftime("%Y-%m-%d"),
            "segments": universe,
            "data_snapshot": snapshot,
            "task": (
                "Return JSON with keys: score_adjustments, blocked_symbols, rationale. "
                "score_adjustments must map symbol->small float in range [-0.15, 0.15]. "
                "blocked_symbols must be a list of symbols from data_snapshot."
            ),
        }
        return {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": json.dumps(user_msg)},
            ],
            "temperature": 0.2,
        }

    def decide(
        self,
        date: pd.Timestamp,
        universe: dict[str, list[str]],
        prices: pd.Series,
        momentum: pd.Series,
        vol: pd.Series,
        base_scores: pd.Series,
    ) -> AgentDecision:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = self._build_payload(
            date=date,
            universe=universe,
            prices=prices,
            momentum=momentum,
            vol=vol,
            base_scores=base_scores,
        )

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_seconds)
            resp.raise_for_status()
            body = resp.json()
            content = body["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except Exception as e:
            return AgentDecision(score_adjustments={}, blocked_symbols=[], rationale=f"agent_error: {e}")

        valid_symbols = set(base_scores.index.tolist())
        raw_adj = parsed.get("score_adjustments", {}) if isinstance(parsed, dict) else {}
        adjustments: dict[str, float] = {}
        if isinstance(raw_adj, dict):
            for k, v in raw_adj.items():
                if k in valid_symbols:
                    try:
                        clipped = max(-self.max_abs_adjustment, min(self.max_abs_adjustment, float(v)))
                        adjustments[k] = clipped
                    except (TypeError, ValueError):
                        continue

        raw_blocked = parsed.get("blocked_symbols", []) if isinstance(parsed, dict) else []
        blocked = []
        if isinstance(raw_blocked, list):
            blocked = [s for s in raw_blocked if isinstance(s, str) and s in valid_symbols]

        rationale = str(parsed.get("rationale", "")) if isinstance(parsed, dict) else ""
        usage = body.get("usage", {}) if isinstance(body, dict) else {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        api_cost_usd = (
            (prompt_tokens / 1000.0) * self.input_cost_per_1k_tokens_usd
            + (completion_tokens / 1000.0) * self.output_cost_per_1k_tokens_usd
        )
        return AgentDecision(
            score_adjustments=adjustments,
            blocked_symbols=blocked,
            rationale=rationale,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            api_cost_usd=float(api_cost_usd),
        )
