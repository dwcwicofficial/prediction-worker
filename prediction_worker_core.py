#!/usr/bin/env python3
"""Prediction Worker operating core. Paper-only. CONTROL law is not optional."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, List, Optional
from timesfm_adapter import TimesFMAdapter, price_path_to_prob

MAX_NICHE_PCT = 0.04
FEE_PROXY = 0.02
MIN_EDGE = 0.03
MIN_LIQUIDITY_USD = 2500.0
MIN_HISTORY = 8

@dataclass
class MarketFrame:
    condition_id: str
    yes_mid: float
    spread: float = 0.04
    liquidity_usd: float = 10000.0
    hours_to_resolve: float = 72.0
    history: List[float] = field(default_factory=list)
    category: str = "unknown"

@dataclass
class Decision:
    action: str
    estimated_prob: float
    edge: float
    reason: str
    size_hint_pct: float = 0.0
    backend: str = "fallback"

class PredictionWorkerCore:
    def __init__(self, adapter: Optional[TimesFMAdapter] = None):
        self.adapter = adapter or TimesFMAdapter(backend="fallback")
        self.brier_n = 0
        self.brier_sum = 0.0

    def ensemble_prob(self, frame: MarketFrame) -> dict:
        market = min(0.95, max(0.05, float(frame.yes_mid)))
        tfm = price_path_to_prob(frame.history or [market], market_price=market, adapter=self.adapter)
        model_p = float(tfm["estimated_prob"])
        blended = min(0.95, max(0.05, 0.45 * model_p + 0.40 * market + 0.15 * 0.50))
        return {"estimated_prob": round(blended, 4), "model_p": round(model_p, 4), "market": round(market, 4), "backend": tfm.get("backend", "fallback")}

    def gate(self, frame: MarketFrame):
        if frame.yes_mid <= 0.05 or frame.yes_mid >= 0.95:
            return "PASS: near-certain quote"
        if frame.liquidity_usd < MIN_LIQUIDITY_USD:
            return "PASS: thin book"
        if frame.hours_to_resolve < 0.5:
            return "PASS: resolving too soon"
        if frame.hours_to_resolve > 24 * 90:
            return "PASS: horizon too long"
        if len(frame.history) < MIN_HISTORY:
            return "PASS: not enough path"
        if frame.spread >= 0.12:
            return "PASS: wide spread"
        return None

    def decide(self, frame: MarketFrame) -> Decision:
        blocked = self.gate(frame)
        if blocked:
            return Decision("PASS", frame.yes_mid, 0.0, blocked)
        ens = self.ensemble_prob(frame)
        p, mid = ens["estimated_prob"], ens["market"]
        cost = max(FEE_PROXY, 0.5 * float(frame.spread)) + FEE_PROXY
        edge_yes = p - mid
        if abs(edge_yes) < MIN_EDGE + cost * 0.25:
            return Decision("PASS", p, round(edge_yes, 4), "PASS: edge inside friction", backend=ens["backend"])
        edge_no = (1.0 - p) - (1.0 - mid)
        if edge_yes >= edge_no and edge_yes > 0:
            action, edge = "BUY_YES", edge_yes
        elif edge_no > 0:
            action, edge = "BUY_NO", edge_no
        else:
            return Decision("PASS", p, round(edge_yes, 4), "PASS: no positive edge")
        hint = min(MAX_NICHE_PCT, max(0.005, abs(edge) * 0.25))
        return Decision(action, p, round(edge, 4), f"{action} ensemble_p={p} mid={mid}", round(hint * 100, 3), ens["backend"])

    def mark_resolution(self, predicted, outcome):
        self.brier_n += 1
        self.brier_sum += (float(predicted) - float(outcome)) ** 2
        return self.brier()

    def brier(self):
        return 0.0 if self.brier_n == 0 else round(self.brier_sum / self.brier_n, 6)

    def decide_batch(self, frames: Iterable[MarketFrame]):
        locked, out = {}, []
        for frame in frames:
            d = self.decide(frame)
            if frame.condition_id in locked and d.action not in {"PASS", locked[frame.condition_id]}:
                d = Decision("PASS", d.estimated_prob, 0.0, "PASS: opposite-side lock")
            if d.action != "PASS":
                locked[frame.condition_id] = d.action
            out.append(d)
        return out
