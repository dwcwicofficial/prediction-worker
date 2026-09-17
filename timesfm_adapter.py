#!/usr/bin/env python3
"""TimesFM adapter for Prediction Worker.

Hot path stays light. The 8-minute loop never loads Google weights.
ALPHA calls forecast() with a 1-D price history and gets point + quantiles.

Backends
  fallback  — local linear residual model (default)
  timesfm25 — google/timesfm-2.5-200m-pytorch  (Apache-2.0)
  timesfm30 — google/timesfm-3.0-pytorch       (non-commercial; opt-in)

Set TIMESFM_BACKEND=timesfm25 after `pip install timesfm[torch]`.
"""
from __future__ import annotations

import math
import os
from typing import Iterable, List, Sequence

SHIP_MODEL = "google/timesfm-2.5-200m-pytorch"
LAB_MODEL = "google/timesfm-3.0-pytorch"


def _as_list(series: Sequence[float]) -> List[float]:
    out = []
    for x in series:
        try:
            v = float(x)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            out.append(v)
    return out


def fallback_forecast(series: Sequence[float], horizon: int = 8) -> dict:
    xs = _as_list(series)
    h = max(1, int(horizon))
    if len(xs) < 2:
        last = xs[0] if xs else 0.5
        point = [last] * h
        return {"backend": "fallback", "point": point, "q10": point[:], "q50": point[:], "q90": point[:], "context": len(xs), "horizon": h}
    n = len(xs)
    mean_x = (n - 1) / 2.0
    mean_y = sum(xs) / n
    var_x = sum((i - mean_x) ** 2 for i in range(n)) or 1.0
    cov = sum((i - mean_x) * (xs[i] - mean_y) for i in range(n))
    slope = cov / var_x
    intercept = mean_y - slope * mean_x
    resid = [xs[i] - (intercept + slope * i) for i in range(n)]
    sigma = math.sqrt(sum(r * r for r in resid) / max(1, n - 2)) if n > 2 else 0.0
    point = [intercept + slope * (n + k) for k in range(h)]
    z10, z90 = -1.28155, 1.28155
    return {"backend": "fallback", "point": point, "q10": [p + z10 * sigma for p in point], "q50": point[:], "q90": [p + z90 * sigma for p in point], "sigma": sigma, "slope": slope, "context": n, "horizon": h}


class TimesFMAdapter:
    def __init__(self, backend: str | None = None):
        self.backend = (backend or os.getenv("TIMESFM_BACKEND") or "fallback").lower()
        self._model = None
        if self.backend in {"timesfm25", "timesfm", "torch"}:
            self._try_load_25()
        elif self.backend in {"timesfm30", "timesfm3"}:
            self._try_load_30()

    def _try_load_25(self) -> None:
        try:
            import numpy as np  # noqa: F401
            import timesfm
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(SHIP_MODEL)
            model.compile(timesfm.ForecastConfig(max_context=512, max_horizon=128, normalize_inputs=True, per_core_batch_size=4, use_continuous_quantile_head=True, infer_is_positive=True, fix_quantile_crossing=True))
            self._model = model
            self.backend = "timesfm25"
        except Exception:
            self._model = None
            self.backend = "fallback"

    def _try_load_30(self) -> None:
        try:
            from timesfm3 import ModelConfig, TimesFM3Evaluator
            self._model = TimesFM3Evaluator(ModelConfig(checkpoint_path=LAB_MODEL, device="cpu"))
            self.backend = "timesfm30"
        except Exception:
            self._model = None
            self.backend = "fallback"

    def forecast(self, series: Sequence[float], horizon: int = 8) -> dict:
        xs = _as_list(series)
        if self._model is None or self.backend == "fallback":
            return fallback_forecast(xs, horizon)
        try:
            import numpy as np
            arr = np.asarray(xs, dtype="float32")
            if self.backend == "timesfm25":
                point, quant = self._model.forecast(horizon=horizon, inputs=[arr])
                q = quant[0]
                return {"backend": "timesfm25", "point": [float(x) for x in point[0]], "q10": [float(x) for x in q[:, 1]], "q50": [float(x) for x in q[:, 5]], "q90": [float(x) for x in q[:, 9]], "context": len(xs), "horizon": horizon, "model": SHIP_MODEL}
            outputs = list(self._model.predict_batch([arr], horizon=horizon, return_quantiles=True))
            first = outputs[0]
            point = first.get("point") or first.get("mean")
            return {"backend": "timesfm30", "point": [float(x) for x in point], "context": len(xs), "horizon": horizon, "model": LAB_MODEL, "license": "non-commercial"}
        except Exception:
            return fallback_forecast(xs, horizon)


def price_path_to_prob(history: Iterable[float], market_price: float, horizon: int = 4, adapter: TimesFMAdapter | None = None) -> dict:
    tfm = adapter or TimesFMAdapter()
    fc = tfm.forecast(list(history), horizon=horizon)
    point = fc.get("point") or [market_price]
    nxt = float(point[0])
    q10 = float((fc.get("q10") or [nxt])[0])
    q90 = float((fc.get("q90") or [nxt])[0])
    width = max(0.0, q90 - q10)
    p_hat = min(0.95, max(0.05, nxt))
    shrink = min(1.0, width / 0.25) if width else 0.0
    blended = p_hat * (1.0 - 0.45 * shrink) + float(market_price) * (0.45 * shrink)
    blended = min(0.95, max(0.05, blended))
    return {"estimated_prob": round(blended, 4), "forecast_mid": round(nxt, 4), "edge": round(blended - float(market_price), 4), "band": [round(q10, 4), round(q90, 4)], "backend": fc.get("backend", "fallback"), "context": fc.get("context", 0)}


if __name__ == "__main__":
    print(price_path_to_prob([0.42, 0.44, 0.43, 0.46, 0.48, 0.47, 0.49, 0.51], 0.50))
