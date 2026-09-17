# TimesFM in Prediction Worker

Google Research decoder-only time-series foundation model (ICML 2024).

INTEL builds the yes-price path. ALPHA runs `price_path_to_prob()` into Kelly. EXEC and CONTROL unchanged.

Default backend is fallback (no 200M download). Production flag:

```bash
pip install "timesfm[torch]"
export TIMESFM_BACKEND=timesfm25
```

Use TimesFM 2.5 weights (Apache-2.0). Do not put TimesFM 3.0 weights on Book I — those are non-commercial.
