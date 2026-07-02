"""Poker44 bot detector — uid7 (Ares90125/poker7), v5 "sanitization fix".

Model: **ExtraTrees + HistGradientBoosting soft-vote ensemble** over the v3
behavioral feature set with the fragile identity / raw-magnitude aggregates
REMOVED (candidate C2 — see features.py FEATURE_NAMES). Those columns went
out-of-distribution on the validator-sanitized live feed and collapsed the raw
predict_proba spread (v3 live raw-std ~0.003, v4 ~0.012); dropping them plus
training on hands passed through the validator's prepare_hand_for_miner
(train==serve) restores a healthy live raw-std. Output = **within-batch rank**,
which matches the validator's ranking-based reward.

IMPORTANT — inference does NOT sanitize. Live chunks arrive already sanitized by
the validator (prepare_hand_for_miner runs validator-side, per hand). Only
TRAINING sanitizes raw benchmark hands (see train_model.py). Sanitizing again
here would double-transform already-sanitized hands and re-introduce skew, so
this path featurizes the incoming chunks directly.

The trained model is the committed `model.joblib` (v5_sani candidate C2).
sklearn loads it at inference. `score_batch(chunks)` returns one rank-based
bot-risk score in [0,1] per chunk.
"""
from __future__ import annotations

import os

import numpy as np
import joblib

from poker44_model.features import chunk_features, FEATURE_NAMES

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        _MODEL = joblib.load(os.path.join(os.path.dirname(__file__), "model.joblib"))
    return _MODEL


def _rank_normalize(vals):
    n = len(vals)
    if n <= 1:
        return [0.5] * n
    order = sorted(range(n), key=lambda i: vals[i])
    out = [0.0] * n
    for pos, i in enumerate(order):
        out[i] = round(pos / (n - 1), 6)
    return out


def _raw_scores(model, chunks):
    # Live chunks are already sanitized by the validator; featurize as-is.
    rows = []
    for c in chunks:
        feats = chunk_features(c)          # compute the feature set ONCE per chunk
        rows.append([feats.get(k, 0.0) for k in FEATURE_NAMES])
    return model.predict_proba(np.array(rows, dtype=float))[:, 1]


def score_batch(chunks):
    """One bot-risk score in [0,1] per chunk, ranked within the batch."""
    chunks = chunks or []
    if not chunks:
        return []
    try:
        return _rank_normalize(list(_raw_scores(_model(), chunks)))
    except Exception:
        return [0.5] * len(chunks)


def score_chunk(chunk):
    """Single-chunk model probability (fallback; batch path is score_batch)."""
    try:
        if not chunk:
            return 0.5
        return round(float(_raw_scores(_model(), [chunk])[0]), 6)
    except Exception:
        return 0.5
