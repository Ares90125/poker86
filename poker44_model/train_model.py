"""Reproducible training for uid7 v5 (sanitization fix) → writes model.joblib.

v3 behavioral features with the fragile identity / raw-magnitude aggregates
removed (candidate C2 subset — see features.py FEATURE_NAMES), scored by an
ExtraTrees + HistGradientBoosting soft-vote ensemble.

The KEY fix: every raw benchmark hand is passed through the validator's
`prepare_hand_for_miner` (the anti-leakage / canonicalization sanitizer, from
poker44/validator/payload_view.py) BEFORE feature extraction, so the training
distribution matches what the validator actually serves miners (train==serve).
Live chunks are already sanitized validator-side, so inference does NOT
re-sanitize — only training does.

    python3 poker44_model/train_model.py --data /root/ares/Poker/train/raw \
        --payload-view /root/ares/Poker/main/poker44/validator/payload_view.py
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import typing

import numpy as np
import joblib
from sklearn.ensemble import (ExtraTreesClassifier,
                              HistGradientBoostingClassifier,
                              VotingClassifier)

from poker44_model.features import chunk_features, FEATURE_NAMES


def _load_sanitizer(pv_path):
    spec = importlib.util.spec_from_file_location("_p44_payload_view", pv_path)
    pv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pv)
    pv.Optional = typing.Optional  # payload_view uses Optional but never imports it
    fn = pv.prepare_hand_for_miner

    def sanitize_chunk(chunk):
        out = []
        for h in (chunk or []):
            try:
                out.append(fn(h))
            except Exception:
                out.append(h)
        return out

    return sanitize_chunk


def load(raw):
    out = []
    for f in sorted(glob.glob(os.path.join(raw, "chunks_*.json"))):
        for rc in json.load(open(f)).get("chunks", []):
            for g, l in zip(rc.get("chunks") or [], rc.get("groundTruth") or []):
                out.append((g, int(l)))
    return out


def build_ensemble(seed=0):
    et = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=4,
                              random_state=seed, n_jobs=-1)
    hgb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03,
                                         max_iter=300, l2_regularization=1.0,
                                         random_state=seed)
    return VotingClassifier(estimators=[("et", et), ("hgb", hgb)], voting="soft")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to train/raw chunk JSON dir")
    ap.add_argument("--payload-view",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "..", "poker44", "validator", "payload_view.py"),
                    help="path to poker44/validator/payload_view.py (the sanitizer); "
                         "defaults to this repo's own copy so daily_update can retrain")
    args = ap.parse_args()

    sanitize_chunk = _load_sanitizer(args.payload_view)

    data = load(args.data)
    rows, y = [], []
    for g, l in data:
        feats = chunk_features(sanitize_chunk(g))   # TRAIN == SERVE: sanitize raw hands
        rows.append([feats.get(k, 0.0) for k in FEATURE_NAMES])
        y.append(l)
    X = np.array(rows, dtype=float)
    y = np.array(y)

    model = build_ensemble(seed=0).fit(X, y)

    out = os.path.join(os.path.dirname(__file__), "model.joblib")
    joblib.dump(model, out)
    print(f"wrote {out} ({len(data)} examples, {len(FEATURE_NAMES)} features)")


if __name__ == "__main__":
    main()
