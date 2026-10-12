"""
장기 일봉 캐시(data/cache/prices_long.pkl)에서 수급 확산 지표 이력을 만들어 data/feature_history.csv 로 저장한다.
이 파일은 작고(약 1MB) git에 올라가며, 매일 갱신(build.py)이 최근 구간을 덮어쓰며 이어 붙인다.

사용법: python scripts/backfill_history.py        # 먼저 일봉 캐시 받기(한 번)
        python scripts/make_feature_history.py
"""
from __future__ import annotations

import os
import pickle
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import signal_model as S  # noqa: E402

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(BASE, "data", "cache", "prices_long.pkl")
OUT = os.path.join(BASE, "data", "feature_history.csv")


def main():
    d = pickle.load(open(CACHE, "rb"))
    P, I = S.align_panel(d["prices"], d["index"])
    bio = [c for c, s in d["sector"].items() if s in S.BIO_SECTORS]
    F = pd.concat([S.breadth_features(P, I, bio), S.index_features(I)], axis=1)
    F.insert(0, "kosdaq", I)
    F["n_stocks"] = P.notna().sum(axis=1)
    F.index.name = "date"
    F.round(6).to_csv(OUT, encoding="utf-8")
    print(f"저장 {OUT}: {len(F)}일 {F.index.min().date()} ~ {F.index.max().date()}, 열 {F.shape[1]}")


if __name__ == "__main__":
    main()
