"""
수급 확산 시그널 연구용: 현재 대시보드 유니버스(1000종목) + 코스닥 지수의 장기 일봉(최대 3000거래일, 약 12년)을
네이버에서 받아 data/cache/ 에 저장한다. 한 번만 받으면 되고, 캐시는 git에 올리지 않는다.

사용법: python scripts/backfill_history.py            # docs/data/latest.json 의 종목 사용
        python scripts/backfill_history.py --refresh  # 캐시 무시하고 다시 받기
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import fetch_naver as F  # noqa: E402

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(BASE, "data", "cache")
OUT = os.path.join(CACHE, "prices_long.pkl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--count", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    if os.path.exists(OUT) and not a.refresh:
        print("캐시가 이미 있습니다:", OUT)
        return

    with open(os.path.join(BASE, "docs", "data", "latest.json"), encoding="utf-8") as f:
        js = json.load(f)
    cols = js["stock_cols"]
    ic, isec, imc = cols.index("code"), cols.index("sector"), cols.index("mcap_bil")
    codes = [r[ic] for r in js["stocks"]]
    sector = {r[ic]: r[isec] for r in js["stocks"]}
    mcap = {r[ic]: r[imc] for r in js["stocks"]}

    index = F.daily_chart("KOSDAQ", a.count)
    print(f"[index] {len(index)}일 {index.index.min().date()} ~ {index.index.max().date()}", flush=True)

    series, failed = {}, []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(F.daily_chart, c, a.count): c for c in codes}
        for i, fu in enumerate(as_completed(futs), 1):
            c = futs[fu]
            try:
                s = fu.result()
                if len(s):
                    series[c] = s
            except Exception as e:  # noqa: BLE001
                failed.append(c)
            if i % 100 == 0:
                print(f"  {i}/{len(futs)}  {time.time() - t0:.0f}s", flush=True)
    prices = pd.DataFrame(series).sort_index()
    prices = prices[[c for c in codes if c in prices.columns]]
    print(f"[prices] {prices.shape[1]}종목 x {prices.shape[0]}일, 실패 {len(failed)}: {failed[:20]}", flush=True)
    with open(OUT, "wb") as f:
        pickle.dump({"prices": prices, "index": index, "sector": sector, "mcap_bil": mcap,
                     "fetched": time.strftime("%Y-%m-%d %H:%M")}, f)
    print("저장:", OUT)


if __name__ == "__main__":
    main()
