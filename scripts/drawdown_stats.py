"""
'최근 낙폭 평균' 탭(④) 데이터 -> docs/data/drawdown.json

종목마다 최근 N거래일 중 '전일보다 내린 날'만 골라 그날 등락률을 평균한다(= 내릴 때 평균 얼마나 내리는가).
코스닥 지수도 같은 방식으로 계산해 옆에 둔다. 종목은 각자 자기가 내린 날만 평균하므로 코스닥이 내린 날과 같지 않다.
창(N)은 5·10·20·40·60거래일. 20거래일 롤링 추이(코스닥 vs 종목 중앙값)도 함께 만든다.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, timezone

import pandas as pd

import rs_engine as E

KST = timezone(timedelta(hours=9))
WINDOWS = (5, 10, 20, 40, 60)
ROLL_N = 20        # 추이 차트의 롤링 창
ROLL_DAYS = 200    # 추이 차트에 보여줄 거래일 수
BIL = 1e9


def _c(v, nd=5):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else round(f, nd)


def window_stats(ret: pd.DataFrame, level: pd.DataFrame, n: int) -> pd.DataFrame:
    """최근 n거래일(마지막 n개 수익률)의 하락일 통계. 행 = 종목."""
    w = ret.iloc[-n:]
    neg = w.where(w < 0)
    mn = w.min()
    return pd.DataFrame({
        "n_valid": w.notna().sum(),                  # 가격이 있었던 거래일 수(신규상장·거래정지는 n보다 작다)
        "n_down": neg.notna().sum(),                 # 내린 날 수
        "avg_down": neg.mean(),                      # 내린 날의 평균 등락률(음수)
        "worst": mn.where(mn < 0),                   # 하루 최대 낙폭(내린 날이 없으면 비움)
        "cum": level.iloc[-1] / level.iloc[-1 - n] - 1.0,   # n거래일 누적 수익률
    })


def compute(prices: pd.DataFrame, index: pd.Series, st: pd.DataFrame) -> dict:
    """st: rs_engine.compute 결과의 res['stock'] (code, name, sector, mcap, mcap_rank)."""
    P, I = E.align(prices, index)
    codes = [c for c in st["code"] if c in P.columns]
    P = P[codes]
    ret = P.pct_change(fill_method=None)
    iret = I.pct_change()
    meta = st.set_index("code").loc[codes]

    ws = {n: window_stats(ret, P, n) for n in WINDOWS}
    kf, kl = iret.to_frame("kosdaq"), I.to_frame("kosdaq")
    kos = {n: window_stats(kf, kl, n).iloc[0] for n in WINDOWS}

    summary = {}
    for n in WINDOWS:
        s = ws[n]
        ok = (s["n_valid"] >= n) & s["avg_down"].notna()     # 가격이 n거래일 모두 있고 내린 날이 있는 종목만(신규상장·거래정지 제외)
        a = s.loc[ok, "avg_down"]
        k = kos[n]["avg_down"]
        summary[str(n)] = {
            "n": int(ok.sum()), "median": _c(a.median()), "mean": _c(a.mean()), "p25": _c(a.quantile(0.25)), "p75": _c(a.quantile(0.75)),
            "worse": int((a < k).sum()), "better": int((a >= k).sum()),
        }

    cols = ["code", "name", "sector", "mcap_bil", "mrank"]
    for n in WINDOWS:
        cols += [f"nv{n}", f"nd{n}", f"ad{n}", f"wr{n}", f"cm{n}"]
    rows = []
    for c in codes:
        m = meta.loc[c]
        row = [c, m["name"], m["sector"], _c(m["mcap"] / BIL, 1), int(m["mcap_rank"])]
        for n in WINDOWS:
            s = ws[n].loc[c]
            row += [int(s["n_valid"]), int(s["n_down"]), _c(s["avg_down"]), _c(s["worst"]), _c(s["cum"])]
        rows.append(row)

    # 20거래일 롤링: 날짜마다 최근 20일 중 내린 날만 평균 (코스닥 vs 종목 중앙값)
    roll = ret.where(ret < 0).rolling(ROLL_N, min_periods=1).mean().iloc[ROLL_N:]
    kroll = iret.where(iret < 0).rolling(ROLL_N, min_periods=1).mean().iloc[ROLL_N:]
    tail = roll.index[-ROLL_DAYS:]
    return {
        "asof": P.index[-1].strftime("%Y-%m-%d"),
        "generated": datetime.now(KST).strftime("%Y-%m-%d %H:%M KST"),
        "windows": list(WINDOWS),
        "kosdaq": {str(n): {"nv": int(kos[n]["n_valid"]), "nd": int(kos[n]["n_down"]), "ad": _c(kos[n]["avg_down"]),
                            "wr": _c(kos[n]["worst"]), "cm": _c(kos[n]["cum"])} for n in WINDOWS},
        "summary": summary,
        "cols": cols,
        "stocks": rows,
        "roll": {"n": ROLL_N, "dates": [d.strftime("%Y-%m-%d") for d in tail],
                 "kosdaq": [_c(v) for v in kroll.loc[tail]], "median": [_c(v) for v in roll.loc[tail].median(axis=1)],
                 "p25": [_c(v) for v in roll.loc[tail].quantile(0.25, axis=1)], "p75": [_c(v) for v in roll.loc[tail].quantile(0.75, axis=1)]},
    }


def write(payload: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
