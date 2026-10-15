"""
'최근 낙폭 평균' 탭(④) 데이터 -> docs/data/drawdown.json

정의(데이터 모음 엑셀의 '최근 낙폭 평균' 시트와 같다): 종목마다 **가장 최근에 내린 날 K일치**(전일보다 하락한 날만)의 등락률을 평균한다.
K일을 다 채우지 못한 종목(신규상장 등)은 있는 만큼만 평균하고 하락일수로 표시한다. 코스닥 지수도 똑같이 계산해 비교한다.
'최근 K거래일'이 아니라 '최근 하락일 K번'이므로, 잘 안 내리는 종목은 그만큼 오래전 하락일까지 거슬러 올라간다(기간 열로 표시).
K는 5·10·20·40·60. 날짜별 롤링 추이(코스닥 vs 종목 중앙값)도 만든다.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import rs_engine as E

KST = timezone(timedelta(hours=9))
WINDOWS = (5, 10, 20, 40, 60)     # 평균에 쓰는 '하락일' 수 K
ROLL_DAYS = 200                    # 추이 차트에 보여줄 거래일 수
BIL = 1e9


def _c(v, nd=5):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else round(f, nd)


def last_k_down(ret: pd.DataFrame, k: int) -> pd.DataFrame:
    """각 열(종목)의 마지막 k개 하락일 통계. 행 = 종목."""
    arr = ret.values
    T, N = arr.shape
    out = {"n_down": np.zeros(N, int), "avg_down": np.full(N, np.nan), "worst": np.full(N, np.nan),
           "span": np.full(N, np.nan), "since": np.full(N, np.nan), "n_valid": ret.notna().sum().values}
    for j in range(N):
        col = arr[:, j]
        idx = np.flatnonzero(col < 0)
        if len(idx) == 0:
            continue
        use = idx[-k:]
        v = col[use]
        out["n_down"][j] = len(use)
        out["avg_down"][j] = v.mean()
        out["worst"][j] = v.min()
        out["span"][j] = T - use[0]          # 가장 오래된 하락일부터 기준일까지 거래일 수
        out["since"][j] = T - 1 - use[-1]    # 마지막 하락일이 며칠 전인지(0 = 기준일에 내림)
    return pd.DataFrame(out, index=ret.columns)


def rolling_k_down(ret: pd.DataFrame, k: int) -> pd.DataFrame:
    """날짜마다 '그날까지의 마지막 k개 하락일 평균'. 하락일이 k개 미만인 날은 비움."""
    arr = ret.values
    T, N = arr.shape
    out = np.full((T, N), np.nan)
    days = np.arange(T)
    for j in range(N):
        col = arr[:, j]
        pos = np.flatnonzero(col < 0)
        if len(pos) < k:
            continue
        cs = np.concatenate([[0.0], np.cumsum(col[pos])])
        m = (cs[k:] - cs[:-k]) / k                   # v[i:i+k] 의 평균
        pts = pos[k - 1:]                            # 그 평균이 완성되는 날짜
        idx = np.searchsorted(pts, days, side="right") - 1
        ok = idx >= 0
        o = np.full(T, np.nan)
        o[ok] = m[idx[ok]]
        out[:, j] = o
    return pd.DataFrame(out, index=ret.index, columns=ret.columns)


def compute(prices: pd.DataFrame, index: pd.Series, st: pd.DataFrame) -> dict:
    """st: rs_engine.compute 결과의 res['stock'] (code, name, sector, mcap, mcap_rank)."""
    P, I = E.align(prices, index)
    codes = [c for c in st["code"] if c in P.columns]
    P = P[codes]
    ret = P.pct_change(fill_method=None)
    iret = I.pct_change().to_frame("kosdaq")
    meta = st.set_index("code").loc[codes]

    ws = {k: last_k_down(ret, k) for k in WINDOWS}
    kos = {k: last_k_down(iret, k).iloc[0] for k in WINDOWS}

    summary = {}
    for k in WINDOWS:
        s = ws[k]
        ok = (s["n_down"] >= k) & s["avg_down"].notna()          # K일을 다 채운 종목만 비교
        a = s.loc[ok, "avg_down"]
        kv = kos[k]["avg_down"]
        summary[str(k)] = {"n": int(ok.sum()), "median": _c(a.median()), "mean": _c(a.mean()), "p25": _c(a.quantile(0.25)), "p75": _c(a.quantile(0.75)),
                           "span_median": _c(s.loc[ok, "span"].median(), 1), "worse": int((a < kv).sum()), "better": int((a >= kv).sum())}

    cols = ["code", "name", "sector", "mcap_bil", "mrank"]
    for k in WINDOWS:
        cols += [f"nd{k}", f"ad{k}", f"wr{k}", f"sp{k}", f"sc{k}"]
    rows = []
    for c in codes:
        m = meta.loc[c]
        row = [c, m["name"], m["sector"], _c(m["mcap"] / BIL, 1), int(m["mcap_rank"])]
        for k in WINDOWS:
            s = ws[k].loc[c]
            row += [int(s["n_down"]), _c(s["avg_down"]), _c(s["worst"]), None if np.isnan(s["span"]) else int(s["span"]),
                    None if np.isnan(s["since"]) else int(s["since"])]
        rows.append(row)

    # 날짜별 롤링(코스닥 vs 종목 중앙값). K마다 따로.
    roll = {}
    tail = None
    for k in WINDOWS:
        rr = rolling_k_down(ret, k)
        kr = rolling_k_down(iret, k)["kosdaq"]
        ok = rr.notna().sum(axis=1) >= 100
        idx = rr.index[ok & kr.notna()]
        idx = idx[-ROLL_DAYS:]
        if tail is None:
            tail = idx
        roll[str(k)] = {"dates": [d.strftime("%Y-%m-%d") for d in idx], "kosdaq": [_c(v) for v in kr.loc[idx]],
                        "median": [_c(v) for v in rr.loc[idx].median(axis=1)],
                        "p25": [_c(v) for v in rr.loc[idx].quantile(0.25, axis=1)], "p75": [_c(v) for v in rr.loc[idx].quantile(0.75, axis=1)]}
    return {
        "asof": P.index[-1].strftime("%Y-%m-%d"),
        "generated": datetime.now(KST).strftime("%Y-%m-%d %H:%M KST"),
        "windows": list(WINDOWS),
        "kosdaq": {str(k): {"nd": int(kos[k]["n_down"]), "ad": _c(kos[k]["avg_down"]), "wr": _c(kos[k]["worst"]),
                            "sp": None if np.isnan(kos[k]["span"]) else int(kos[k]["span"])} for k in WINDOWS},
        "summary": summary,
        "cols": cols,
        "stocks": rows,
        "roll": roll,
    }


def write(payload: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
