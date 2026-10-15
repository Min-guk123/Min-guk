"""
섹터 이평선 위 비율(⑥ 탭) -> docs/data/sector_ma.json

섹터(내 섹터 정리)별로, 주가가 자기 N일 이동평균 이상인 종목의 비율. 높을수록 섹터 강함.
  - 이동평균: 10·15·20·60일, 20주(=100거래일), 120·200일
  - 기준: 종목 수(위 종목 수 ÷ 계산 가능한 종목 수) / 시총 가중(위 종목 시총 합 ÷ 섹터 시총 합). 3종목 미만인 날은 비운다.
  - 가격은 수정주가(액면분할에도 끊기지 않음). 시총은 엑셀의 일별 시가총액, 엑셀 이전 구간은 수정주가 x 현재 주식수로 근사.
  - 장기 약세 기준 40%(20주·120·200일선): 이 아래에 머문 날 수('약세 N일째')와, 코스닥 12년 자료로 그 기준을 다시 검증한 결과.
"""
from __future__ import annotations

import json
import math
import warnings
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

KST = timezone(timedelta(hours=9))
warnings.filterwarnings("ignore", message="Mean of empty slice")
MAS = [("10", "10일", 10), ("15", "15일", 15), ("20", "20일", 20), ("60", "60일", 60), ("100", "20주", 100), ("120", "120일", 120), ("200", "200일", 200)]
WEAK_MAS = ("100", "120", "200")
WEAK = 0.40
DAILY = 260          # 일별로 보내는 최근 거래일 수(3달·6달·1년 차트)
MIN_N = 3            # 섹터에서 계산 가능한 종목이 이보다 적은 날은 비운다
BIL = 1e9


def _i(v):
    return -1 if (v is None or (isinstance(v, float) and math.isnan(v))) else int(round(v * 100))


def _arr(x):
    """0~1 비율 배열 -> 정수 퍼센트(결측 -1) 리스트."""
    return np.where(np.isnan(x), -1, np.rint(x * 100)).astype(int).tolist()


def _c(v, nd=4):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else round(f, nd)


def compute(P: pd.DataFrame, I: pd.Series, st: pd.DataFrame, caps_actual: pd.DataFrame | None = None) -> dict:
    """P: 장기 수정주가(날짜 x 종목코드), I: 코스닥 지수, st: rs_engine 의 res['stock'], caps_actual: 엑셀 일별 시총(원)."""
    codes = [c for c in st["code"] if c in P.columns]
    P = P[codes]
    meta = st.set_index("code").loc[codes]
    sector_of = meta["sector"]
    counts = sector_of.value_counts()
    secs = [s for s in counts.index if counts[s] >= MIN_N]                  # 종목 수 많은 순
    S = len(secs)
    mem = np.zeros((len(codes), S))
    for j, s in enumerate(secs):
        mem[(sector_of.values == s), j] = 1.0
    T = len(P)

    # 시총 패널: 근사(수정주가 x 현재 주식수) 후, 엑셀 일별 시총이 있는 날은 그 값으로 교체
    capT = meta["mcap"].astype(float)
    cap = P * (capT / P.iloc[-1].replace(0, np.nan))
    if caps_actual is not None:
        ca = caps_actual.reindex(index=P.index, columns=codes)
        cap = cap.where(ca.isna(), ca)
    capv = cap.fillna(0.0).values

    start = min(T - 1, 219)                       # 가장 긴 이동평균(200일) 준비 구간 이후부터 보낸다
    idx = P.index[start:]
    wk = pd.Series(idx, index=idx).groupby([idx.isocalendar().year.values, idx.isocalendar().week.values]).last().values
    wk_pos = P.index.get_indexer(pd.DatetimeIndex(wk))
    d_pos = np.arange(T - DAILY, T)
    d_pos = d_pos[d_pos >= start]

    ratio, weak, stock_disp, raw_ratio = {}, {}, {}, {}
    for key, label, w in MAS:
        ma = P.rolling(w, min_periods=w).mean()
        valid = (ma.notna() & P.notna())
        above = (P >= ma) & valid
        V, A = valid.values.astype(np.float64), above.values.astype(np.float64)
        cv, ca_ = V @ mem, A @ mem                                          # 섹터별 계산 가능/위 종목 수 (T x S)
        capv_v, capa = (capv * V) @ mem, (capv * A) @ mem
        with np.errstate(divide="ignore", invalid="ignore"):
            r_cnt = np.where(cv >= MIN_N, ca_ / cv, np.nan)
            r_cap = np.where((cv >= MIN_N) & (capv_v > 0), capa / capv_v, np.nan)
            all_cnt = np.where(V.sum(1) >= MIN_N, A.sum(1) / V.sum(1), np.nan)
            all_cap = np.where((V.sum(1) >= MIN_N) & ((capv * V).sum(1) > 0), (capv * A).sum(1) / (capv * V).sum(1), np.nan)
        rc = np.column_stack([all_cnt, r_cnt])                               # 0열 = 섹터 정리 전체
        rp = np.column_stack([all_cap, r_cap])
        raw_ratio[key] = rc
        ratio[key] = {"cnt": {"d": [_arr(rc[d_pos, j]) for j in range(S + 1)], "w": [_arr(rc[wk_pos, j]) for j in range(S + 1)]},
                      "cap": {"d": [_arr(rp[d_pos, j]) for j in range(S + 1)], "w": [_arr(rp[wk_pos, j]) for j in range(S + 1)]}}
        if key in WEAK_MAS:                                                  # 40% 아래에 연속으로 머문 날 수(마지막 날 기준)
            wd = {}
            for bname, mat in (("cnt", rc), ("cap", rp)):
                below = (mat < WEAK) & ~np.isnan(mat)
                n_days = []
                for j in range(S + 1):
                    k = 0
                    for t in range(T - 1, -1, -1):
                        if below[t, j]:
                            k += 1
                        else:
                            break
                    n_days.append(k)
                wd[bname] = n_days
            weak[key] = wd
        disp = (P.iloc[-1] / ma.iloc[-1] - 1.0)
        stock_disp[key] = disp

    # 섹터 등락률(1일·1주·1달): 구성종목 단순평균 + 오른 종목 수
    rets = {}
    for h in (1, 5, 20):
        r = P.iloc[-1] / P.iloc[-1 - h] - 1.0
        ok = r.notna().values.astype(np.float64)
        rv = np.nan_to_num(r.values)
        n = ok @ mem
        up = ((r.values > 0) & r.notna().values).astype(np.float64) @ mem
        with np.errstate(divide="ignore", invalid="ignore"):
            avg = (rv * ok) @ mem / n
        rets[str(h)] = [{"avg": _c(avg[j], 5), "up": int(up[j]), "n": int(n[j])} for j in range(S)]
    r_all = P.iloc[-1] / P.iloc[-2] - 1.0

    # 종목 표(이격도): 가격 ÷ 이동평균 − 1
    cols = ["code", "name", "sector", "mcap_bil"] + [f"dp{k}" for k, _, _ in MAS]
    stocks = []
    for c in codes:
        row = [c, meta.at[c, "name"], meta.at[c, "sector"], _c(meta.at[c, "mcap"] / BIL, 1)]
        row += [_c(stock_disp[k].get(c), 4) for k, _, _ in MAS]
        stocks.append(row)

    payload = {
        "asof": P.index[-1].strftime("%Y-%m-%d"),
        "generated": datetime.now(KST).strftime("%Y-%m-%d %H:%M KST"),
        "mas": [{"key": k, "label": lb, "days": d} for k, lb, d in MAS],
        "weak_line": WEAK, "weak_mas": list(WEAK_MAS),
        "sectors": ["섹터 정리 전체"] + secs,
        "n_stocks": [int(len(codes))] + [int(counts[s]) for s in secs],
        "dates_d": [P.index[i].strftime("%Y-%m-%d") for i in d_pos],
        "dates_w": [P.index[i].strftime("%Y-%m-%d") for i in wk_pos],
        "ratio": ratio, "weak_days": weak, "returns": rets,
        "kosdaq": {"last": _c(I.iloc[-1], 2), "chg1d": _c(I.iloc[-1] / I.iloc[-2] - 1.0, 5)},
        "stock_cols": cols, "stocks": stocks,
        "history_start": P.index[0].strftime("%Y-%m-%d"),
    }
    payload["weak_check"] = validate_weak_line(P, raw_ratio, secs, mem)
    return payload


def validate_weak_line(P: pd.DataFrame, raw_ratio: dict, secs: list, mem: np.ndarray, horizon: int = 120, split: str = "2020-07-01") -> dict:
    """'장기 약세 기준 40%' 검증: 위 비율이 40% 아래였던 섹터-일의 이후 horizon 거래일 초과수익(섹터 평균 − 그날 전체 섹터 평균)."""
    sec_idx = [j + 1 for j, s in enumerate(secs) if s != "개별주" and mem[:, j].sum() >= 5]
    fwd = P.shift(-horizon) / P - 1.0
    ok = fwd.notna().values.astype(np.float64)
    fv = np.nan_to_num(fwd.values)
    n = ok @ mem
    with np.errstate(divide="ignore", invalid="ignore"):
        F = np.where(n >= 3, (fv * ok) @ mem / n, np.nan)                    # T x S 섹터별 이후 수익률
    cols = [j - 1 for j in sec_idx]
    F = F[:, cols]
    ex = F - np.nanmean(F, axis=1, keepdims=True)
    dates = P.index
    mid = dates < pd.Timestamp(split)
    out = {"horizon": horizon, "split": split, "rows": []}
    for key in WEAK_MAS:
        R = raw_ratio[key][:, sec_idx]
        rec = {"ma": key}
        for lab, mask in (("all", np.ones(len(dates), bool)), ("first", mid), ("second", ~mid)):
            m = mask[:, None] & ~np.isnan(R) & ~np.isnan(ex)
            below = m & (R < WEAK)
            above = m & (R >= WEAK)
            rec[lab] = {"below": _c(ex[below].mean() if below.any() else np.nan, 5), "above": _c(ex[above].mean() if above.any() else np.nan, 5),
                        "n_below": int(below.sum()), "n_above": int(above.sum())}
        out["rows"].append(rec)
    return out


def write(payload: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
