"""
시장 국면 점수(⑦ 탭) -> docs/data/market_phase.json   (참고 대시보드 '시장 국면 점수'의 코스닥 버전)

지표 3개
  1) 밸류 하단 종목 비율(배점 3, 거꾸로): OP밴드 배수(시총 ÷ 12개월 선행 영업이익 추정치)가 자기 최근 3년 하위 10% 아래인 종목 비율. op_band.py
  2) 공격-방어 상대강도(1달, 배점 2): 공격 섹터(코스닥 지수 대비 베타 상위 13개) 동일가중 20거래일 수익률 - 방어 섹터 수익률
  3) 120일선 위 종목 비율(배점 2): 주가가 자기 120일 이동평균 위에 있는 코스닥 보통주(상장 250거래일 이상) 비율
점수: 세 지표를 각각 '그 시점까지의 과거 값 대비 백분위'(0~100, 미래 정보 없음)로 바꿔 배점대로 평균하고 65일 EMA 를 국면 점수로 쓴다.
국면: 점수가 그 시점까지 분포의 하위 1/3이면 낮음, 상위 1/3이면 높음, 그 사이는 중간. 40거래일(8주) 전보다 오르는 중이면
      낮음 ① 반등시작 · 중간 ② 중소형 확산 · 높음 ③ 전면 확산, 내리는 중이면 높음 ④ 고점 · 중간 ⑤ 중소형 하락 · 낮음 ⑥ 대형주까지 바닥.
      (8주 변화가 ±2점 이내면 직전 방향 유지, 낮음·높음 기준선은 2점 넘게 넘어야 수준 변경)
"""
from __future__ import annotations

import json
import math
import warnings
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import signal_model as SMD

KST = timezone(timedelta(hours=9))
warnings.filterwarnings("ignore", message="Mean of empty slice")
DEFENSE = ["유틸리티", "통신", "음식료", "보험", "은행", "지주", "유통"]
N_OFFENSE = 13
WEIGHTS = {"V": 3, "A": 2, "M": 2}
EMA_SPAN = 65
LOOK = 40
HYST = 2.0
MIN_PCT = 250          # 백분위를 매기려면 필요한 최소 관측일
MIN_PCT_V = 60         # 밸류 하단은 이력이 짧아 더 낮춤
PHASES = [
    {"id": 1, "name": "① 반등시작", "color": "#a5d6a7", "desc": "점수 낮음·상승 - 바닥권에서 올라오기 시작"},
    {"id": 2, "name": "② 중소형 확산", "color": "#51cf66", "desc": "점수 중간·상승 - 상승이 중소형으로 퍼지기 시작, 바닥권 종목은 아직 많음"},
    {"id": 3, "name": "③ 전면 확산", "color": "#2b8a3e", "desc": "점수 높음·상승 - 대부분 종목이 오르는 국면"},
    {"id": 4, "name": "④ 고점", "color": "#ffa94d", "desc": "점수 높음·하락 - 확산이 꺾이기 시작(고점 판정 뒤에도 지수가 더 오른 경우가 많아 매도 신호가 아님)"},
    {"id": 5, "name": "⑤ 중소형 하락", "color": "#ff6b4a", "desc": "점수 중간·하락 - 하락이 중소형으로 퍼지는 중"},
    {"id": 6, "name": "⑥ 대형주까지 바닥", "color": "#c92a2a", "desc": "점수 낮음·하락 - 대부분 종목이 바닥권"},
]


def _c(v, nd=4):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else round(f, nd)


def _lst(s, nd=4):
    return [_c(v, nd) for v in s.values]


def ma120_above(P: pd.DataFrame, min_listed: int = 250, min_stocks: int = 100) -> pd.Series:
    """120일선 위 종목 비율(상장 250거래일 이상 종목만)."""
    ma = P.rolling(120, min_periods=120).mean()
    listed = P.notna().cumsum() >= min_listed
    valid = ma.notna() & P.notna() & listed
    above = (P >= ma) & valid
    n = valid.sum(axis=1)
    return (above.sum(axis=1) / n).where(n >= min_stocks)


def sector_matrix(codes, sector_of, secs):
    mem = np.zeros((len(codes), len(secs)))
    for j, s in enumerate(secs):
        mem[(sector_of.reindex(codes).values == s), j] = 1.0
    return mem


def sector_mean(X: pd.DataFrame, mem: np.ndarray, min_n: int = 3) -> np.ndarray:
    """종목 x 날짜 행렬(X: 날짜 x 종목)의 섹터별 단순평균(결측 제외), 종목이 min_n 미만이면 NaN. 반환: 날짜 x 섹터."""
    ok = X.notna().values.astype(np.float64)
    v = np.nan_to_num(X.values)
    n = ok @ mem
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(n >= min_n, (v * ok) @ mem / n, np.nan)


def offense_defense(P: pd.DataFrame, I: pd.Series, st: pd.DataFrame):
    """(공격-방어 1달 상대강도 Series, 섹터 정보 dict). 공격 = 코스닥 지수 베타 상위 13개 섹터, 방어 = 고정 목록."""
    codes = [c for c in st["code"] if c in P.columns]
    sector_of = st.set_index("code")["sector"]
    counts = sector_of.loc[codes].value_counts()
    secs = list(counts.index)
    mem = sector_matrix(codes, sector_of, secs)
    ret = P[codes].pct_change(fill_method=None)
    ir = I.pct_change()
    sr = sector_mean(ret, mem, 3)                                   # 날짜 x 섹터 일별 수익률(동일가중)
    betas = {}
    iv = ir.values
    for j, s in enumerate(secs):
        x = sr[:, j]
        ok = ~np.isnan(x) & ~np.isnan(iv)
        if ok.sum() >= 750 and counts[s] >= 5 and s != "개별주":
            xv, yv = x[ok], iv[ok]
            betas[s] = float(np.cov(xv, yv)[0, 1] / np.var(yv, ddof=1))
    defense = [s for s in DEFENSE if s in counts.index and counts[s] >= 3]
    offense = sorted([s for s in betas if s not in DEFENSE], key=lambda s: betas[s], reverse=True)[:N_OFFENSE]     # 방어 목록에 있는 섹터는 공격 후보에서 뺀다
    r20 = sector_mean(P[codes] / P[codes].shift(20) - 1.0, mem, 3)
    ix = {s: j for j, s in enumerate(secs)}
    off = np.nanmean(r20[:, [ix[s] for s in offense]], axis=1)
    dfn = np.nanmean(r20[:, [ix[s] for s in defense]], axis=1) if defense else np.full(len(P), np.nan)
    A = pd.Series(off - dfn, index=P.index)
    info = {"offense": [{"name": s, "beta": _c(betas[s], 3), "n": int(counts[s])} for s in offense],
            "defense": [{"name": s, "beta": _c(betas.get(s), 3), "n": int(counts[s])} for s in defense],
            "betas": [{"name": s, "beta": _c(b, 3), "n": int(counts[s])} for s, b in sorted(betas.items(), key=lambda kv: -kv[1])]}
    return A, info


def pct_score(x: pd.Series, min_periods: int) -> pd.Series:
    return SMD.expanding_pct(x, min_periods) * 100.0


def assign_phase(ema: pd.Series):
    """(phase id 1~6 또는 0(없음) 배열, 낮음 기준선, 높음 기준선, level, direction)."""
    v = ema.values
    T = len(v)
    lo = np.full(T, np.nan)
    hi = np.full(T, np.nan)
    phase = np.zeros(T, dtype=int)
    valid = ~np.isnan(v)
    hist = []
    level, direction = None, None
    for t in range(T):
        if not valid[t]:
            continue
        hist.append(v[t])
        if len(hist) < MIN_PCT:
            continue
        lo[t], hi[t] = np.quantile(hist, [1 / 3, 2 / 3])
        x = v[t]
        if level is None:
            level = 0 if x < lo[t] else 2 if x > hi[t] else 1
        else:                                                   # 기준선을 2점 넘게 넘어야 수준 변경
            if level == 0 and x > lo[t] + HYST:
                level = 1
            if level == 1 and x > hi[t] + HYST:
                level = 2
            elif level == 1 and x < lo[t] - HYST:
                level = 0
            elif level == 2 and x < hi[t] - HYST:
                level = 1
                if x < lo[t] - HYST:
                    level = 0
        if t >= LOOK and not np.isnan(v[t - LOOK]):
            d = x - v[t - LOOK]
            if direction is None:
                direction = 1 if d > 0 else -1
            elif d > HYST:
                direction = 1
            elif d < -HYST:
                direction = -1
        if direction is None:
            continue
        phase[t] = (level + 1) if direction > 0 else (6 - level)       # 상승: 0→①,1→②,2→③ / 하락: 2→④,1→⑤,0→⑥
    return phase, lo, hi


def phase_stats(I: pd.Series, phase: np.ndarray) -> list:
    """국면별 이후 20·60거래일 코스닥 수익률(우리 자료로 확인한 값)."""
    f20 = I.shift(-20) / I - 1.0
    f60 = I.shift(-60) / I - 1.0
    ph = pd.Series(phase, index=I.index)
    out = []
    for p in [0] + [d["id"] for d in PHASES]:
        m = (ph == p) if p else (ph > 0)
        a, b = f20[m].dropna(), f60[m].dropna()
        out.append({"id": p, "n": int(m.sum()), "f20": _c(a.mean(), 4), "up20": _c((a > 0).mean(), 3) if len(a) else None,
                    "f60": _c(b.mean(), 4), "up60": _c((b > 0).mean(), 3) if len(b) else None, "n20": int(len(a)), "n60": int(len(b))})
    return out


def compute(P: pd.DataFrame, I: pd.Series, st: pd.DataFrame, V: pd.Series | None = None, v_info: dict | None = None) -> dict:
    """P: 장기 수정주가, I: 코스닥 지수, st: res['stock'], V: 밸류 하단 비율(0~1, 날짜 index) 또는 None."""
    codes = [c for c in st["code"] if c in P.columns]
    P = P[codes]
    M = ma120_above(P)
    A, sec_info = offense_defense(P, I, st)
    Vs = V.reindex(P.index) if V is not None else pd.Series(np.nan, index=P.index)
    pM, pA = pct_score(M, MIN_PCT), pct_score(A, MIN_PCT)
    pV = pct_score(Vs, MIN_PCT_V)
    w_v = WEIGHTS["V"] * ~pV.isna()
    num = WEIGHTS["A"] * pA + WEIGHTS["M"] * pM + WEIGHTS["V"] * (100.0 - pV).fillna(0.0)
    den = WEIGHTS["A"] + WEIGHTS["M"] + w_v
    daily = (num / den).where(pA.notna() & pM.notna())
    ema = daily.dropna().ewm(span=EMA_SPAN, adjust=False).mean().reindex(P.index)
    first = ema.first_valid_index()
    pos0 = P.index.get_loc(first) + EMA_SPAN                      # EMA 준비 구간은 버린다
    ema.iloc[:pos0] = np.nan
    phase, lo, hi = assign_phase(ema)
    ok = phase > 0
    start = P.index[np.argmax(ok)] if ok.any() else P.index[pos0]
    start_pos = P.index.get_loc(start)
    sl = slice(start_pos, None)
    idx = P.index[sl]
    ser = lambda x, nd=4: _lst((x if isinstance(x, pd.Series) else pd.Series(x, index=P.index)).iloc[sl], nd)

    n = len(P)
    last = n - 1
    now_phase = int(phase[last])
    now_level = "낮음" if ema.iloc[last] < lo[last] else "높음" if ema.iloc[last] > hi[last] else "중간"
    recent = []
    for t in range(last, max(last - 15, start_pos) - 1, -1):
        recent.append({"date": P.index[t].strftime("%Y-%m-%d"), "phase": int(phase[t]), "score": _c(ema.iloc[t], 2),
                       "d1": _c(ema.iloc[t] - ema.iloc[t - 1], 2) if t > 0 else None, "d40": _c(ema.iloc[t] - ema.iloc[t - LOOK], 2) if t >= LOOK else None,
                       "V": _c(Vs.iloc[t], 4), "A": _c(A.iloc[t], 4), "M": _c(M.iloc[t], 4)})
    changed = None
    for t in range(last, start_pos, -1):
        if phase[t] != phase[t - 1]:
            changed = {"date": P.index[t].strftime("%Y-%m-%d"), "from": int(phase[t - 1]), "to": int(phase[t]), "days": int(last - t + 1)}
            break
    v_first = Vs.first_valid_index()
    payload = {
        "asof": P.index[-1].strftime("%Y-%m-%d"),
        "generated": datetime.now(KST).strftime("%Y-%m-%d %H:%M KST"),
        "start": idx[0].strftime("%Y-%m-%d"),
        "params": {"ema": EMA_SPAN, "look": LOOK, "hyst": HYST, "weights": WEIGHTS, "defense": DEFENSE, "n_offense": N_OFFENSE},
        "phases": PHASES,
        "dates": [d.strftime("%Y-%m-%d") for d in idx],
        "kosdaq": [_c(v, 2) for v in I.reindex(idx).values],
        "score": ser(ema, 2), "lo": ser(pd.Series(lo, index=P.index), 2), "hi": ser(pd.Series(hi, index=P.index), 2), "daily": ser(daily, 2),
        "phase": [int(x) for x in phase[sl]],
        "M": ser(M, 4), "A": ser(A, 4), "V": ser(Vs, 4),
        "pM": ser(pM, 1), "pA": ser(pA, 1), "pV": ser(pV, 1),
        "now": {"date": P.index[-1].strftime("%Y-%m-%d"), "phase": now_phase, "level": now_level, "score": _c(ema.iloc[last], 1),
                "lo": _c(lo[last], 1), "hi": _c(hi[last], 1), "score_8w": _c(ema.iloc[last - LOOK], 1), "daily": _c(daily.iloc[last], 1),
                "daily_avg20": _c(daily.iloc[last - 19:last + 1].mean(), 1),
                "M": _c(M.iloc[last], 4), "pM": _c(pM.iloc[last], 0), "A": _c(A.iloc[last], 4), "pA": _c(pA.iloc[last], 0),
                "V": _c(Vs.iloc[last], 4), "pV": _c(pV.iloc[last], 0), "kosdaq": _c(I.iloc[-1], 2), "kosdaq_1m": _c(I.iloc[-1] / I.iloc[-21] - 1, 4),
                "strip": [int(x) for x in phase[last - 29:last + 1]], "strip_dates": [P.index[t].strftime("%Y-%m-%d") for t in range(last - 29, last + 1)]},
        "sectors": sec_info,
        "recent": recent, "changed": changed,
        "phase_stats": phase_stats(I.reindex(P.index), phase),
        "v_info": dict(v_info or {}, first=v_first.strftime("%Y-%m-%d") if v_first is not None else None),
        "n_ma": int(P.notna().sum(axis=1).iloc[-1]),
    }
    return payload


def write(payload: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))


def run(P, I, st, caps, op_path, hist_csv, out_path):
    """밸류 하단 비율(OP밴드)을 준비해 국면 점수를 계산하고 JSON 으로 쓴다.
    OP 엑셀이 있으면 새로 계산해 hist_csv 에 저장하고, 없으면 저장된 이력을 쓴다(마지막 값을 최대 40거래일 이어 쓰고 stale 표시)."""
    import op_band as OB
    V, v_info = None, {}
    codes = set(st["code"])
    if op_path and os.path.exists(op_path):
        try:
            op, caps_op = OB.read_workbook(op_path)
            op = op.loc[:, [c for c in op.columns if c[0] in codes]]
            capsm = caps_op if caps is None else caps.combine_first(caps_op.reindex(columns=caps.columns.union(caps_op.columns))).sort_index()
            ratio, n, _ = OB.bottom_ratio(capsm, op)
            V = ratio.dropna()
            pd.DataFrame({"ratio": V.round(5), "n": n.reindex(V.index)}).to_csv(hist_csv, index_label="date")
            v_info = {"op_file": os.path.basename(op_path), "op_last": op.index.max().strftime("%Y-%m-%d"), "n_stocks": int(n.iloc[-1]),
                      "n_covered": int(len({c[0] for c in op.columns})), "stale": False}
        except Exception as e:  # noqa: BLE001
            print(f"[market_phase] OP 엑셀을 읽지 못해 저장된 이력을 씁니다: {type(e).__name__}: {e}")
    if V is None and os.path.exists(hist_csv):
        h = pd.read_csv(hist_csv, index_col=0, parse_dates=True)
        V = h["ratio"].dropna()
        v_info = {"stale": True, "last": V.index.max().strftime("%Y-%m-%d"), "n_stocks": int(h["n"].dropna().iloc[-1])}
        V = V.reindex(V.index.union(P.index)).ffill(limit=40)
    if v_info:
        v_info.update({"min_obs": OB.MIN_OBS, "window": OB.WINDOW, "quantile": OB.Q})
    payload = compute(P, I, st, V=V, v_info=v_info)
    write(payload, out_path)
    return payload
