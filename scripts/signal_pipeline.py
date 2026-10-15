"""
매일 갱신용: 수급 확산 시그널을 계산해 docs/data/signal.json 으로 쓴다.

  1) data/feature_history.csv(약 12년치 지표 이력) 를 읽고
  2) 이번 실행의 종목 패널(엑셀/네이버)로 최근 지표를 다시 계산해 덮어쓰고 이어 붙인다
     (엑셀 모드이면 엑셀보다 최신인 날짜는 네이버에서 받아 보충: 실패해도 무시)
  3) 낙관 점수(0~100) → 기준 숏 비중(50/75/100%)별 권장 숏 비중과 과거 성과 곡선을 만든다

파라미터는 config/signal_params.json (연구로 정한 값, 매일 다시 최적화하지 않음).
연구 결과(예측력 표, 5분위, 시뮬레이션)는 research_signal.py report 가 docs/data/signal_research.json 에 쓴다.
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import signal_model as S  # noqa: E402

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HIST = os.path.join(BASE, "data", "feature_history.csv")
PARAMS = os.path.join(BASE, "config", "signal_params.json")
OUT = os.path.join(BASE, "docs", "data", "signal.json")
KST = timezone(timedelta(hours=9))

CORE = [c for c in S.BREADTH_FEATURES if not c.startswith("bio")] + S.INDEX_FEATURES + S.RISK_FEATURES
DEFAULTS = {
    "features": [c for c in S.BREADTH_FEATURES if not c.startswith("bio")],
    "min_periods": 250, "amp": 0.25, "smooth": 10, "lo": 0.5, "hi": 1.0, "cost": 0.001,
    "bases": {"저평가": 0.5, "중립": 0.75, "고평가": 1.0},
    "bands": [0.2, 0.4, 0.6, 0.8],
}


def load_params() -> dict:
    p = dict(DEFAULTS)
    if os.path.exists(PARAMS):
        with open(PARAMS, encoding="utf-8-sig") as f:
            p.update(json.load(f))
    return p


def _features(prices: pd.DataFrame, index: pd.Series, codes, bio_codes) -> pd.DataFrame:
    P, I = S.align_panel(prices[[c for c in codes if c in prices.columns]], index)
    F = pd.concat([S.breadth_features(P, I, bio_codes), S.index_features(I)], axis=1)
    F.insert(0, "kosdaq", I)
    F["n_stocks"] = P.notna().sum(axis=1)
    return F


def _merge(hist: pd.DataFrame, new: pd.DataFrame, only_after=None) -> pd.DataFrame:
    """새로 계산한 행(핵심 지표가 모두 있는 날)을 이력에 덮어쓰거나 이어 붙인다."""
    ok = new[new[CORE].notna().all(axis=1)]
    if only_after is not None:
        ok = ok[ok.index > only_after]
    if ok.empty:
        return hist
    out = hist.reindex(hist.index.union(ok.index))
    for c in out.columns:
        if c in ok.columns:
            out.loc[ok.index, c] = ok[c]
    return out.sort_index()


def _naver_topup(codes, bio_codes, count=320) -> pd.DataFrame | None:
    try:
        import fetch_naver as FN
        idx = FN.daily_chart("KOSDAQ", count)
        with ThreadPoolExecutor(max_workers=8) as ex:
            res = list(ex.map(lambda c: (c, FN.daily_chart(c, count)), codes))
        prices = pd.DataFrame({c: s for c, s in res if len(s)}).sort_index()
        return _features(prices, idx, codes, bio_codes)
    except Exception as e:  # noqa: BLE001  (네트워크 실패 시 엑셀 기준일까지만 사용)
        print(f"[signal] 네이버 보충 실패: {e}")
        return None


def _clean(o):
    """JSON 에 NaN/inf 가 들어가면 브라우저가 못 읽으므로 None 으로 바꾼다."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if (np.isnan(o) or np.isinf(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def _curve(w: pd.Series, r: pd.Series, cost: float, start):
    """순노출(=100%-숏) 부분의 손익 곡선: 신호 가감 vs 같은 평균 숏을 계속 유지."""
    ww = w.loc[start:]
    rr = r.loc[ww.index]
    wl = ww.shift(1)
    net = (1.0 - wl) * rr - cost * ww.diff().abs().fillna(0.0)
    wbar = float(wl.mean())
    static = (1.0 - wbar) * rr
    nav = lambda x: 100.0 * (1.0 + x.fillna(0.0)).cumprod()
    ns, nt = nav(net), nav(static)
    dd = lambda n: n / n.cummax() - 1.0
    return {"w_mean": wbar, "nav_sig": ns, "nav_static": nt, "dd_sig": dd(ns), "dd_static": dd(nt)}


def _yearly(w: pd.Series, r: pd.Series, cost: float, start) -> list:
    """달력 연도별 순노출 손익: 점수 가감 vs 같은 평균 숏 고정 (한 해의 큰 급락에 기댄 결과인지 확인용)."""
    ww = w.loc[start:]
    rr = r.loc[ww.index]
    wl = ww.shift(1)
    net = (1.0 - wl) * rr - cost * ww.diff().abs().fillna(0.0)
    static = (1.0 - float(wl.mean())) * rr
    comp = lambda x: float((1.0 + x.fillna(0.0)).prod() - 1.0)
    out = []
    for y in sorted(set(ww.index.year)):
        m = ww.index.year == y
        out.append({"year": int(y), "idx": comp(rr[m]), "sig": comp(net[m]), "static": comp(static[m]),
                    "w": float(wl[m].mean()), "partial": bool(y == ww.index[-1].year)})
    return out


def run(prices, index, res, source: str, topup: bool = True, long=None) -> dict | None:
    P = load_params()
    st = res["stock"]
    codes = st["code"].tolist()
    bio_codes = st.loc[st["sector"].isin(S.BIO_SECTORS), "code"].tolist()
    if not os.path.exists(HIST):
        print("[signal] data/feature_history.csv 가 없어 시그널 계산을 건너뜁니다 (make_feature_history.py 를 먼저 실행).")
        return None
    hist = pd.read_csv(HIST, index_col="date", parse_dates=True)

    new = _features(prices, index, codes, bio_codes)
    merged = _merge(hist, new)
    last_panel = new[new[CORE].notna().all(axis=1)].index.max()
    end = last_panel if pd.notna(last_panel) else merged[CORE].dropna().index.max()
    if topup and source == "dataguide":
        extra = _features(long[0], long[1], codes, bio_codes) if long is not None else _naver_topup(codes, bio_codes)   # 이미 받은 장기 이력이 있으면 재사용
        if extra is not None and pd.notna(last_panel):
            merged = _merge(merged, extra, only_after=last_panel)
            ok = extra[extra[CORE].notna().all(axis=1)]
            if len(ok):
                end = max(end, ok.index.max())
    merged = merged.loc[:end]      # 이번 실행에서 확인한 날짜까지만(예전 실행이 남긴 장중 값 등은 잘라낸다)
    merged.round(6).to_csv(HIST, encoding="utf-8", index_label="date")

    feats = P["features"]
    Zall = S.expanding_z(merged[feats], P["min_periods"])
    comp = S.composite(Zall, {k: 1 for k in feats})
    pbull = S.expanding_pct(comp, P["min_periods"])
    kos = merged["kosdaq"]
    r = kos.pct_change()
    start = pbull.first_valid_index()
    cost, amp, sm, lo, hi = P["cost"], P["amp"], P["smooth"], P["lo"], P["hi"]

    idx = pbull.loc[start:].index
    out_bases, now_bases = {}, {}
    for name, b in P["bases"].items():
        w = S.short_weight_band(pbull, b, amp, sm, lo, hi)
        w_raw = S.short_weight_band(pbull, b, amp, 1, lo, hi)
        ts = S.timing_stats(w.loc[start:], r.loc[start:], cost)
        cv = _curve(w, r, cost, start)
        rnd = lambda s, n=4: [None if pd.isna(v) else round(float(v), n) for v in s.loc[idx]]
        out_bases[name] = {
            "base": b, "w": rnd(w), "nav_sig": rnd(cv["nav_sig"], 2), "nav_static": rnd(cv["nav_static"], 2),
            "dd_sig": rnd(cv["dd_sig"]), "dd_static": rnd(cv["dd_static"]),
            "yearly": _yearly(w, r, cost, start),
            "stats": {"w_mean": ts.get("w_mean"), "turnover": ts.get("turnover"), "d_sharpe": ts.get("d_sharpe"), "d_mdd": ts.get("d_mdd"),
                      "ir": ts.get("ir"), "sig": ts.get("net"), "static": ts.get("static")},
        }
        now_bases[name] = {"base": b, "w": float(w.iloc[-1]), "w_raw": float(w_raw.iloc[-1]),
                           "w_prev_week": float(w.iloc[-6]) if len(w) > 6 else None}

    # 지표별 현재 상태(표준화 점수, 역사 백분위)
    comps = []
    for k in feats:
        s = merged[k]
        zk = Zall[k]
        pk = S.expanding_pct(s.where(Zall[k].notna()), P["min_periods"]).iloc[-1] if pd.notna(Zall[k].iloc[-1]) else np.nan
        comps.append({"key": k, "label": S.FEATURE_LABEL.get(k, k), "value": None if pd.isna(s.iloc[-1]) else round(float(s.iloc[-1]), 4),
                      "z": None if pd.isna(zk.iloc[-1]) else round(float(zk.iloc[-1]), 2), "pct": None if pd.isna(pk) else round(float(pk), 3)})
    adr20 = 100 * np.exp(merged["adr20"].iloc[-1])
    bio20 = 100 * np.exp(merged["bio20"].iloc[-1]) if pd.notna(merged["bio20"].iloc[-1]) else None

    pb_last = float(pbull.iloc[-1])
    bands = P["bands"]
    band = int(np.searchsorted(bands, pb_last, side="right"))   # 0(가장 위험) ~ 4(가장 낙관)
    payload = {
        "asof": merged.index[-1].strftime("%Y-%m-%d"),
        "generated": datetime.now(KST).strftime("%Y-%m-%d %H:%M KST"),
        "source": source, "n_stocks": int(merged["n_stocks"].iloc[-1]),
        "history": [merged.index[0].strftime("%Y-%m-%d"), merged.index[-1].strftime("%Y-%m-%d")],
        "params": {k: P[k] for k in ("amp", "smooth", "lo", "hi", "cost", "bases", "bands", "min_periods")} | {"n_features": len(feats)},
        "dates": [d.strftime("%Y-%m-%d") for d in idx],
        "kosdaq": [round(float(v), 2) for v in kos.loc[idx]],
        "pbull": [None if pd.isna(v) else round(float(v), 4) for v in pbull.loc[idx]],
        "bases": out_bases,
        "now": {"date": merged.index[-1].strftime("%Y-%m-%d"), "pbull": pb_last, "score": round(pb_last * 100, 1), "band": band,
                "pbull_prev_week": float(pbull.iloc[-6]) if len(pbull) > 6 else None, "kosdaq": float(kos.iloc[-1]),
                "kosdaq_chg_1w": float(kos.iloc[-1] / kos.iloc[-6] - 1) if len(kos) > 6 else None,
                "adr20": round(float(adr20), 1), "bio_adr20": None if bio20 is None else round(float(bio20), 1),
                "bases": now_bases, "components": comps,
                "recent": [{"date": d.strftime("%Y-%m-%d"), "score": round(float(v) * 100, 1)} for d, v in pbull.iloc[-15:].items()]},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(_clean(payload), f, ensure_ascii=False, separators=(",", ":"))
    print(f"[signal] 기준일 {payload['asof']}, 낙관 점수 {payload['now']['score']}, "
          + ", ".join(f"{k} {v['w']:.0%}" for k, v in now_bases.items()))
    return payload
