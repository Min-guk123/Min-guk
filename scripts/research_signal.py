"""
수급 확산 시그널 연구(오프라인). 장기 일봉 캐시(data/cache/prices_long.pkl)로
  1) 지표별 예측력(IC)과 통계적 유의성
  2) 지표를 평균해 만든 종합 점수의 5분위별 성과
  3) 숏 비중 규칙의 표본 외(walk-forward) 성과와 유의성
을 계산한다. 결과는 config/signal_params.json 과 docs/data/signal_research.json 에 저장.

사용법: python scripts/research_signal.py ic        # 1단계: 지표별 예측력
"""
from __future__ import annotations

import io
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import signal_model as S  # noqa: E402

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(BASE, "data", "cache")
HS = (5, 10, 20, 60)


def load_cache():
    d = pickle.load(open(os.path.join(CACHE, "prices_long.pkl"), "rb"))
    P, I = S.align_panel(d["prices"], d["index"])
    bio = [c for c, s in d["sector"].items() if s in S.BIO_SECTORS]
    return P, I, bio, d


def get_features():
    f = os.path.join(CACHE, "features_long_v2.pkl")
    if os.path.exists(f):
        return pickle.load(open(f, "rb"))
    P, I, bio, d = load_cache()
    F = pd.concat([S.breadth_features(P, I, bio), S.index_features(I)], axis=1)
    T = S.forward_targets(I, HS)
    pickle.dump((F, T, I), open(f, "wb"))
    return F, T, I


def rankvec(x: np.ndarray) -> np.ndarray:
    r = pd.Series(x).rank().values
    r = r - r.mean()
    return r / (np.sqrt((r ** 2).sum()) or 1.0)


def ic_with_perm(f: pd.Series, y: pd.Series, h: int):
    """스피어만 IC와, 신호를 원형으로 밀어(자기상관 유지) 만든 귀무분포 대비 p값."""
    df = pd.concat([f, y], axis=1).dropna()
    if len(df) < 300:
        return np.nan, np.nan, len(df)
    a, b = rankvec(df.iloc[:, 0].values), rankvec(df.iloc[:, 1].values)
    n = len(a)
    cc = np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(b)), n)   # lag k 에서의 상관
    kmin = max(60, 2 * h)
    lags = np.r_[np.arange(kmin, n - kmin)]
    null = cc[lags]
    obs = cc[0]
    p = float((np.abs(null) >= abs(obs)).mean())
    return float(obs), p, n


def stage_ic(out):
    F, T, I = get_features()
    rows = []
    mid = pd.Timestamp("2020-07-01")
    for c in S.BREADTH_FEATURES + S.INDEX_FEATURES:
        if c not in F:
            continue
        rec = {"feature": c, "label": S.FEATURE_LABEL.get(c, c)}
        for h in HS:
            ic, p, n = ic_with_perm(F[c], T[f"fwd{h}"], h)
            rec[f"ic{h}"], rec[f"p{h}"] = ic, p
        ic, p, n = ic_with_perm(F[c], T["fmdd20"], 20)
        rec["ic_dd"], rec["p_dd"] = ic, p
        # 기간을 반으로 나눴을 때 부호가 같은지(안정성)
        for h in (20,):
            a = F[c][F.index < mid]; b = F[c][F.index >= mid]
            ia, _, _ = ic_with_perm(a, T[f"fwd{h}"].reindex(a.index), h)
            ib, _, _ = ic_with_perm(b, T[f"fwd{h}"].reindex(b.index), h)
            rec["ic20_first"], rec["ic20_second"] = ia, ib
        rows.append(rec)
    R = pd.DataFrame(rows)
    out.write(f"표본: {F.dropna(how='all').index.min().date()} ~ {F.index.max().date()}  ({len(F)}일)\n")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
    show = R.copy()
    for c in show.columns:
        if c.startswith(("ic", "p")) and c != "feature":
            show[c] = show[c].astype(float).round(3)
    out.write(show.drop(columns=["feature"]).to_string(index=False))
    out.write("\n")
    R.to_pickle(os.path.join(CACHE, "ic_table.pkl"))


def get_z():
    f = os.path.join(CACHE, "z_long_v2.pkl")
    if os.path.exists(f):
        return pickle.load(open(f, "rb"))
    F, T, I = get_features()
    Z = S.expanding_z(F)
    pickle.dump(Z, open(f, "wb"))
    return Z


def bucket_table(score_pct: pd.Series, T: pd.DataFrame, bins=5, start=None, end=None):
    df = pd.concat([score_pct.rename("s"), T[["fwd5", "fwd20", "fmdd20"]]], axis=1).dropna()
    if start is not None:
        df = df[df.index >= start]
    if end is not None:
        df = df[df.index < end]
    df["b"] = pd.cut(df["s"], np.linspace(0, 1, bins + 1), include_lowest=True, labels=False)
    g = df.groupby("b")
    t = pd.DataFrame({"n": g.size(), "fwd20_%": g["fwd20"].mean() * 100, "hit20": g["fwd20"].apply(lambda x: (x > 0).mean()),
                      "fwd5_%": g["fwd5"].mean() * 100, "fmdd20_%": g["fmdd20"].mean() * 100})
    base = pd.Series({"n": len(df), "fwd20_%": df["fwd20"].mean() * 100, "hit20": (df["fwd20"] > 0).mean(),
                      "fwd5_%": df["fwd5"].mean() * 100, "fmdd20_%": df["fmdd20"].mean() * 100}, name="전체")
    return pd.concat([t, base.to_frame().T]).round(3)


def stage_comp(out):
    F, T, I = get_features()
    Z = get_z()
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
    fams = {"ADR": ["adr5", "adr10", "adr20", "adr40", "adr60"], "이긴비율": ["beat5", "beat10", "beat20", "beat60"],
            "이평위": ["ama20", "ama60", "ama120"], "신고저": ["nhl60", "nhl120"], "동일가중": ["ews20", "ews60"],
            "바이오ADR": ["bio20", "bio60"], "전체": S.BREADTH_FEATURES, "전체(바이오제외)": [c for c in S.BREADTH_FEATURES if not c.startswith("bio")],
            "지수추세": S.INDEX_FEATURES}
    out.write("== 평균화한 종합 점수의 예측력(IC, p값): 부호 +1(넓게 오를수록 낙관) ==\n")
    comp = {}
    for name, cols in fams.items():
        c = S.composite(Z, {k: 1 for k in cols})
        comp[name] = c
        rec = [name]
        for h in HS:
            ic, p, n = ic_with_perm(c, T[f"fwd{h}"], h)
            rec += [f"{ic:+.3f}", f"{p:.2f}"]
        ic, p, n = ic_with_perm(c, T["fmdd20"], 20)
        rec += [f"{ic:+.3f}", f"{p:.2f}"]
        out.write("  ".join(f"{x:>8}" for x in rec) + "\n")
    out.write("(열: 5일IC p | 10일IC p | 20일IC p | 60일IC p | 20일최대낙폭IC p)\n\n")
    pc = S.expanding_pct(comp["전체(바이오제외)"])
    pc.to_pickle(os.path.join(CACHE, "pct_all.pkl"))
    mid = pd.Timestamp("2020-07-01")
    for lab, st, en in (("전체 구간", None, None), ("2014~2020.6", None, mid), ("2020.7~2026", mid, None)):
        out.write(f"== 종합 점수(바이오제외 전체) 5분위별 이후 성과: {lab} ==\n")
        out.write(bucket_table(pc, T, 5, st, en).to_string() + "\n\n")
    out.write("== 10분위(극단 확인) 전체 구간 ==\n")
    out.write(bucket_table(pc, T, 10).to_string() + "\n")


def ols_hac(y, X, L=25):
    """OLS + Newey-West(HAC) t값. 겹치는 20일 수익률 때문에 오차가 서로 붙어 있으므로 HAC로 보정."""
    X = np.column_stack([np.ones(len(X)), X])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    n, k = X.shape
    U = X * e[:, None]
    S0 = U.T @ U
    for l in range(1, L + 1):
        G = U[l:].T @ U[:-l]
        S0 += (1 - l / (L + 1)) * (G + G.T)
    XtXi = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(XtXi @ S0 @ XtXi))
    return b, b / se, 1 - e.var() / y.var()


def stage_risk(out):
    F, T, I = get_features()
    Z = get_z()
    comp = S.composite(Z, {k: 1 for k in S.BREADTH_FEATURES if not k.startswith("bio")})
    D = pd.DataFrame({"breadth": comp, "lrv20": np.log(F["rv20"]), "imom60": F["imom60"], "idist": F["idist120"]})
    D = pd.concat([D, T[["fwd20", "fmdd20", "fvol20"]]], axis=1).dropna()
    D = D[D.index >= D.index[0] + pd.Timedelta(days=0)]
    for c in ("breadth", "lrv20", "imom60", "idist"):
        D[c] = (D[c] - D[c].mean()) / D[c].std()
    mid = pd.Timestamp("2020-07-01")
    out.write(f"표본 {D.index.min().date()} ~ {D.index.max().date()} ({len(D)}일), 설명변수는 표준화, 괄호는 HAC t값\n\n")
    for ycol, lab in (("fmdd20", "이후 20일 최대낙폭(클수록 덜 빠짐)"), ("fvol20", "이후 20일 변동성"), ("fwd20", "이후 20일 수익률(방향)")):
        out.write(f"[{lab}]\n")
        for name, cols in (("변동성만", ["lrv20"]), ("수급확산만", ["breadth"]), ("변동성+수급확산", ["lrv20", "breadth"]),
                           ("변동성+수급확산+지수추세", ["lrv20", "breadth", "imom60"])):
            for per, sub in (("전체", D), ("전반", D[D.index < mid]), ("후반", D[D.index >= mid])):
                b, t, r2 = ols_hac(sub[ycol].values, sub[cols].values)
                cs = "  ".join(f"{c}={b[i + 1]:+.4f}({t[i + 1]:+.1f})" for i, c in enumerate(cols))
                out.write(f"  {name:<16}{per}  R2={r2:.3f}  {cs}\n")
        out.write("\n")
    # 표본 외(walk-forward) R2: 매 21거래일 재추정, 겹치는 구간은 제외
    out.write("[표본 외 예측력: 확장 윈도우, 최소 750일 학습, 21일마다 재추정]\n")
    idx = D.index
    for ycol, lab in (("fmdd20", "20일 최대낙폭"), ("fvol20", "20일 변동성"), ("fwd20", "20일 수익률")):
        res = {}
        for name, cols in (("변동성만", ["lrv20"]), ("수급확산만", ["breadth"]), ("변동성+수급확산", ["lrv20", "breadth"])):
            pred = pd.Series(np.nan, index=idx); base = pd.Series(np.nan, index=idx)
            for t0 in range(750, len(idx), 21):
                tr = D.iloc[:t0 - 20]                      # 20일 뒤 값이 확정된 행만
                b = np.linalg.lstsq(np.column_stack([np.ones(len(tr)), tr[cols].values]), tr[ycol].values, rcond=None)[0]
                te = D.iloc[t0:t0 + 21]
                pred.iloc[t0:t0 + 21] = np.column_stack([np.ones(len(te)), te[cols].values]) @ b
                base.iloc[t0:t0 + 21] = tr[ycol].mean()
            m = pred.notna()
            sse = ((D[ycol][m] - pred[m]) ** 2).sum(); sst = ((D[ycol][m] - base[m]) ** 2).sum()
            res[name] = 1 - sse / sst
        out.write(f"  {lab:<10} " + "  ".join(f"{k}: {v:+.3f}" for k, v in res.items()) + "\n")
    out.write("(표본 외 R2: 0보다 커야 '평균값으로 찍는 것'보다 낫다)\n")


def forecast_vol(F, T, comp, t_min=750, step=21, cols=("lrv20", "breadth")):
    """확장 윈도우로 '이후 20일 변동성'을 예측(표본 외). 20일 뒤 값이 확정된 행만 학습에 쓴다."""
    D = pd.DataFrame({"breadth": comp, "lrv20": np.log(F["rv20"]), "fvol20": T["fvol20"]})
    X = D[list(cols)]
    ok = X.notna().all(axis=1)
    idx = D.index[ok]
    pred = pd.Series(np.nan, index=D.index)
    cols = list(cols)
    for t0 in range(t_min, len(idx), step):
        d_tr = D.loc[idx[:t0 - 20]].dropna()
        b = np.linalg.lstsq(np.column_stack([np.ones(len(d_tr)), d_tr[cols].values]), d_tr["fvol20"].values, rcond=None)[0]
        te = D.loc[idx[t0:t0 + step], cols]
        pred.loc[te.index] = np.column_stack([np.ones(len(te)), te.values]) @ b
    return pred


def risk_scores(F, T, Z):
    """위험 점수(0~1, 클수록 위험)들. 모두 그 날짜까지의 자료만 쓴다."""
    comp = S.composite(Z, {k: 1 for k in S.BREADTH_FEATURES if not k.startswith("bio")})
    s = {}
    s["수급확산"] = 1 - S.expanding_pct(comp)
    s["변동성"] = S.expanding_pct(F["rv20"])
    s["평균(수급+변동성)"] = (s["수급확산"] + s["변동성"]) / 2
    fc = forecast_vol(F, T, comp)
    s["예측모형"] = S.expanding_pct(fc, min_periods=120)
    s["_comp"], s["_fc"] = comp, fc
    return s


def perm_p(score, r, base, amp, smooth, cost, start, n=1500, seed=1, stat="ir"):
    """점수를 원형으로 밀어(자기상관 유지) 만든 귀무분포 대비 타이밍 효과의 p값."""
    rng = np.random.default_rng(seed)
    sc = score.loc[start:].copy()
    rr = r.loc[sc.index]
    def f(x):
        w = S.short_weight(1 - x, base - amp, base + amp, smooth)   # x: 위험점수 → w 크게
        return S.timing_stats(w, rr, cost).get(stat, np.nan)
    # 위의 short_weight 는 낙관(pbull)을 받으므로 위험점수에서 1 - x 를 넘긴다
    obs = f(sc)
    vals = sc.values.copy()
    m = len(vals)
    null = []
    for k in rng.integers(90, m - 90, n):
        null.append(f(pd.Series(np.roll(vals, int(k)), index=sc.index)))
    null = np.array(null)
    return obs, float((null >= obs).mean()), null


def stage_strat(out):
    F, T, I = get_features()
    Z = get_z()
    r = I.pct_change()
    sc = risk_scores(F, T, Z)
    names = ["수급확산", "변동성", "평균(수급+변동성)", "예측모형"]
    start = max(sc[n].first_valid_index() for n in names)
    out.write(f"공통 평가 구간: {start.date()} ~ {I.index[-1].date()}  ({len(I.loc[start:])}일)\n")
    cost = 0.001
    base, amp, smooth = 0.40, 0.25, 5
    pd.set_option("display.width", 250)
    # 1) 기준: 롱만 / 고정 헤지
    rr = r.loc[start:]
    out.write("롱 100%(헤지 없음): " + json.dumps({k: round(v, 3) for k, v in S.perf(rr).items()}, ensure_ascii=False) + "\n\n")
    rows = []
    for n in names:
        w = S.short_weight(1 - sc[n], base - amp, base + amp, smooth)
        ts = S.timing_stats(w.loc[start:], rr, cost)
        obs, p, _ = perm_p(sc[n], r, base, amp, smooth, cost, start, n=1000)
        rows.append({"전략": n, "평균숏": ts["w_mean"], "Sharpe(신호)": ts["net"]["sharpe"], "Sharpe(고정헤지)": ts["static"]["sharpe"],
                     "ΔSharpe": ts["d_sharpe"], "MDD(신호)": ts["net"]["mdd"], "MDD(고정헤지)": ts["static"]["mdd"], "ΔMDD": ts["d_mdd"],
                     "변동성(신호)": ts["net"]["vol"], "변동성(고정)": ts["static"]["vol"], "IR": ts["ir"], "연회전": ts["turnover"], "p값": p})
    R = pd.DataFrame(rows).set_index("전략").round(3)
    out.write(f"[기본 설정: 평균 숏 {base:.0%} 중심, 위험 점수에 따라 ±{amp:.0%}p, 평활 {smooth}일, 비용 {cost * 1e4:.0f}bp]\n")
    out.write(R.to_string() + "\n\n")
    out.write("p값: 같은 점수를 시간축에서 임의로 밀어 넣었을 때보다 타이밍 효과(IR)가 좋을 확률의 반대. 작을수록 '우연이 아니다'.\n\n")
    # 2) 구간 나눠 보기
    mids = [pd.Timestamp("2022-01-01")]
    for lab, a, b in (("전반", start, mids[0]), ("후반", mids[0], I.index[-1] + pd.Timedelta(days=1))):
        rows = []
        for n in names:
            w = S.short_weight(1 - sc[n], base - amp, base + amp, smooth)
            ww = w.loc[a:b]; r2 = r.loc[ww.index]
            ts = S.timing_stats(ww, r2, cost)
            rows.append({"전략": n, "ΔSharpe": ts["d_sharpe"], "ΔMDD": ts["d_mdd"], "IR": ts["ir"]})
        out.write(f"[{lab}: {a.date()} ~ {min(b, I.index[-1]).date()}]\n" + pd.DataFrame(rows).set_index("전략").round(3).to_string() + "\n\n")
    # 3) 파라미터 격자(평균화): 예측모형·평균·변동성·수급확산
    out.write("[격자 평균: base 0.2~0.6 x amp 0~0.4 x 평활 {1,5,10,20} 의 ΔSharpe / ΔMDD(%p)]\n")
    for n in names:
        ds, dm = [], []
        for b0 in (0.2, 0.3, 0.4, 0.5, 0.6):
            for a0 in (0.1, 0.2, 0.3, 0.4):
                for sm in (1, 5, 10, 20):
                    lo, hi = max(b0 - a0, 0), min(b0 + a0, 1)
                    w = S.short_weight(1 - sc[n], lo, hi, sm)
                    ts = S.timing_stats(w.loc[start:], rr, cost)
                    ds.append(ts["d_sharpe"]); dm.append(ts["d_mdd"] * 100)
        ds, dm = np.array(ds), np.array(dm)
        out.write(f"  {n:<14} ΔSharpe 평균 {ds.mean():+.3f} (최소 {ds.min():+.3f}, 최대 {ds.max():+.3f}, 양수 {np.mean(ds > 0):.0%})"
                  f" | ΔMDD 평균 {dm.mean():+.1f}%p (개선 {np.mean(dm > 0):.0%})\n")


FAMS = {"ADR": ["adr5", "adr10", "adr20", "adr40", "adr60"], "이긴비율": ["beat5", "beat10", "beat20", "beat60"],
        "이평위": ["ama20", "ama60", "ama120"], "신고저": ["nhl60", "nhl120"], "동일가중": ["ews20", "ews60"],
        "바이오ADR": ["bio20", "bio60"]}
ALLB = [c for c in S.BREADTH_FEATURES if not c.startswith("bio")]


def stage_strat2(out):
    F, T, I = get_features()
    Z = get_z()
    r = I.pct_change()
    cost = 0.001
    pd.set_option("display.width", 250)
    scores = {"평균(전체 지표)": S.composite(Z, {k: 1 for k in ALLB})}
    for n, cols in FAMS.items():
        scores[n] = S.composite(Z, {k: 1 for k in cols})
    risk = {n: 1 - S.expanding_pct(c) for n, c in scores.items()}
    start = risk["평균(전체 지표)"].first_valid_index()
    start = max(start, risk["ADR"].first_valid_index(), risk["바이오ADR"].first_valid_index())
    out.write(f"평가 구간 {start.date()} ~ {I.index[-1].date()} ({len(I.loc[start:])}일)\n")
    rr = r.loc[start:]
    out.write("롱 100%: " + json.dumps({k: round(v, 3) for k, v in S.perf(rr).items()}, ensure_ascii=False) + "\n\n")
    base, amp, smooth = 0.40, 0.25, 5
    blocks = [("전반", start, pd.Timestamp("2020-01-01")), ("중반", pd.Timestamp("2020-01-01"), pd.Timestamp("2023-01-01")),
              ("후반", pd.Timestamp("2023-01-01"), I.index[-1] + pd.Timedelta(days=1))]
    rows = []
    for n, x in risk.items():
        w = S.short_weight(1 - x, base - amp, base + amp, smooth)
        ts = S.timing_stats(w.loc[start:], rr, cost)
        obs, p, _ = perm_p(x, r, base, amp, smooth, cost, start, n=1000)
        rec = {"점수": n, "평균숏": ts["w_mean"], "ΔSharpe": ts["d_sharpe"], "ΔMDD%p": ts["d_mdd"] * 100, "IR": ts["ir"], "p값": p, "연회전": ts["turnover"]}
        for lab, a, b in blocks:
            ww = w.loc[a:b]; ts2 = S.timing_stats(ww, r.loc[ww.index], cost)
            rec[f"IR_{lab}"] = ts2.get("ir", np.nan); rec[f"dMDD_{lab}"] = ts2.get("d_mdd", np.nan) * 100
        rows.append(rec)
    out.write(f"[설정 base {base:.0%} ± {amp:.0%}p, 평활 {smooth}일, 비용 {cost * 1e4:.0f}bp]\n")
    out.write(pd.DataFrame(rows).set_index("점수").round(3).to_string() + "\n\n")

    x = risk["평균(전체 지표)"]
    out.write("[평균 점수: base x amp 격자 (평활 5일)]  ΔSharpe / ΔMDD(%p) / 평균숏\n")
    for b0 in (0.2, 0.3, 0.4, 0.5, 0.6):
        line = []
        for a0 in (0.0, 0.1, 0.2, 0.3, 0.4):
            lo, hi = max(b0 - a0, 0), min(b0 + a0, 1)
            w = S.short_weight(1 - x, lo, hi, 5)
            ts = S.timing_stats(w.loc[start:], rr, cost)
            line.append(f"{ts['d_sharpe']:+.3f}/{ts['d_mdd'] * 100:+.1f}/{ts['w_mean']:.2f}")
        out.write(f"  base {b0:.1f}: " + " | ".join(line) + "\n")
    out.write("  (열: amp 0.0, 0.1, 0.2, 0.3, 0.4)\n\n")
    for sm in (1, 10, 20):
        w = S.short_weight(1 - x, 0.15, 0.65, sm)
        ts = S.timing_stats(w.loc[start:], rr, cost)
        out.write(f"  평활 {sm:>2}일 (15~65%): ΔSharpe {ts['d_sharpe']:+.3f}  ΔMDD {ts['d_mdd'] * 100:+.1f}%p  연회전 {ts['turnover']:.1f}\n")
    for c in (0.0, 0.003, 0.005):
        w = S.short_weight(1 - x, 0.15, 0.65, 5)
        ts = S.timing_stats(w.loc[start:], rr, c)
        out.write(f"  비용 {c * 1e4:.0f}bp: ΔSharpe {ts['d_sharpe']:+.3f}  ΔMDD {ts['d_mdd'] * 100:+.1f}%p\n")

    # ADR 기준선(80/100/120): 선생님이 쓰는 해석
    out.write("\n[ADR 20일 구간별 이후 성과 (ADR=100*exp(로그ADR))]\n")
    for col in ("adr20", "bio20"):
        adr = 100 * np.exp(F[col])
        b = pd.cut(adr, [0, 80, 100, 120, 1e9], labels=["<80", "80~100", "100~120", ">120"])
        df = pd.concat([b.rename("b"), T[["fwd5", "fwd20", "fmdd20", "fvol20"]]], axis=1).dropna()
        g = df.groupby("b", observed=True)
        t = pd.DataFrame({"일수": g.size(), "fwd5_%": g["fwd5"].mean() * 100, "fwd20_%": g["fwd20"].mean() * 100,
                          "상승비율": g["fwd20"].apply(lambda s: (s > 0).mean()), "fmdd20_%": g["fmdd20"].mean() * 100,
                          "fvol20_%": g["fvol20"].mean() * 100}).round(2)
        out.write(f"  {col}\n" + t.to_string() + "\n")

    # 추세 구간별(지수 120일선 위/아래) 방향 예측력
    out.write("\n[지수 120일선 위/아래에서의 수급확산 점수 예측력: fwd20 ~ breadth, HAC t값]\n")
    D = pd.concat([scores["평균(전체 지표)"].rename("b"), F["idist120"].rename("d"), T[["fwd20", "fmdd20"]]], axis=1).dropna()
    D["b"] = (D["b"] - D["b"].mean()) / D["b"].std()
    for lab, m in (("120일선 위", D["d"] > 0), ("120일선 아래", D["d"] <= 0)):
        sub = D[m]
        for y in ("fwd20", "fmdd20"):
            bb, tt, r2 = ols_hac(sub[y].values, sub[["b"]].values, L=25)
            out.write(f"  {lab:<8} n={len(sub):>4}  {y}: 계수 {bb[1]:+.4f}  t={tt[1]:+.1f}\n")


def perm_generic(make_w, score, r, start, stat, cost, n=1500, seed=3):
    """make_w(score)->숏비중. 점수만 원형으로 밀어(자기상관 유지) 타이밍 정보가 없을 때의 분포를 만든다."""
    rng = np.random.default_rng(seed)
    sc = score.loc[start:]
    rr = r.loc[sc.index]
    obs = S.timing_stats(make_w(sc), rr, cost).get(stat, np.nan)
    vals = sc.values
    m = len(vals)
    null = np.array([S.timing_stats(make_w(pd.Series(np.roll(vals, int(k)), index=sc.index)), rr, cost).get(stat, np.nan)
                     for k in rng.integers(90, m - 90, n)])
    return obs, float((null >= obs).mean()), float(np.nanmean(null))


def wf_family_scores(Z, T, first="2019-01-01", min_ic=0.0, refit_months=12):
    """해마다 과거 자료(미래 20일이 확정된 날까지)만으로 '이후 20일 최대낙폭 IC > min_ic' 인 지표군만 골라 평균."""
    fam_scores = {n: S.composite(Z, {k: 1 for k in cols}) for n, cols in FAMS.items()}
    idx = Z.index
    out = pd.Series(np.nan, index=idx)
    picks = {}
    dates = pd.date_range(first, idx[-1], freq=f"{refit_months}MS")
    dates = list(dates) + [idx[-1] + pd.Timedelta(days=1)]
    for a, b in zip(dates[:-1], dates[1:]):
        tr_end = idx[idx < a]
        if len(tr_end) < 300:
            continue
        cut = tr_end[-21]                                   # 20일 뒤 값이 확정된 날까지
        chosen = []
        for n, s in fam_scores.items():
            d = pd.concat([s, T["fmdd20"]], axis=1).loc[:cut].dropna()
            if len(d) > 300 and d.iloc[:, 0].rank().corr(d.iloc[:, 1].rank()) > min_ic:
                chosen.append(n)
        picks[str(a.date())] = chosen
        seg = idx[(idx >= a) & (idx < b)]
        if chosen:
            out.loc[seg] = pd.concat([fam_scores[n] for n in chosen], axis=1).mean(axis=1).loc[seg]
    return out, picks


def stage_strat3(out):
    F, T, I = get_features()
    Z = get_z()
    r = I.pct_change()
    cost = 0.001
    pd.set_option("display.width", 250)
    base, amp, smooth = 0.40, 0.25, 10
    sc_all = S.composite(Z, {k: 1 for k in ALLB})
    abs_cols = FAMS["ADR"] + FAMS["이평위"] + FAMS["신고저"]
    sc_abs = S.composite(Z, {k: 1 for k in abs_cols})
    sc_abs4 = S.composite(Z, {k: 1 for k in abs_cols + FAMS["바이오ADR"]})
    sc_wf, picks = wf_family_scores(Z, T)
    out.write("[해마다 과거 자료만으로 고른 지표군]\n")
    for k, v in picks.items():
        out.write(f"  {k}: {v}\n")
    risk = {"전체 평균": 1 - S.expanding_pct(sc_all), "절대지표 평균(ADR+이평위+신고저)": 1 - S.expanding_pct(sc_abs),
            "절대지표+바이오ADR": 1 - S.expanding_pct(sc_abs4), "연도별 선택(표본 외)": 1 - S.expanding_pct(sc_wf, 120)}
    start_all = max(v.first_valid_index() for k, v in risk.items() if k != "연도별 선택(표본 외)")
    start_wf = risk["연도별 선택(표본 외)"].first_valid_index()
    gate = (F["idist120"] < 0).astype(float)                 # 지수가 120일선 아래

    def mk(gated):
        def f(x):
            w = S.short_weight(1 - x, base - amp, base + amp, smooth)
            if gated:
                g = gate.reindex(x.index).fillna(0.0)
                w = base + (w - base) * g
                w = w.ewm(span=smooth, adjust=False, min_periods=1).mean() if smooth > 1 else w
            return w
        return f

    rows = []
    for name, x in risk.items():
        start = start_wf if "표본 외" in name else start_all
        for gated in (False, True):
            f = mk(gated)
            ww = f(x.loc[start:])
            rr = r.loc[ww.index]
            ts = S.timing_stats(ww, rr, cost)
            _, p_ir, _ = perm_generic(f, x, r, start, "ir", cost, n=800)
            _, p_dd, _ = perm_generic(f, x, r, start, "d_mdd", cost, n=800)
            rows.append({"점수": name, "추세게이트": "O" if gated else "X", "시작": str(start.date()), "평균숏": ts["w_mean"],
                         "ΔSharpe": ts["d_sharpe"], "ΔMDD%p": ts["d_mdd"] * 100, "IR": ts["ir"], "p(IR)": p_ir, "p(ΔMDD)": p_dd, "연회전": ts["turnover"]})
    out.write(f"\n[base {base:.0%} ± {amp:.0%}p, 평활 {smooth}일, 비용 {cost * 1e4:.0f}bp]\n")
    out.write(pd.DataFrame(rows).round(3).to_string(index=False) + "\n")
    # 같은 구간(연도별 선택 시작일 이후)으로 맞춰 다시 비교
    out.write(f"\n[동일 구간 {start_wf.date()} ~ 비교]\n")
    rows = []
    for name, x in risk.items():
        for gated in (False, True):
            ww = mk(gated)(x.loc[start_wf:]); ts = S.timing_stats(ww, r.loc[ww.index], cost)
            rows.append({"점수": name, "게이트": "O" if gated else "X", "ΔSharpe": ts["d_sharpe"], "ΔMDD%p": ts["d_mdd"] * 100, "IR": ts["ir"]})
    out.write(pd.DataFrame(rows).round(3).to_string(index=False) + "\n")


def make_w_fn(gate, lo, hi, smooth):
    """위험점수(0~1) -> 숏 비중. gate 가 있으면 지수가 120일선 아래일 때만 점수로 가감하고, 위에서는 중간값(base)."""
    base = (lo + hi) / 2
    def f(x):
        w = S.short_weight(1 - x, lo, hi, 1)
        if gate is not None:
            g = gate.reindex(x.index).fillna(0.0)
            w = base + (w - base) * g
        return w.ewm(span=smooth, adjust=False, min_periods=1).mean().where(w.notna()) if smooth > 1 else w
    return f


def stage_final0(out):
    F, T, I = get_features()
    Z = get_z()
    r = I.pct_change()
    x = 1 - S.expanding_pct(S.composite(Z, {k: 1 for k in ALLB}))
    start = x.first_valid_index()
    gate = (F["idist120"] < 0).astype(float)
    cost = 0.001
    blocks = [("2016.9~2019", start, pd.Timestamp("2020-01-01")), ("2020~2022", pd.Timestamp("2020-01-01"), pd.Timestamp("2023-01-01")),
              ("2023~2026.10", pd.Timestamp("2023-01-01"), I.index[-1] + pd.Timedelta(days=1))]
    pd.set_option("display.width", 250)
    out.write(f"평가 구간 {start.date()} ~ {I.index[-1].date()}\n\n[구간별 ΔMDD(%p) / IR — 평활 10일, base 40±25]\n")
    for gl, g in (("게이트 없음", None), ("게이트(120일선 아래만)", gate)):
        f = make_w_fn(g, 0.15, 0.65, 10)
        w = f(x)
        line = []
        for lab, a, b in blocks:
            ww = w.loc[a:b]; ts = S.timing_stats(ww, r.loc[ww.index], cost)
            line.append(f"{lab}: ΔMDD {ts['d_mdd'] * 100:+.1f}, ΔSharpe {ts['d_sharpe']:+.3f}, IR {ts['ir']:+.2f}")
        out.write(f"  {gl:<22} " + " | ".join(line) + "\n")
    out.write("\n[게이트 적용: base x amp 격자, 평활 10일]  ΔSharpe / ΔMDD%p / 연회전 / 평균숏\n")
    for b0 in (0.2, 0.3, 0.4, 0.5):
        line = []
        for a0 in (0.1, 0.2, 0.3, 0.4):
            lo, hi = max(b0 - a0, 0), min(b0 + a0, 1)
            w = make_w_fn(gate, lo, hi, 10)(x)
            ts = S.timing_stats(w.loc[start:], r.loc[start:], cost)
            line.append(f"{ts['d_sharpe']:+.3f}/{ts['d_mdd'] * 100:+.1f}/{ts['turnover']:.1f}/{ts['w_mean']:.2f}")
        out.write(f"  base {b0:.1f}: " + " | ".join(line) + "\n")
    out.write("  (열: amp 0.1, 0.2, 0.3, 0.4)\n\n[비용 민감도: 게이트, base 40±25, 평활 10]\n")
    for c in (0.0, 0.001, 0.003, 0.005, 0.01):
        w = make_w_fn(gate, 0.15, 0.65, 10)(x)
        ts = S.timing_stats(w.loc[start:], r.loc[start:], c)
        out.write(f"  {c * 1e4:>3.0f}bp: ΔSharpe {ts['d_sharpe']:+.3f}  ΔMDD {ts['d_mdd'] * 100:+.1f}%p  신호 Sharpe {ts['net']['sharpe']:.3f} / 고정 {ts['static']['sharpe']:.3f}\n")
    out.write("\n[평활 x 게이트: ΔSharpe / ΔMDD / 연회전]\n")
    for sm in (1, 5, 10, 20, 40):
        a = S.timing_stats(make_w_fn(gate, 0.15, 0.65, sm)(x).loc[start:], r.loc[start:], cost)
        out.write(f"  평활 {sm:>2}일: {a['d_sharpe']:+.3f} / {a['d_mdd'] * 100:+.1f} / {a['turnover']:.1f}\n")
    # 점수 5분위별 이후 성과 (게이트 무관, 전체/반반)
    pb = 1 - x
    mid = pd.Timestamp("2021-07-01")
    for lab, st, en in (("전체", None, None), ("전반", None, mid), ("후반", mid, None)):
        df = pd.concat([pb.rename("s"), T[["fwd5", "fwd20", "fmdd20", "fvol20"]]], axis=1).dropna()
        if st is not None: df = df[df.index >= st]
        if en is not None: df = df[df.index < en]
        df["b"] = pd.cut(df["s"], np.linspace(0, 1, 6), include_lowest=True, labels=False)
        g = df.groupby("b")
        t = pd.DataFrame({"일수": g.size(), "fwd20%": g["fwd20"].mean() * 100, "상승비율": g["fwd20"].apply(lambda s: (s > 0).mean()),
                          "최대낙폭%": g["fmdd20"].mean() * 100, "변동성%": g["fvol20"].mean() * 100}).round(2)
        out.write(f"\n[낙관점수 5분위(0=가장 약함) 이후 성과: {lab}]\n" + t.to_string() + "\n")


def stage_fund(out):
    """펀드 구조(롱 100% + 지수 숏 50~100%)에 맞춘 가감 폭 시뮬레이션."""
    F, T, I = get_features()
    Z = get_z()
    r = I.pct_change()
    pb = S.expanding_pct(S.composite(Z, {k: 1 for k in ALLB}))
    start = pb.first_valid_index()
    rr = r.loc[start:]
    pd.set_option("display.width", 250)
    out.write(f"평가 구간 {start.date()} ~ {I.index[-1].date()} ({len(rr)}일). 순노출 = 100% - 숏 비중. 롱 북은 지수 베타 1로 가정.\n\n")
    out.write("[기준 숏 비중 x 가감 폭: ΔSharpe / ΔMDD(%p, 순노출 100% 환산) / 연회전 — 평활 10일, 비용 10bp]\n")
    for b0 in (0.5, 0.625, 0.75, 0.875, 1.0):
        line = []
        for a0 in (0.0625, 0.125, 0.25, 0.375, 0.5):
            w = S.short_weight_band(pb, b0, a0, 10)
            ts = S.timing_stats(w.loc[start:], rr, 0.001)
            line.append(f"{ts['d_sharpe']:+.3f}/{ts['d_mdd'] * 100:+.1f}/{ts['turnover']:.1f}")
        out.write(f"  base {b0:.3f}: " + " | ".join(line) + "\n")
    out.write("  (열: amp 0.0625, 0.125, 0.25, 0.375, 0.5)\n\n")
    f = lambda x: S.short_weight_band(x, 0.75, 0.25, 10)
    sc = 1 - pb
    for stat, lab in (("ir", "IR"), ("d_mdd", "ΔMDD"), ("d_sharpe", "ΔSharpe")):
        obs, p, nm = perm_generic(lambda x: S.short_weight_band(1 - x, 0.75, 0.25, 10), sc, r, start, stat, 0.001, n=1500)
        out.write(f"  base 75% ±25%p 유의성[{lab}]: 관측 {obs:+.4f}, 임의 배치 평균 {nm:+.4f}, p={p:.3f}\n")
    ts = S.timing_stats(f(pb).loc[start:], rr, 0.001)
    out.write("\n[base 75% ±25%p 상세]\n" + json.dumps({k: (round(v, 4) if isinstance(v, float) else {kk: round(vv, 4) for kk, vv in v.items()} if isinstance(v, dict) else v)
                                                      for k, v in ts.items()}, ensure_ascii=False, indent=1) + "\n")
    # 구간별
    blocks = [("2016.9~2019", start, pd.Timestamp("2020-01-01")), ("2020~2022", pd.Timestamp("2020-01-01"), pd.Timestamp("2023-01-01")),
              ("2023~2026.10", pd.Timestamp("2023-01-01"), I.index[-1] + pd.Timedelta(days=1))]
    w = f(pb)
    out.write("\n[구간별 base 75% ±25%p]\n")
    for lab, a, b in blocks:
        ww = w.loc[a:b]; t2 = S.timing_stats(ww, r.loc[ww.index], 0.001)
        out.write(f"  {lab}: ΔSharpe {t2['d_sharpe']:+.3f}  ΔMDD {t2['d_mdd'] * 100:+.1f}%p  IR {t2['ir']:+.2f}  신호 MDD {t2['net']['mdd'] * 100:.1f}% / 고정 {t2['static']['mdd'] * 100:.1f}%\n")
    # 가장 큰 급락 구간에서의 숏 비중 (사례)
    out.write("\n[주요 급락 구간에서 권장 숏 비중 (base 75 ±25)]\n")
    for lab, a, b in (("2018 하락", "2018-01-26", "2018-10-30"), ("2020 코로나", "2020-02-14", "2020-03-19"), ("2021말~2022", "2021-09-01", "2022-09-30"),
                      ("2024 하락", "2024-07-01", "2024-08-05"), ("최근 3개월", "2026-07-12", "2026-10-12")):
        seg = I.loc[a:b]
        if len(seg) < 3: continue
        ww = w.loc[seg.index]
        out.write(f"  {lab}: 지수 {seg.iloc[0]:.0f}→{seg.iloc[-1]:.0f} ({seg.iloc[-1] / seg.iloc[0] - 1:+.1%}), 숏 비중 시작 {ww.iloc[0]:.0%} → 끝 {ww.iloc[-1]:.0%}, 평균 {ww.mean():.0%}\n")
    out.write(f"\n[오늘({I.index[-1].date()})] 낙관 점수 백분위 {pb.iloc[-1]:.3f}, 권장 숏 비중(base 75±25) {w.iloc[-1]:.1%}\n")
    out.write("최근 10일 점수: " + ", ".join(f"{v:.2f}" for v in pb.iloc[-10:]) + "\n")


def _perf_pack(ts):
    g = lambda d, k: None if not d else d.get(k)
    return {"sharpe": g(ts.get("net"), "sharpe"), "mdd": g(ts.get("net"), "mdd"), "vol": g(ts.get("net"), "vol"), "ret": g(ts.get("net"), "ret"),
            "static_sharpe": g(ts.get("static"), "sharpe"), "static_mdd": g(ts.get("static"), "mdd"), "static_vol": g(ts.get("static"), "vol"),
            "static_ret": g(ts.get("static"), "ret"), "d_sharpe": ts.get("d_sharpe"), "d_mdd": ts.get("d_mdd"), "ir": ts.get("ir"),
            "turnover": ts.get("turnover"), "w_mean": ts.get("w_mean")}


def stage_report(out):
    """docs/data/signal_research.json 생성: 대시보드 ④ 탭이 보여주는 모든 연구 결과."""
    from signal_pipeline import _clean
    from datetime import datetime
    F, T, I = get_features()
    Z = get_z()
    r = I.pct_change()
    with open(os.path.join(BASE, "config", "signal_params.json"), encoding="utf-8-sig") as fp:   # 숫자의 출처는 이 파일 하나
        PJ = json.load(fp)
    cost, amp, smooth, lo, hi = PJ["cost"], PJ["amp"], PJ["smooth"], PJ["lo"], PJ["hi"]
    comp = S.composite(Z, {k: 1 for k in ALLB})
    pb = S.expanding_pct(comp)
    start = pb.first_valid_index()
    split = pd.Timestamp("2021-07-01")
    R = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
         "sample": {"start": str(F.index[0].date()), "end": str(F.index[-1].date()), "days": int(len(F)),
                    "eval_start": str(start.date()), "n_stocks_first": int(pd.read_csv(os.path.join(BASE, "data", "feature_history.csv"), index_col=0)["n_stocks"].iloc[0]),
                    "n_stocks_last": 1000, "split": str(split.date())}}
    # 1) 지표별 예측력
    fam_of = {c: n for n, cols in FAMS.items() for c in cols}
    rows = []
    for c in S.BREADTH_FEATURES + ["imom60", "rv20"]:
        rec = {"key": c, "label": S.FEATURE_LABEL.get(c, c), "family": fam_of.get(c, "지수 기준선")}
        for tgt, h in (("fwd5", 5), ("fwd20", 20), ("fwd60", 60), ("fmdd20", 20), ("fvol20", 20)):
            ic, p, _ = ic_with_perm(F[c], T[tgt], h)
            rec[tgt] = {"ic": ic, "p": p}
        rows.append(rec)
    R["ic"] = rows
    cc = {}
    for tgt, h in (("fwd5", 5), ("fwd10", 10), ("fwd20", 20), ("fwd60", 60), ("fmdd20", 20), ("fvol20", 20)):
        ic, p, _ = ic_with_perm(comp, T[tgt], h)
        cc[tgt] = {"ic": ic, "p": p}
    R["composite_ic"] = cc
    # 2) 5분위 성과
    qs = {}
    for lab, st_, en_ in (("all", None, None), ("first", None, split), ("second", split, None)):
        df = pd.concat([pb.rename("s"), T[["fwd5", "fwd20", "fmdd20", "fvol20"]]], axis=1).dropna()
        if st_ is not None: df = df[df.index >= st_]
        if en_ is not None: df = df[df.index < en_]
        df["b"] = pd.cut(df["s"], np.linspace(0, 1, 6), include_lowest=True, labels=False)
        g = df.groupby("b")
        qs[lab] = [{"n": int(g.size()[i]), "fwd20": float(g["fwd20"].mean()[i]), "hit": float((df[df.b == i]["fwd20"] > 0).mean()),
                    "fmdd20": float(g["fmdd20"].mean()[i]), "fvol20": float(g["fvol20"].mean()[i])} for i in range(5)]
        qs[lab + "_base"] = {"fwd20": float(df["fwd20"].mean()), "hit": float((df["fwd20"] > 0).mean()), "fmdd20": float(df["fmdd20"].mean()), "fvol20": float(df["fvol20"].mean()), "n": int(len(df))}
    R["quintiles"] = qs
    # 3) 위험 회귀(변동성 통제) + 표본 외 R2
    D = pd.DataFrame({"breadth": comp, "lrv20": np.log(F["rv20"])})
    D = pd.concat([D, T[["fwd20", "fmdd20", "fvol20"]]], axis=1).dropna()
    for c in ("breadth", "lrv20"):
        D[c] = (D[c] - D[c].mean()) / D[c].std()
    reg = {}
    for y in ("fvol20", "fmdd20", "fwd20"):
        rec = {}
        for per, sub in (("all", D), ("first", D[D.index < split]), ("second", D[D.index >= split])):
            b1, t1, r1 = ols_hac(sub[y].values, sub[["lrv20"]].values)
            b2, t2, r2 = ols_hac(sub[y].values, sub[["lrv20", "breadth"]].values)
            rec[per] = {"r2_vol": r1, "r2_both": r2, "breadth_coef": b2[2], "breadth_t": t2[2], "vol_coef": b2[1], "vol_t": t2[1]}
        idx = D.index; oos = {}
        for name, cols in (("vol", ["lrv20"]), ("breadth", ["breadth"]), ("both", ["lrv20", "breadth"])):
            pred = pd.Series(np.nan, index=idx); basep = pd.Series(np.nan, index=idx)
            for t0 in range(750, len(idx), 21):
                tr = D.iloc[:t0 - 20]
                bb = np.linalg.lstsq(np.column_stack([np.ones(len(tr)), tr[cols].values]), tr[y].values, rcond=None)[0]
                te = D.iloc[t0:t0 + 21]
                pred.iloc[t0:t0 + 21] = np.column_stack([np.ones(len(te)), te[cols].values]) @ bb
                basep.iloc[t0:t0 + 21] = tr[y].mean()
            m = pred.notna()
            oos[name] = float(1 - ((D[y][m] - pred[m]) ** 2).sum() / ((D[y][m] - basep[m]) ** 2).sum())
        rec["oos_r2"] = oos
        reg[y] = rec
    R["regression"] = reg
    # 4) ADR 구간
    ab = {}
    for col in ("adr20", "bio20"):
        adr = 100 * np.exp(F[col])
        b = pd.cut(adr, [0, 80, 100, 120, 1e9], labels=["<80", "80~100", "100~120", ">120"])
        df = pd.concat([b.rename("b"), T[["fwd20", "fmdd20", "fvol20"]]], axis=1).dropna()
        g = df.groupby("b", observed=True)
        ab[col] = [{"band": k, "n": int(len(v)), "fwd20": float(v["fwd20"].mean()), "hit": float((v["fwd20"] > 0).mean()),
                    "fmdd20": float(v["fmdd20"].mean()), "fvol20": float(v["fvol20"].mean())} for k, v in g]
    R["adr_bands"] = ab
    # 5) 펀드 구조(숏 50~100%)에서의 시뮬레이션
    rr = r.loc[start:]
    blocks = [("2016.9~2019", start, pd.Timestamp("2020-01-01")), ("2020~2022", pd.Timestamp("2020-01-01"), pd.Timestamp("2023-01-01")),
              ("2023~2026", pd.Timestamp("2023-01-01"), I.index[-1] + pd.Timedelta(days=1))]
    def run_ts(score_pb, base=0.75, a=amp, sm=smooth, c=cost, lo_=lo, hi_=hi, s0=None, s1=None):
        w = S.short_weight_band(score_pb, base, a, sm, lo_, hi_)
        w = w.loc[s0 or start:s1] if s1 is not None or s0 is not None else w.loc[start:]
        return S.timing_stats(w, r.loc[w.index], c), w
    ts, w75 = run_ts(pb)
    fund = {"base": 0.75, "amp": amp, "smooth": smooth, "cost": cost, "stats": _perf_pack(ts), "long": ts.get("long")}
    sc = 1 - pb
    for stat in ("ir", "d_mdd", "d_sharpe"):
        obs, p, nm = perm_generic(lambda x: S.short_weight_band(1 - x, 0.75, amp, smooth, lo, hi), sc, r, start, stat, cost, n=1500)
        fund[f"p_{stat}"] = p
    fund["blocks"] = []
    for lab, a, b in blocks:
        ww = w75.loc[a:b]; t2 = S.timing_stats(ww, r.loc[ww.index], cost)
        fund["blocks"].append({"label": lab, **_perf_pack(t2)})
    bases = [0.5, 0.625, 0.75, 0.875, 1.0]; amps = [0.0625, 0.125, 0.25, 0.375, 0.5]
    grid = {"bases": bases, "amps": amps, "d_sharpe": [], "d_mdd": [], "turnover": []}
    for b0 in bases:
        a1, a2, a3 = [], [], []
        for a0 in amps:
            t2 = S.timing_stats(S.short_weight_band(pb, b0, a0, smooth, lo, hi).loc[start:], rr, cost)
            a1.append(t2["d_sharpe"]); a2.append(t2["d_mdd"]); a3.append(t2["turnover"])
        grid["d_sharpe"].append(a1); grid["d_mdd"].append(a2); grid["turnover"].append(a3)
    fund["grid"] = grid
    fund["cost_sens"] = []
    for c in (0.0, 0.001, 0.003, 0.005, 0.01):
        t2 = S.timing_stats(S.short_weight_band(pb, 0.75, amp, smooth, lo, hi).loc[start:], rr, c)
        fund["cost_sens"].append({"cost_bp": c * 1e4, "d_sharpe": t2["d_sharpe"], "d_mdd": t2["d_mdd"]})
    fund["smooth_sens"] = []
    for sm in (1, 5, 10, 20, 40):
        t2 = S.timing_stats(S.short_weight_band(pb, 0.75, amp, sm, lo, hi).loc[start:], rr, cost)
        fund["smooth_sens"].append({"smooth": sm, "d_sharpe": t2["d_sharpe"], "d_mdd": t2["d_mdd"], "turnover": t2["turnover"]})
    eps = []
    for lab, a, b in (("2018 하락", "2018-01-26", "2018-10-30"), ("2020 코로나", "2020-02-14", "2020-03-19"), ("2021말~2022 하락", "2021-09-01", "2022-09-30"),
                      ("2024 하락", "2024-07-01", "2024-08-05")):
        seg = I.loc[a:b]; ww = w75.loc[seg.index]
        eps.append({"label": lab, "from": a, "to": b, "idx_chg": float(seg.iloc[-1] / seg.iloc[0] - 1), "w_start": float(ww.iloc[0]), "w_end": float(ww.iloc[-1]), "w_mean": float(ww.mean())})
    fund["episodes"] = eps
    R["fund"] = fund
    # 6) 지표군별 비교(같은 구조, 같은 기간)
    fams = []
    for name, cols in list(FAMS.items()) + [("전체 평균(채택)", ALLB)]:
        c = S.composite(Z, {k: 1 for k in cols})
        pbf = S.expanding_pct(c)
        s0 = max(start, pbf.first_valid_index())
        f = lambda x: S.short_weight_band(1 - x, 0.75, amp, smooth, lo, hi)
        t2 = S.timing_stats(f(1 - (1 - pbf)).loc[s0:] if False else S.short_weight_band(pbf, 0.75, amp, smooth, lo, hi).loc[s0:], r.loc[s0:], cost)
        _, p, _ = perm_generic(f, 1 - pbf, r, s0, "d_mdd", cost, n=500)
        bl = []
        for lab, a, b in blocks:
            ww = S.short_weight_band(pbf, 0.75, amp, smooth, lo, hi).loc[max(a, s0):b]
            t3 = S.timing_stats(ww, r.loc[ww.index], cost)
            bl.append(t3.get("d_mdd"))
        fams.append({"name": name, "n": len(cols), "d_sharpe": t2["d_sharpe"], "d_mdd": t2["d_mdd"], "p_mdd": p, "blocks_mdd": bl})
    R["families"] = fams
    # 7) 변동성으로 조절하면? (비교용)
    pv = S.expanding_pct(F["rv20"])
    s0 = pv.first_valid_index()
    tv = S.timing_stats(S.short_weight_band(1 - pv, 0.75, amp, smooth, lo, hi).loc[max(s0, start):], r.loc[max(s0, start):], cost)
    R["vol_timing"] = {"d_sharpe": tv["d_sharpe"], "d_mdd": tv["d_mdd"]}
    # 8) 추세 구간별
    dd = pd.concat([comp.rename("b"), F["idist120"].rename("d"), T[["fwd20", "fmdd20"]]], axis=1).dropna()
    dd["b"] = (dd["b"] - dd["b"].mean()) / dd["b"].std()
    reg2 = {}
    for lab, m in (("above", dd["d"] > 0), ("below", dd["d"] <= 0)):
        sub = dd[m]; rec = {"n": int(len(sub))}
        for y in ("fwd20", "fmdd20"):
            bb, tt, _ = ols_hac(sub[y].values, sub[["b"]].values)
            rec[y] = {"coef": bb[1], "t": tt[1]}
        reg2[lab] = rec
    R["trend_regime"] = reg2
    # 9) 가감 폭별 효과 (한 해에 기대는지 보려고 '가장 좋은 해'를 뺀 평균도 함께)
    at = []
    for a0 in (0.0625, 0.125, 0.25, 0.375, 0.5):
        w = S.short_weight_band(pb, 0.75, a0, smooth, lo, hi).loc[start:]
        ts0 = S.timing_stats(w, rr, cost)
        wl = w.shift(1); net = (1 - wl) * rr - cost * w.diff().abs().fillna(0.0); stc = (1 - wl.mean()) * rr
        yd = [float((1 + net[w.index.year == y]).prod() - (1 + stc[w.index.year == y]).prod()) for y in sorted(set(w.index.year))]
        best = max(yd)
        at.append({"amp": a0, "d_mdd": ts0["d_mdd"], "d_sharpe": ts0["d_sharpe"], "d_ret": ts0["d_ret"], "turnover": ts0["turnover"],
                   "wins": int(sum(v > 0 for v in yd)), "years": len(yd), "best": best, "mean_ex_best": float((sum(yd) - best) / (len(yd) - 1))})
    R["amp_table"] = at
    # 10) 단순 추세로 같은 규칙을 쓰면? (수급 확산이 추세 이상의 정보를 주는지)
    pt = S.expanding_pct(F["idist120"])
    pm = S.expanding_pct(F["imom60"])
    trend_s = pd.concat([pt, pm], axis=1).mean(axis=1)
    dtmp = pd.concat([pb.rename("b"), trend_s.rename("t")], axis=1).dropna()
    beta = np.polyfit(dtmp["t"], dtmp["b"], 1)
    resid_pct = S.expanding_pct(dtmp["b"] - np.polyval(beta, dtmp["t"]))
    cands = [("수급 확산 점수(채택)", pb), ("지수 120일선 이격", pt), ("지수 추세(120일선+3달 평균)", trend_s),
             ("수급 확산 + 추세 평균", (pb + trend_s) / 2), ("수급 확산에서 추세 성분을 뺀 것", resid_pct)]
    tc = []
    for name, sc_ in cands:
        s0 = max(start, sc_.first_valid_index())
        w = S.short_weight_band(sc_, 0.75, amp, smooth, lo, hi).loc[s0:]
        t2 = S.timing_stats(w, r.loc[w.index], cost)
        _, p_dd, _ = perm_generic(lambda x: S.short_weight_band(1 - x, 0.75, amp, smooth, lo, hi), 1 - sc_, r, s0, "d_mdd", cost, n=600)
        tc.append({"name": name, "d_sharpe": t2["d_sharpe"], "d_mdd": t2["d_mdd"], "ir": t2["ir"], "p_mdd": p_dd})
    R["trend_cmp"] = tc
    path = os.path.join(BASE, "docs", "data", "signal_research.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean(R), f, ensure_ascii=False, separators=(",", ":"))
    out.write(f"저장 {path} ({os.path.getsize(path):,} bytes)\n")
    out.write(f"base75±25: ΔSharpe {fund['stats']['d_sharpe']:+.3f} ΔMDD {fund['stats']['d_mdd'] * 100:+.1f}%p, p(IR)={fund['p_ir']:.3f} p(ΔMDD)={fund['p_d_mdd']:.3f} p(ΔSharpe)={fund['p_d_sharpe']:.3f}\n")
    out.write("families: " + "; ".join(f"{x['name']} dMDD {x['d_mdd'] * 100:+.1f} p={x['p_mdd']:.2f}" for x in fams) + "\n")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "ic"
    out = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    if stage == "ic":
        stage_ic(out)
    elif stage == "comp":
        stage_comp(out)
    elif stage == "risk":
        stage_risk(out)
    elif stage == "strat":
        stage_strat(out)
    elif stage == "strat2":
        stage_strat2(out)
    elif stage == "strat3":
        stage_strat3(out)
    elif stage == "final0":
        stage_final0(out)
    elif stage == "fund":
        stage_fund(out)
    elif stage == "report":
        stage_report(out)
    out.flush()
