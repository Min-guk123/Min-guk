"""
위험 예측 점수(⑧ 탭) -> docs/data/risk_score.json

재료(후보 지표) 3묶음 + 기준선 - 매일, 12년치 (장기 수정주가 패널로 계산)
  A) 수급 확산(16): ADR, 코스닥을 이긴 종목 비율, 이동평균 위 종목 비율, 신고가-신저가, 동일가중-지수 괴리 (signal_model, ⑤ 탭과 같은 지표)
  B) 섹터 이평선 위 비율(25): 이동평균(20·60·20주·120·200일)별 섹터 평균, 강한 섹터(60%↑) 비중, 약한 섹터(40%↓) 비중, 섹터 간 편차, 20일 변화 (⑥ 탭 정의)
  C) 최근 낙폭 평균과 '심해지는지'(19): 종목별 '가장 최근 하락일 K일치' 평균(④ 탭 정의, K=10·20·60)의 중앙값, 코스닥 자신의 값, 둘의 배수,
     20일 전보다 더 깊어진 종목 비중, 20일 변화, 1년 평균 대비, 최근 20일 중 내린 날 비중
  기준선) 코스닥 지수 자신의 최근 변동성(10·20·60·250일). 변동성은 원래 자기 과거가 가장 잘 맞히므로, 새 지표들이 거기에 '추가로' 보태는 정보를 따로 잰다.

무엇을 맞히나: 앞으로 20거래일 코스닥 변동성(위험). 방향(오를지 내릴지)은 같은 방법으로 시험했지만 표본 외에서 맞히지 못했다(표에 그대로 싣는다).
모형(2단계): 1단계 지수 변동성 4개로 OLS(HAR) -> 2단계 A·B·C 지표(그 날짜까지의 평균·표준편차로 표준화)를 1단계 오차에 릿지.
검증: 매 시점에 '결과가 확정된 날'까지만으로 다시 학습하는 walk-forward(21거래일마다), 규제 강도는 시간 순서 블록 교차검증. 2018-10 이후가 표본 외.
"""
from __future__ import annotations

import json
import math
import os
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import drawdown_stats as DS
import signal_model as SMD

KST = timezone(timedelta(hours=9))
warnings.filterwarnings("ignore", message="Mean of empty slice")

SEC_MAS = (20, 60, 100, 120, 200)
KS = (10, 20, 60)
MIN_SEC = 5            # 섹터 요약에 쓰는 섹터의 최소 종목 수
HI, LO = 0.60, 0.40    # 강한/약한 섹터 기준(이동평균 위 비율)
ZMIN = 250             # 표준화에 필요한 최소 관측일
HAR = ["lrv10", "lrv20", "lrv60", "lrv250"]
H = 20                 # 예측 구간(거래일)
STEP = 21              # 재학습 주기
TRAIN_MIN = 500        # 첫 예측 전 최소 학습 표본
OOS_START = pd.Timestamp("2018-10-01")
GRID = (100, 300, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6)
RISK_MINP = 120        # 위험 점수 백분위에 필요한 최소 예측 이력
EMA_SPAN = 10
LVL_LO, LVL_HI, LVL_HYST = 100 / 3, 200 / 3, 3.0
LOOK, DIR_HYST = 20, 0.12
BIG_DD = -0.07

GROUPS = {"A": "수급 확산", "B": "섹터 이평선 위 비율", "C": "최근 낙폭 평균", "H": "지수 자신의 변동성"}
LEVELS = [{"id": 0, "name": "낮음", "color": "#2f9e44"}, {"id": 1, "name": "보통", "color": "#fab005"}, {"id": 2, "name": "높음", "color": "#e03131"}]
# 화면에 보여줄 현재 입력 값(지표, 표시 형식). pct = 비율(%), spct = 부호 있는 %, x2 = 소수 둘째 자리
INPUT_SHOW = [("lrv20", "pct"), ("idd250", "spct"),
              ("adr20", "x2"), ("beat20", "pct"), ("ama60", "pct"), ("nhl60", "spct"), ("ews20", "spct"),
              ("smean60", "pct"), ("shi60", "pct"), ("slo60", "pct"), ("smean120", "pct"), ("smean60_d20", "spct"),
              ("dd_med20", "pct2"), ("dd_idx20", "pct2"), ("dd_ratio20", "x2"), ("dd_deep20", "pct"), ("dd_rel20", "spct"), ("dn_share20", "pct")]
STATES = [
    {"id": 1, "name": "① 안정 지속", "color": "#2f9e44", "desc": "위험 점수 낮음·줄어드는 중 - 앞으로 한 달 변동성이 작을 것으로 예상"},
    {"id": 2, "name": "② 불안 싹틈", "color": "#fab005", "desc": "위험 점수 낮음·커지는 중 - 아직 조용하지만 예상 변동성이 올라오기 시작"},
    {"id": 3, "name": "③ 진정 중", "color": "#8ce99a", "desc": "위험 점수 보통·줄어드는 중 - 흔들림이 가라앉는 중"},
    {"id": 4, "name": "④ 불안 확대", "color": "#ff922b", "desc": "위험 점수 보통·커지는 중 - 예상 변동성이 커지는 중"},
    {"id": 5, "name": "⑤ 위험 완화 시작", "color": "#ff8787", "desc": "위험 점수 높음·줄어드는 중 - 큰 흔들림 뒤 진정되기 시작(방향은 알 수 없음)"},
    {"id": 6, "name": "⑥ 위험 고조", "color": "#c92a2a", "desc": "위험 점수 높음·커지는 중 - 앞으로 한 달 큰 등락이 예상(오를지 내릴지는 알 수 없음)"},
]


def _c(v, nd=4):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else round(f, nd)


def _lst(x, nd=4):
    return [_c(v, nd) for v in (x.values if hasattr(x, "values") else x)]


# ------------------------------------------------------------------ 지표 만들기
def sector_membership(codes, sector_of: pd.Series):
    """(섹터 이름 목록, 멤버십 행렬 N x S, 요약에 쓰는 섹터 마스크)."""
    s = sector_of.reindex(codes)
    counts = s.value_counts()
    secs = list(counts.index)
    mem = np.zeros((len(codes), len(secs)))
    for j, name in enumerate(secs):
        mem[(s.values == name), j] = 1.0
    elig = np.array([(counts[x] >= MIN_SEC and x != "개별주") for x in secs])
    return secs, mem, elig


def sector_ratio(P: pd.DataFrame, mem: np.ndarray, w: int, min_n: int = 3) -> np.ndarray:
    """섹터별 '자기 w일 이동평균 이상인 종목 비율'(날짜 x 섹터). 계산 가능한 종목이 min_n 미만이면 NaN."""
    ma = P.rolling(w, min_periods=w).mean()
    valid = (ma.notna() & P.notna())
    V = valid.values.astype(np.float64)
    A = ((P >= ma) & valid).values.astype(np.float64)
    cv, ca = V @ mem, A @ mem
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(cv >= min_n, ca / cv, np.nan)


def build_features(P: pd.DataFrame, I: pd.Series, st: pd.DataFrame):
    """(원시 지표 DataFrame, 지표 -> 묶음(A/B/C/D) dict). 날짜 index = P.index. 양수 낙폭 = 크기."""
    codes = [c for c in st["code"] if c in P.columns]
    P = P[codes]
    I = I.reindex(P.index)
    ret = P.pct_change(fill_method=None)
    iret = I.pct_change()
    cols, group = {}, {}

    def put(name, s, g):
        cols[name] = pd.Series(np.asarray(s, dtype=float), index=P.index)
        group[name] = g

    FA = SMD.breadth_features(P, I, None)                                  # A) 수급 확산 16개 (바이오 제외)
    for c in FA.columns:
        put(c, FA[c].reindex(P.index).values, "A")

    secs, mem, elig = sector_membership(codes, st.set_index("code")["sector"])     # B) 섹터 이평선 위 비율
    for w in SEC_MAS:
        R = sector_ratio(P, mem, w)[:, elig]
        with np.errstate(all="ignore"):
            n = np.sum(~np.isnan(R), axis=1)
            mean = np.where(n >= 5, np.nanmean(R, axis=1), np.nan)
            hi = np.where(n >= 5, np.nansum(R >= HI, axis=1) / np.maximum(n, 1), np.nan)
            lo = np.where(n >= 5, np.nansum(R < LO, axis=1) / np.maximum(n, 1), np.nan)
            sd = np.where(n >= 5, np.nanstd(R, axis=1), np.nan)
        put(f"smean{w}", mean, "B")
        put(f"shi{w}", hi, "B")
        put(f"slo{w}", lo, "B")
        put(f"sdisp{w}", sd, "B")
        put(f"smean{w}_d20", pd.Series(mean, index=P.index).diff(20).values, "B")

    for k in KS:                                                           # C) 최근 낙폭 평균과 심해지는지
        AD = DS.rolling_k_down(ret, k)                                     # 종목별 '마지막 k개 하락일' 평균(음수)
        ik = DS.rolling_k_down(iret.to_frame("i"), k)["i"]
        med = -AD.median(axis=1)
        put(f"dd_med{k}", med.values, "C")
        put(f"dd_idx{k}", (-ik).values, "C")
        put(f"dd_ratio{k}", (med / (-ik)).values, "C")
        prev = AD.shift(20)
        both = AD.notna() & prev.notna()
        deeper = ((AD < prev) & both).sum(axis=1) / both.sum(axis=1).replace(0, np.nan)
        put(f"dd_deep{k}", deeper.where(both.sum(axis=1) >= 100).values, "C")
        put(f"dd_chg{k}", (med - med.shift(20)).values, "C")
        put(f"dd_rel{k}", (med / med.rolling(250, min_periods=120).mean() - 1.0).values, "C")
    dn = (ret < 0).where(ret.notna())
    put("dn_share20", dn.rolling(20, min_periods=15).mean().mean(axis=1).values, "C")

    li = np.log(I)                                                         # 기준선: 지수 자신
    for k in (20, 60, 120):
        put(f"imom{k}", (li - li.shift(k)).values, "D")
    put("idist120", (li - np.log(I.rolling(120).mean())).values, "D")
    put("idd250", (li - li.rolling(250, min_periods=120).max()).values, "D")
    put("idd60", (li - li.rolling(60, min_periods=40).max()).values, "D")
    ann = np.sqrt(SMD.ANN)
    for k, mp in ((10, 10), (20, 20), (60, 60), (250, 200)):
        put(f"lrv{k}", np.log(iret.rolling(k, min_periods=mp).std() * ann).values, "D")
    return pd.DataFrame(cols, index=P.index), group


def build_targets(I: pd.Series) -> pd.DataFrame:
    """이후 5·10·20·60거래일 로그수익률, 이후 20일 최대낙폭·변동성(연율), 큰 낙폭 여부."""
    T = SMD.forward_targets(I, (5, 10, 20, 60))
    T["lfvol20"] = np.log(T["fvol20"])
    T["big_dd"] = (T["fmdd20"] <= BIG_DD).astype(float).where(T["fmdd20"].notna())
    return T


def theory_signs(F_cols, group):
    """묶음 점수에 쓰는 이론 부호: 수급 확산·섹터 이평선은 높을수록 낙관(+), 낙폭 평균은 클수록 비관(-). 섹터 간 편차는 방향이 불분명해 뺀다."""
    s = {}
    for c in F_cols:
        g = group[c]
        if g == "A":
            s[c] = 1.0
        elif g == "B":
            s[c] = 0.0 if c.startswith("sdisp") else (-1.0 if c.startswith("slo") else 1.0)
        elif g == "C":
            s[c] = -1.0
    return s


def group_scores(Z: pd.DataFrame, group: dict) -> pd.DataFrame:
    sg = theory_signs(Z.columns, group)
    out = {}
    for g in "ABC":
        cs = {c: v for c, v in sg.items() if group[c] == g and v != 0}
        out[g] = SMD.composite(Z, cs)
    return pd.DataFrame(out)


# ------------------------------------------------------------------ 통계 도구
def rankvec(x):
    r = pd.Series(x).rank().values
    r = r - r.mean()
    return r / (np.sqrt((r ** 2).sum()) or 1.0)


def ic_perm(f: pd.Series, y: pd.Series, h: int):
    """스피어만 IC 와, 신호를 원형으로 밀어(자기상관 유지) 만든 귀무분포 대비 p값. 반환 (ic, p, n)."""
    df = pd.concat([f, y], axis=1).dropna()
    if len(df) < 300:
        return np.nan, np.nan, len(df)
    a, b = rankvec(df.iloc[:, 0].values), rankvec(df.iloc[:, 1].values)
    n = len(a)
    cc = np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(b)), n)
    kmin = max(60, 2 * h)
    null = cc[np.arange(kmin, n - kmin)]
    return float(cc[0]), float((np.abs(null) >= abs(cc[0])).mean()), n


def hac_t(x, lag):
    """평균의 HAC(뉴이-웨스트) t값."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 30:
        return np.nan
    xc = x - x.mean()
    lrv = (xc ** 2).mean()
    for l in range(1, lag + 1):
        lrv += 2 * (1 - l / (lag + 1)) * (xc[l:] * xc[:-l]).mean()
    return float(x.mean() / np.sqrt(max(lrv, 1e-18) / n))


def expanding_mean_pred(y: pd.Series, h: int, first: int) -> pd.Series:
    """t 시점 예측 = (t-h-1 까지 확정된) 목표의 평균. 모형이 이 단순 평균보다 나은지 재는 기준."""
    v = y.values
    out = np.full(len(v), np.nan)
    cs = np.nancumsum(np.nan_to_num(v))
    cn = np.cumsum(~np.isnan(v))
    for t in range(first, len(v)):
        k = t - h - 1
        if k >= 0 and cn[k] > 100:
            out[t] = cs[k] / cn[k]
    return pd.Series(out, index=y.index)


# ------------------------------------------------------------------ 릿지 / 2단계 모형
def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float):
    xm, ym = X.mean(axis=0), y.mean()
    Xc, yc = X - xm, y - ym
    A = Xc.T @ Xc + lam * np.eye(X.shape[1])
    w = np.linalg.solve(A, Xc.T @ yc)
    return w, ym - xm @ w


def blocked_cv_lambda(X: np.ndarray, y: np.ndarray, h: int, grid=GRID, k: int = 5) -> float:
    """시간 순서 블록 교차검증(앞뒤 h일 제외)으로 규제 강도를 고른다."""
    n = len(y)
    edges = np.linspace(0, n, k + 1).astype(int)
    best, best_err = grid[len(grid) // 2], np.inf
    for lam in grid:
        err = 0.0
        for i in range(k):
            a, b = edges[i], edges[i + 1]
            tr = np.r_[0:max(a - h, 0), min(b + h, n):n]
            if len(tr) < 200:
                continue
            w, b0 = ridge_fit(X[tr], y[tr], lam)
            err += float(np.mean((y[a:b] - (X[a:b] @ w + b0)) ** 2)) * (b - a)
        if err < best_err:
            best, best_err = lam, err
    return best


def walk_forward(Xz: pd.DataFrame, y: pd.Series, h: int, first: int, step: int = STEP, grid=GRID) -> pd.Series:
    """릿지 walk-forward. 각 시점 t0 에서 '결과가 확정된 날(j + h < t0)'만으로 학습해 t0 ~ t0+step-1 을 예측."""
    X, yv = Xz.values, y.values
    ok = ~np.isnan(X).any(axis=1) & ~np.isnan(yv)
    T = len(y)
    pred = np.full(T, np.nan)
    for t0 in range(first, T, step):
        tr = np.flatnonzero(ok & (np.arange(T) <= t0 - h - 1))
        if len(tr) < TRAIN_MIN:
            continue
        lam = blocked_cv_lambda(X[tr], yv[tr], h, grid)
        w, b0 = ridge_fit(X[tr], yv[tr], lam)
        te = np.arange(t0, min(t0 + step, T))
        te = te[~np.isnan(X[te]).any(axis=1)]
        pred[te] = X[te] @ w + b0
    return pd.Series(pred, index=Xz.index)


def _ols(X, y):
    A = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(A, y, rcond=None)[0]


def _ols_pred(b, X):
    return b[0] + X @ b[1:]


def two_stage(F: pd.DataFrame, Z: pd.DataFrame, y: pd.Series, feat_cols, first: int, ok: np.ndarray, h: int = H, step: int = STEP):
    """1단계: 지수 변동성(HAR 4개) OLS, 2단계: feat_cols(표준화 지표)를 1단계 오차에 릿지. 매 재학습 시점마다 그때까지 확정된 자료만 쓴다.
    반환 dict: pred(로그 변동성 예측), har(1단계만), smear(오차 보정 계수), fits(재학습별 계수)."""
    T = len(F)
    Xh = F[HAR].values
    Xz = Z[feat_cols].values if feat_cols else None
    yv = y.values
    pred, har, smear = np.full(T, np.nan), np.full(T, np.nan), np.full(T, np.nan)
    fits = []
    for t0 in range(first, T, step):
        tr = np.flatnonzero(ok & (np.arange(T) <= t0 - h - 1))
        if len(tr) < TRAIN_MIN:
            continue
        b = _ols(Xh[tr], yv[tr])
        res = yv[tr] - _ols_pred(b, Xh[tr])
        te = np.arange(t0, min(t0 + step, T))
        te = te[~np.isnan(Xh[te]).any(axis=1)]
        har[te] = _ols_pred(b, Xh[te])
        if Xz is None:
            pred[te] = har[te]
            smear[te] = np.mean(np.exp(res))
            fits.append({"t0": t0, "b": b, "xh_mean": Xh[tr].mean(axis=0), "w": None, "n": len(tr)})
            continue
        lam = blocked_cv_lambda(Xz[tr], res, h)
        wz, b0 = ridge_fit(Xz[tr], res, lam)
        te2 = te[~np.isnan(Xz[te]).any(axis=1)]
        pred[te2] = _ols_pred(b, Xh[te2]) + Xz[te2] @ wz + b0
        smear[te2] = np.mean(np.exp(res - (Xz[tr] @ wz + b0)))
        fits.append({"t0": t0, "b": b, "lam": lam, "w": wz, "b0": b0, "xh_mean": Xh[tr].mean(axis=0), "xz_mean": Xz[tr].mean(axis=0), "n": len(tr)})
    idx = F.index
    return {"pred": pd.Series(pred, index=idx), "har": pd.Series(har, index=idx), "smear": pd.Series(smear, index=idx), "fits": fits}


def compare_to(pred: pd.Series, ref: pd.Series, y: pd.Series, mean_b: pd.Series, first: int, h: int = H) -> dict:
    """표본 외 R²(과거 평균 대비), IC, 그리고 ref(보통 지수 변동성만 쓴 모형) 대비 MSE 개선·Clark-West t·Diebold-Mariano t·연도 승수."""
    df = pd.concat([pred.rename("p"), y.rename("y"), mean_b.rename("m"), ref.rename("r")], axis=1).iloc[first:].dropna()
    e_m, e_p, e_r = (df["y"] - df["m"]).values, (df["y"] - df["p"]).values, (df["y"] - df["r"]).values
    ic, pv, n = ic_perm(df["p"], df["y"], h)
    out = {"r2": 1 - (e_p ** 2).sum() / (e_m ** 2).sum(), "ic": ic, "p": pv, "n": int(len(df)),
           "gain": 1 - (e_p ** 2).sum() / (e_r ** 2).sum(),
           "cw_t": hac_t(e_r ** 2 - (e_p ** 2 - (df["r"].values - df["p"].values) ** 2), h),
           "dm_t": hac_t(e_r ** 2 - e_p ** 2, h)}
    wins = tot = 0
    for _, g in df.groupby(df.index.year):
        if len(g) < 100:
            continue
        tot += 1
        wins += int(((g["y"] - g["p"]) ** 2).sum() < ((g["y"] - g["r"]) ** 2).sum())
    out["wins"], out["years"] = wins, tot
    return out


# ------------------------------------------------------------------ 상태(국면) 판정
def assign_states(score: pd.Series, lvs: pd.Series):
    """위험 점수(0~100) -> 수준(낮음/보통/높음, 33.3·66.7 기준 ±3점 이력 현상) x 방향(예상 변동성이 20일 전보다 커지는 중/줄어드는 중) 6상태.
    방향은 점수(백분위)가 아니라 평활한 로그 예상 변동성(lvs)의 20일 변화로 본다: 백분위는 위험이 큰 장세에서 100 근처에 붙어 변화가 안 보이기 때문.
    로그 변화가 ±0.12(약 ±13%) 안이면 직전 방향 유지. 반환 (상태 id 0~6 배열(0=없음), 수준 0/1/2, 방향 ±1)."""
    v = score.values
    lv = lvs.values
    T = len(v)
    state = np.zeros(T, dtype=int)
    lvl_a = np.full(T, -1, dtype=int)
    dir_a = np.zeros(T, dtype=int)
    level = direction = None
    for t in range(T):
        x = v[t]
        if np.isnan(x):
            continue
        if level is None:
            level = 0 if x < LVL_LO else 2 if x > LVL_HI else 1
        else:
            if level == 0 and x > LVL_LO + LVL_HYST:
                level = 1
            if level == 1 and x > LVL_HI + LVL_HYST:
                level = 2
            elif level == 1 and x < LVL_LO - LVL_HYST:
                level = 0
            elif level == 2 and x < LVL_HI - LVL_HYST:
                level = 1
                if x < LVL_LO - LVL_HYST:
                    level = 0
        if t >= LOOK and not np.isnan(lv[t - LOOK]) and not np.isnan(lv[t]):
            d = lv[t] - lv[t - LOOK]
            if direction is None:
                direction = 1 if d > 0 else -1
            elif d > DIR_HYST:
                direction = 1
            elif d < -DIR_HYST:
                direction = -1
        if direction is None:
            continue
        lvl_a[t], dir_a[t] = level, direction
        # 낮음·줄어듦 ①, 낮음·커짐 ②, 보통·줄어듦 ③, 보통·커짐 ④, 높음·줄어듦 ⑤, 높음·커짐 ⑥
        state[t] = 2 * level + (2 if direction > 0 else 1)
    return state, lvl_a, dir_a


# ------------------------------------------------------------------ 표 만들기
LABEL_A = SMD.FEATURE_LABEL


def label(name: str) -> str:
    if name in LABEL_A:
        return LABEL_A[name]
    mawin = lambda w: "20주(100일)" if w == "100" else f"{w}일"
    for pre, fmt in (("smean", "섹터 평균 {m}선 위 비율"), ("shi", "{m}선 위 비율 60%↑ 섹터 비중"), ("slo", "{m}선 위 비율 40%↓ 섹터 비중"), ("sdisp", "{m}선 위 비율 섹터 간 편차")):
        if name.startswith(pre):
            body = name[len(pre):]
            tail = ""
            if body.endswith("_d20"):
                body, tail = body[:-4], " (20일 변화)"
            return fmt.format(m=mawin(body)) + tail
    for pre, fmt in (("dd_med", "종목 중앙값 최근 하락 {k}일 평균 낙폭"), ("dd_idx", "코스닥 최근 하락 {k}일 평균 낙폭"), ("dd_ratio", "종목÷코스닥 낙폭 배수({k}일치)"),
                     ("dd_deep", "20일 전보다 낙폭이 깊어진 종목 비중({k}일치)"), ("dd_chg", "종목 중앙값 낙폭 20일 변화({k}일치)"), ("dd_rel", "종목 중앙값 낙폭, 1년 평균 대비({k}일치)")):
        if name.startswith(pre):
            return fmt.format(k=name[len(pre):])
    if name == "dn_share20":
        return "최근 20일 중 내린 날 비중(종목 평균)"
    return {"lrv10": "코스닥 변동성 10일", "lrv20": "코스닥 변동성 20일", "lrv60": "코스닥 변동성 60일", "lrv250": "코스닥 변동성 250일",
            "idd250": "코스닥 250일 고점 대비", "idd60": "코스닥 60일 고점 대비"}.get(name, name)


def _resid(s: pd.Series, X: pd.DataFrame) -> pd.Series:
    df = pd.concat([s.rename("s"), X], axis=1).dropna()
    A = np.column_stack([np.ones(len(df)), df[X.columns].values])
    b = np.linalg.lstsq(A, df["s"].values, rcond=None)[0]
    return pd.Series(df["s"].values - A @ b, index=df.index)


def indicator_table(Z: pd.DataFrame, F: pd.DataFrame, group: dict, y: pd.Series, top: int = 6) -> list:
    """지표별 이후 20일 변동성과의 IC(전체)와 '지수 자신의 변동성을 걷어낸 뒤의' 부분 IC. 묶음별 상위 top 개."""
    ctrl = Z[HAR]
    ry = _resid(y, ctrl)
    rows = []
    for c in [c for c in F.columns if group[c] in "ABC"]:
        ic, p, n = ic_perm(Z[c], y, H)
        ic2, p2, n2 = ic_perm(_resid(Z[c], ctrl), ry, H)
        rows.append({"name": c, "label": label(c), "group": group[c], "ic": _c(ic, 3), "p": _c(p, 3), "pic": _c(ic2, 3), "pp": _c(p2, 3)})
    out = []
    for g in "ABC":
        sub = sorted([r for r in rows if r["group"] == g], key=lambda r: -abs(r["pic"] or 0))
        out += sub[:top]
    return out


def count_runs(arr: np.ndarray, ids) -> dict:
    """arr 에서 id 값이 '연속해서 나타난 덩어리' 수 = 서로 떨어진 에피소드 수(겹치는 일별 표본이 실제로는 몇 번의 사례인지)."""
    out = {}
    for i in ids:
        m = (arr == i)
        out[i] = (int(m[0]) + int((m[1:] & ~m[:-1]).sum())) if len(m) else 0
    return out


def level_table(df: pd.DataFrame, key: str, ids: list, runs: dict | None = None) -> list:
    """key 열 값(수준/상태)별 이후 성과. df 열: fwd5 fwd20 fwd60 fmdd20 fvol20 pv(예상 변동성, 소수)."""
    out = []
    for i in ids:
        g = df[df[key] == i]
        if len(g) == 0:
            out.append({"id": i, "n": 0, "runs": 0})
            continue
        out.append({"id": i, "n": int(len(g)), "runs": (runs or {}).get(i), "pv": _c(g["pv"].mean() * 100, 1), "rv": _c(g["fvol20"].mean() * 100, 1),
                    "dd": _c(g["fmdd20"].mean(), 4), "big": _c((g["fmdd20"] <= BIG_DD).mean(), 3),
                    "r5": _c(g["fwd5"].mean(), 4), "r20": _c(g["fwd20"].mean(), 4), "up20": _c((g["fwd20"] > 0).mean(), 3),
                    "r60": _c(g["fwd60"].mean(), 4), "n60": int(g["fwd60"].notna().sum())})
    return out


def hedge_table(pb_new: pd.Series, pb_old: pd.Series, r: pd.Series) -> list:
    """기준 숏 비중(밸류에이션 룰) 50·75·100% 에 ±12.5%p 가감했을 때, '같은 평균 숏을 계속 유지'한 것과의 차이. 새 점수는 위험이 높을수록 숏을 늘린다."""
    rows = []
    for nm, pb in (("⑧ 위험 점수(위험 높으면 숏 가산)", pb_new), ("⑤ 낙관 점수(수급 확산)", pb_old)):
        for base in (0.5, 0.75, 1.0):
            ww = SMD.short_weight_band(pb, base, 0.125, 10, 0.5, 1.0)
            t = SMD.timing_stats(ww, r)
            if not t:
                continue
            rows.append({"model": nm, "base": base, "w_mean": _c(t["w_mean"], 3), "d_sharpe": _c(t["d_sharpe"], 3), "d_mdd": _c(t["d_mdd"], 4),
                         "ir": _c(t["ir"], 3), "mdd_static": _c(t["static"]["mdd"], 4), "mdd_sig": _c(t["net"]["mdd"], 4)})
    return rows


def hedge_yearly(pb: pd.Series, r: pd.Series, base: float = 0.75, amp: float = 0.125) -> pd.Series:
    """연도별 (점수로 가감한 순수익 - 같은 평균 숏을 고정한 순수익) 합. 한두 해에 기댄 결과인지 보려는 표."""
    ww = SMD.short_weight_band(pb, base, amp, 10, 0.5, 1.0)
    df = pd.concat([ww.rename("w"), r.rename("r")], axis=1).dropna()
    wl = df["w"].shift(1)
    d = df.dropna()
    net = (1.0 - wl.loc[d.index]) * d["r"] - 0.001 * df["w"].diff().abs().loc[d.index].fillna(0.0)
    static = (1.0 - wl.loc[d.index].mean()) * d["r"]
    diff = net - static
    return diff.groupby(diff.index.year).sum()


# ------------------------------------------------------------------ 본 계산
def compute(P: pd.DataFrame, I: pd.Series, st: pd.DataFrame, mp: dict | None = None) -> dict:
    """P: 장기 수정주가, I: 코스닥 지수, st: res['stock'], mp: ⑦ 탭 결과(market_phase.compute 의 payload, 비교 행용·없어도 됨)."""
    P, I = SMD.align_panel(P, I)
    F, group = build_features(P, I, st)
    T = build_targets(I)
    Z = SMD.expanding_z(F, ZMIN)
    cols = {g: [c for c in F.columns if group[c] == g] for g in "ABCD"}
    abc = cols["A"] + cols["B"] + cols["C"]
    y = T["lfvol20"]
    dates = F.index
    nT = len(dates)

    valid = Z[abc].notna().all(axis=1) & F[HAR].notna().all(axis=1)
    pos0 = int(np.argmax(valid.values))
    first = max(int(np.searchsorted(dates, OOS_START)), pos0 + TRAIN_MIN + H + 1)
    ok = (valid & y.notna()).values

    full = two_stage(F, Z, y, abc, first, ok)
    har = two_stage(F, Z, y, [], first, ok)
    variants = {"HAR": har, "HAR+A": two_stage(F, Z, y, cols["A"], first, ok), "HAR+B": two_stage(F, Z, y, cols["B"], first, ok),
                "HAR+C": two_stage(F, Z, y, cols["C"], first, ok), "HAR+A+B+C": full}
    mean_b = expanding_mean_pred(y, H, first)
    pred = full["pred"]
    smear = full["smear"]

    ablation = []
    names = {"HAR": "지수 변동성만(기준선)", "HAR+A": "+ 수급 확산", "HAR+B": "+ 섹터 이평선 위 비율", "HAR+C": "+ 최근 낙폭 평균(심해짐)", "HAR+A+B+C": "+ 세 묶음 모두(이 탭의 점수)"}
    for k, v in variants.items():
        rec = compare_to(v["pred"], har["pred"], y, mean_b, first)
        ablation.append({"key": k, "name": names[k], "r2": _c(rec["r2"], 3), "ic": _c(rec["ic"], 3), "p": _c(rec["p"], 3), "gain": _c(rec["gain"], 3) if k != "HAR" else None,
                         "cw_t": _c(rec["cw_t"], 2) if k != "HAR" else None, "dm_t": _c(rec["dm_t"], 2) if k != "HAR" else None,
                         "wins": rec["wins"] if k != "HAR" else None, "years": rec["years"]})

    # 연도별
    dfy = pd.concat([pred.rename("p"), har["pred"].rename("r"), y.rename("y"), mean_b.rename("m")], axis=1).iloc[first:].dropna()
    years = []
    for yr, g in dfy.groupby(dfy.index.year):
        e_m, e_p, e_r = g["y"] - g["m"], g["y"] - g["p"], g["y"] - g["r"]
        years.append({"year": int(yr), "n": int(len(g)), "r2": _c(1 - (e_p ** 2).sum() / (e_m ** 2).sum(), 3), "gain": _c(1 - (e_p ** 2).sum() / (e_r ** 2).sum(), 3),
                      "pv": _c(np.exp(g["p"]).mean() * 100, 1), "rv": _c(np.exp(g["y"]).mean() * 100, 1)})

    # 점수·상태: 예상 로그 변동성을 10일 EMA 로 평활 -> 그 시점까지 이력 중 백분위(0~100) = 위험 점수
    lvs = pred.dropna().ewm(span=EMA_SPAN, adjust=False).mean().reindex(dates)
    score = SMD.expanding_pct(lvs.dropna(), RISK_MINP).reindex(dates) * 100.0
    risk_pct = score
    state, lvl, dirn = assign_states(score, lvs)
    s_first = int(np.argmax(~np.isnan(score.values)))
    sl = slice(s_first, None)
    idx = dates[sl]

    G = group_scores(Z, group)
    Gp = pd.DataFrame({g: SMD.expanding_pct(G[g].dropna(), ZMIN).reindex(dates) * 100.0 for g in "ABC"})

    # 표본 외 이후 성과표
    pv_mean = np.exp(pred) * smear                                      # 평균 기준으로 보정한 예상 변동성(소수)
    df = pd.concat([pv_mean.rename("pv"), T[["fwd5", "fwd20", "fwd60", "fmdd20", "fvol20"]], pd.Series(lvl, index=dates, name="lvl"), pd.Series(state, index=dates, name="state")], axis=1)
    df = df.iloc[s_first:].dropna(subset=["pv", "fvol20", "fwd20"])
    levels = level_table(df, "lvl", [0, 1, 2], count_runs(lvl[s_first:], [0, 1, 2]))
    states = level_table(df, "state", [1, 2, 3, 4, 5, 6], count_runs(state[s_first:], [1, 2, 3, 4, 5, 6]))
    allrow = level_table(df.assign(all=1), "all", [1])[0]
    mid = df.index[len(df) // 2]                                          # 표본 외 구간을 날짜 기준 반으로 나눠 관계가 안정적인지 본다
    lvl_s = pd.Series(lvl, index=dates)
    levels_split = {"split": mid.strftime("%Y-%m-%d"),
                    "first": level_table(df[df.index < mid], "lvl", [0, 1, 2], count_runs(lvl_s[(lvl_s.index >= dates[s_first]) & (lvl_s.index < mid)].values, [0, 1, 2])),
                    "second": level_table(df[df.index >= mid], "lvl", [0, 1, 2], count_runs(lvl_s[lvl_s.index >= mid].values, [0, 1, 2]))}

    # 방향 시험표
    dir_rows = []
    sc = {"⑧ 안정도(예상 변동성의 반대)": -pred, "수급 확산 묶음(⑤ 낙관 점수)": G["A"], "섹터 이평선 위 비율 묶음": G["B"], "낙폭 평균 묶음(깊을수록 비관)": G["C"],
          "세 묶음 단순 평균": G[["A", "B", "C"]].mean(axis=1)}
    oos_start = dates[first]
    hs = (5, 20, 60)
    for nm, s in sc.items():
        rec = {"name": nm}
        for h_ in hs:
            ic, p, n = ic_perm(s[s.index >= oos_start], T[f"fwd{h_}"], h_)
            rec[f"ic{h_}"], rec[f"p{h_}"] = _c(ic, 3), _c(p, 3)
        ic, p, n = ic_perm(s[s.index >= oos_start], T["lfvol20"], H)
        rec["icv"], rec["pv_"] = _c(ic, 3), _c(p, 3)
        dir_rows.append(rec)
    rec = {"name": "과거로 학습한 방향 모형(릿지, 세 묶음 전부)"}
    for h_ in hs:
        pr = walk_forward(Z[abc], T[f"fwd{h_}"], h_, first)
        ic, p, n = ic_perm(pr, T[f"fwd{h_}"], h_)
        rec[f"ic{h_}"], rec[f"p{h_}"] = _c(ic, 3), _c(p, 3)
    rec["icv"], rec["pv_"] = None, None
    dir_rows.append(rec)
    if mp and mp.get("score"):                       # 참고: ⑦ 시장 국면 점수(높을수록 상승 국면 쪽)를 같은 표본 외 구간으로 시험
        s7 = pd.Series(mp["score"], index=pd.to_datetime(mp["dates"]), dtype=float)
        s7 = s7[s7.index >= oos_start]
        rec = {"name": "참고: ⑦ 시장 국면 점수(기존 점수)"}
        for h_ in hs:
            ic, p, n = ic_perm(s7, T[f"fwd{h_}"], h_)
            rec[f"ic{h_}"], rec[f"p{h_}"] = _c(ic, 3), _c(p, 3)
        ic, p, n = ic_perm(s7, T["lfvol20"], H)
        rec["icv"], rec["pv_"] = _c(ic, 3), _c(p, 3)
        dir_rows.append(rec)

    # 숏 비중 가감 시험
    rr = I.pct_change()
    st_ = dates[first]
    pb_new = 1.0 - score.dropna() / 100.0
    pb_old = SMD.expanding_pct(G["A"][G["A"].index >= st_].dropna(), RISK_MINP)
    pb_old = pb_old[pb_old.index >= pb_new.index[0]]
    hedge = hedge_table(pb_new, pb_old, rr)
    hy8, hy5 = hedge_yearly(pb_new, rr), hedge_yearly(pb_old, rr)
    hedge_years = [{"year": int(y), "r8": _c(hy8.get(y), 4), "r5": _c(hy5.get(y), 4)} for y in hy8.index]

    indicators = indicator_table(Z, F, group, y)

    # 현재 값
    last = nT - 1
    fit = full["fits"][-1]
    xh, xz = F[HAR].values[last], Z[abc].values[last]
    eff = {"H": float(fit["b"][1:] @ (xh - fit["xh_mean"]))}
    wz = fit["w"] * 1.0
    zz = np.nan_to_num(xz - fit["xz_mean"])
    for g in "ABC":
        m = np.array([group[c] == g for c in abc])
        eff[g] = float(np.sum(wz[m] * zz[m]))
    base_log = float(_ols_pred(fit["b"], fit["xh_mean"]))
    smr = float(smear.iloc[last]) if not np.isnan(smear.iloc[last]) else 1.0
    pv_now = float(np.exp(pred.iloc[last]) * smr) * 100
    base_vol = float(np.exp(base_log) * smr) * 100
    har_now = float(np.exp(har["pred"].iloc[last]) * float(har["smear"].iloc[last])) * 100
    inputs = []
    for name, fmt in INPUT_SHOW:
        v = F[name].iloc[last]
        if name == "lrv20":
            v = math.exp(v)
        inputs.append({"name": name, "label": label(name), "group": group[name], "fmt": fmt, "v": _c(v, 4), "z": _c(Z[name].iloc[last], 2) if name in Z else None})
    nowstate = int(state[last])
    ch20 = _c((math.exp(lvs.iloc[last] - lvs.iloc[last - LOOK]) - 1) * 100, 0) if last >= LOOK else None      # 평활 예상 변동성의 20일 변화(%)
    contrib = [{"key": "H", "name": GROUPS["H"], "mult": _c(math.exp(eff["H"]), 3)}] + [{"key": g, "name": GROUPS[g], "mult": _c(math.exp(eff[g]), 3)} for g in "ABC"]
    strip_n = 30
    now = {
        "date": dates[last].strftime("%Y-%m-%d"), "state": nowstate, "level": int(lvl[last]), "dir": int(dirn[last]), "score": _c(score.iloc[last], 1), "change20": ch20,
        "risk_pct": _c(risk_pct.iloc[last], 1), "vol": _c(pv_now, 1), "vol_daily": _c(pv_now / math.sqrt(SMD.ANN), 2), "vol_har": _c(har_now, 1), "vol_base": _c(base_vol, 1),
        "vol_now20": _c(math.exp(F["lrv20"].iloc[last]) * 100, 1), "kosdaq": _c(I.iloc[last], 2), "kosdaq_1m": _c(I.iloc[last] / I.iloc[last - 20] - 1, 4),
        "kosdaq_dd250": _c(F["idd250"].iloc[last], 4),
        "gA": _c(Gp["A"].iloc[last], 0), "gB": _c(Gp["B"].iloc[last], 0), "gC": _c(Gp["C"].iloc[last], 0),
        "gA_20": _c(Gp["A"].iloc[last - 20], 0), "gB_20": _c(Gp["B"].iloc[last - 20], 0), "gC_20": _c(Gp["C"].iloc[last - 20], 0),
        "contrib": contrib, "inputs": inputs, "n_stocks": int(P.notna().iloc[last].sum()),
        "strip": [int(x) for x in lvl[last - strip_n + 1:last + 1]], "strip_dates": [dates[t].strftime("%Y-%m-%d") for t in range(last - strip_n + 1, last + 1)],
        "lam": _c(fit.get("lam"), 0), "n_train": int(fit["n"]), "fit_date": dates[fit["t0"]].strftime("%Y-%m-%d"),
    }
    changed = None
    for t in range(last, s_first, -1):
        if lvl[t] != lvl[t - 1]:
            changed = {"date": dates[t].strftime("%Y-%m-%d"), "from": int(lvl[t - 1]), "to": int(lvl[t]), "days": int(last - t + 1)}
            break
    recent = [{"date": dates[t].strftime("%Y-%m-%d"), "vol": _c(pv_mean.iloc[t] * 100, 1), "score": _c(score.iloc[t], 1), "level": int(lvl[t]),
               "kosdaq": _c(I.iloc[t], 2), "ret": _c(I.iloc[t] / I.iloc[t - 1] - 1, 4)} for t in range(last, last - 12, -1)]

    ser = lambda x, nd=3: _lst((x if isinstance(x, pd.Series) else pd.Series(x, index=dates)).iloc[sl], nd)
    rv = (T["fvol20"] * 100)
    return {
        "asof": dates[-1].strftime("%Y-%m-%d"), "generated": datetime.now(KST).strftime("%Y-%m-%d %H:%M KST"),
        "start": idx[0].strftime("%Y-%m-%d"), "oos_start": dates[first].strftime("%Y-%m-%d"), "data_start": dates[0].strftime("%Y-%m-%d"),
        "params": {"horizon": H, "step": STEP, "ema": EMA_SPAN, "look": LOOK, "lvl": [round(LVL_LO, 1), round(LVL_HI, 1)], "hyst": LVL_HYST, "dir_hyst": round(math.exp(DIR_HYST) - 1, 3), "min_train": TRAIN_MIN,
                   "n_features": {g: len(cols[g]) for g in "ABCD"}, "oos_days": int(nT - first), "refits": len(full["fits"])},
        "states": STATES, "groups": GROUPS, "level_defs": LEVELS,
        "dates": [d.strftime("%Y-%m-%d") for d in idx],
        "kosdaq": [_c(v, 2) for v in I.reindex(idx).values],
        "vol": ser(pv_mean * 100, 1), "vol_har": ser(np.exp(har["pred"]) * har["smear"] * 100, 1), "vol_real": ser(rv, 1),
        "score": ser(score, 1), "level": [int(x) for x in lvl[sl]], "state": [int(x) for x in state[sl]],
        "gA": ser(Gp["A"], 0), "gB": ser(Gp["B"], 0), "gC": ser(Gp["C"], 0),
        "now": now, "changed": changed, "recent": recent,
        "ablation": ablation, "years": years, "direction": dir_rows, "levels": levels, "levels_split": levels_split, "state_stats": states, "all": allrow,
        "hedge": hedge, "hedge_years": hedge_years, "indicators": indicators,
    }


def write(payload: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))


def run(P, I, st, out_path, mp=None):
    payload = compute(P, I, st, mp=mp)
    write(payload, out_path)
    return payload
