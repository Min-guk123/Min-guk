"""
수급 확산(breadth) 시그널 라이브러리.

종목 일봉 패널(날짜 x 종목)과 코스닥 지수로 수급 확산 지표를 만들고, 그것을 하나의 종합 점수로 평균한 뒤
'숏 비중'으로 바꾸는 함수들. 연구(research_signal.py)와 매일 갱신(build.py)이 같이 쓴다.

원칙
  - 모든 표준화·순위는 그 날짜까지의 자료만 쓴다(expanding). 미래 정보를 쓰지 않는다.
  - 오늘 신호로 정한 숏 비중은 다음 거래일 수익률부터 적용한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252
BIO_SECTORS = ("제약바이오", "주사제")

# 지표 이름 → 한글 설명 (대시보드 표에 그대로 쓴다)
FEATURE_LABEL = {
    "adr5": "ADR 5일", "adr10": "ADR 10일", "adr20": "ADR 20일", "adr40": "ADR 40일", "adr60": "ADR 60일",
    "beat5": "이긴 종목 비율 1주", "beat10": "이긴 종목 비율 2주", "beat20": "이긴 종목 비율 1달", "beat60": "이긴 종목 비율 3달",
    "ama20": "20일선 위 종목 비율", "ama60": "60일선 위 종목 비율", "ama120": "120일선 위 종목 비율",
    "nhl60": "신고가-신저가(60일)", "nhl120": "신고가-신저가(120일)",
    "ews20": "동일가중-지수 괴리 1달", "ews60": "동일가중-지수 괴리 3달",
    "bio20": "바이오 ADR 20일", "bio60": "바이오 ADR 60일",
    "rv20": "지수 변동성 20일", "rv60": "지수 변동성 60일",
    "imom20": "지수 모멘텀 1달", "imom60": "지수 모멘텀 3달", "imom120": "지수 모멘텀 6달", "idist120": "지수 120일선 이격",
}
BREADTH_FEATURES = ["adr5", "adr10", "adr20", "adr40", "adr60", "beat5", "beat10", "beat20", "beat60",
                    "ama20", "ama60", "ama120", "nhl60", "nhl120", "ews20", "ews60", "bio20", "bio60"]
INDEX_FEATURES = ["imom20", "imom60", "imom120", "idist120"]
RISK_FEATURES = ["rv20", "rv60"]


def align_panel(prices: pd.DataFrame, index: pd.Series):
    """지수와 같은 거래일만 남긴다(일부 종목의 오래된 거래일 행 제거)."""
    index = index.dropna().sort_index()
    index = index[~index.index.duplicated()]
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated()]
    common = prices.index.intersection(index.index)
    return prices.loc[common], index.loc[common]


def _adr(ret: pd.DataFrame, ks=(5, 10, 20, 40, 60), prefix="adr") -> dict:
    adv = (ret > 0).sum(axis=1).astype(float)
    dec = (ret < 0).sum(axis=1).astype(float)
    return {f"{prefix}{k}": np.log((adv.rolling(k).sum() + 1.0) / (dec.rolling(k).sum() + 1.0)) for k in ks}


def breadth_features(prices: pd.DataFrame, index: pd.Series, bio_codes=None) -> pd.DataFrame:
    """날짜별 수급 확산 지표. 값이 클수록 '넓게 오르는 중'."""
    P, I = align_panel(prices, index)
    ret = P.pct_change(fill_method=None)
    F = {}
    F.update(_adr(ret))                                   # ADR: 상승종목 누적 / 하락종목 누적 (로그)
    for k in (5, 10, 20, 60):                             # 코스닥을 이긴 종목 비율
        r = P / P.shift(k) - 1.0
        ir = I / I.shift(k) - 1.0
        v = r.notna()
        F[f"beat{k}"] = (r.sub(ir, axis=0).gt(0) & v).sum(axis=1) / v.sum(axis=1).replace(0, np.nan)
    for k in (20, 60, 120):                               # 이동평균 위 종목 비율
        ma = P.rolling(k, min_periods=int(k * 0.8)).mean()
        v = ma.notna() & P.notna()
        F[f"ama{k}"] = ((P > ma) & v).sum(axis=1) / v.sum(axis=1).replace(0, np.nan)
    for k in (60, 120):                                   # 신고가 종목 비율 - 신저가 종목 비율
        mp = int(k * 0.8)
        hi, lo = P.rolling(k, min_periods=mp).max(), P.rolling(k, min_periods=mp).min()
        v = hi.notna() & P.notna()
        F[f"nhl{k}"] = (((P >= hi) & v).sum(axis=1) - ((P <= lo) & v).sum(axis=1)) / v.sum(axis=1).replace(0, np.nan)
    ew = np.log1p(ret.mean(axis=1).fillna(0.0)).cumsum()  # 동일가중 지수 대비 코스닥(시총가중) 괴리
    li = np.log(I)
    for k in (20, 60):
        F[f"ews{k}"] = (ew - ew.shift(k)) - (li - li.shift(k))
    if bio_codes is not None:
        cols = [c for c in P.columns if c in set(bio_codes)]
        if len(cols) >= 30:
            F.update(_adr(ret[cols], ks=(20, 60), prefix="bio"))
    return pd.DataFrame(F, index=P.index)


def index_features(index: pd.Series) -> pd.DataFrame:
    """지수 자체 추세 지표(수급 확산이 이것보다 나은지 비교하는 기준선)."""
    I = index.dropna().sort_index()
    li = np.log(I)
    F = {f"imom{k}": li - li.shift(k) for k in (20, 60, 120)}
    F["idist120"] = li - np.log(I.rolling(120).mean())
    dr = I.pct_change()
    F["rv20"] = dr.rolling(20).std() * np.sqrt(ANN)       # 최근 변동성(위험 기준선)
    F["rv60"] = dr.rolling(60).std() * np.sqrt(ANN)
    return pd.DataFrame(F, index=I.index)


def forward_targets(index: pd.Series, hs=(5, 10, 20, 60)) -> pd.DataFrame:
    """t 시점 종가 기준 이후 h거래일 로그수익률과, 20일 동안의 최대낙폭."""
    I = index.dropna().sort_index()
    li = np.log(I)
    T = {f"fwd{h}": li.shift(-h) - li for h in hs}
    arr = I.values
    n = len(arr)
    mdd = np.full(n, np.nan)
    for t in range(n - 20):
        seg = arr[t + 1:t + 21] / arr[t] - 1.0
        mdd[t] = min(seg.min(), 0.0)
    T["fmdd20"] = pd.Series(mdd, index=I.index)
    T["fvol20"] = I.pct_change().rolling(20).std().shift(-20) * np.sqrt(ANN)   # 이후 20일 변동성
    return pd.DataFrame(T, index=I.index)


def expanding_z(F: pd.DataFrame, min_periods: int = 250, clip: float = 3.0) -> pd.DataFrame:
    """그 날짜까지의 평균·표준편차로만 표준화(미래 정보 없음)."""
    mu = F.expanding(min_periods).mean()
    sd = F.expanding(min_periods).std()
    return ((F - mu) / sd.replace(0, np.nan)).clip(-clip, clip)


def expanding_pct(s: pd.Series, min_periods: int = 250) -> pd.Series:
    """그 날짜 값이 지금까지 값들 중 몇 %에 해당하는지(0~1). 미래 정보 없음."""
    v = s.values.astype(float)
    out = np.full(len(v), np.nan)
    for t in range(len(v)):
        if np.isnan(v[t]):
            continue
        h = v[:t + 1]
        h = h[~np.isnan(h)]
        if len(h) >= min_periods:
            out[t] = ((h[:-1] < v[t]).sum() + 0.5 * (h[:-1] == v[t]).sum()) / max(len(h) - 1, 1)
    return pd.Series(out, index=s.index)


def composite(Z: pd.DataFrame, signs: dict) -> pd.Series:
    """부호를 맞춘 표준화 점수의 단순평균 = '평균화'. 클수록 낙관(오를 가능성)."""
    cols = [c for c, s in signs.items() if s != 0 and c in Z.columns]
    if not cols:
        return pd.Series(np.nan, index=Z.index)
    sg = pd.Series({c: signs[c] for c in cols})
    return (Z[cols] * sg).mean(axis=1, skipna=True).where(Z[cols].notna().sum(axis=1) >= max(1, len(cols) // 2))


def short_weight(pbull: pd.Series, w_lo: float, w_hi: float, smooth: int = 1) -> pd.Series:
    """낙관 점수 백분위(0~1) → 숏 비중. 낙관이면 w_lo, 비관이면 w_hi, 사이는 직선. 지수이동평균으로 완만하게."""
    w = w_hi - (w_hi - w_lo) * pbull
    if smooth and smooth > 1:
        w = w.ewm(span=smooth, adjust=False, min_periods=1).mean().where(w.notna())
    return w


def short_weight_band(pbull: pd.Series, base: float, amp: float, smooth: int = 10,
                      lo: float = 0.5, hi: float = 1.0) -> pd.Series:
    """기준 숏 비중(base, 밸류에이션 룰이 정한 값)에 수급 확산 점수로 ±amp 가감하고 [lo, hi] 안으로 자른다.
    pbull = 낙관 점수 백분위(0~1). 점수가 낮을수록(위험) 숏을 늘리고, 높을수록 줄인다."""
    w = (base + amp * (1.0 - 2.0 * pbull)).clip(lo, hi)
    if smooth and smooth > 1:
        w = w.ewm(span=smooth, adjust=False, min_periods=1).mean().where(pbull.notna())
    return w


def perf(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 20:
        return {}
    eq = (1 + r).cumprod()
    mu, sd = r.mean() * ANN, r.std() * np.sqrt(ANN)
    mdd = float((eq / eq.cummax() - 1).min())
    cagr = float(eq.iloc[-1] ** (ANN / len(r)) - 1)
    return {"ret": float(mu), "cagr": cagr, "vol": float(sd), "sharpe": float(mu / sd) if sd > 0 else np.nan,
            "mdd": mdd, "calmar": float(cagr / abs(mdd)) if mdd < 0 else np.nan}


def hedged_returns(w: pd.Series, r: pd.Series, cost: float = 0.001) -> pd.Series:
    """롱 북이 코스닥 지수와 같이 움직인다고 보고, 숏 비중 w(전일 신호)로 헤지한 순수익률.
    cost = 숏 비중 100%를 바꿀 때 드는 비용(왕복 비율, 0.001 = 10bp)."""
    w = w.reindex(r.index)
    return (1.0 - w.shift(1)) * r - cost * w.diff().abs().fillna(0.0)


def timing_stats(w: pd.Series, r: pd.Series, cost: float = 0.001) -> dict:
    """신호로 숏 비중을 조절한 결과를 '같은 평균 숏 비중을 계속 유지한 경우'와 비교한다.
    (평균 노출이 같으니 지수가 올랐다/내렸다 자체의 영향은 빠지고, 타이밍 효과만 남는다.)"""
    df = pd.concat([w.rename("w"), r.rename("r")], axis=1).dropna()
    if len(df) < 60:
        return {}
    wl = df["w"].shift(1)
    d = df.dropna()
    net = (1.0 - wl.loc[d.index]) * d["r"] - cost * df["w"].diff().abs().loc[d.index].fillna(0.0)
    wbar = float(wl.loc[d.index].mean())
    static = (1.0 - wbar) * d["r"]
    diff = net - static
    sd = diff.std()
    out = {"w_mean": wbar, "ir": float(diff.mean() / sd * np.sqrt(ANN)) if sd > 0 else np.nan,
           "d_ret": float(diff.mean() * ANN), "turnover": float(df["w"].diff().abs().mean() * ANN)}
    pn, ps, pf = perf(net), perf(static), perf(d["r"])
    out.update({"net": pn, "static": ps, "long": pf,
                "d_sharpe": pn.get("sharpe", np.nan) - ps.get("sharpe", np.nan),
                "d_mdd": pn.get("mdd", np.nan) - ps.get("mdd", np.nan)})
    return out
