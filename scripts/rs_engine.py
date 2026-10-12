"""
코스닥 상대강도 계산 엔진.

'데이터 모음' 엑셀의 VBA(UpdateSectorReturns_TimeSeries)와 같은 방식으로 세 표를 만든다.
  - 섹터별 상세    : 종목별 1일·1주·1개월·2개월·3개월 수익률 (1·5·20·40·60거래일)
  - 섹터 상대강도  : 섹터 소속 종목 수익률의 단순평균 - 코스닥 지수 수익률 (1주·2주·1달·2달·3달 = 5·10·20·40·60거래일)
                    + 기간별 등수, 섹터 시총 비중(유니버스 시총 합 = 100%)
  - 개별종목 상대강도: 종목 수익률 - 코스닥 지수 수익률 (1주·2주·1달·2달·3달·6달 = 5·10·20·40·60·120거래일)
원본 VBA와 다른 점은 비교 지수가 코스피가 아니라 코스닥이라는 것 하나다.

수익률은 '기준일 종가 / N거래일 전 종가 - 1' (행 기준 N칸 위, VBA의 startRow + lookback 과 동일).
N거래일 전 가격이 없으면(신규상장 등) 그 칸은 비운다(VBA의 "-").

확산도(breadth) 지표는 이번에 추가한 것: 날짜별로 '코스닥 대비 초과수익 > 0' 종목 비율과
섹터별 초과수익 순위의 추이. 유니버스는 기준일의 시총 상위 1000개로 고정해 과거로 되돌려 계산한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DETAIL_H = [("1일", 1), ("1주", 5), ("1개월", 20), ("2개월", 40), ("3개월", 60)]
SECTOR_H = [("1주", 5), ("2주", 10), ("1달", 20), ("2달", 40), ("3달", 60), ("6달", 120)]
STOCK_H = [("1주", 5), ("2주", 10), ("1달", 20), ("2달", 40), ("3달", 60), ("6달", 120)]
UNCLASSIFIED = "미분류"


def ret_n(panel: pd.DataFrame, n: int, at: int = -1) -> pd.Series:
    """panel: 날짜 오름차순 x 종목. at번째 행 기준 n거래일 수익률."""
    if len(panel) < n + 1 + (-at - 1):
        return pd.Series(np.nan, index=panel.columns)
    cur = panel.iloc[at]
    past = panel.iloc[at - n]
    r = cur / past - 1
    r[(cur <= 0) | (past <= 0)] = np.nan
    return r


def idx_ret(index: pd.Series, n: int, at: int = -1) -> float:
    if len(index) < n + 1 + (-at - 1):
        return np.nan
    a, b = index.iloc[at], index.iloc[at - n]
    return float(a / b - 1) if (a > 0 and b > 0) else np.nan


def align(prices: pd.DataFrame, index: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """지수와 종목 가격을 같은 거래일로 맞춘다. 기준일 = 지수와 종목 가격이 모두 있는 마지막 날."""
    prices = prices.sort_index()
    index = index.dropna().sort_index()
    common = prices.index.intersection(index.index)
    prices, index = prices.loc[common], index.loc[common]
    # 맨 끝에 종목 가격이 거의 없는 날(장중/휴장 잔여행)은 잘라낸다
    valid = prices.notna().sum(axis=1)
    if len(valid):
        thresh = max(10, int(valid.max() * 0.5))
        last_ok = valid[valid >= thresh].index.max()
        prices, index = prices.loc[:last_ok], index.loc[:last_ok]
    return prices, index


def compute(prices: pd.DataFrame, index: pd.Series, universe: pd.DataFrame,
            sector_map: pd.DataFrame, top_n: int = 1000, history_days: int = 120) -> dict:
    """
    prices   : 날짜 x 종목코드(6자리) 수정종가
    index    : 날짜 -> 코스닥 지수
    universe : columns [code, name, mcap] (mcap 단위: 원) - 기준일 시가총액
    sector_map: columns [code, sector]
    """
    prices, index = align(prices, index)
    asof = prices.index[-1]

    uni = universe.dropna(subset=["mcap"]).copy()
    uni = uni[uni["code"].isin(prices.columns)]
    uni = uni[prices.iloc[-1].reindex(uni["code"]).gt(0).values]
    uni = uni.sort_values("mcap", ascending=False).head(top_n).reset_index(drop=True)
    uni["mcap_rank"] = np.arange(1, len(uni) + 1)
    smap = dict(zip(sector_map["code"], sector_map["sector"]))
    uni["sector"] = uni["code"].map(smap).fillna(UNCLASSIFIED)
    codes = uni["code"].tolist()
    P = prices[codes]

    # ---------- 개별 종목 수익률 / 초과수익률 ----------
    idx_r = {lab: idx_ret(index, n) for lab, n in STOCK_H}
    stock = uni[["code", "name", "sector", "mcap", "mcap_rank"]].copy()
    for lab, n in DETAIL_H:
        stock["r_" + lab] = ret_n(P, n).reindex(codes).values
    for lab, n in STOCK_H:
        r = ret_n(P, n).reindex(codes).values
        stock["raw_" + lab] = r
        stock["ex_" + lab] = r - idx_r[lab] if not np.isnan(idx_r[lab]) else np.nan
    for lab, _ in STOCK_H:
        stock["rk_" + lab] = stock["ex_" + lab].rank(ascending=False, method="min")
        stock["srk_" + lab] = stock.groupby("sector")["ex_" + lab].rank(ascending=False, method="min")

    # ---------- 섹터 상대강도 (VBA: 섹터 내 종목 단순평균 - 지수) ----------
    total_cap = stock["mcap"].sum()
    rows = []
    for sec, g in stock.groupby("sector"):
        row = {"sector": sec, "n": int(len(g)), "mcap": float(g["mcap"].sum()),
               "weight": float(g["mcap"].sum() / total_cap)}
        for lab, n in SECTOR_H:
            raw = g["raw_" + lab].dropna()
            row["cnt_" + lab] = int(len(raw))
            row["ex_" + lab] = float(raw.mean() - idx_r[lab]) if len(raw) and not np.isnan(idx_r[lab]) else np.nan
            row["med_" + lab] = float(raw.median() - idx_r[lab]) if len(raw) and not np.isnan(idx_r[lab]) else np.nan
            ex = g["ex_" + lab].dropna()
            row["win_" + lab] = float((ex > 0).mean()) if len(ex) else np.nan
        rows.append(row)
    sector = pd.DataFrame(rows)
    for lab, _ in SECTOR_H:
        sector["rk_" + lab] = sector["ex_" + lab].rank(ascending=False, method="min")

    # ---------- 확산도: 날짜별 초과수익 종목 비율 + 섹터 순위 추이 ----------
    H = min(history_days, len(P) - 21)
    breadth = []
    sec_hist = {lab: {} for lab in ("1주", "1달")}
    sec_of = stock.set_index("code")["sector"]
    for k in range(H, 0, -1):
        at = -k
        d = P.index[at]
        rec = {"date": d.strftime("%Y-%m-%d"), "kosdaq": float(index.iloc[at])}
        for lab, n in (("1주", 5), ("1달", 20)):
            ir = idx_ret(index, n, at)
            r = ret_n(P, n, at)
            ex = (r - ir).dropna()
            rec["pct_" + lab] = float((ex > 0).mean()) if len(ex) else None
            rec["idx_" + lab] = ir
            if len(ex):
                s = (r.dropna().groupby(sec_of).mean() - ir)
                s = s.drop(UNCLASSIFIED, errors="ignore")
                rk = s.rank(ascending=False, method="min")
                sec_hist[lab][rec["date"]] = {"ex": s.round(5).to_dict(), "rk": rk.astype(int).to_dict()}
                rec["secpos_" + lab] = float((s > 0).mean())
        breadth.append(rec)

    # 섹터 1주 등수 변화(5거래일 전 대비)
    dates_hist = list(sec_hist["1주"].keys())
    if len(dates_hist) >= 6:
        prev = sec_hist["1주"][dates_hist[-6]]["rk"]
        sector["rk_1주_prev"] = sector["sector"].map(prev)
    else:
        sector["rk_1주_prev"] = np.nan

    return {
        "asof": asof.strftime("%Y-%m-%d"),
        "index_ret": idx_r,
        "index_level": float(index.iloc[-1]),
        "universe_size": int(len(stock)),
        "cutoff_mcap": float(stock["mcap"].min()) if len(stock) else None,
        "stock": stock,
        "sector": sector,
        "breadth": breadth,
        "sector_hist": sec_hist,
    }
