"""
장기 가격 이력: 엑셀(DataGuide, 1년치)에 네이버 일봉(수정주가, 최대 3000거래일 ≈ 12년)을 이어 붙인다.

- 엑셀 기간은 엑셀 값을 우선한다(사용자의 공식 데이터). 엑셀 이전 구간만 네이버로 채운다.
- 이어 붙이는 지점의 수준 차이(권리락 조정 시점 차이 등)는 종목별 배율로 맞춘다
  (엑셀과 겹치는 첫 10거래일의 중앙 배율을 그 이전 네이버 구간에 곱한다).
- 네이버 수집이 실패하면 data/cache/prices_long.pkl(지난번에 받은 것)을 쓰고, 그것도 없으면 None.
- 장중에 받으면 마감 전 오늘 봉은 fetch_naver 가 뺀다.
"""
from __future__ import annotations

import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import fetch_naver as FN

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(BASE, "data", "cache", "prices_long.pkl")


def fetch(codes, count: int = 3000, workers: int = 8):
    """네이버에서 종목들과 코스닥 지수의 일봉 수집. 일부 종목만 실패하면 그 종목은 건너뛴다(5% 넘게 실패하면 예외)."""
    idx = FN.daily_chart("KOSDAQ", count)

    def one(c):
        try:
            return c, FN.daily_chart(c, count)
        except Exception:  # noqa: BLE001
            return c, None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(one, codes))
    ok = {c: s for c, s in res if s is not None and len(s)}
    if len(ok) < 0.95 * len(codes):
        raise RuntimeError(f"네이버 일봉 수집 실패 종목이 너무 많음: {len(codes) - len(ok)}/{len(codes)}")
    return pd.DataFrame(ok).sort_index(), idx


def _splice_cols(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A = 네이버(날짜 x 종목), B = 엑셀(날짜 x 종목). 같은 달력·같은 열 순서. 엑셀 구간은 엑셀 값, 그 앞뒤는 네이버 x 배율."""
    out = B.copy()
    for j in range(B.shape[1]):
        a, b = A[:, j], B[:, j]
        okb = ~np.isnan(b)
        if not okb.any():
            out[:, j] = a
            continue
        ov = okb & ~np.isnan(a) & (a > 0) & (b > 0)
        if not ov.any():
            out[:, j] = np.where(okb, b, a)
            continue
        ratio = b[ov] / a[ov]
        s_old = float(np.median(ratio[:10]))      # 엑셀 시작 부근 배율 -> 그 이전 네이버 구간
        s_new = float(np.median(ratio[-3:]))      # 엑셀 끝 부근 배율 -> 그 이후 네이버 구간
        t0 = int(np.argmax(okb))
        t1 = len(okb) - 1 - int(np.argmax(okb[::-1]))
        col = b.copy()
        col[:t0] = a[:t0] * s_old
        col[t1 + 1:] = a[t1 + 1:] * s_new
        out[:, j] = col
    return out


def splice(Pn: pd.DataFrame, In: pd.Series, Px: pd.DataFrame, Ix: pd.Series):
    """네이버(Pn, In)와 엑셀(Px, Ix)을 이어 붙인 (가격 패널, 지수)."""
    cal = In.index.union(Ix.index).sort_values()
    cols = list(dict.fromkeys(list(Px.columns) + list(Pn.columns)))
    A = Pn.reindex(index=cal, columns=cols).values.astype(float)
    B = Px.reindex(index=cal, columns=cols).values.astype(float)
    P = pd.DataFrame(_splice_cols(A, B), index=cal, columns=cols)
    I = pd.Series(_splice_cols(In.reindex(cal).values.astype(float)[:, None], Ix.reindex(cal).values.astype(float)[:, None])[:, 0], index=cal)
    # 달력 끝에 엑셀도 네이버도 값이 없는 날(오늘 아직 안 나온 봉)은 잘라낸다
    last = P.notna().sum(axis=1)
    keep = last[last >= max(10, int(last.max() * 0.5))].index.max()
    return P.loc[:keep], I.loc[:keep].dropna()


def load_long(excel_prices: pd.DataFrame, excel_index: pd.Series, codes, use_net: bool = True):
    """(장기 가격 패널, 장기 지수, 정보 dict). 네이버 수집이 불가능하면 캐시, 그것도 없으면 엑셀 구간만."""
    info = {"source": None, "naver_last": None, "n_codes": 0}
    Pn = In = None
    if use_net:
        t0 = time.time()
        try:
            Pn, In = fetch(list(codes))
            os.makedirs(os.path.dirname(CACHE), exist_ok=True)
            with open(CACHE, "wb") as f:
                pickle.dump({"prices": Pn, "index": In, "fetched": time.strftime("%Y-%m-%d %H:%M")}, f)
            info["source"] = f"네이버 새로 수집({time.time() - t0:.0f}초)"
        except Exception as e:  # noqa: BLE001
            print(f"[longdata] 네이버 수집 실패: {e}")
    if Pn is None and os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            d = pickle.load(f)
        Pn, In = d["prices"], d["index"]
        info["source"] = f"네이버 캐시({d.get('fetched', '?')})"
    Px, Ix = excel_prices.sort_index(), excel_index.dropna().sort_index()
    cols = [c for c in codes if c in Px.columns or (Pn is not None and c in Pn.columns)]
    if Pn is None:
        info["source"] = "엑셀만(장기 이력 없음)"
        return Px[[c for c in cols if c in Px.columns]], Ix, info
    P, I = splice(Pn[[c for c in cols if c in Pn.columns]], In, Px[[c for c in cols if c in Px.columns]], Ix)
    info["naver_last"] = str(In.index.max().date())
    info["n_codes"] = int(P.shape[1])
    return P[[c for c in cols if c in P.columns]], I, info
