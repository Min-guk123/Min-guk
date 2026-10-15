"""
밸류 하단 종목 비율(OP밴드) - 시가총액 ÷ 12개월 선행 영업이익 추정치 배수가 '자기 최근 3년 하위 10%' 아래인 종목의 비율.

정의(참고 대시보드의 OP밴드 트래커와 같은 방식)
  - 선행OP = w x 당해연도 영업이익 추정치 + (1 - w) x 차년도 추정치,  w = (13 - 월) / 12   (매달 분모가 조금씩 굴러감)
  - 배수 = 시가총액 ÷ 선행OP  (선행OP <= 0 이면 배수를 비운다)
  - 바텀 = 그 종목 배수의 최근 3년(756거래일) 하위 10% 값(그 시점까지의 데이터만 사용)
  - 비율 = 배수 < 바텀 인 종목 수 ÷ (배수와 바텀이 모두 있는 종목 수)

원자료는 DataGuide 형식의 엑셀(코스닥 데이터_v5 의 '영업이익 추정치'·'시가총액' 시트)이다. 이 파일에는 2025-06 이후 일별 추정치만 있어
'2년 이상 이력' 조건을 쓸 수 없으므로 MIN_OBS(기본 120거래일)로 낮춰 쓴다. 더 긴 이력을 같은 형식으로 받으면 MIN_OBS 를 2년(500)으로 올리면 된다.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

WINDOW = 756      # 3년
MIN_OBS = 120     # 바텀을 계산하는 데 필요한 최소 관측일(원래는 2년 이상 이력)
Q = 0.10


def _find(rows, label, cols=(0, 1)):
    for i, r in enumerate(rows[:24]):
        for c in cols:
            if c < len(r) and r[c] is not None and str(r[c]).strip() == label:
                return i, c
    return None, None


def _read_ts(ws, with_fy: bool):
    """DataGuide 시계열 시트 -> (DataFrame[날짜 x (코드 또는 (코드, FY))])."""
    rows = list(ws.iter_rows(values_only=True))
    code_r, _ = _find(rows, "Code")
    date_r, date_c = _find(rows, "D A T E")
    base_r, _ = _find(rows, "Base Date")
    if code_r is None or date_r is None:
        raise ValueError(f"DataGuide 시계열 형식이 아닙니다: {ws.title}")
    codes = rows[code_r]
    body = [r for r in rows[date_r + 1:] if r and len(r) > date_c and hasattr(r[date_c], "year")]
    dates = pd.to_datetime([r[date_c] for r in body])
    data = {}
    for c in range(date_c + 1, len(codes)):
        if not codes[c]:
            continue
        code = str(codes[c]).lstrip("A")
        key = (code, str(rows[base_r][c])[:4]) if with_fy else code
        data[key] = [r[c] if c < len(r) else None for r in body]
    df = pd.DataFrame(data, index=dates).apply(pd.to_numeric, errors="coerce").sort_index()
    df = df[~df.index.duplicated(keep="last")].dropna(how="all")
    if with_fy:
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=["code", "fy"])
    return df


def read_workbook(path: str):
    """(영업이익 추정치[날짜 x (코드,FY)] 단위 원, 시가총액[날짜 x 코드] 단위 원). 시트 이름은 '영업이익 추정치', '시가총액'."""
    import openpyxl
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        op = _read_ts(wb["영업이익 추정치"], with_fy=True) * 1000.0      # 천원 -> 원
        caps = _read_ts(wb["시가총액"], with_fy=False)
    return op, caps


def forward_op(op: pd.DataFrame, dates: pd.DatetimeIndex, codes) -> pd.DataFrame:
    """날짜별 12개월 선행 영업이익(원). 추정치는 마지막 값을 이어 쓴다(컨센서스는 천천히 바뀜)."""
    op_d = op.reindex(op.index.union(dates)).sort_index().ffill().reindex(dates)
    fys = sorted({int(f) for f in op.columns.get_level_values("fy")})
    out = pd.DataFrame(np.nan, index=dates, columns=codes)
    for y in sorted(set(dates.year)):
        rows = dates.year == y
        if not rows.any():
            continue
        def fy(year):
            if str(year) in op_d.columns.get_level_values("fy"):
                return op_d.xs(str(year), axis=1, level="fy").reindex(columns=codes)
            return pd.DataFrame(np.nan, index=dates, columns=codes)
        o1 = fy(y).loc[rows].values
        o2 = fy(y + 1).loc[rows].values
        w = ((13 - dates[rows].month) / 12.0).values[:, None]
        both = ~np.isnan(o1) & ~np.isnan(o2)
        f = np.where(both, w * o1 + (1 - w) * o2, np.where(~np.isnan(o2), o2, o1))   # 당해 추정치가 없으면 차년도만, 차년도가 없으면 당해만
        out.loc[rows] = f
    return out.where(out > 0)


def bottom_ratio(caps: pd.DataFrame, op: pd.DataFrame, q: float = Q, window: int = WINDOW, min_obs: int = MIN_OBS):
    """(비율 Series, 대상 종목 수 Series, 배수 DataFrame). 날짜 = caps 와 op 가 겹치는 구간."""
    start = op.index.min()
    caps = caps.loc[caps.index >= start]
    codes = [c for c in sorted(set(op.columns.get_level_values("code")) & set(caps.columns))]
    fwd = forward_op(op, caps.index, codes)
    mult = caps[codes] / fwd
    thr = mult.rolling(window, min_periods=min_obs).quantile(q)
    ok = mult.notna() & thr.notna()
    below = (mult < thr) & ok
    n = ok.sum(axis=1)
    ratio = (below.sum(axis=1) / n).where(n >= 30)
    return ratio, n, mult
