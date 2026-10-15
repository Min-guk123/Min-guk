"""
데이터를 받아(네이버 또는 DataGuide 엑셀) 상대강도를 계산하고 대시보드용 파일을 쓴다.

  docs/data/latest.json                 대시보드가 읽는 데이터
  docs/downloads/kosdaq_rs_latest.xlsx  엑셀 3시트(섹터별 상세/섹터 상대강도/개별종목 상대강도), 원본과 같은 배치
  data/snapshots/YYYY-MM-DD_{sector,stock}.csv  일자별 기록

사용법
  python scripts/build.py                       # 네이버에서 수집(매일 자동 실행이 쓰는 방식)
  python scripts/build.py --source dataguide --file "코스닥 분류.xlsx"
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import rs_engine as E  # noqa: E402

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KST = timezone(timedelta(hours=9))
BIL = 1e9  # 십억원


def clean(v, nd=5):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, nd)


def load_sector_map():
    m = pd.read_csv(os.path.join(BASE, "config", "sector_map.csv"), dtype=str, encoding="utf-8-sig")
    m["code"] = m["code"].str.strip().str.zfill(6)
    return m


def to_json(res, source):
    st, se = res["stock"], res["sector"]
    s_cols = ["code", "name", "sector", "mcap_rank"]
    stocks = []
    for r in st.itertuples(index=False):
        d = r._asdict()
        stocks.append([d["code"], d["name"], d["sector"], int(d["mcap_rank"]), clean(d["mcap"] / BIL, 1)]
                      + [clean(d["r_" + l]) for l, _ in E.DETAIL_H]
                      + [clean(d["ex_" + l]) for l, _ in E.STOCK_H])
    sectors = []
    for r in se.itertuples(index=False):
        d = r._asdict()
        sectors.append({
            "sector": d["sector"], "n": d["n"], "mcap": clean(d["mcap"] / BIL, 1), "weight": clean(d["weight"]),
            "ex": [clean(d["ex_" + l]) for l, _ in E.SECTOR_H],
            "med": [clean(d["med_" + l]) for l, _ in E.SECTOR_H],
            "win": [clean(d["win_" + l], 4) for l, _ in E.SECTOR_H],
            "rk_prev_1w": clean(d["rk_1주_prev"], 0),
        })
    unclassified = st.loc[st["sector"] == E.UNCLASSIFIED, ["code", "name", "mcap_rank"]].values.tolist()
    return {
        "asof": res["asof"],
        "generated": datetime.now(KST).strftime("%Y-%m-%d %H:%M KST"),
        "source": source,
        "index_level": clean(res["index_level"], 2),
        "index_ret": {k: clean(v) for k, v in res["index_ret"].items()},
        "universe_size": res["universe_size"],
        "cutoff_mcap_bil": clean((res["cutoff_mcap"] or 0) / BIL, 1),
        "detail_h": [l for l, _ in E.DETAIL_H],
        "stock_h": [l for l, _ in E.STOCK_H],
        "sector_h": [l for l, _ in E.SECTOR_H],
        "stock_cols": ["code", "name", "sector", "mcap_rank", "mcap_bil"]
                      + ["r_" + l for l, _ in E.DETAIL_H] + ["ex_" + l for l, _ in E.STOCK_H],
        "stocks": stocks,
        "sectors": sectors,
        "breadth": [{k: clean(v) for k, v in b.items()} for b in res["breadth"]],
        "sector_hist": res["sector_hist"],
        "unclassified": unclassified,
    }


def write_excel(res, path):
    """원본 '데이터 모음' 시트 배치를 따른 3시트 + 섹터 구성. 금액은 십억원."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    st, se, asof, ir = res["stock"], res["sector"], res["asof"], res["index_ret"]
    green = PatternFill("solid", fgColor="00B050")
    hdr_font = Font(bold=True, color="FFFFFF")
    red, blue = Font(color="D21E1E"), Font(color="1E64D2")
    wb = Workbook()

    def header(ws, row, labels):
        for j, lab in enumerate(labels, 1):
            c = ws.cell(row=row, column=j, value=lab)
            c.fill, c.font, c.alignment = green, hdr_font, Alignment(horizontal="center")

    def pct(ws, r, c, v):
        cell = ws.cell(row=r, column=c, value=None if pd.isna(v) else float(v))
        if cell.value is None:
            cell.value = "-"
            cell.alignment = Alignment(horizontal="center")
        else:
            cell.number_format = "0.00%"
            if v > 0:
                cell.font = red
            elif v < 0:
                cell.font = blue

    # 섹터별 상세
    ws = wb.active
    ws.title = "섹터별 상세"
    ws["A1"] = f"기준일자: {asof}"
    ws["A1"].font = Font(bold=True, color="808080")
    labs = [l for l, _ in E.DETAIL_H]
    header(ws, 3, ["섹터명", "종목명", *labs, "시총(십억원)"])
    d = st.sort_values(["sector", "r_1주"], ascending=[True, False], na_position="last")
    for i, r in enumerate(d.itertuples(index=False), 4):
        dd = r._asdict()
        ws.cell(row=i, column=1, value=dd["sector"])
        ws.cell(row=i, column=2, value=dd["name"])
        for j, l in enumerate(labs, 3):
            pct(ws, i, j, dd["r_" + l])
        ws.cell(row=i, column=3 + len(labs), value=round(dd["mcap"] / BIL, 1)).number_format = "#,##0.0"
    ws.freeze_panes = "C4"
    ws.auto_filter.ref = f"A3:{get_column_letter(3 + len(labs))}{3 + len(d)}"

    # 섹터 상대강도
    ws = wb.create_sheet("섹터 상대강도")
    slabs = [l for l, _ in E.SECTOR_H]
    ws["A1"] = f"기준일자: {asof}  |  코스닥 대비 초과수익률 (섹터 내 종목 단순평균 - 코스닥)"
    ws["A1"].font = Font(bold=True, color="808080")
    ws["A2"] = "코스닥 수익률"
    for j, l in enumerate(slabs, 2):
        v = ir.get(l)
        ws.cell(row=2, column=j, value=f"{l} {v:.2%}" if v is not None and not pd.isna(v) else f"{l} -")
    header(ws, 4, ["섹터명", "코스닥비중", "종목수", "시총(십억원)", *slabs, *[f"{l} 등수" for l in slabs]])
    d = se.sort_values("ex_1주", ascending=False, na_position="last")
    for i, r in enumerate(d.itertuples(index=False), 5):
        dd = r._asdict()
        ws.cell(row=i, column=1, value=dd["sector"])
        ws.cell(row=i, column=2, value=dd["weight"]).number_format = "0.00%"
        ws.cell(row=i, column=3, value=dd["n"])
        ws.cell(row=i, column=4, value=round(dd["mcap"] / BIL, 1)).number_format = "#,##0.0"
        for j, l in enumerate(slabs, 5):
            pct(ws, i, j, dd["ex_" + l])
            rk = dd["rk_" + l]
            c = ws.cell(row=i, column=j + len(slabs), value=None if pd.isna(rk) else int(rk))
            c.alignment = Alignment(horizontal="center")
            if c.value is not None and c.value <= 3:
                c.fill, c.font = PatternFill("solid", fgColor="FFE699"), Font(bold=True)
    ws.freeze_panes = "B5"

    # 개별종목 상대강도
    ws = wb.create_sheet("개별종목 상대강도")
    ws["A1"] = f"기준일자: {asof}  |  개별종목 코스닥 대비 초과수익률"
    ws["A1"].font = Font(bold=True, color="808080")
    ws["A2"] = "코스닥 수익률"
    klabs = [l for l, _ in E.STOCK_H]
    for j, l in enumerate(klabs, 3):
        v = ir.get(l)
        ws.cell(row=2, column=j, value=f"{l} {v:.2%}" if v is not None and not pd.isna(v) else f"{l} -")
    header(ws, 4, ["섹터명", "종목명", *klabs, "시총(십억원)", "시총순위"])
    d = st[st["ex_1주"].notna()].sort_values("ex_1주", ascending=False)
    for i, r in enumerate(d.itertuples(index=False), 5):
        dd = r._asdict()
        ws.cell(row=i, column=1, value=dd["sector"])
        ws.cell(row=i, column=2, value=dd["name"])
        for j, l in enumerate(klabs, 3):
            pct(ws, i, j, dd["ex_" + l])
        ws.cell(row=i, column=3 + len(klabs), value=round(dd["mcap"] / BIL, 1)).number_format = "#,##0.0"
        ws.cell(row=i, column=4 + len(klabs), value=int(dd["mcap_rank"]))
    for i in range(5, min(25, 5 + len(d))):
        ws.cell(row=i, column=3).fill = PatternFill("solid", fgColor="FFE699")
    ws.freeze_panes = "C5"
    ws.auto_filter.ref = f"A4:{get_column_letter(4 + len(klabs))}{4 + len(d)}"

    for w in wb.worksheets:
        w.column_dimensions["A"].width = 16
        w.column_dimensions["B"].width = 20
        for col in range(3, 20):
            w.column_dimensions[get_column_letter(col)].width = 11
    wb.save(path)


def local_cfg() -> dict:
    p = os.path.join(BASE, "config", "local.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8-sig") as f:
            return json.load(f)
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["naver", "dataguide"], default="naver")
    ap.add_argument("--file", help="DataGuide 엑셀 경로(--source dataguide)")
    ap.add_argument("--top", type=int, default=1000)
    ap.add_argument("--op", help="영업이익 추정치 엑셀(코스닥 데이터_v5 형식). 없으면 config/local.json 의 op_path")
    args = ap.parse_args()

    if args.source == "dataguide":
        from load_dataguide import load
        prices, index, universe = load(args.file)
        src = "DataGuide 엑셀(수정주가)"
    else:
        from fetch_naver import fetch
        prices, index, universe = fetch(top_n=args.top)
        src = "네이버 금융(수정주가)"

    res = E.compute(prices, index, universe, load_sector_map(), top_n=args.top)
    payload = to_json(res, src)

    os.makedirs(os.path.join(BASE, "docs", "data"), exist_ok=True)
    os.makedirs(os.path.join(BASE, "docs", "downloads"), exist_ok=True)
    os.makedirs(os.path.join(BASE, "data", "snapshots"), exist_ok=True)
    with open(os.path.join(BASE, "docs", "data", "latest.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    write_excel(res, os.path.join(BASE, "docs", "downloads", "kosdaq_rs_latest.xlsx"))
    snap = os.path.join(BASE, "data", "snapshots", res["asof"])
    res["sector"].to_csv(snap + "_sector.csv", index=False, encoding="utf-8-sig")
    keep = ["code", "name", "sector", "mcap_rank", "mcap"] + [c for c in res["stock"].columns if c.startswith(("r_", "ex_"))]
    res["stock"][keep].to_csv(snap + "_stock.csv", index=False, encoding="utf-8-sig")

    caps = prices.attrs.get("caps")          # 엑셀 일별 시총(없으면 None)
    try:  # 최근 낙폭 평균(④ 탭)
        import drawdown_stats as DS
        DS.write(DS.compute(prices, index, res["stock"]), os.path.join(BASE, "docs", "data", "drawdown.json"))
    except Exception as e:  # noqa: BLE001
        print(f"[drawdown] 낙폭 평균 계산 실패(대시보드 나머지는 정상): {type(e).__name__}: {e}")

    long = None
    try:  # 장기 가격 이력(엑셀 + 네이버 12년) - ⑥ 섹터 이평선 위 비율, ⑦ 시장 국면 점수, ⑤ 시그널이 함께 쓴다
        import longdata as LD
        Px, Ix = E.align(prices, index)
        long = LD.load_long(Px, Ix, res["stock"]["code"].tolist())
        print(f"[longdata] {long[2]['source']} · {long[0].shape[1]}종목 x {long[0].shape[0]}일 ({long[0].index[0].date()} ~ {long[0].index[-1].date()})")
    except Exception as e:  # noqa: BLE001
        print(f"[longdata] 장기 이력 준비 실패(⑥⑦ 건너뜀): {type(e).__name__}: {e}")

    if long is not None:
        try:  # 섹터 이평선 위 비율(⑥ 탭)
            import sector_ma as SM
            SM.write(SM.compute(long[0], long[1], res["stock"], caps_actual=caps), os.path.join(BASE, "docs", "data", "sector_ma.json"))
        except Exception as e:  # noqa: BLE001
            print(f"[sector_ma] 섹터 이평선 위 비율 계산 실패(대시보드 나머지는 정상): {type(e).__name__}: {e}")
        try:  # 시장 국면 점수(⑦ 탭)
            import market_phase as MP
            op_path = args.op or local_cfg().get("op_path")
            pl = MP.run(long[0], long[1], res["stock"], caps, op_path, os.path.join(BASE, "data", "valuation_history.csv"),
                        os.path.join(BASE, "docs", "data", "market_phase.json"))
            nw = pl["now"]
            print(f"[market_phase] {nw['date']} 국면 {pl['phases'][nw['phase'] - 1]['name'] if nw['phase'] else '-'} · 점수 {nw['score']} (낮음<{nw['lo']} 높음>{nw['hi']})")
        except Exception as e:  # noqa: BLE001
            print(f"[market_phase] 시장 국면 점수 계산 실패(대시보드 나머지는 정상): {type(e).__name__}: {e}")

    try:  # 수급 확산 시그널(⑤ 탭). 실패해도 위 결과는 이미 저장됐으므로 계속 진행
        import signal_pipeline as SP
        SP.run(prices, index, res, args.source, long=long)
    except Exception as e:  # noqa: BLE001
        print(f"[signal] 시그널 계산 실패(대시보드 나머지는 정상): {type(e).__name__}: {e}")

    n_un = len(payload["unclassified"])
    print(f"완료: 기준일 {res['asof']}, 유니버스 {res['universe_size']}종목, "
          f"1000위 시총 {payload['cutoff_mcap_bil']}십억원, 미분류 {n_un}종목")
    if n_un:
        print("  미분류(config/sector_map.csv에 추가 필요):", payload["unclassified"][:30])


if __name__ == "__main__":
    main()
