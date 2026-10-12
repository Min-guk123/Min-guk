"""
네이버 금융에서 코스닥 시가총액 순위와 종목별/지수 일봉을 받아 엔진 입력을 만든다 (API 키 불필요).

  1) 시가총액 순위 : m.stock.naver.com/api/stocks/marketValue/KOSDAQ  (실패 시 finance.naver.com 시가총액 페이지)
  2) 종목 일봉     : fchart.stock.naver.com/sise.nhn?symbol=<코드>&timeframe=day  (수정주가)
  3) 코스닥 지수   : fchart.stock.naver.com/sise.nhn?symbol=KOSDAQ

스팩·우선주는 유니버스에서 뺀 뒤 시총 상위 N개(기본 1000) + 여유분만 일봉을 받는다.
"""
from __future__ import annotations

import io
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Referer": "https://finance.naver.com/"}
SESSION = requests.Session()
SESSION.headers.update(UA)
EXCLUDE_NAME = re.compile(r"스팩|SPAC|기업인수목적")


def _get(url, params=None, tries=4, timeout=20):
    last = None
    for i in range(tries):
        try:
            r = SESSION.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last = e
        time.sleep(1.0 + i)
    raise RuntimeError(f"요청 실패 {url} {params}: {last}")


def _num(v):
    if v is None:
        return None
    s = str(v).replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _to_won(values: pd.Series) -> pd.Series:
    """네이버가 주는 시가총액 단위(억원/백만원/원)를 크기로 판별해 원 단위로 맞춘다.
    코스닥 1위 종목 시총은 수조~수십조원 수준이라 최댓값의 자릿수로 단위를 알 수 있다."""
    mx = values.max()
    if mx < 5e6:        # 억원 (예: 300,000억 = 30조)
        return values * 1e8
    if mx < 5e9:        # 백만원
        return values * 1e6
    return values       # 원


def listing_api() -> pd.DataFrame:
    rows, page = [], 1
    while True:
        r = _get("https://m.stock.naver.com/api/stocks/marketValue/KOSDAQ",
                 {"page": page, "pageSize": 100})
        js = r.json()
        stocks = js.get("stocks") or []
        if not stocks:
            break
        for s in stocks:
            rows.append({"code": s.get("itemCode"), "name": s.get("stockName"),
                         "mcap_raw": _num(s.get("marketValue")),
                         "end_type": s.get("stockEndType", "stock")})
        total = js.get("totalCount") or 0
        if page * 100 >= total:
            break
        page += 1
        time.sleep(0.15)
    df = pd.DataFrame(rows).dropna(subset=["code", "mcap_raw"])
    df = df[df["end_type"].fillna("stock").eq("stock")]
    df["mcap"] = _to_won(df["mcap_raw"])
    return df[["code", "name", "mcap"]]


def listing_html(max_pages: int = 40) -> pd.DataFrame:
    """finance.naver.com 시가총액 순위 페이지(코스닥=sosok 1, 시가총액 단위 억원)."""
    out = []
    for page in range(1, max_pages + 1):
        r = _get("https://finance.naver.com/sise/sise_market_sum.naver", {"sosok": 1, "page": page})
        html = r.content.decode("euc-kr", errors="ignore")
        codes = re.findall(r'href="/item/main\.naver\?code=([0-9A-Z]{6})" class="tltle">([^<]+)</a>', html)
        if not codes:
            break
        tbl = pd.read_html(io.StringIO(html), match="종목명")[0]
        tbl = tbl.dropna(subset=["종목명"])
        tbl = tbl[tbl["종목명"].isin([n for _, n in codes])]
        capmap = dict(zip(tbl["종목명"], tbl["시가총액"]))
        for code, name in codes:
            v = _num(capmap.get(name))
            if v is not None:
                out.append({"code": code, "name": name, "mcap": v * 1e8})
        time.sleep(0.2)
    return pd.DataFrame(out)


def kosdaq_listing() -> pd.DataFrame:
    try:
        df = listing_api()
        if len(df) > 1000:
            print(f"[listing] 모바일 API {len(df)}종목")
            return df
        print(f"[listing] 모바일 API 결과 부족({len(df)}) - HTML로 재시도")
    except Exception as e:  # noqa: BLE001
        print(f"[listing] 모바일 API 실패: {e} - HTML로 재시도")
    df = listing_html()
    print(f"[listing] HTML {len(df)}종목")
    return df


def daily_chart(symbol: str, count: int) -> pd.Series:
    r = _get("https://fchart.stock.naver.com/sise.nhn",
             {"symbol": symbol, "timeframe": "day", "count": count, "requestType": 0})
    items = re.findall(r'<item data="([^"]+)"', r.text)
    d, c = [], []
    for it in items:
        p = it.split("|")
        if len(p) >= 5 and p[4] not in ("", "null"):
            d.append(p[0])
            c.append(float(p[4]))
    return pd.Series(c, index=pd.to_datetime(d, format="%Y%m%d"), name=symbol)


def fetch(top_n: int = 1000, buffer: int = 60, count: int = 300, workers: int = 8):
    lst = kosdaq_listing()
    lst = lst[~lst["name"].str.contains(EXCLUDE_NAME)]
    lst = lst[lst["code"].str[-1] == "0"]          # 우선주 제외(종목코드 끝자리 0이 보통주)
    lst = lst.sort_values("mcap", ascending=False).drop_duplicates("code")
    target = lst.head(top_n + buffer)

    index = daily_chart("KOSDAQ", count)
    print(f"[index] 코스닥 지수 {len(index)}일, 마지막 {index.index.max().date()}")

    series, failed = {}, []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(daily_chart, c, count): c for c in target["code"]}
        for i, f in enumerate(as_completed(futs), 1):
            c = futs[f]
            try:
                s = f.result()
                if len(s):
                    series[c] = s
            except Exception as e:  # noqa: BLE001
                failed.append(c)
            if i % 200 == 0:
                print(f"  일봉 {i}/{len(futs)}")
    if failed:
        print(f"[prices] 실패 {len(failed)}종목: {failed[:20]}")
    prices = pd.DataFrame(series).sort_index()
    print(f"[prices] {prices.shape[1]}종목 x {prices.shape[0]}일")
    return prices, index, lst.reset_index(drop=True)
