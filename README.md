# 코스닥 상대강도 대시보드

코스닥 시가총액 상위 1000종목의 **종목 상대강도 순위**, **섹터 상대강도 순위**, **수급 확산(코스닥을 이긴 종목 비율)** 을
매 영업일 장 마감 후 자동으로 계산해 GitHub Pages에 올립니다.

계산 방식은 '데이터 모음' 엑셀의 `섹터별 상세` · `섹터 상대강도` · `개별종목 상대강도` 시트(VBA `UpdateSectorReturns_TimeSeries`)와
같고, 비교 지수만 코스피 대신 코스닥입니다. 2026-10-02 기준으로 원본 엑셀과 겹치는 코스닥 447종목의 1일~3개월 수익률이
소수점 15자리까지 일치하는 것을 확인했습니다.

## 구성

| 경로 | 내용 |
|---|---|
| `config/sector_map.csv` | 종목코드 → 섹터 분류표 (1,200종목). `basis` 열에 분류 근거 |
| `scripts/fetch_naver.py` | 네이버 금융에서 시총 순위·일봉(수정주가)·코스닥 지수 수집 (API 키 불필요) |
| `scripts/load_dataguide.py` | DataGuide 엑셀을 입력으로 쓸 때 |
| `scripts/rs_engine.py` | 상대강도 계산 (VBA 로직 이식 + 확산도) |
| `scripts/build.py` | 수집 → 계산 → `docs/` 결과 파일 작성 |
| `docs/index.html` | 홈 |
| `docs/kosdaq_rs.html` | 대시보드 |
| `docs/data/latest.json` | 대시보드 데이터 |
| `docs/downloads/kosdaq_rs_latest.xlsx` | 엑셀(원본 3시트 배치, 금액 십억원) |
| `data/snapshots/` | 날짜별 섹터/종목 결과 기록 |
| `.github/workflows/daily.yml` | 평일 16:20 KST 자동 실행 |
| `scripts/watch_excel.py` | 엑셀 저장을 감지해 자동 갱신 + GitHub 업로드 (아래 '엑셀 자동 갱신') |
| `scripts/drawdown_stats.py` | ④ 탭(최근 낙폭 평균): 종목·코스닥의 '내린 날만' 평균 낙폭 → `docs/data/drawdown.json` |
| `scripts/signal_model.py` | ⑤ 탭: 수급 확산 지표·낙관 점수·숏 비중 함수 |
| `scripts/signal_pipeline.py` | ⑤ 탭: 매일 갱신 때 시그널 계산 → `docs/data/signal.json` |
| `scripts/backfill_history.py` | (한 번) 네이버에서 12년치 일봉 캐시 받기 (`data/cache/`, git 제외) |
| `scripts/make_feature_history.py` | (한 번) 일봉 캐시 → `data/feature_history.csv` 만들기 |
| `scripts/research_signal.py` | 연구(검증) → `docs/data/signal_research.json` (`report` 단계) |
| `config/signal_params.json` | ⑤ 탭 파라미터(가감 폭, 평활, 비용, 기준 숏 비중) |
| `data/feature_history.csv` | 수급 확산 지표 이력 (2014~, 매일 갱신 때 이어 붙음) |

## 처음 올리는 순서 (10분)

1. GitHub에서 새 저장소를 만듭니다. 예: `kosdaq-rs-dashboard` (Public이어야 무료 Pages 사용 가능).
2. 이 폴더 전체를 올립니다.
   ```bash
   cd kosdaq-rs
   git init -b main
   git add .
   git commit -m "코스닥 상대강도 대시보드"
   git remote add origin https://github.com/<계정>/kosdaq-rs-dashboard.git
   git push -u origin main
   ```
   (명령어 대신 GitHub 웹에서 'Add file → Upload files'로 폴더 내용을 끌어다 놓아도 됩니다. `.github` 폴더까지 올라가야 합니다.)
3. 저장소 **Settings → Pages** → Source: `Deploy from a branch`, Branch: `main` / 폴더 `/docs` → Save.
   1~2분 뒤 `https://<계정>.github.io/kosdaq-rs-dashboard/` 에서 열립니다.
4. 저장소 **Settings → Actions → General → Workflow permissions** 를 `Read and write permissions` 로 바꿉니다(결과를 커밋하려면 필요).
5. **Actions** 탭 → `코스닥 상대강도 일일 갱신` → `Run workflow` 로 한 번 수동 실행해 정상 동작을 확인합니다.
   이후에는 평일 16:20 KST에 자동으로 돕니다(GitHub 사정으로 10~30분 늦게 시작될 수 있음).

## 섹터 분류 고치기

`config/sector_map.csv` 에서 `sector` 값을 바꾸고 커밋하면 다음 실행부터 반영됩니다.
1000위 안에 새로 들어온 종목이 분류표에 없으면 `미분류` 로 표시되고, 대시보드 상단과 Actions 로그에 종목이 나옵니다. 그 종목을 표에 한 줄 추가하면 됩니다.

`basis` 열 의미
- `기존 섹터맵`: 데이터 모음 '섹터별 구성 종목' 시트 분류 그대로 (456종목)
- `Claude 분류`: 같은 섹터 체계로 새로 분류
- `Claude 분류(웹 확인)` / `(네이버 업종 참고)`: 사업 내용을 검색하거나 네이버 업종과 대조해 정함
- `미확인→개별주`: 사업 내용을 확인하지 못해 개별주로 둔 종목 (모두 1000위 밖)

## 실행 시간 바꾸기

`.github/workflows/daily.yml` 의 `cron: "20 7 * * 1-5"` 는 UTC 기준입니다(KST −9시간). 예: 17:00 KST → `"0 8 * * 1-5"`.

## 수동 갱신 (DataGuide 엑셀)

```bash
pip install -r requirements.txt
python scripts/build.py --source dataguide --file "코스닥 분류.xlsx"
```
엑셀 형식: 시트1 = 종목별 `수정주가`·`시가총액` 일별 시계열(내림차순), 시트2 = 코스닥 `종가지수`. 네이버 수집이 막혔을 때 이 방법으로 갱신 후 커밋하면 됩니다.

## 엑셀 자동 갱신 (내 PC)

엑셀(`코스닥 분류.xlsx`)을 저장하면 대시보드를 다시 계산해 GitHub에 올립니다.

1. Python, Git 설치 후 `pip install -r requirements.txt`
2. `config/local.json` 에 `{"excel_path": "C:\\경로\\코스닥 분류.xlsx"}` 를 적습니다(git에 올라가지 않음).
3. `powershell -ExecutionPolicy Bypass -File install_autostart.ps1` — 로그인 때마다 감시 프로그램이 자동으로 켜집니다(관리자 권한 불필요).
   수동 실행은 `start_watcher.bat`, 한 번만 갱신하려면 `python scripts/watch_excel.py --once`.

## ④ 최근 낙폭 평균 탭

종목마다 최근 N거래일(5·10·20·40·60) 중 **전일보다 내린 날만** 골라 그날 등락률을 평균한 값(= 내릴 때 평균 얼마나 내리는가)과,
코스닥 지수의 같은 값을 나란히 보여줍니다. 종목은 각자 자기가 내린 날만 평균하므로 코스닥이 내린 날과 같지 않습니다.
가격이 N거래일 모두 있는 종목만 비교하는 것이 기본이고(신규상장·거래정지 종목은 끌 수 있음), 날짜별 20거래일 롤링 추이(코스닥 vs 종목 중앙값)도 있습니다.
`build.py` 가 돌 때마다 `docs/data/drawdown.json` 을 다시 씁니다.

## ⑤ 시그널·숏 비중 탭

수급 확산 지표 16개(ADR, 코스닥을 이긴 종목 비율, 이동평균 위 종목 비율, 신고가−신저가, 동일가중−지수 괴리)를 평균해
'낙관 점수'(0~100)를 만들고, 밸류에이션 룰이 정한 기준 숏 비중(저평가 50 · 중립 75 · 고평가 100%)에 ±amp 를 가감합니다.
연구 결과는 **방향(오를지 내릴지)은 예측하지 못하고, 앞으로의 변동성·낙폭 위험을 구분**하며, 숏 가감의 효과는 작고 한두 번의 급락에 기대고 있다는 것입니다.
자세한 근거와 한계는 탭 안의 '방법과 한계'에 있습니다.

처음 한 번(또는 데이터를 새로 받고 싶을 때):

```bash
python scripts/backfill_history.py          # 네이버에서 12년치 일봉(약 10초)
python scripts/make_feature_history.py      # data/feature_history.csv
python scripts/research_signal.py report    # docs/data/signal_research.json (약 1분)
```

이후에는 `build.py` 가 돌 때마다 최근 지표를 이어 붙이고 `docs/data/signal.json` 을 다시 씁니다
(엑셀 모드에서는 엑셀보다 최신인 날짜를 네이버에서 보충하며, 실패해도 무시. 장중(평일 15:40 전)에 받으면 마감 전인 오늘 봉은 자동으로 뺍니다).
`config/signal_params.json` 의 `amp` 를 키우면(예: 0.25) 더 강하게 가감하고, `cost` 는 숏 100%를 바꿀 때의 비용입니다.
파라미터는 매일 다시 최적화하지 않으므로 분기마다 `research_signal.py report` 로 재검증하세요.

## 계산 정의

- 기간: 1일=1, 1주=5, 2주=10, 1달=20, 2달=40, 3달=60, 6달=120 거래일
- 종목 초과수익률 = 종목 수익률 − 코스닥 수익률
- 섹터 초과수익률 = 섹터 소속 종목 수익률 단순평균 − 코스닥 수익률 (중앙값·이긴 종목 비율도 제공)
- 코스닥비중 = 섹터 시총 ÷ 1000종목 시총 합
- 유니버스 = 기준일 시총 상위 1000종목(스팩·우선주 제외), 매일 재선정
- 수급 확산 = 날짜별 '코스닥 대비 초과수익 > 0' 종목 비율(기준일 1000종목 고정으로 과거 재계산)
