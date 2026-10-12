"""
엑셀 파일이 저장(수정)되면 자동으로 대시보드를 다시 계산해 GitHub에 올린다.

사용법
  python scripts/watch_excel.py                 # config/local.json 의 excel_path 를 감시
  python scripts/watch_excel.py --once          # 지금 한 번만 갱신하고 종료
  python scripts/watch_excel.py --file "C:\\경로\\코스닥 분류.xlsx"

config/local.json (git에 올라가지 않음)
  {"excel_path": "C:\\Users\\내이름\\Desktop\\코스닥 분류.xlsx"}
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG = os.path.join(BASE, "config", "local.json")
POLL_SEC = 5          # 파일 확인 주기
SETTLE_SEC = 20       # 마지막 변경 후 이만큼 조용하면 저장이 끝난 것으로 본다
PUBLISH = ["docs/data", "docs/downloads", "data/snapshots"]


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def run(cmd, check=True):
    r = subprocess.run(cmd, cwd=BASE, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    return r


def stamp(path):
    st = os.stat(path)
    return (st.st_mtime, st.st_size)


def wait_until_stable(path):
    """엑셀이 파일을 쓰는 중에는 크기·시간이 계속 바뀌므로 조용해질 때까지 기다린다."""
    last = stamp(path)
    quiet_since = time.time()
    while time.time() - quiet_since < SETTLE_SEC:
        time.sleep(2)
        cur = stamp(path)
        if cur != last:
            last, quiet_since = cur, time.time()
    return last


def update(path):
    log(f"계산 시작: {path}")
    run([sys.executable, os.path.join("scripts", "build.py"), "--source", "dataguide", "--file", path])

    run(["git", "pull", "--rebase", "--autostash"], check=False)  # 서버(Actions)가 먼저 올린 커밋과 충돌 방지
    run(["git", "add"] + PUBLISH)
    if run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        log("바뀐 내용이 없어 올리지 않았습니다.")
        return
    run(["git", "commit", "-m", f"엑셀 갱신 {datetime.now():%Y-%m-%d %H:%M}"])
    run(["git", "push"])
    log("GitHub에 올렸습니다. 1~2분 뒤 사이트에 반영됩니다.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()

    path = a.file
    if not path and os.path.exists(CONFIG):
        with open(CONFIG, encoding="utf-8-sig") as f:
            path = json.load(f).get("excel_path")
    if not path:
        sys.exit("감시할 엑셀 경로가 없습니다. config/local.json 에 excel_path 를 적거나 --file 로 지정하세요.")
    path = os.path.abspath(path)

    if a.once:
        update(path)
        return

    log(f"감시 시작: {path}  (이 창을 닫으면 멈춥니다)")
    seen = stamp(path) if os.path.exists(path) else None
    while True:
        try:
            if os.path.exists(path) and stamp(path) != seen:
                log("엑셀 변경 감지, 저장이 끝나길 기다립니다...")
                seen = wait_until_stable(path)
                try:
                    update(path)
                except Exception as e:  # 한 번 실패해도 감시는 계속
                    log(f"실패: {e}")
        except OSError as e:
            log(f"파일 접근 오류(엑셀이 저장 중일 수 있음): {e}")
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()

