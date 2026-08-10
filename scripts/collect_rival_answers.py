"""상대 챗봇(yunju.work/grant-rules) 53문항 라이브 답변 수집.

비교 목적(교수님 요구 7번)의 1회성 수집. 상대 서비스 예의를 지킨다:
- 질의 간 고정 간격(기본 30초) — 53문항을 약 36분에 걸쳐 분산
- 재시도는 문항당 1회만, 실패는 기록하고 넘어감
- 상대 UI 기본 설정 그대로(쾌속 모델, brief, cache=1) — 비교 조건을
  상대에게 유리하지도 불리하지도 않게. 모드·응답시간·cache_hit을 같이
  기록해 비교표에 조건을 명시한다

append 방식: 중단 후 같은 --out으로 재실행하면 남은 문항부터 이어간다.

사용 예:
  python3 scripts/collect_rival_answers.py \
      --out processed/eval/20260806-rival-answers.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://yunju.work/grant-rules/api/ask_stream"
UA = "pnu-docs-chatbot-benchmark (academic comparison; contact: sosojini02@gmail.com)"


def ask(query: str, model: str, timeout: float) -> dict:
    params = urllib.parse.urlencode({
        "q": query, "detail": "brief", "model": model, "cache": "1", "limit": "5",
    })
    req = urllib.request.Request(f"{BASE}?{params}", headers={"User-Agent": UA})
    t0 = time.time()
    final, cache_hit, decision = None, None, None
    with urllib.request.urlopen(req, timeout=timeout) as res:
        for raw in res:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data: "):
                continue
            try:
                d = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if d.get("kind") == "answer" and d.get("phase") == "final":
                final = d
            elif d.get("kind") == "decision":
                decision = d
            if "cache_hit" in d:
                cache_hit = d["cache_hit"]
    if final is None:
        raise RuntimeError("final answer 이벤트 없음")
    ans = final.get("answer") or {}
    return {
        "answer": ans.get("text"),
        "citations": ans.get("citations") or ans.get("sources"),
        "decision": (decision or {}).get("decision"),
        "cache_hit": cache_hit,
        "elapsed_s": round(time.time() - t0, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-file", type=Path,
                    default=Path("config/grant-rules-answer-eval.jsonl"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="gemini-3.5-flash-lite",
                    help="상대 UI 기본값(쾌속). 성실/심층은 별도 샘플링으로만")
    ap.add_argument("--sleep", type=float, default=30.0,
                    help="질의 간 간격(초). 부하 예의상 30 미만으로 낮추지 말 것")
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()

    # 답변이 실제로 있는 레코드만 완료로 인정 — 네트워크 단절 등으로 남은
    # error 레코드는 재실행 시 다시 수집한다.
    done = set()
    if args.out.exists():
        for line in args.out.open():
            try:
                rec = json.loads(line)
                if rec.get("answer"):
                    done.add(rec["id"])
            except (json.JSONDecodeError, KeyError):
                pass

    rows = [json.loads(l) for l in args.eval_file.open()]
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(todo)}/{len(rows)}문항 수집 시작 (간격 {args.sleep}s)")

    with args.out.open("a", encoding="utf-8") as out:
        for i, r in enumerate(todo):
            rec = {"id": r["id"], "query": r["query"], "model": args.model,
                   "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            for attempt in (1, 2):  # 재시도 1회
                try:
                    rec.update(ask(r["query"], args.model, args.timeout))
                    break
                except Exception as e:  # noqa: BLE001 — 기록 후 진행
                    rec["error"] = f"attempt{attempt}: {e}"
                    if attempt == 1:
                        time.sleep(args.sleep)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            status = "ok" if rec.get("answer") else "FAIL"
            print(f"[{i+1}/{len(todo)}] {r['id']} {status} "
                  f"{rec.get('elapsed_s', '-')}s cache={rec.get('cache_hit')}")
            if i < len(todo) - 1:
                time.sleep(args.sleep)
    print("완료")


if __name__ == "__main__":
    main()
