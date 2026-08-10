"""상대 챗봇(yunju.work) 답변을 우리와 동일한 judge로 채점.

비교 조건 통일이 핵심: `evaluate_grant_generation.py`의 JUDGE_PROMPT와
call_gemini_judge를 그대로 import해서 쓴다. 프롬프트·온도·모델이 우리
G-esnow 채점과 완전히 같아야 점수를 나란히 놓을 수 있다.

append 방식. 사용 예 (3.1 트랙으로 통일):
  GEMINI_MODEL=gemini-3.1-flash-lite python3 scripts/judge_rival_answers.py \
      --answers processed/eval/20260806-rival-answers.jsonl \
      --out processed/eval/20260806-rival-judged.jsonl --sleep 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate_grant_generation import call_gemini_judge, load_env  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", type=Path, required=True)
    ap.add_argument("--eval-file", type=Path,
                    default=Path("config/grant-rules-answer-eval.jsonl"))
    ap.add_argument("--env-file", type=Path, default=Path(".env"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sleep", type=float, default=4.0)
    args = ap.parse_args()

    load_env(args.env_file)

    refs = {r["id"]: r for r in map(json.loads, args.eval_file.open())}
    answers = {r["id"]: r for r in map(json.loads, args.answers.open())
               if r.get("answer")}

    done = set()
    if args.out.exists():
        for line in args.out.open():
            try:
                rec = json.loads(line)
                if (rec.get("judge") or {}).get("score") is not None:
                    done.add(rec["id"])
            except (json.JSONDecodeError, KeyError):
                pass

    todo = [i for i in answers if i not in done]
    print(f"{len(todo)}문항 채점 (완료 {len(done)}, 답변 없음 {len(refs) - len(answers)})")

    with args.out.open("a", encoding="utf-8") as out:
        for n, qid in enumerate(sorted(todo)):
            a, r = answers[qid], refs[qid]
            judge = call_gemini_judge(r["query"], r["reference_answer"], a["answer"])
            rec = {"id": qid, "judge": judge, "model": a.get("model"),
                   "cache_hit": a.get("cache_hit"), "elapsed_s": a.get("elapsed_s")}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{n+1}/{len(todo)}] {qid} score={judge.get('score')}")
            time.sleep(args.sleep)
    print("완료")


if __name__ == "__main__":
    main()
