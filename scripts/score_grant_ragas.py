#!/usr/bin/env python3
"""RAGAS context recall로 검색 컨텍스트를 채점한다.

`export_grant_contexts.py`가 만든 샘플을 읽어 기준답안의 각 주장이 검색된
컨텍스트로 뒷받침되는지 LLM이 판정한다(RAGAS `ContextRecall`). 자체 정규식
프록시와 달리 수치가 없는 답변도 채점되고, 부분 점수가 나온다.

판정 LLM은 Gemini의 OpenAI 호환 엔드포인트로 붙인다. 무료 등급 한도를 넘지
않도록 호출 간격을 강제하고(--rpm), 503/429에는 재시도 후 폴백 모델로 넘어간다.

append 방식이라 중단돼도 같은 --out으로 재실행하면 남은 문항부터 이어간다.

사용 예 (ragas venv에서 실행):
  .parser-tools/venvs/ragas/bin/python scripts/score_grant_ragas.py \
      --samples processed/eval/20260804-ctx-challenger.jsonl \
      --out processed/eval/20260804-ragas-challenger.jsonl --rpm 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path


def load_env(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


class RateLimiter:
    """무료 등급 RPM을 넘지 않도록 호출 간 최소 간격을 강제한다."""

    def __init__(self, rpm: float) -> None:
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        if self.interval <= 0:
            return
        async with self._lock:
            gap = time.monotonic() - self._last
            if gap < self.interval:
                await asyncio.sleep(self.interval - gap)
            self._last = time.monotonic()


async def score_one(metrics, limiter, sample, attempts: int) -> tuple[str, float, str | None]:
    """한 라운드에서 모든 모델을 한 번씩 시도한다.

    실패는 대부분 rate limit(429)이 아니라 특정 모델의 일시적 과부하(503)이고,
    그럴 때 다른 모델은 즉시 응답한다. 그래서 모델별로 재시도를 소진하는 대신
    라운드마다 모델을 한 바퀴 돌고, 그 라운드가 전부 실패했을 때만 백오프한다.

    반환: (사용 모델, 점수, 실패 사유).
    """

    last_error: str | None = None
    for attempt in range(attempts):
        for model, metric in metrics:
            await limiter.wait()
            try:
                result = await metric.ascore(
                    user_input=sample["user_input"],
                    retrieved_contexts=sample["retrieved_contexts"],
                    reference=sample["reference"],
                )
                return model, float(result.value), None
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
        await asyncio.sleep(3 * (attempt + 1))
    return "", float("nan"), last_error


async def run(args) -> None:
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import ContextRecall

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("GEMINI_API_KEY가 없습니다 (--env-file 확인)")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    models = [args.model] + [
        name.strip()
        for name in os.environ.get("GEMINI_FALLBACK_MODELS", "").split(",")
        if name.strip() and name.strip() != args.model
    ]
    metrics = [
        (name, ContextRecall(llm=llm_factory(name, provider="openai", client=client)))
        for name in models
    ]

    samples = [
        json.loads(line)
        for line in args.samples.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    done: set[str] = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["id"])
            except Exception:  # noqa: BLE001
                pass
    todo = [s for s in samples if s["id"] not in done]
    print(f"샘플 {len(samples)} / 이미 채점 {len(done)} / 남은 {len(todo)}  "
          f"(rpm={args.rpm}, model={args.model})")

    limiter = RateLimiter(args.rpm)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    # RateLimiter가 전역 호출 간격을 강제하므로 동시 실행을 올려도 한도를 넘지
    # 않는다. 동시성은 그 한도를 실제로 다 쓰게 만드는 역할만 한다.
    gate = asyncio.Semaphore(max(1, args.concurrency))
    write_lock = asyncio.Lock()
    completed = 0

    with args.out.open("a", encoding="utf-8") as sink:

        async def worker(sample) -> None:
            nonlocal completed
            async with gate:
                model, score, error = await score_one(
                    metrics, limiter, sample, args.attempts
                )
            record = {
                "id": sample["id"],
                "context_recall": None if score != score else score,
                "model": model or None,
                "error": error,
                "n_contexts": len(sample["retrieved_contexts"]),
                "section": (sample.get("meta") or {}).get("section"),
            }
            async with write_lock:
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()
                completed += 1
                shown = "실패" if record["context_recall"] is None else f"{score:.3f}"
                print(f"  [{completed:>2d}/{len(todo)}] {sample['id']:<12s} {shown}"
                      f"  ({time.time() - started:.0f}s)", flush=True)

        await asyncio.gather(*(worker(sample) for sample in todo))

    elapsed = time.time() - started
    if todo:
        print(f"\n처리량: {len(todo)}문항 / {elapsed:.0f}초 = "
              f"{elapsed / len(todo):.1f}초․문항, 실효 {len(todo) / elapsed * 60:.1f} RPM "
              f"(상한 {args.rpm})")

    rows = [
        json.loads(line)
        for line in args.out.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scored = [r["context_recall"] for r in rows if r.get("context_recall") is not None]
    failed = len(rows) - len(scored)
    print(f"\n=== {args.samples.name} ===")
    print(f"채점 {len(scored)}문항 / 실패 {failed}")
    if scored:
        print(f"  context recall 평균 : {statistics.mean(scored):.4f}")
        print(f"  중앙값              : {statistics.median(scored):.4f}")
        print(f"  1.0 (완전 뒷받침)   : {sum(1 for v in scored if v >= 0.999)}문항")
        print(f"  0.0 (근거 없음)     : {sum(1 for v in scored if v <= 0.001)}문항")
    print(f"wrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--env-file", type=Path, default=Path(".env"))
    ap.add_argument("--model", default=None, help="기본값은 .env의 GEMINI_MODEL")
    ap.add_argument("--rpm", type=float, default=10.0,
                    help="분당 호출 상한. 무료 등급 여유를 두고 낮게 잡는다")
    ap.add_argument("--concurrency", type=int, default=6,
                    help="동시 실행 수. rpm 상한은 RateLimiter가 따로 지키므로 "
                         "이 값은 그 상한을 실제로 채우는 역할만 한다")
    ap.add_argument("--attempts", type=int, default=3,
                    help="라운드 수. 라운드마다 모델 후보를 한 바퀴 시도한다")
    args = ap.parse_args()

    load_env(args.env_file)
    if not args.model:
        args.model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
