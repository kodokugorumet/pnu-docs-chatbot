#!/usr/bin/env python3
"""53문항 연구비 규정 평가셋의 답변 생성 + 기준답안 대비 채점.

라우팅 적용 BM25 검색 → Gemini 생성 → Gemini 채점(LLM-as-judge, 0~2점).
기준답안이 있으므로 채점 프롬프트에 기준답안의 핵심 사실(금액·조건·절차)
포함 여부를 명시적으로 묻는다. corpus 미수집 문항도 생성은 수행해
"모른다고 답하는지"(정직성)를 함께 기록한다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bm25_search import search_bm25_candidates, search_index  # noqa: E402
from rag.grant_router import route, filter_hits  # noqa: E402
from rag import generators  # noqa: E402


def load_env(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


JUDGE_PROMPT = """당신은 연구비 규정 상담 답변의 채점자입니다.

[질문]
{question}

[기준답안] (부산대학교 연구비관리부 공식 답변)
{reference}

[채점 대상 답변]
{answer}

기준답안의 핵심 사실(금액·비율·기한 등 수치, 허용/불허 조건, 필요 절차·서류)이
채점 대상 답변에 정확히 담겼는지 평가하세요. 기준답안에 없는 내용이 추가된 것은
사실과 모순되지 않는 한 감점하지 않습니다.

점수 기준:
- 2: 핵심 사실이 모두 정확함 (수치·조건·결론 일치)
- 1: 결론 방향은 맞으나 핵심 수치·조건 일부 누락 또는 부정확
- 0: 결론이 틀렸거나, 근거 없는 수치 제시, 또는 무관한 답변

JSON 한 줄로만 답하세요: {{"score": 0|1|2, "reason": "한 문장 근거"}}"""


def call_gemini_judge(question: str, reference: str, answer: str, retries: int = 4) -> dict:
    import urllib.request

    api_key = os.environ["GEMINI_API_KEY"]
    model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    )
    prompt = JUDGE_PROMPT.format(question=question, reference=reference, answer=answer)
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 200},
    }).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, body,
                headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            )
            payload = json.load(urllib.request.urlopen(req, timeout=45))
            text = payload["candidates"][0]["content"]["parts"][0]["text"].strip()
            if text.startswith("```"):
                text = text.strip("`\n")
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text.strip())
        except Exception as exc:  # noqa: BLE001 - 재시도 후 기록
            if attempt == retries - 1:
                return {"score": None, "reason": f"judge_error: {exc}"}
            time.sleep(5 * (attempt + 1))
    return {"score": None, "reason": "unreachable"}


def litm_order(hits: list[dict]) -> list[dict]:
    """문서 단위 Lost-in-the-Middle 가장자리 배치.

    입력은 검색 순위순. 문서군은 첫 등장 순서를 강도로 삼고, 같은 문서 안은
    원래 순서(원문 조문 순서)를 유지한다(D24). 배치는 1,3,5,…,6,4,2 —
    가장 강한 군을 맨 앞, 다음 강한 군을 맨 뒤, 약한 군을 가운데로.
    """
    groups: dict = {}
    order = []
    for h in hits:
        key = h.get("document_id") or h.get("chunk_id") or id(h)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(h)
    front, back = [], []
    for pos, key in enumerate(order):
        (front if pos % 2 == 0 else back).append(groups[key])
    return [h for g in front + back[::-1] for h in g]


def load_context_map(path: Path) -> dict:
    """export_grant_contexts.py 산출물을 id → (contexts, sources, scope)로 로드."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        meta = rec.get("meta") or {}
        out[rec["id"]] = (
            rec["retrieved_contexts"],
            meta.get("sources") or [],
            meta.get("scope"),
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=Path, default=Path("processed/index/grant-rules-20260803.sqlite"))
    ap.add_argument("--eval-file", type=Path, default=Path("config/grant-rules-answer-eval.jsonl"))
    ap.add_argument("--env-file", type=Path, default=Path(".env"))
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N문항만 (0=전체)")
    ap.add_argument("--out", type=Path, default=Path("processed/eval/20260803-grant-generation.jsonl"))
    ap.add_argument("--sleep", type=float, default=2.0, help="문항 간 대기(초)")
    ap.add_argument("--contexts", type=Path, default=None,
                    help="export_grant_contexts.py 산출물 주입 — 내부 검색을 건너뛰고 "
                         "검색 1회 → 생성·채점 N회 구조로 만든다 (RAGAS 트랙과 동일)")
    ap.add_argument("--litm-order", action="store_true",
                    help="컨텍스트를 문서 단위 가장자리 배치로 재정렬 (D24)")
    ap.add_argument("--rejudge", action="store_true",
                    help="--out 파일에서 judge 실패(score=None) 레코드만 재채점. "
                         "답변은 보존돼 있으므로 문항당 judge 1호출로 복구된다 "
                         "(2026-08-06 429 장애 대응). 생성 실패 레코드는 파일에서 "
                         "제거해 일반 재실행이 다시 생성하게 한다")
    args = ap.parse_args()

    load_env(args.env_file)
    items = [json.loads(l) for l in args.eval_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        items = items[: args.limit]

    ctx_map = load_context_map(args.contexts) if args.contexts else None

    if args.rejudge:
        ref = {it["id"]: it for it in items}
        recs = [json.loads(l) for l in args.out.read_text(encoding="utf-8").splitlines() if l.strip()]
        kept, fixed, dropped = [], 0, 0
        for r in recs:
            broken_gen = r.get("generation", {}).get("error") or not r.get("answer")
            if broken_gen:
                dropped += 1  # 일반 재실행이 다시 생성하도록 제거
                continue
            if r["judge"].get("score") is None and r["id"] in ref:
                it = ref[r["id"]]
                r["judge"] = call_gemini_judge(
                    it["query"], it["reference_answer"], r["answer"]
                )
                if r["judge"].get("score") is not None:
                    fixed += 1
                time.sleep(args.sleep)
            kept.append(r)
        tmp = args.out.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as sink:
            for r in kept:
                sink.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(args.out)
        graded = [r["judge"]["score"] for r in kept if isinstance(r["judge"].get("score"), int)]
        print(f"rejudge: 복구 {fixed}, 생성실패 제거 {dropped}, 채점 완료 {len(graded)}/{len(kept)}")
        if graded:
            print(f"mean={sum(graded)/len(graded):.3f}, 2점={graded.count(2)}, "
                  f"1점={graded.count(1)}, 0점={graded.count(0)}")
        return

    done_ids = set()
    if args.out.exists():
        for l in args.out.read_text(encoding="utf-8").splitlines():
            try:
                done_ids.add(json.loads(l)["id"])
            except Exception:  # noqa: BLE001
                pass

    scores = []
    with args.out.open("a", encoding="utf-8") as sink:
        for it in items:
            if it["id"] in done_ids:
                continue
            if ctx_map is not None:
                contexts, sources, scope = ctx_map[it["id"]]
                hits = [
                    {
                        "text": ctx,
                        "preview": ctx,
                        "institution": src.get("institution"),
                        "file_name": src.get("file_name"),
                        "chunk_id": src.get("chunk_id"),
                        "document_id": src.get("document_id"),
                    }
                    for ctx, src in zip(contexts, sources)
                ]
                if args.litm_order:
                    hits = litm_order(hits)
            else:
                scope = route(it["query"])
                # 문서 커버리지는 다양성 검색으로 확보하고, 상위 문서의 추가 청크
                # (조문+별표 조합)는 원시 후보에서 보충한다.
                diverse = filter_hits(
                    search_index(args.index, it["query"], 30, None, include_text=True),
                    scope, 5, it["query"],
                )
                raw = search_bm25_candidates(args.index, it["query"], 60, None, include_text=True)
                raw = [h for h in raw if scope is None or (h.get("institution") or "") in scope]
                seen = {h.get("chunk_id") for h in diverse}
                top_docs = {h.get("document_id") for h in diverse[:3]}
                per_doc: dict = {}
                extra = []
                for h in raw:
                    doc = h.get("document_id")
                    if doc not in top_docs or h.get("chunk_id") in seen:
                        continue
                    if per_doc.get(doc, 0) >= 2:
                        continue
                    per_doc[doc] = per_doc.get(doc, 0) + 1
                    seen.add(h.get("chunk_id"))
                    extra.append(h)
                hits = (diverse + extra)[: args.top_k + 3]
                if args.litm_order:
                    hits = litm_order(hits)
            try:
                result = generators.generate(it["query"], hits, requested="gemini")
                answer = result.text
                gen_meta = result.metadata()
            except Exception as exc:  # noqa: BLE001
                answer = ""
                gen_meta = {"error": str(exc)}
            time.sleep(args.sleep)
            judge = call_gemini_judge(it["query"], it["reference_answer"], answer or "(생성 실패)")
            record = {
                "id": it["id"],
                "section": it["section"],
                "query": it["query"],
                "scope": scope,
                "retrieved": [
                    {"institution": h.get("institution"), "file": (h.get("file_name") or "")[:60]}
                    for h in hits
                ],
                "answer": answer,
                "generation": gen_meta,
                "judge": judge,
            }
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            s = judge.get("score")
            scores.append(s)
            print(f"{it['id']}: score={s} ({(judge.get('reason') or '')[:60]})")
            time.sleep(args.sleep)

    graded = [s for s in scores if isinstance(s, int)]
    if graded:
        print(f"\nthis run: {len(graded)} graded, mean={sum(graded)/len(graded):.2f}, "
              f"full(2)={graded.count(2)}, partial(1)={graded.count(1)}, fail(0)={graded.count(0)}")


if __name__ == "__main__":
    main()
