#!/usr/bin/env python3
"""Build a self-contained visual review board for paired service runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record["id"]): record for record in records}


def script_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<" + "\\/")


def build_payload(
    summary_path: Path,
    cases_path: Path,
    runs_a: list[Path],
    runs_b: list[Path],
) -> dict[str, Any]:
    summary = load_json(summary_path)
    cases = by_id(load_jsonl(cases_path))
    records_a = [by_id(load_jsonl(path)) for path in runs_a]
    records_b = [by_id(load_jsonl(path)) for path in runs_b]
    labels = summary.get("labels") or {"a": "A", "b": "B"}
    if not isinstance(labels, dict):
        raise ValueError("summary.labels must be an object")
    label_a = labels.get("a")
    label_b = labels.get("b")
    if not isinstance(label_a, str) or not label_a.strip():
        raise ValueError("summary.labels.a must be a non-empty string")
    if not isinstance(label_b, str) or not label_b.strip():
        raise ValueError("summary.labels.b must be a non-empty string")

    questions: list[dict[str, Any]] = []
    for row in summary["questions"]:
        case_id = str(row["id"])
        case = cases[case_id]
        a_rows = [run[case_id] for run in records_a]
        b_rows = [run[case_id] for run in records_b]
        questions.append(
            {
                **row,
                "query": case.get("query"),
                "reference": case.get("reference"),
                "answers_a": [record.get("answer") for record in a_rows],
                "answers_b": [record.get("answer") for record in b_rows],
                "reasons_a": [
                    (record.get("judge") or {}).get("reason") for record in a_rows
                ],
                "reasons_b": [
                    (record.get("judge") or {}).get("reason") for record in b_rows
                ],
                "sources_a": a_rows[0].get("sources") or [],
                "sources_b": b_rows[0].get("sources") or [],
            }
        )
    return {
        "generated_from": {
            "summary": str(summary_path),
            "cases": str(cases_path),
            "runs_a": [str(path) for path in runs_a],
            "runs_b": [str(path) for path in runs_b],
        },
        "summary": {
            "labels": {"a": label_a, "b": label_b},
            "a": summary["a"],
            "b": summary["b"],
            "comparison": summary["comparison"],
        },
        "questions": questions,
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>서비스 45문항 A/B 직접 비교</title>
<style>
:root{--ink:#15231f;--muted:#5b6d67;--paper:#f5f7f2;--card:#fff;--line:#d8e0da;--a:#476b5a;--b:#136f63;--up:#087a52;--down:#b13b3b;--same:#66736e;--soft:#eaf1ec}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 ui-sans-serif,-apple-system,BlinkMacSystemFont,"Noto Sans KR",sans-serif}
header{padding:42px max(24px,calc((100vw - 1240px)/2));background:#183f37;color:#fff}h1{margin:0 0 8px;font-size:clamp(28px,4vw,44px);letter-spacing:-.04em}header p{margin:0;color:#d7e8e1;max-width:850px}.wrap{max-width:1240px;margin:auto;padding:24px}.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.metric{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px}.metric .label{color:var(--muted);font-size:13px}.metric strong{display:block;margin-top:4px;font-size:25px}.metric small{color:var(--muted)}.warning{margin:16px 0;padding:14px 16px;background:#fff7e5;border:1px solid #e8c87c;border-radius:12px}.toolbar{position:sticky;top:0;z-index:4;display:flex;gap:10px;flex-wrap:wrap;margin:22px 0;padding:12px;background:rgba(245,247,242,.94);backdrop-filter:blur(10px);border-bottom:1px solid var(--line)}input,select{min-height:42px;border:1px solid #b9c6bf;border-radius:10px;background:#fff;padding:0 12px;font:inherit}input{flex:1;min-width:260px}.count{margin-left:auto;align-self:center;color:var(--muted)}.case{background:var(--card);border:1px solid var(--line);border-left:6px solid var(--same);border-radius:14px;margin:12px 0;padding:18px}.case.up{border-left-color:var(--up)}.case.down{border-left-color:var(--down)}.topline{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.id{font-weight:750}.badge{font-size:12px;padding:3px 8px;border-radius:999px;background:var(--soft);color:#355348}.badge.up{background:#e3f5eb;color:#086b48}.badge.down{background:#fde9e7;color:#9b2929}.badge.same{background:#edf0ef;color:#5e6965}.question{font-size:19px;font-weight:700;letter-spacing:-.02em;margin:10px 0}.scoreline{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:12px;margin:12px 0}.score{padding:10px 12px;border-radius:10px;background:#f2f5f3}.score b{font-size:20px}.arrow{color:var(--muted)}.flags{display:flex;gap:8px;flex-wrap:wrap;color:var(--muted);font-size:13px}details{margin-top:12px;border-top:1px solid var(--line);padding-top:12px}summary{cursor:pointer;font-weight:700}.reference{background:#f6f2e8;border-radius:10px;padding:12px;margin:12px 0}.answers{display:grid;grid-template-columns:1fr 1fr;gap:14px}.answer-col h3{margin:0 0 8px}.run{border:1px solid var(--line);border-radius:10px;padding:12px;margin:8px 0}.run .run-head{display:flex;justify-content:space-between;font-weight:700}.answer{white-space:pre-wrap;margin:8px 0}.reason{color:var(--muted);font-size:13px;border-top:1px dashed var(--line);padding-top:7px}.sources{font-size:12px;color:var(--muted)}.sources li{margin:3px 0}@media(max-width:760px){.metrics{grid-template-columns:1fr 1fr}.answers{grid-template-columns:1fr}.scoreline{grid-template-columns:1fr}.arrow{display:none}.count{width:100%;margin-left:0}}
</style>
</head>
<body>
<header><h1>서비스 45문항 A/B 직접 비교</h1><p id="subtitle"></p></header>
<main class="wrap">
  <section id="metrics" class="metrics"></section>
  <div class="warning"><strong>해석 주의:</strong> <span id="warning-text"></span></div>
  <div class="toolbar">
    <input id="search" placeholder="질문·ID·기준답안 검색">
    <select id="outcome"><option value="all">전체 결과</option><option value="down">악화만</option><option value="up">개선만</option><option value="same">동일만</option></select>
    <select id="category"><option value="all">전체 영역</option></select>
    <span id="count" class="count"></span>
  </div>
  <section id="cases"></section>
</main>
<script>const DATA=__PAYLOAD__;
const esc=v=>String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const LABEL_A=DATA.summary.labels.a,LABEL_B=DATA.summary.labels.b;
const pct=v=>(v*100).toFixed(1)+"%";const n=v=>Number(v).toFixed(3);const outcome=q=>q.delta_b_minus_a>0?"up":q.delta_b_minus_a<0?"down":"same";
function metric(label,a,b,fmt=n){return `<article class="metric"><span class="label">${label}</span><strong>${fmt(a)} → ${fmt(b)}</strong><small>Δ ${(b-a)>=0?"+":""}${fmt(b-a)}</small></article>`}
function renderMetrics(){const a=DATA.summary.a,b=DATA.summary.b,c=DATA.summary.comparison;document.querySelector("#metrics").innerHTML=metric("문서 Hit@5",a.retrieval.hit_rate,b.retrieval.hit_rate,pct)+metric("MRR",a.retrieval.mrr,b.retrieval.mrr)+metric("Any-Gold-Chunk@8 (DEV)",a.retrieval.evidence_hit_rate,b.retrieval.evidence_hit_rate,pct)+metric("생성 n=3 평균",a.mean,b.mean)+`<article class="metric"><span class="label">문항 승/무/패</span><strong>${c.wins_b} / ${c.ties} / ${c.losses_b}</strong><small>${esc(LABEL_B)} 기준</small></article>`}
function sourceList(rows){const seen=new Set;return `<ul class="sources">${rows.filter(x=>{const k=(x.source_title||"")+"|"+(x.chunk_id||"");if(seen.has(k))return false;seen.add(k);return true}).map(x=>`<li>${esc(x.source_title||"제목 없음")} · ${esc(x.chunk_id||"")}</li>`).join("")}</ul>`}
function runs(q,key){const answers=q["answers_"+key],scores=q["scores_"+key],reasons=q["reasons_"+key];return answers.map((answer,i)=>`<article class="run"><div class="run-head"><span>run ${i+1}</span><span>${scores[i]}점</span></div><div class="answer">${esc(answer)}</div><div class="reason">${esc(reasons[i])}</div></article>`).join("")}
function caseHtml(q){const o=outcome(q),delta=q.delta_b_minus_a;return `<article class="case ${o}"><div class="topline"><span class="id">${esc(q.id)}</span><span class="badge">${esc(q.category)}</span>${q.role?`<span class="badge">${esc(q.role)}</span>`:""}<span class="badge ${o}">${o==="up"?"개선":o==="down"?"악화":"동일"} ${delta>0?"+":""}${delta.toFixed(2)}</span></div><div class="question">${esc(q.query)}</div><div class="scoreline"><div class="score"><small>${esc(LABEL_A)}</small><br><b>${q.mean_a.toFixed(2)}</b> <span>${esc(JSON.stringify(q.scores_a))}</span></div><span class="arrow">→</span><div class="score"><small>${esc(LABEL_B)}</small><br><b>${q.mean_b.toFixed(2)}</b> <span>${esc(JSON.stringify(q.scores_b))}</span></div></div><div class="flags"><span>문서 ${q.retrieval_hit_a?"✓":"✕"}→${q.retrieval_hit_b?"✓":"✕"}</span><span>순위 ${q.rank_a??"-"}→${q.rank_b??"-"}</span><span>근거 ${q.evidence_hit_a?"✓":"✕"}→${q.evidence_hit_b?"✓":"✕"}</span><span>sources ${q.sources_changed?"변경":"동일"}</span></div><details><summary>기준답안·3회 답변·출처 펼치기</summary><div class="reference"><strong>기준답안</strong><br>${esc(q.reference)}</div><div class="answers"><section class="answer-col"><h3>${esc(LABEL_A)}</h3>${runs(q,"a")}${sourceList(q.sources_a)}</section><section class="answer-col"><h3>${esc(LABEL_B)}</h3>${runs(q,"b")}${sourceList(q.sources_b)}</section></div></details></article>`}
function render(){const term=document.querySelector("#search").value.trim().toLowerCase(),out=document.querySelector("#outcome").value,cat=document.querySelector("#category").value;const rows=DATA.questions.filter(q=>(out==="all"||outcome(q)===out)&&(cat==="all"||q.category===cat)&&(!term||[q.id,q.query,q.reference].join(" ").toLowerCase().includes(term)));document.querySelector("#count").textContent=`${rows.length} / ${DATA.questions.length}문항`;document.querySelector("#cases").innerHTML=rows.map(caseHtml).join("")}
const ci=DATA.summary.comparison.bootstrap.ci95,includesZero=ci[0]<=0&&ci[1]>=0;document.querySelector("#subtitle").textContent=`${LABEL_A}와 ${LABEL_B}의 반복 답변, Judge 점수, 검색 문서·DEV gold chunk를 문항별로 비교합니다. 평균보다 악화 문항과 근거 변화부터 검토하세요.`;document.querySelector("#warning-text").textContent=`생성 평균 차이의 family-cluster bootstrap 95% CI가 0을 ${includesZero?"포함합니다":"포함하지 않습니다"}. 이 화면은 DEV 분석이며 Judge 점수는 정답률이 아닌 0–2점 rubric 결과입니다.`;renderMetrics();[...new Set(DATA.questions.map(q=>q.category))].sort().forEach(c=>document.querySelector("#category").insertAdjacentHTML("beforeend",`<option value="${esc(c)}">${esc(c)}</option>`));["search","outcome","category"].forEach(id=>document.querySelector("#"+id).addEventListener(id==="search"?"input":"change",render));render();
</script>
</body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--runs-a", type=Path, nargs="+", required=True)
    parser.add_argument("--runs-b", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"immutable output path already exists: {args.out}")
    payload = build_payload(args.summary, args.cases, args.runs_a, args.runs_b)
    output = HTML_TEMPLATE.replace("__PAYLOAD__", script_json(payload))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        handle.write(output)
    print(f"wrote {args.out} ({len(payload['questions'])} questions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
