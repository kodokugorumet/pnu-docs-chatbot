"""Parent-child 확장 — 검색은 작은 청크로, LLM 컨텍스트는 섹션 전체로.

근거(2026-08-05): 남은 실패 16문항은 정답 문서는 찾았지만 그 안에서 엉뚱한
조각을 고른 문제이고, 근거가 조문+별표처럼 같은 섹션 안에 흩어져 있는 경우가
많다(8/4 여비 분석). 청킹이 표를 독립 청크로 떼어내므로(core/chunking.py),
같은 section_path 묶음을 부모로 쓰면 조문과 딸린 표가 재결합된다.

- 부모 = (document_id, section_path) 그룹을 chunk_index 순으로 병합
- section_path가 없는 청크(이 인덱스의 23.9%)는 chunk_index 이웃 창 폴백
- 같은 부모에 속한 히트는 하나로 합쳐진다(중복 제거) — 빈 슬롯만큼
  다른 근거가 살아남는 부수 효과
- 예산 2중: 부모당 `per_parent_chars`, 문항 전체 `total_chars`.
  판정자가 긴 컨텍스트에서 근거를 놓치는 것을 실측했으므로(8/4, 낙폭 0.403)
  기준선과 총량을 맞춰야 공정 비교가 된다(D19 함의 4). 히트 자신의 본문은
  예산과 무관하게 항상 포함한다 — 검색 결정 자체를 확장이 뒤집으면 안 된다.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def expand_hits(
    index_path: Path,
    hits: list[dict[str, Any]],
    *,
    per_parent_chars: int = 3000,
    total_chars: int = 13000,
    neighbor_window: int = 2,
) -> tuple[list[str], list[dict[str, Any]]]:
    """히트 순서를 유지하며 각 히트를 부모 컨텍스트로 넓힌다.

    반환: (컨텍스트 문자열 리스트, 히트별 확장 메타 리스트).
    이미 앞 부모에 포함된 히트는 건너뛰므로 리스트가 히트 수보다 짧을 수 있다.
    """

    con = sqlite3.connect(str(index_path))
    try:
        used: set[str] = set()
        contexts: list[str] = []
        meta: list[dict[str, Any]] = []
        running_total = 0

        for hit in hits:
            chunk_id = hit.get("chunk_id")
            if not chunk_id or chunk_id in used:
                continue
            row = con.execute(
                "SELECT document_id, chunk_index, section_path_json, text "
                "FROM chunks WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
            if row is None:
                # 인덱스 불일치 — 확장 없이 원문만 사용
                text = hit.get("text") or hit.get("preview") or ""
                contexts.append(text)
                meta.append({"chunk_id": chunk_id, "n_chunks": 1,
                             "chars": len(text), "mode": "missing"})
                running_total += len(text)
                continue
            document_id, chunk_index, section_path_json, own_text = row

            if section_path_json and section_path_json not in ("null", "[]"):
                siblings = con.execute(
                    "SELECT chunk_id, chunk_index, text FROM chunks "
                    "WHERE document_id = ? AND section_path_json = ? "
                    "ORDER BY chunk_index",
                    (document_id, section_path_json),
                ).fetchall()
                mode = "section"
            else:
                siblings = con.execute(
                    "SELECT chunk_id, chunk_index, text FROM chunks "
                    "WHERE document_id = ? AND chunk_index BETWEEN ? AND ? "
                    "ORDER BY chunk_index",
                    (document_id, chunk_index - neighbor_window,
                     chunk_index + neighbor_window),
                ).fetchall()
                mode = "neighbor"

            # 히트 본문은 무조건 포함하고, 남는 예산만큼 히트에 가까운
            # 형제부터 앞뒤 번갈아 붙인다(문서 순서는 최종 조립에서 복원).
            budget = min(
                per_parent_chars,
                max(len(own_text), total_chars - running_total),
            )
            anchor = next(
                (i for i, s in enumerate(siblings) if s[0] == chunk_id), 0
            )
            chosen = {anchor}
            size = len(siblings[anchor][2]) if siblings else len(own_text)
            offset = 1
            while True:
                added = False
                for pos in (anchor - offset, anchor + offset):
                    if 0 <= pos < len(siblings):
                        sib_id, _, sib_text = siblings[pos]
                        if sib_id in used or pos in chosen:
                            continue
                        if size + len(sib_text) > budget:
                            continue
                        chosen.add(pos)
                        size += len(sib_text)
                        added = True
                if not added:
                    break
                offset += 1

            ordered = sorted(chosen)
            parts = [siblings[i][2] for i in ordered] if siblings else [own_text]
            for i in ordered:
                used.add(siblings[i][0])
            used.add(chunk_id)

            text = "\n\n".join(parts)
            contexts.append(text)
            meta.append({
                "chunk_id": chunk_id,
                "n_chunks": len(ordered) if siblings else 1,
                "chars": len(text),
                "mode": mode,
            })
            running_total += len(text)

        return contexts, meta
    finally:
        con.close()
