"""Cross-encoder 리랭커 — BM25 후보를 (질의, 본문) 쌍 채점으로 재정렬한다.

BM25는 용어 커버리지로 순위를 정하므로(2026-08-04 분석) 장황한 조문이 금액 든
짧은 표를 이긴다. cross-encoder는 질의와 본문을 한 입력으로 함께 읽어 관련도를
직접 채점하므로, 이 편향과 독립적인 2차 판정을 제공한다.

`sentence-transformers`가 필요하므로 embedding venv에서만 import할 것.
모델은 첫 채점 호출 때 지연 로딩한다(bge-reranker-v2-m3 약 2.3GB 다운로드).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

# Qwen3-Reranker 계열은 seq-cls 변환판이라도 원본 학습에 쓰인 채팅 템플릿을
# 그대로 입혀야 한다. 날것 (query, doc) 쌍을 넣으면 점수 보정이 무너진다
# (2026-08-05 실측: 템플릿 없이 context recall 0.4522 → 0.1386으로 붕괴).
_QWEN3_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements "
    'based on the Query and the Instruct provided. Note that the answer can '
    'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
_QWEN3_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_QWEN3_DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


class CrossEncoderReranker:
    """dict 행 리스트를 (query, text) 쌍 점수로 재정렬한다.

    각 행에 `cross_encoder_score`를 남겨 측정 조건을 추적할 수 있게 하고,
    동점은 안정 정렬로 원래 BM25 순위를 유지한다.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        max_length: int = 1024,
        batch_size: int = 8,
        device: str | None = None,
        instruction: str | None = None,
        doc_char_limit: int = 1500,
    ) -> None:
        self.model_name = model_name
        self._max_length = max_length
        self._batch_size = batch_size
        self._device = device
        self._instruction = instruction or _QWEN3_DEFAULT_INSTRUCTION
        # 템플릿 꼬리(<|im_end|>...<think>)가 문서 뒤에 붙으므로, 토크나이저
        # 절단에 꼬리가 잘려나가지 않도록 문서를 먼저 글자 수로 자른다.
        self._doc_char_limit = doc_char_limit
        self._is_qwen3 = "qwen3-reranker" in model_name.lower()
        if self._is_qwen3:
            # 템플릿(~150토큰) + 한국어 1,500자 문서가 1024를 넘을 수 있어
            # 꼬리 절단을 피하려면 여유가 필요하다.
            self._max_length = max(self._max_length, 2048)
        self._model: Any = None

    def _format_pair(self, query: str, document: str) -> list[str]:
        if not self._is_qwen3:
            return [query, document]
        document = document[: self._doc_char_limit]
        return [
            f"{_QWEN3_PREFIX}<Instruct>: {self._instruction}\n<Query>: {query}\n",
            f"<Document>: {document}{_QWEN3_SUFFIX}",
        ]

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name,
                max_length=self._max_length,
                device=self._device,
            )
        return self._model

    def rerank(
        self, query: str, rows: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        out = [dict(row) for row in rows]
        if not out:
            return out
        pairs = [
            self._format_pair(query, (row.get("text") or row.get("preview") or ""))
            for row in out
        ]
        scores = self._load().predict(
            pairs, batch_size=self._batch_size, show_progress_bar=False
        )
        for row, score in zip(out, scores):
            row["cross_encoder_score"] = float(score)
        # 안정 정렬: 점수가 같으면 BM25 순위(입력 순서)를 그대로 둔다.
        out.sort(key=lambda row: -row["cross_encoder_score"])
        return out
