# 2026-09-14 제출 근거 패키지

이 디렉터리는 Git에서 제외되는 `processed/` 평가 원본 중 최종 보고서에 직접
필요한 작은 산출물을 추적 가능한 형태로 보존한다. 현재 파일은 2026-08-31
historical 생성 결과와 2026-09-01 로컬 검색·구성요소 검증을 함께 보존한 중간
패키지다. 정본 프로토콜에 따라 2026-09-09 결과 동결 때 `final/` 정본을 추가한다.

## 서비스 baseline 대 검색 튜닝 n=3

- `20260831-svc-gen-baseline-run{1,2,3}.jsonl`
- `20260831-svc-gen-tuned-run{1,2,3}.jsonl`
- `service-ab-summary.json`
- `service-ab-per-question.csv`
- `interim-service-n3-manifest.json`
- `service-ab-summary-metric-corrected-20260901.json`
- `service-ab-per-question-metric-corrected-20260901.csv`
- `service-ab-review-metric-corrected-20260901.html`

엄격 집계 결과:

| 지표 | Baseline | 검색 튜닝 | 차이 |
|---|---:|---:|---:|
| 문서 Hit@5 | 0.644 | 0.889 | +0.245 |
| MRR | 0.494 | 0.723 | +0.229 |
| Any-Gold-Chunk@5 | 0.467 | 0.600 | +0.133 |
| GoldChunkRecall@5 | 0.315 | 0.411 | +0.096 |
| Any-Gold-Chunk@8 | 0.511 | 0.622 | +0.111 |
| GoldChunkRecall@8 | 0.359 | 0.456 | +0.096 |
| 생성 n=3 평균 | 0.7185 | 0.7481 | +0.0296 |

과거 JSONL의 `evidence_hit`은 실제 `/chat` context 8개 전체를 검사했으므로
@5가 아니라 Any-Gold-Chunk@8이다. 정정 집계는
`service-ab-summary-metric-corrected-20260901.json`과 대응 CSV를 정본으로
사용한다. DEV gold는 flat chunk ID 목록이며 atomic evidence metric이 아니다.

문항별 생성 결과는 10승·29무·6패이며, 역할 변형을 기본 질문과 같은 family로
묶은 paired cluster bootstrap 10,000회의 평균 차이 95% 신뢰구간은
`[-0.1407, 0.1951]`이다. 따라서 검색은 동일 45문항 DEV에서 개선 지표가
재현됐다고만 표현하며, 생성 개선은 통계적으로 확정하지 않는다.

`interim-service-n3-manifest.json`에는 생성 당시 `search_api.py`와
`bm25_search.py`가 uncommitted였고 이후 후처리·일반 제목 규칙이 수정되어
정확한 producing-code snapshot을 복구할 수 없다는 한계를 명시했다. 여섯
JSONL 자체와 인덱스·평가셋 hash는 보존돼 있다.

## 2026-08-31/09-01 snapshot의 외부 전송 없는 검증

- `20260831-svc-retrieval-current-smoke.jsonl`
- `postprocessor-offline-ab.json`
- `current-code-audit-manifest.json`

해당 snapshot의 로컬 extractive 검색 재검증은 Hit@5 40/45, MRR 0.723,
Any-Gold-Chunk@8 28/45로 n=3 생성 당시 검색 수치와 동일했다.

날짜·짧은 사실 후처리 회귀는 각 평가 문항의 공식 기준답안을 known-good
draft로 사용했으며 외부 API를 호출하지 않았다. Critical-value 보존 평균은
0.7426에서 1.0000으로 증가했고, 20문항 개선·25문항 동일·악화 0문항이었다.
이는 splitter 단위 안전성이지 실제 모델 답변 품질 점수가 아니다.

`current-code-audit-manifest.json`은 이름과 달리 2026-08-31/09-01 당시 파일
hash를 기록한 중간 manifest다. 이후 evaluator, `search_api.py`, 테스트가
변경됐으므로 현재 working tree의 final manifest로 사용하지 않는다.

## cap2 대 cap4 retrieval DEV 직접 비교

- 원본: `cap2-service-extractive-dev45-20260901.jsonl`,
  `cap4-service-extractive-dev45-20260901.jsonl`
- 집계: `cap2-vs-cap4-retrieval-dev45.json`, `.csv`
- 직접 비교: `cap2-vs-cap4-retrieval-dev45.html`,
  `cap2-vs-cap4-retrieval-dev45-preview.png`

cap2→cap4에서 문서 Hit@5와 MRR은 40/45, 0.723으로 같았다.
Any-Gold-Chunk@8은 28→30, All-Gold-Chunks@8은 13→16,
GoldChunkRecall@8은 0.456→0.500으로 늘었다. 반면 p50은
142.4→166.9ms, 평균 unique document는 6.13→5.04였고 context 구성 또는
순서가 31/45에서 바뀌었다. 단일 순차 latency run과 extractive 출력 비교이므로
cap4는 실제 paired generation 전까지 후보이며 기본값은 cap2다.

## Claim grounding guard microbenchmark

- 평가셋: `config/pnu-grounding-adversarial-eval.jsonl` (77건)
- 결과: `grounding-guard-ablation.json`, `.csv`

허용/금지, 비교연산자와 복합 금액·수량, 인상/인하/동결, 연도·주체·값 결합,
부정·검토·예정 modality, 표 행, 날짜 시작·종료 경계를 다룬 합성·공지형
microbenchmark에서 guard off의 label 정확도는 0.4416, label+reason joint
정확도는 0.4156, false positive는 43건이었다. Guard on은 두 정확도 1.0000,
false positive/negative 0건이었다. 이 수치는 claim attribution 규칙의 표적
회귀 결과이며 검색·LLM 생성·실서비스 성능이 아니다. 일반 NLI가 아니므로
의무·필요 관계(예: `제출해야 한다`와 `제출할 필요가 없다`)는 범위 밖이다.

## Working-tree 전체 게이트

`local-full-check-20260901.log`에는 현재 작업 트리에서 Python 테스트 348건 실행
(342 pass·6 skip), ESLint·TypeScript·Vite production build 통과가 기록돼
있다. 이는 clean final commit 결과가 아니라 2026-09-01 중간 snapshot이다.

## 재현 명령

```bash
PYTHONDONTWRITEBYTECODE=1 .parser-tools/venvs/core/bin/python \
  scripts/analyze_service_ab.py \
  --legacy-inline \
  --runs-a evidence/20260914/20260831-svc-gen-baseline-run1.jsonl \
           evidence/20260914/20260831-svc-gen-baseline-run2.jsonl \
           evidence/20260914/20260831-svc-gen-baseline-run3.jsonl \
  --runs-b evidence/20260914/20260831-svc-gen-tuned-run1.jsonl \
           evidence/20260914/20260831-svc-gen-tuned-run2.jsonl \
           evidence/20260914/20260831-svc-gen-tuned-run3.jsonl \
  --label-a baseline --label-b service-tuning \
  --bootstrap 10000 \
  --json-out evidence/20260914/service-ab-summary-metric-corrected-20260901.json \
  --csv-out evidence/20260914/service-ab-per-question-metric-corrected-20260901.csv

PYTHONDONTWRITEBYTECODE=1 .parser-tools/venvs/core/bin/python \
  scripts/build_service_ab_review.py \
  --summary evidence/20260914/service-ab-summary-metric-corrected-20260901.json \
  --cases config/pnu-service-answer-eval.jsonl \
  --runs-a evidence/20260914/20260831-svc-gen-baseline-run1.jsonl \
           evidence/20260914/20260831-svc-gen-baseline-run2.jsonl \
           evidence/20260914/20260831-svc-gen-baseline-run3.jsonl \
  --runs-b evidence/20260914/20260831-svc-gen-tuned-run1.jsonl \
           evidence/20260914/20260831-svc-gen-tuned-run2.jsonl \
           evidence/20260914/20260831-svc-gen-tuned-run3.jsonl \
  --out evidence/20260914/service-ab-review-metric-corrected-20260901.html

python3 scripts/analyze_service_retrieval_ab.py \
  --cases config/pnu-service-answer-eval.jsonl \
  --run-a evidence/20260914/cap2-service-extractive-dev45-20260901.jsonl \
  --run-b evidence/20260914/cap4-service-extractive-dev45-20260901.jsonl \
  --label-a cap2 --label-b cap4 \
  --json-out evidence/20260914/cap2-vs-cap4-retrieval-dev45.json \
  --csv-out evidence/20260914/cap2-vs-cap4-retrieval-dev45.csv \
  --html-out evidence/20260914/cap2-vs-cap4-retrieval-dev45.html

PYTHONDONTWRITEBYTECODE=1 .parser-tools/venvs/core/bin/python \
  scripts/evaluate_postprocessor_regression.py \
  --out evidence/20260914/postprocessor-offline-ab.json \
  --fail-on-regression

python3 scripts/evaluate_grounding_guard.py \
  --cases config/pnu-grounding-adversarial-eval.jsonl \
  --output-json evidence/20260914/grounding-guard-ablation.json \
  --output-csv evidence/20260914/grounding-guard-ablation.csv
```

## 아직 필요한 정본

- 신규 holdout 36문항과 parser audit 18문서 작성·교차 검수·hash 동결
- DEV 8조건 및 holdout 4조건 retrieval one-shot
- 생성 198개와 structured Judge 234회
- 후처리 수정의 실제 생성 표적 n=3 및 필요 시 전체 45문항 n=3
- cap4 대 cap2 실제 paired generation
- 고정 답변에 대한 judge 반복 일치도
- 경쟁 서비스 최신 재수집과 양방향 blind A/B
- 사람 2인의 63개 출력 blind 평가·adjudication
- clean final commit에서의 전체 회귀와
  `evidence/20260914/final/manifest.json`
- 최종 보고서 PDF와 SHA-256

2026-09-01 현재 위 신규 생성·Judge 실행은 부산대 검색 context의 외부 Gemini
전송에 대한 명시적 승인 전이라 수행하지 않았다. Historical n=3은 이미 보존된
별도 snapshot이며 현재 코드의 신규 결과로 간주하지 않는다.

최종 실행은 위 historical inline JSONL 형식을 재사용하지 않는다. 답변은
`evaluate_service_answers.py`, 판정은 `judge_service_answers.py`가 서로 다른
append-only JSONL에 저장하고 `answer_id`·`answer_sha256`으로 연결한다.
