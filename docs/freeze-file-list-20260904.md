# 2026-09-04 final evaluation code freeze 승인용 파일 목록

## 상태와 staging 원칙

- 기준: 최초 `git status --short --untracked-files=all`의 기존 130개 항목과 이 문서,
  이후 승인된 C2 실험 코드·테스트 4개.
- 승인된 staging 대상: **131개 경로**(수정 9, 신규 scripts 39, 신규 tests 34,
  신규 config 6, 신규 docs 15, 신규 evidence 28).
- holdout embargo 대상 4개는 내용 확인 없이 별도 보류한다. 사람 검수·sign-off가
  끝난 뒤 별도 명시 승인으로만 staging한다.
- `processed/`와 `.env`는 `.gitignore`가 각각 18행과 9행에서 제외한다.
  따라서 분석 결과 JSON/CSV와 비밀값은 이 freeze commit에 들어가지 않는다.
- `git add -A`, `git add .`, wildcard staging은 사용하지 않는다. 승인된 정확한
  경로만 `git add -- <path...>`로 올린다.

## 수정 파일 — 포함 권고 9개

| 경로 | 포함 근거 |
|---|---|
| `scripts/bm25_search.py` | 서비스 BM25 검색·평가 동작의 동결 대상 구현이다. |
| `scripts/evaluate_grant_generation.py` | 연구비 생성 평가의 재현성·오류 처리를 반영한다. |
| `scripts/rag/generators.py` | 생성 prompt·후처리의 동결 대상 구현이다. |
| `scripts/rag/role_router.py` | 역할 입력의 안전한 정규화·routing 변경을 포함한다. |
| `scripts/search_api.py` | 실제 서비스 검색·생성 API의 동결 대상 구현이다. |
| `src/api/rag.ts` | 프런트엔드와 RAG API 사이 계약 변경을 반영한다. |
| `tests/test_api_hardening.py` | 서비스 검색·후처리·보안 회귀를 고정한다. |
| `tests/test_rag_generators.py` | 생성기 prompt·후처리 회귀를 고정한다. |
| `tests/test_role_router.py` | 역할 routing 변경의 회귀를 고정한다. |

## 신규 scripts — 포함 승인 39개

| 경로 | 포함 근거 |
|---|---|
| `scripts/analyze_final_generation_gfc.py` | 최종 질문 단위 GFC·paired 통계를 재현한다. |
| `scripts/analyze_generation_failures.py` | 검색·근거·Judge 실패 원인을 교차 분해한다. |
| `scripts/analyze_gfc_pairs.py` | DEV 반복 생성의 paired GFC 효과를 계산한다. |
| `scripts/analyze_holdout_retrieval.py` | 동결 holdout 검색 headline 지표를 계산한다. |
| `scripts/analyze_judge_human_calibration.py` | Judge–사람 calibration gate를 계산한다. |
| `scripts/analyze_postprocessor_judge_pairs.py` | 후처리 전후 Judge pair의 제한적 진단을 만든다. |
| `scripts/analyze_postprocessor_pairs.py` | 동일 초안 후처리 변화와 eligibility를 감사한다. |
| `scripts/analyze_rule_activation.py` | 동결 규칙의 관측·발동 가능률을 분리 집계한다. |
| `scripts/analyze_service_ab.py` | 서비스 생성 A/B 요약을 재현한다. |
| `scripts/analyze_service_retrieval_ab.py` | 서비스 검색 A/B 지표를 재현한다. |
| `scripts/build_answer_review_packet.py` | condition-blind 사람 답변 검토 패킷을 만든다. |
| `scripts/build_dev_source_manifest.py` | DEV source binding manifest를 만든다. |
| `scripts/build_evidence_manifest.py` | 최종 산출물 SHA manifest를 만든다. |
| `scripts/build_holdout_review_packet.py` | holdout 독립 검수용 blinded packet을 만든다. |
| `scripts/build_holdout_signoff.py` | 두 사람 응답을 strict 검증해 sign-off한다. |
| `scripts/build_judge_repeat_selection.py` | 반복 Judge 표본의 답변·순서·SHA를 고정한다. |
| `scripts/build_service_ab_review.py` | 서비스 A/B 육안 비교 화면을 만든다. |
| `scripts/evaluate_grounding_guard.py` | grounding guard adversarial 평가를 실행한다. |
| `scripts/evaluate_grounded_claims_v2.py` | frozen C1 검색 근거를 재사용해 C2 quote-bound 생성 실험을 수집한다. |
| `scripts/evaluate_oracle_context_answers.py` | retrieval을 분리한 oracle-context 진단을 실행한다. |
| `scripts/evaluate_parser_audit.py` | 파서별 source-bound anchor 보존을 평가한다. |
| `scripts/evaluate_postprocessor_regression.py` | 후처리 허용·거부 회귀셋을 평가한다. |
| `scripts/evaluate_service_answers.py` | 서비스 답변·trace 수집과 append/resume 계약을 구현한다. |
| `scripts/final_generation_slots.py` | 최종 generation slot 정의와 순서를 공유한다. |
| `scripts/holdout_gold.py` | holdout gold를 안전하게 정규화·검증한다. |
| `scripts/immutable_outputs.py` | no-clobber·fsync 기반 immutable publish를 제공한다. |
| `scripts/judge_service_answers.py` | 분리된 Judge v11 판정과 감점 guard를 구현한다. |
| `scripts/parser_audit.py` | 파서 감사 공통 schema와 검사를 제공한다. |
| `scripts/project_raw_draft_answers.py` | 저장 초안의 현재 후처리 projection을 만든다. |
| `scripts/run_dev_retrieval_matrix.py` | DEV 검색 matrix 실행을 재현한다. |
| `scripts/run_final_generation_schedule.py` | 최종 생성의 고정 schedule·WAL·lock을 집행한다. |
| `scripts/run_final_retrieval_schedule.py` | 최종 검색 4-lane schedule을 집행한다. |
| `scripts/service_eval_artifacts.py` | answer/judgment artifact 계약과 hash 검증을 공유한다. |
| `scripts/summarize_generation_runs.py` | 독립 생성 반복 결과를 집계한다. |
| `scripts/summarize_judge_repeats.py` | 고정 답변 Judge 반복 일치율을 집계한다. |
| `scripts/validate_parser_audit.py` | 파서 감사 입력·출력 gate를 검증한다. |
| `scripts/validate_service_eval.py` | 서비스 평가셋 schema·분리를 검증한다. |
| `scripts/validate_service_holdout.py` | holdout DEV 분리·corpus·sign-off gate를 검증한다. |
| `scripts/rag/grounded_claims_v2.py` | claim별 원문 quote binding과 deterministic 검증을 구현한다. |

## 신규 tests — 포함 승인 34개

| 경로 | 포함 근거 |
|---|---|
| `tests/test_analyze_final_generation_gfc.py` | 최종 GFC paired 통계·terminal fallback을 검증한다. |
| `tests/test_analyze_generation_failures.py` | 실패 다중 레이블·주원인 분할·정본 일치를 검증한다. |
| `tests/test_analyze_gfc_pairs.py` | DEV GFC pair 집계와 bootstrap 계약을 검증한다. |
| `tests/test_analyze_holdout_retrieval.py` | holdout 검색 분석과 schedule binding을 검증한다. |
| `tests/test_analyze_judge_human_calibration.py` | calibration 지표·fail-closed gate를 검증한다. |
| `tests/test_analyze_postprocessor_judge_pairs.py` | 후처리 Judge pair 결합·eligibility를 검증한다. |
| `tests/test_analyze_postprocessor_pairs.py` | 후처리 projection 비교·원천 binding을 검증한다. |
| `tests/test_analyze_rule_activation.py` | observed/possible/trace_absent 분리를 검증한다. |
| `tests/test_analyze_service_ab.py` | 생성 A/B 집계 회귀를 검증한다. |
| `tests/test_analyze_service_retrieval_ab.py` | 검색 A/B 집계 회귀를 검증한다. |
| `tests/test_build_answer_review_packet.py` | blinded packet·private mapping 분리를 검증한다. |
| `tests/test_build_dev_source_manifest.py` | DEV manifest의 source hash binding을 검증한다. |
| `tests/test_build_holdout_review_packet.py` | holdout packet 생성의 no-clobber·blind 조건을 검증한다. |
| `tests/test_build_holdout_signoff.py` | 두 reviewer strict merge와 SHA binding을 검증한다. |
| `tests/test_build_judge_repeat_selection.py` | 반복 Judge selection manifest 고정을 검증한다. |
| `tests/test_evaluate_grounding_guard.py` | grounding guard ablation 집계를 검증한다. |
| `tests/test_evaluate_grounded_claims_v2.py` | C2 collector의 dry-run·승인·무결성·실패 경로를 검증한다. |
| `tests/test_evaluate_oracle_context_answers.py` | oracle 진단의 제외 표지·provenance를 검증한다. |
| `tests/test_evaluate_service_answers.py` | collector retry·resume·terminal row 계약을 검증한다. |
| `tests/test_evidence_manifest.py` | 최종 evidence manifest 완전성과 SHA를 검증한다. |
| `tests/test_holdout_gold.py` | holdout gold 정규화와 fail-closed 검증을 고정한다. |
| `tests/test_immutable_outputs.py` | immutable output의 symlink·overwrite 거부를 검증한다. |
| `tests/test_judge_service_answers.py` | Judge v11 schema·guard·artifact 결합을 검증한다. |
| `tests/test_parser_audit.py` | 파서 감사 schema·anchor 검사를 검증한다. |
| `tests/test_postprocessor_regression.py` | 허용 claim과 음성 회귀를 함께 고정한다. |
| `tests/test_project_raw_draft_answers.py` | 동일 초안 projection의 provenance를 검증한다. |
| `tests/test_run_dev_retrieval_matrix.py` | DEV matrix 실행·출력 binding을 검증한다. |
| `tests/test_run_final_generation_schedule.py` | 생성 schedule·WAL·lock·poison 처리를 검증한다. |
| `tests/test_run_final_retrieval_schedule.py` | 검색 schedule·lane·append-only 계약을 검증한다. |
| `tests/test_service_eval_artifacts.py` | 공통 answer/judgment schema와 SHA 검증을 고정한다. |
| `tests/test_service_retrieval_tuning.py` | 동결 검색 규칙의 positive/negative 회귀를 고정한다. |
| `tests/test_summarize_judge_repeats.py` | Judge 반복 일치율·selection binding을 검증한다. |
| `tests/test_validate_service_holdout.py` | holdout 네 gate와 sign-off 검증을 고정한다. |
| `tests/test_grounded_claims_v2.py` | quote·숫자·관계 방향의 fail-closed 검증을 고정한다. |

## 신규 config — 포함 권고 6개

| 경로 | 포함 근거 |
|---|---|
| `config/model-pricing-20260902.json` | 비용 계산에 사용한 날짜 고정 가격 snapshot이다. |
| `config/pnu-grounding-adversarial-eval.jsonl` | grounding guard의 positive/negative 평가셋이다. |
| `config/pnu-parser-audit-v1.jsonl` | 18문서·54 anchor 파서 감사 정의다. |
| `config/pnu-service-answer-eval-v1.jsonl` | 최초 서비스 DEV 생성 평가셋 이력을 보존한다. |
| `config/pnu-service-answer-eval.jsonl` | 현재 45문항 DEV 생성 평가 정본이다. |
| `config/pnu-service-dev-source-manifest.json` | DEV 문항과 source bytes를 결합하는 manifest다. |

## 신규 docs — 포함 권고 15개

| 경로 | 포함 근거 |
|---|---|
| `docs/codex-brief-20260904.md` | 세션 간 동결 상태와 남은 작업을 인계한다. |
| `docs/dev45-generation-root-cause-20260902.md` | 생성 병목의 근거 기반 원인 분석을 보존한다. |
| `docs/dev45-retrieval-matrix-20260902.md` | 검색 구성별 DEV 비교와 채택 판단을 보존한다. |
| `docs/evaluation-protocol-20260914.md` | 사전 고정 지표·통계·Judge-human protocol을 정의한다. |
| `docs/final-eval-runbook-20260914.md` | freeze부터 최종 평가까지 실행 절차를 고정한다. |
| `docs/final-report-20260914.md` | 제출 보고서 본문과 한계·근거 경로를 포함한다. |
| `docs/parser-audit-20260914.md` | 다형식 파서 감사 방법과 결과를 설명한다. |
| `docs/plan-20260914-final.md` | 9월 14일 제출까지의 작업 범위와 순서를 보존한다. |
| `docs/progress-log-20260901.md` | 날짜별 명령·결과·SHA의 정본 작업 로그다. |
| `docs/report-evidence-ledger-20260914.md` | 보고서 수치와 증거 artifact의 대응표다. |
| `docs/requirements-traceability-20260914.md` | 요구사항과 구현·평가 증거를 추적한다. |
| `docs/rule-inventory-20260904.md` | 동결 규칙 68개의 위치·트리거·과적합 분류다. |
| `docs/theory-background-20260914.md` | RAG 검색·생성·LLM Judge 이론 배경을 정리한다. |
| `docs/worklog-20260831-service-track.md` | 서비스 트랙의 초기 실험·결정 이력을 보존한다. |
| `docs/freeze-file-list-20260904.md` | 승인받을 정확한 staging 범위와 제외 대상을 고정한다. |

## 신규 evidence — 포함 권고 28개

| 경로 | 포함 근거 |
|---|---|
| `evidence/20260914/20260831-svc-gen-baseline-run1.jsonl` | historical C0 run1 원판이다. |
| `evidence/20260914/20260831-svc-gen-baseline-run2.jsonl` | historical C0 run2 원판이다. |
| `evidence/20260914/20260831-svc-gen-baseline-run3.jsonl` | historical C0 run3 원판이다. |
| `evidence/20260914/20260831-svc-gen-tuned-run1.jsonl` | historical C1 run1 원판이다. |
| `evidence/20260914/20260831-svc-gen-tuned-run2.jsonl` | historical C1 run2 원판이다. |
| `evidence/20260914/20260831-svc-gen-tuned-run3.jsonl` | historical C1 run3 원판이다. |
| `evidence/20260914/20260831-svc-retrieval-current-smoke.jsonl` | historical 검색 smoke 원판이다. |
| `evidence/20260914/README.md` | evidence 파일의 의미·해석 범위를 안내한다. |
| `evidence/20260914/cap2-service-extractive-dev45-20260901.jsonl` | cap2 DEV45 검색 응답 원판이다(약 3.88 MB). |
| `evidence/20260914/cap2-vs-cap4-retrieval-dev45-preview.png` | cap2/cap4 비교의 143 KB 시각 증거다. |
| `evidence/20260914/cap2-vs-cap4-retrieval-dev45.csv` | cap 비교의 문항별 기계판독 표다. |
| `evidence/20260914/cap2-vs-cap4-retrieval-dev45.html` | cap 비교 육안 검토 화면이다. |
| `evidence/20260914/cap2-vs-cap4-retrieval-dev45.json` | cap 비교 구조화 요약이다. |
| `evidence/20260914/cap4-service-extractive-dev45-20260901.jsonl` | cap4 DEV45 검색 응답 원판이다(약 3.79 MB). |
| `evidence/20260914/current-code-audit-manifest.json` | current-code 감사 입력·SHA를 결합한다. |
| `evidence/20260914/grounding-guard-ablation.csv` | guard ablation의 문항별 표다. |
| `evidence/20260914/grounding-guard-ablation.json` | guard ablation 구조화 요약이다. |
| `evidence/20260914/interim-service-n3-manifest.json` | historical 생성 n=3 artifact SHA manifest다. |
| `evidence/20260914/local-performance-eval-snapshot-20260901.json` | 로컬 성능 평가 환경·결과 snapshot이다. |
| `evidence/20260914/postprocessor-offline-ab.json` | 동일 초안 후처리 A/B 진단이다. |
| `evidence/20260914/service-ab-per-question-metric-corrected-20260901.csv` | 수정 지표 기준 문항별 A/B다. |
| `evidence/20260914/service-ab-per-question.csv` | 최초 문항별 A/B 이력을 보존한다. |
| `evidence/20260914/service-ab-review-metric-corrected-20260901.html` | 수정 지표 A/B 육안 검토 화면이다. |
| `evidence/20260914/service-ab-review.html` | 최초 A/B 검토 화면 이력을 보존한다. |
| `evidence/20260914/service-ab-summary-metric-corrected-20260901.json` | 수정 지표 A/B 집계 정본이다. |
| `evidence/20260914/service-ab-summary.json` | 최초 A/B 집계 이력을 보존한다. |
| `evidence/20260914/single-numeric-facet-dev45-replay.json` | 숫자 facet 회귀의 DEV replay 증거다. |
| `evidence/20260914/tuning-freeze-snapshot-20260904.json` | 동결 코드·index SHA와 품질 gate snapshot이다. |

## 보류 — 이번 freeze 승인 범위에서 제외 4개

다음 파일은 지시서의 holdout embargo 때문에 내용·해시를 확인하지 않았다. 임시
파일이나 폐기 대상이라는 뜻이 아니라, 사람 검수가 끝나기 전 코드 freeze와 섞지
않는다는 뜻이다.

| 경로 | 보류 근거 |
|---|---|
| `config/pnu-service-answer-holdout-v2.draft.jsonl` | 질문 원문이므로 사람 sign-off 전 읽기·수정·staging을 보류한다. |
| `docs/holdout-v2-human-review.md` | 검토 패킷 내용이므로 embargo 해제 전 접근·staging을 보류한다. |
| `evidence/holdout-v2-reviewer-a.json` | Reviewer A 응답은 36문항 PENDING이며 별도 승인 대상이다. |
| `evidence/holdout-v2-reviewer-b.json` | Reviewer B 응답은 36문항 PENDING이며 별도 승인 대상이다. |

## 비밀·임시·대용량 점검

- `.env`는 ignore 상태이며 포함 금지다.
- embargo 4개를 제외한 작업 파일을 실제 값 형태의 Gemini/OpenAI key 패턴으로
  파일명만 검사했으며 매치가 없었다. 비밀값 자체는 출력하지 않았다.
- 5 MB를 넘는 신규 단일 파일은 없다. 가장 큰 두 파일은 위 cap2/cap4 JSONL이며
  텍스트 evidence라 포함을 권고하되, 저장소 용량을 줄이려면 사용자 판단으로
  두 파일만 별도 artifact storage로 이관할 수 있다.
- `processed/` 아래의 기존·신규 answer, Judge, 분석 JSON/CSV/README는 ignore
  상태이며 수정·삭제하지 않는다.

## 승인 뒤 실행할 명령

2026-09-04 사용자가 현재까지 만든 코드의 git merge를 승인했다. 실제 실행 시에는
위 **포함 승인 131개 경로를 정확히 열거한 pathspec**을 만들고, 경로 수·누락·추가가 없는지
대조한 뒤 `git add --pathspec-from-file=<승인된 파일> --pathspec-file-nul` 형태로
staging한다. `git add -A`와 `git add .`은 사용하지 않는다.

```text
commit: chore: freeze PNU final evaluation code
tag:    pnu-eval-code-freeze-20260904-v1
```

Staging 후에는 `git diff --cached --check`, staged path exact-set 비교,
`git diff --cached --stat`를 먼저 보여준다. Commit·tag는 이 문서의 목록에 대한
사용자 명시 승인 뒤에만 실행하며 push는 별도 승인 사항이다.
