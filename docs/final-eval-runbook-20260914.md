# 부산대학교 RAG 최종 평가 실행 Runbook

> 기준 시점: **2026-09-02 (Asia/Seoul)**
> 제출일: **2026-09-14**
> 방법론 정본: [`evaluation-protocol-20260914.md`](evaluation-protocol-20260914.md)
> 목적: holdout을 본 뒤 판단을 바꾸지 않고 C0/C1 검색·생성·Judge·사람 평가를
> 재현 가능하게 끝내는 운영 절차

이 문서는 명령을 실제 저장소 CLI와 대조한 실행 안내서다. 현재는 **NO-GO**다.
B1·B3·B4 구현과 관련 로컬 회귀는 끝났지만, 실제 사람 2인 holdout sign-off,
최종 holdout filename, 사용자 승인을 받은 clean code-freeze가 아직 없다. 따라서
final holdout 검색 수집·외부 생성·Judge 호출은 시작하지 않는다.

## 1. 9월 2일 현재 상태

| 항목 | 현재 값 | 판정 |
|---|---|---|
| Holdout draft | 36문항 = Core 27 + Challenge 9 | 작성 완료 |
| Evidence option | 93개, corpus 대조 93/93 | 기계 gate 통과 |
| Holdout SHA-256 | `2fbedad63531491d3c069a5de6fe51fff623f7e571b9170c6a9aa7452e2d51f9` | 현재 draft에만 유효 |
| 사람 2인 sign-off | `evidence/holdout-v2-signoff.json` 없음 | **BLOCK** |
| 최종 holdout 파일 | `config/pnu-service-answer-holdout-v2.jsonl` 없음 | **BLOCK** |
| 전체 품질 검사 | 637 tests OK(6 skip), ESLint·TypeScript·Vite build PASS | 2026-09-02 dirty snapshot; clean freeze에서 재실행 필요 |
| Working tree | 수정·미추적 파일 존재 | **BLOCK** |
| Code-freeze commit/tag | 없음 | **BLOCK** |
| Oracle-context collector | `evaluate_oracle_context_answers.py`, 영역별 `multi_evidence` 1개 = 9개 | 구현·관련 60 tests PASS; final 미실행 |
| Calibration protocol | v1.1, balanced accuracy ≥ 0.80 포함 | 코드와 동기화 완료 |
| 가격 snapshot | `config/model-pricing-20260902.json` | 공식 Google paid-standard, 2026-09-02 고정 |
| DEV 생성 smoke | `p0g`, 3문항, C0/C1 답변 수집 완료 | DEV 진단만 가능 |
| Judge | `pnu-grounded-fully-correct-v11`, DEV 조건별 3회 완료 | final 미실행 |
| Final holdout 검색·생성·사람 평가 | 미실행 | 정상 |
| Final schedule/분석 도구 | B1/B3/B4 구현·로컬 검증 완료 | 실제 holdout은 위 BLOCK 해소 뒤 실행 |

정본 BM25 index 3개의 pin은 다음과 같다. Phase C retrieval은 세 profile을
사용하고, generation은 Cascade만 사용한다.

| Profile | Index SHA-256 | Corpus revision |
|---|---|---|
| Baseline | `a993f00d222177adf668fa7069869f0a79576166fba9234662c37a7b657b94e7` | `20260725-pnu-curated-baseline-v5:baseline:997f28ba153d0702f773a09a` |
| Challenger | `6c1aab850a0b1ac572de9123eee8acfb21007afffaeb6b6e66c7dffc6875ece4` | `20260725-pnu-curated-challenger-v5:challenger:7d441a11d96b052609b96337` |
| Cascade | `a4c1ca6318cb14721866a6512f3dfeacdca58c62ecff98ea7b81df2e75c74c31` | `20260725-pnu-curated-cascade-v5:cascade:5d1b5fee3d2eafa1a7c77d33` |

모든 서비스 lane은 `context_chunks_per_document=2`, 실제 최종 context는
`top_k=8`로 고정한다.

Judge v11의 검증된 설정은 다음과 같다. CLI 기본 `max_output_tokens=1200`에
의존하지 않고 DEV v8과 같은 `1600`을 명시한다.

```text
judge_model: gemini-3.1-flash-lite
rubric_version: pnu-grounded-fully-correct-v11
temperature: 0.0
max_output_tokens: 1600
input_projection: compact-observed-v1
quote_validation: line-list-marker-canonical-v1
```

## 2. CLI/protocol 대조 결과

### 2.1 해결된 final hard stop

#### B1. Frozen generation schedule과 단일 승인 경로

`scripts/run_final_generation_schedule.py`가 189개 direct service slot을 정본
schedule로 만든다. Core는 `case → generation run → condition` 순으로, seed
`20260914`의 영역 층화 AB/BA 41/40을 사용한다. Challenge는 C1 run 1→3이다.
Schedule은 holdout byte/canonical SHA, 선택 ID, index와 source-manifest SHA,
Git commit, model/config, 전역 `call_order`, 9개 output artifact를 고정한다.

Live collection은 이 runner가 발급한 정확한 내부 authorization으로 한 slot씩만
collector에 전달된다. 시작 직전에 사람 sign-off를 포함한 holdout gate, 현재 파일
byte hash(sign-off가 가리키는 packet·Reviewer A/B 원본 포함), clean local Git
commit, server/index/model pin, 기존 전역 prefix를 다시 검사한다. 직접
`evaluate_service_answers.py --only --allow-partial`을 호출하는 것은
final 승인 경로가 아니다.

각 output root에는 `O_EXCL` 방식의 `.final-generation-run.lock`이 걸려 동시 runner가
같은 slot을 두 번 호출할 수 없다. 각 논리 slot은 fsync된 WAL의 정확한
`slot_started → slot_completed` 전이를 가져야 한다. 성공은 answer, 복구 불가능한
실패는 terminal service-error record로 한 번만 종결하며 uncertain/poisoned slot은
자동 재생하지 않는다. 최대 세 번의 transport attempt와 retry 사유도 record에
보존된다.

#### B3. Frozen 4-lane retrieval과 전용 headline 분석기

`scripts/run_final_retrieval_schedule.py`가 Core 27 × PAR-B/PAR-CH/C0/C1 = 108개를
case-major 정본 schedule로 만든다. PAR-B/CH의 parser/index와 C0/C1의 동일 Cascade
index를 각각 pin하며 provider는 `extractive`로 고정돼 외부 LLM을 호출하지 않는다.
Live run은 generation과 같은 gate 재검증, 정확한 authorization, append-only WAL,
전역 prefix audit, `O_EXCL` `.final-retrieval-run.lock`을 적용한다.

`scripts/analyze_holdout_retrieval.py`의 final 4-lane mode는 schedule과 108개
answer/WAL provenance를 다시 감사한 뒤 required claim의 evidence option을 기준으로
Source Hit@1/3/5, Evidence Recall·All-Evidence@5/@8, raw BM25의 MRR@50·Candidate
Recall@50, latency, 질문/family paired bootstrap과 exact McNemar, parser 비교의 Holm
보정을 계산한다. Legacy `retrieval_hit`은 headline으로 쓰지 않는다.

#### B4. 생성 run 단위 GFC와 calibration fallback

`scripts/analyze_final_generation_gfc.py`는 C0/C1 각각 서로 다른 generation
run 1/2/3을 결합하고 질문/family `n=27`에서 paired bootstrap 10,000회,
paired sign-flip, 2/3 majority exact McNemar를 계산한다. 같은 답변의 Judge repeat는
생성 반복으로 세지 않는다. `build_judge_repeat_selection.py`와
`summarize_judge_repeats.py`는 고정 stability subset을 answer hash·순서·selection
hash로 결합하며, `build_evidence_manifest.py`도 이 partial group을 명시적으로
검증한다.

Terminal generation slot은 가짜 답변이나 Judge 점수를 만들지 않는다. 분석 시
service error, GFC=0으로 포함해 질문 `n=27`을 유지한다. 특히 run 1 terminal은
Judge headline의 effective calibration gate를 강제로 FAIL로 만들며, 성공한 run 1
답변의 adjudicated human label과 terminal 자동 GFC=0을 결합한 사람 fallback이
headline이 된다. Calibration threshold를 통과했을 때만 3-run LLM-Judge 결과가
headline이다.

#### 공통 산출물 안전성

Schedule은 exclusive-create하고 수집 artifact는 append-only audit를 거친다. 분석
JSON/CSV/completion manifest, selection, review packet, repeat summary, evidence
manifest의 publication layer는 기존 경로와 symlink/hardlink alias를 거부한다.
여러 파일을 만드는 도구는 companion을 먼저 fsync·불변 publish하고 권위
JSON/manifest를 마지막에 게시한다. 기존 결과를 덮어써서 “완료” 상태를 만들 수 없다.

### 2.2 기타 해결됨

#### B2. Oracle-context 9개 수집기

`scripts/evaluate_oracle_context_answers.py`가 별도 fail-closed collector로
구현됐고 관련 60 tests가 PASS했다. 선택 규칙은 답변을 보기 전에 고정한
`holdout-core`의 `multi_evidence` 문항을 9개 Core 영역에서 정확히 하나씩 뽑는
것이다. 이 collector는 draft 경로, 4개 holdout gate 실패, gold context 잘림,
provider/model/prompt/config 불일치, 기존 output을 모두 거부하고 case·sign-off·
corpus·선택·gold provenance hash를 각 row에 저장한다.

Oracle 9개는 정본 생성 198개에는 포함하지만 `diagnostic_only=true`,
`service_performance_eligible=false`이므로 C0/C1 서비스 성능에는 합산하지 않는다.
실제 외부 호출은 사람 sign-off와 code freeze 뒤 Step 6에서만 한다.

#### B5. Calibration gate protocol/code 동기화

프로토콜 v1.1과 `analyze_judge_human_calibration.py`는 Core 54개에서 raw agreement
≥ 0.80, balanced accuracy ≥ 0.80, Cohen's kappa ≥ 0.60, macro-F1 ≥ 0.75의 네
기준을 모두 요구하도록 동기화됐다. Balanced accuracy가 undefined인 경우도
fail-closed다. Threshold는 사람 label을 보기 전에 동결됐으며 결과를 본 뒤
완화하지 않는다.

## 3. 목표 실행량과 표본 단위

| 단계 | 계산 | 외부 LLM 호출 계획 | 논리적 slot |
|---|---:|---:|---:|
| Phase C Core retrieval-only: PAR-B/PAR-CH/C0/C1 | 27 × 4 | 0 | 108 |
| C0/C1 Core generation | 27 × 2 × 3 | 162 | 162 |
| C1 Challenge generation | 9 × 3 | 27 | 27 |
| C1 oracle-context 진단 | 9 × 1 | 9 | 9 |
| Judge 최초 판정 | 198 × 1 | 198 | 198 |
| 고정 답변 18개 Judge 추가 반복 | 18 × 2 | 36 | 36 |
| 합계 | generator 198 + Judge 234 | **432** | 생성 198 + 판정 234 |

432는 terminal error가 없는 사전 계획값이다. 기술적 retry는 이 수에 포함하지
않고 별도 집계한다. Generator terminal error가 나면 성공 answer row와 terminal
error slot의 합이 198이어야 하며, 해당 slot은 Judge를 억지로 호출하지 않고
service error 및 GFC=0으로 포함한다. 따라서 실제 Judge 호출은 실패 수만큼 계획보다
적을 수 있고, 목표 숫자를 맞추기 위한 dummy/repeat 호출은 금지한다. Provider/model
fallback은 성공으로 숨기지 않고 더 좋은 답을 얻으려고 다시 뽑지 않는다.

### 3.1 비용 보고 계약

가격은 실행 코드에 하드코딩하지 않고 `$PRICING_SNAPSHOT`을 입력 artifact로 쓴다.
Generator `gemini-3.5-flash-lite`는 paid-standard 1M token당 input `$0.30`,
output(사고 token 포함) `$2.50`; Judge `gemini-3.1-flash-lite`는 input `$0.25`,
output(사고 token 포함) `$1.50`로 동결돼 있다. 먼저 snapshot 자체를 검사한다.

```bash
jq -e '
  .schema_version == "pnu.model-pricing-snapshot/v1" and
  .captured_at == "2026-09-02T15:50:00+09:00" and
  .currency == "USD" and
  .billing_unit == "per_1m_tokens" and
  .source.provider == "Google Gemini Developer API" and
  .source.pricing_tier == "paid_standard" and
  .source.accessed_date == "2026-09-02" and
  .models["gemini-3.5-flash-lite"].input_text_usd_per_1m_tokens == 0.30 and
  .models["gemini-3.5-flash-lite"].output_including_thinking_usd_per_1m_tokens == 2.50 and
  .models["gemini-3.1-flash-lite"].input_text_usd_per_1m_tokens == 0.25 and
  .models["gemini-3.1-flash-lite"].output_including_thinking_usd_per_1m_tokens == 1.50 and
  .estimation_policy.use_provider_reported_usage_only == true and
  .estimation_policy.missing_usage == "report usage=unavailable; do not estimate from characters"
  ' "$PRICING_SNAPSHOT"
```

Provider가 반환한 input/output token usage가 각 실제 시도에 보존됐을 때만 snapshot
rate로 비용을 합산한다. Retry도 실제 billable usage가 있으면 포함한다. 9월 2일
현재 generator/Judge artifact 계약에는 provider-reported token usage가 없으므로,
freeze 전에 별도 구현·검증하지 않는 한 최종 비용은 `usage=unavailable`로 쓰고
논리적 호출 수·실제 시도 수만 보고한다. 문자 수, prompt 길이, max token으로
금액을 추정하지 않는다.

사람 평가는 terminal error가 없을 때 generation run 1에서 다음 최대 63개를 쓴다.

```text
C0 Core 27 + C1 Core 27 + C1 Challenge 9 = 63 answers
독립 평가: 63 × 2명 = 126 ratings
최종 adjudication: 모든 63개에 1개씩 = 63 labels
calibration gate 표본: Core 54 answers
최종 C0/C1 paired 통계 표본: Core question family 27개, n=27
```

Run 1 terminal slot은 blind packet/Judge/사람 label에서 제외하므로 실제 rating 수는
그만큼 감소한다. 이때 B4 effective calibration gate는 FAIL이고, 사람 fallback은
성공 답변의 adjudication과 terminal 자동 GFC=0을 결합해 Core `n=27`을 유지한다.

한 split/run당 한 파일을 쓰는 설계라면 예상 answer artifact는 retrieval 4개,
generation 10개다. Generation 10개는 C0 Core 3, C1 Core 3, C1 Challenge 3,
C1 oracle 1이다. 최초 Judge artifact 10개와 고정 18개용 추가 Judge artifact
4개를 합쳐 최대 14개가 된다. Service collector가 만드는 retrieval 4개와 direct
generation 9개에는 시도/WAL sidecar가 따라온다. Retrieval은 non-empty error가
있으면 poison으로 중단한다. Generation 성공 slot에는 error가 없어야 하고 terminal
service-error slot에는 정확히 한 error sidecar가 있어야 한다. Oracle collector는
sidecar/resume이 없고, 실패한 partial
파일을 보존한 채 새 generation run ID와 새 경로로 다시 시작한다.

## 4. 절대 하지 말 것

1. Holdout의 질문, 검색 결과, 생성 답변, Judge 결과를 본 뒤 검색 규칙·prompt·
   postprocessor·gold·threshold를 바꾸지 않는다. 바꾸면 이 holdout을 DEV로
   강등하고 새 holdout을 만든다.
2. 기존 `.answers.jsonl`, `.judgments.jsonl`, 사람 label, 통계, manifest를
   덮어쓰지 않는다. 재실행은 새 experiment/run ID와 새 파일을 쓴다.
3. Judge repeats를 독립 표본 `n`이나 generation n=3으로 세지 않는다.
4. C0/C1에서 corpus, parser, retrieval mode, context 수, generator model,
   prompt/postprocessor를 다르게 하지 않는다. 차이는 retrieval tuning ON/OFF뿐이다.
5. `--allow-unpinned`, provider `auto`, 인자 없는 `search_api.py`, `/search@50`을
   final에 사용하지 않는다.
6. 낮은 점수, 부분 답변, 보기 싫은 출력 때문에 재생성하지 않는다.
7. Judge error row를 정상 점수처럼 집계하지 않는다. Judge CLI는 error row가
   있어도 종료 코드 0일 수 있으므로 별도 완전성 검사를 반드시 실행한다.
8. `analyze_judge_human_calibration.py`의 종료 코드 0만 보고 gate PASS로 보지
   않는다. 출력 JSON의 `summary.protocol_gate.passed`를 검사한다.
9. Blind 검토가 끝나기 전에 private mapping, condition, provider/model 정보를
   Reviewer A/B에게 공개하지 않는다.
10. `git add -A`, `git add .`, force tag 이동, 기존 result directory 재사용을
    하지 않는다.

## 5. 공통 변수와 append-only 작업공간

아래 명령은 저장소 루트의 zsh에서 실행하는 템플릿이다. API key 값을 명령행,
로그, 문서에 쓰지 않는다. `.env`도 출력하거나 commit하지 않는다.

```bash
set -eu
set -o pipefail
set -o noclobber
umask 077

cd /Users/leehyunwoo/project/pnu-docs-chatbot

HOLDOUT_DRAFT=config/pnu-service-answer-holdout-v2.draft.jsonl
HOLDOUT=config/pnu-service-answer-holdout-v2.jsonl
DEV_MANIFEST=config/pnu-service-dev-source-manifest.json
PRICING_SNAPSHOT=config/model-pricing-20260902.json
SOURCE_MANIFEST_SHA=1fa7e0f2f5d03a2a1249584d91f7232c9b2cb4642b39115bf3c9e6a871845a2f

CANONICAL_DRAFT_SHA=2fbedad63531491d3c069a5de6fe51fff623f7e571b9170c6a9aa7452e2d51f9
DRAFT_SHA=$(shasum -a 256 "$HOLDOUT_DRAFT" | awk '{print $1}')
if [ "$DRAFT_SHA" = "$CANONICAL_DRAFT_SHA" ]; then
  ACTIVE_REVIEW_PACKET=docs/holdout-v2-human-review.md
  ACTIVE_REVIEW_A=evidence/holdout-v2-reviewer-a.json
  ACTIVE_REVIEW_B=evidence/holdout-v2-reviewer-b.json
  ACTIVE_SIGNOFF=evidence/holdout-v2-signoff.json
else
  ACTIVE_REVIEW_PACKET="docs/holdout-v2-human-review-$DRAFT_SHA.md"
  ACTIVE_REVIEW_A="evidence/holdout-v2-reviewer-a-$DRAFT_SHA.json"
  ACTIVE_REVIEW_B="evidence/holdout-v2-reviewer-b-$DRAFT_SHA.json"
  ACTIVE_SIGNOFF="evidence/holdout-v2-signoff-$DRAFT_SHA.json"
fi

CASCADE_INDEX=processed/index/pnu-20260725-curated-cascade-v5-allow-suspect.sqlite
BASELINE_INDEX=processed/index/pnu-20260725-curated-baseline-v5-allow-suspect.sqlite
CHALLENGER_INDEX=processed/index/pnu-20260725-curated-challenger-v5-allow-suspect.sqlite
BASELINE_CORPUS_REVISION=20260725-pnu-curated-baseline-v5:baseline:997f28ba153d0702f773a09a
CHALLENGER_CORPUS_REVISION=20260725-pnu-curated-challenger-v5:challenger:7d441a11d96b052609b96337
CORPUS_REVISION=20260725-pnu-curated-cascade-v5:cascade:5d1b5fee3d2eafa1a7c77d33

EXPERIMENT_ID=pnu-service-final-20260909-v1
RUN_ROOT=processed/eval/final-20260909/$EXPERIMENT_ID
RETRIEVAL_SCHEDULE=$RUN_ROOT/meta/final-retrieval-schedule.json
GENERATION_SCHEDULE=$RUN_ROOT/meta/final-generation-schedule.json
```

`ACTIVE_*` 경로는 현재 draft bytes의 SHA에서 매번 결정한다. 사람 검토에 시간이 걸려
새 셸을 열더라도 이 공통 변수 블록만 다시 실행하면 같은 packet·A/B 응답·sign-off
경로를 복원한다. `PACKET_SHA`는 active packet을 생성·검증한 뒤에만 계산한다.

`RUN_ROOT`는 code-freeze 뒤 한 번만 만든다. 같은 이름이 있으면 중단하고 새
experiment ID를 정한다.

```bash
test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT"/{meta,retrieval,generation,judge,human,stats,logs}
```

출력 파일을 만드는 각 명령 전에는 `test ! -e "$OUT"`를 실행한다. 수집 도중
중단된 동일 run만 collector의 append/resume 규칙으로 이어 간다. 이미 완성된
run을 다시 실행하지 않는다. Resume은 기존 row의 `collector_config_sha256`과 새
실행의 config hash가 완전히 같을 때만 허용된다. `--sleep`, `--timeout`, retry,
출력·sidecar 경로를 포함해 collector 인자를 하나라도 바꾸지 않는다. 바꿔야 하면
기존 산출물을 보존하고 새 experiment/run ID와 새 경로를 사용한다.

Freeze 전 dirty worktree에서 DEV만 재현할 때는 `--expected-index-sha256`와
`--expected-git-commit`을 주지 않는다. 둘 중 하나라도 지정하면 `/health` gate가
clean startup worktree를 요구한다. 이때 무핀 실행으로 풀지 말고
`--expected-corpus-revision`과 `--expected-source-manifest-sha256`를 유지해 코퍼스
정체성을 고정한다. 최종 holdout schedule은 예외 없이 clean code-freeze의 Git·
index·source manifest를 모두 pin한다.

## 6. Step 1 — 사람 2인 holdout sign-off

### 6.1 모든 draft SHA의 기계 gate

사람에게 배포하기 전에 sign-off 없이 세 기계 gate를 먼저 통과시킨다. Draft를
수정할 때마다 이 단계부터 다시 시작한다.

```bash
python3 -B scripts/validate_service_holdout.py \
  --cases "$HOLDOUT_DRAFT" \
  --dev-manifest "$DEV_MANIFEST" \
  --corpus-index "$CASCADE_INDEX" \
  --pre-review \
  --json | jq -e '
    .ok == true and
    (.gates | keys | sort) == ["corpus", "dev", "schema"] and
    .case_count == 36 and
    .core_count == 27 and
    .challenge_count == 9 and
    .gates.corpus.verified_evidence_option_count == 93
  '
```

### 6.2 배포 전 packet·빈 응답 생성과 검증

이 하위 단계는 **사람이 JSON을 편집하기 전에만** 실행한다. 응답을 회수한 뒤에는
완료된 응답이 빈 template과 다른 것이 정상이므로 `--check-templates`를 다시 실행하지
않는다. Active packet/A/B가 모두 없는 새 SHA에서는 생성하고, 이미 빈 template을
생성한 상태로 셸만 다시 연 경우에는 no-clobber 생성은 건너뛰고 같은 검증만 수행한다.
한쪽 A/B만 존재하는 partial 상태는 아래 `test`에서 중단하고 원인을 조사한다.

```bash
if test ! -e "$ACTIVE_REVIEW_PACKET"; then
  test ! -e "$ACTIVE_REVIEW_A"
  test ! -e "$ACTIVE_REVIEW_B"
  python3 -B scripts/build_holdout_review_packet.py \
    --cases "$HOLDOUT_DRAFT" \
    --output "$ACTIVE_REVIEW_PACKET"
fi

python3 -B scripts/build_holdout_review_packet.py \
  --cases "$HOLDOUT_DRAFT" \
  --output "$ACTIVE_REVIEW_PACKET" \
  --check

if test ! -e "$ACTIVE_REVIEW_A" || test ! -e "$ACTIVE_REVIEW_B"; then
  test ! -e "$ACTIVE_REVIEW_A"
  test ! -e "$ACTIVE_REVIEW_B"
  python3 -B scripts/build_holdout_signoff.py \
    --create-templates \
    --cases "$HOLDOUT_DRAFT" \
    --review-packet "$ACTIVE_REVIEW_PACKET" \
    --review-a "$ACTIVE_REVIEW_A" \
    --review-b "$ACTIVE_REVIEW_B"
fi

python3 -B scripts/build_holdout_signoff.py \
  --check-templates \
  --cases "$HOLDOUT_DRAFT" \
  --review-packet "$ACTIVE_REVIEW_PACKET" \
  --review-a "$ACTIVE_REVIEW_A" \
  --review-b "$ACTIVE_REVIEW_B"

PACKET_SHA=$(shasum -a 256 "$ACTIVE_REVIEW_PACKET" | awk '{print $1}')
```

Reviewer A에게는 읽기 전용 active packet과 `$ACTIVE_REVIEW_A`만, Reviewer B에게는
같은 packet과 `$ACTIVE_REVIEW_B`만 전달한다. 서로의 응답 파일은 병합 전까지 공개하지
않는다. 각 36문항에서 다음 네 항목을 확인한다.

- `source_verified`
- `gold_verified`
- `answerability_verified`
- `label_verified`

각 reviewer는 자기 JSON의 placeholder `reviewer_id`를 실제 식별자로 바꾸고,
독립 검토를 끝낸 뒤에만 `independent_review_confirmed=true`로 바꾼다. 확인한 항목만
JSON boolean `true`, 최종 통과 문항만 `decision="PASS"`로 기록한다. 자동으로
값을 채우거나 다른 reviewer 파일을 복사하지 않는다.

한 문항이라도 REVISE/BLOCK이면 merge하지 않는다. Draft를 수정하고 세 기계 gate를
다시 통과시킨다. 공통 변수 블록을 다시 실행하면 새 cases SHA가 포함된 **새 active
packet·A/B 응답·sign-off 경로**가 자동으로 선택된다. 6.2에서 새 파일을 만든 뒤 두
사람 모두 36문항 전체를 다시 검토한다. Canonical packet과 이전 응답·sign-off는
덮어쓰지 않는다.

### 6.3 완료 응답 회수·resume·strict merge

두 독립 응답이 모두 끝났을 때만 병합한다. 병합기는 exact cases/packet SHA,
36개 case 순서, 동일한 두 reviewer roster, strictly-true 네 check, `PASS`, 응답
파일 SHA를 검증하고 새 sign-off를 immutable/no-clobber로 만든다.

검토가 끝난 뒤 새 셸에서 재개한다면 **5절 공통 변수와 6.1 기계 gate만** 다시
실행한 다음 아래 블록으로 간다. 6.2의 빈-template 검사는 완료 응답에 실행하지 않는다.
한 문항이라도 REVISE/BLOCK이면 아래 merge를 실행하지 않고 앞 문단의 새-SHA 절차로
돌아간다.

```bash
PACKET_SHA=$(shasum -a 256 "$ACTIVE_REVIEW_PACKET" | awk '{print $1}')

python3 -B scripts/build_holdout_signoff.py \
  --merge \
  --cases "$HOLDOUT_DRAFT" \
  --review-packet "$ACTIVE_REVIEW_PACKET" \
  --review-a "$ACTIVE_REVIEW_A" \
  --review-b "$ACTIVE_REVIEW_B" \
  --output "$ACTIVE_SIGNOFF"

jq -e \
  --arg draft_sha "$DRAFT_SHA" \
  --arg packet_sha "$PACKET_SHA" \
  '.cases_sha256 == $draft_sha and
   .review_packet_sha256 == $packet_sha and
   (.review_response_sha256s | length) == 2 and
   (.case_signoffs | length) == 36' \
  "$ACTIVE_SIGNOFF"
```

Validator는 sign-off JSON만 신뢰하지 않는다. 그 manifest가 가리키는 읽기 전용
packet과 A/B 원본 응답을 다시 열어 SHA, reviewer slot/ID, 각 case 판정과 메모까지
대조한다. 문항마다 reviewer 조합을 바꾸거나 제3자를 섞어도 실패한다.

사람 sign-off 뒤 먼저 draft에 네 gate를 실행한다.

```bash
python3 -B scripts/validate_service_holdout.py \
  --cases "$HOLDOUT_DRAFT" \
  --dev-manifest "$DEV_MANIFEST" \
  --corpus-index "$CASCADE_INDEX" \
  --signoff "$ACTIVE_SIGNOFF" \
  --json | jq -e '
    .ok == true and
    .case_count == 36 and
    .core_count == 27 and
    .challenge_count == 9 and
    .gates.corpus.verified_evidence_option_count == 93 and
    .gates.signoff.approved_case_count == 36 and
    .gates.signoff.validated_review_response_count == 2
  '
```

주의: `--dev-cases config/pnu-service-answer-eval.jsonl`을 추가하지 않는다. 그
legacy DEV 파일에는 v2 source identity 필드가 없어서 잘못된 preflight failure를
만든다. 정본 DEV 분리 입력은 `$DEV_MANIFEST`다.

## 7. Step 2 — 최종 holdout과 code freeze

이 단계의 파일 복사, commit, tag는 **사용자 명시 승인 뒤에만** 수행한다. 현재
세션에서는 수행하지 않는다.

승인 뒤 final filename을 처음 만들 때만 다음 형태를 사용한다.

```bash
test "$DRAFT_SHA" = "$(jq -er '.cases_sha256' "$ACTIVE_SIGNOFF")"
python3 -B - "$HOLDOUT_DRAFT" "$HOLDOUT" "$DRAFT_SHA" <<'PY'
import hashlib
import sys
from pathlib import Path

from scripts.immutable_outputs import publish_immutable_texts

source = Path(sys.argv[1])
target = Path(sys.argv[2])
expected_sha = sys.argv[3]
payload = source.read_bytes()
actual_sha = hashlib.sha256(payload).hexdigest()
if actual_sha != expected_sha:
    raise SystemExit(
        f"draft changed before final publication: {actual_sha} != {expected_sha}"
    )
try:
    text = payload.decode("utf-8")
except UnicodeDecodeError as exc:
    raise SystemExit(f"draft must be UTF-8: {exc}") from exc
publish_immutable_texts({target: text}, authoritative_path=target)
if source.read_bytes() != payload or target.read_bytes() != payload:
    raise SystemExit("draft/final bytes changed during immutable publication")
PY
cmp -s "$HOLDOUT_DRAFT" "$HOLDOUT"
shasum -a 256 "$HOLDOUT"
```

출력 SHA가 sign-off의 `cases_sha256`과 정확히 같아야 한다. Final filename으로
validator를 다시 실행해 네 gate가 모두 PASS하는지 확인한다.

`authoring_status=draft_unreviewed`는 case가 처음 작성된 시점의 상태를 기록하는
필드다. 최종 승인 상태의 정본은 SHA-bound 외부 sign-off이며, byte-copy 후 이 필드
하나만 바꿔 sign-off 결합을 깨뜨리지 않는다.

```bash
python3 -B scripts/validate_service_holdout.py \
  --cases "$HOLDOUT" \
  --dev-manifest "$DEV_MANIFEST" \
  --corpus-index "$CASCADE_INDEX" \
  --signoff "$ACTIVE_SIGNOFF" \
  --json | jq -e --arg expected_sha "$DRAFT_SHA" '
    .ok == true and
    .cases_sha256 == $expected_sha and
    .case_count == 36 and
    .gates.corpus.verified_evidence_option_count == 93 and
    .gates.signoff.approved_case_count == 36 and
    .gates.signoff.validated_review_response_count == 2
  '
```

B1·B3·B4 구현과 2026-09-02 저장소 전체 검사는 통과했다. 로컬 fixture socket이
허용된 환경에서 `bun run check` 결과는 **637 tests OK(6 skip)**였고
ESLint·TypeScript·Vite production build도 PASS했다. 다만 dirty 작업 snapshot의
결과이므로 Code-freeze 직전에는 다음 저장소 전체 품질 gate를 다시 실행한다.

2026-09-04 서비스 튜닝 동결 스냅샷은
`evidence/20260914/tuning-freeze-snapshot-20260904.json`에 기록했다. 이때
`scripts/search_api.py`의 `/health` `startup_code_sha256` 방식 SHA는
`9a8865db361d6d3a044d26fc57698f56840540ed573a5f37bb50ef684c1cf4a5`였고,
세 index SHA도 같은 스냅샷에 고정했다. `git diff --check`, unittest 747건
(6 skip), lint, build가 최종적으로 통과했다. 이후 서비스 튜닝 코드는 변경하지
않고 분석 도구·테스트·문서만 준비하며, 아래 gate는 그 작업까지 끝난 뒤 다시
실행한다.

```bash
git diff --check
python3 -B -m unittest discover -s tests -p 'test_*.py'
bun run lint
bun run build
git status --short
```

2026-09-02의 637/6 결과와 2026-09-04 서비스 튜닝 동결의 747/6 결과는 각각 그
시점의 snapshot이다. 분석 도구·테스트·문서까지 포함한 최종 freeze 직전에는 위
명령을 다시 실행하고 실제 test/skip 수를 이 문서와 manifest metadata에 기록한다.
실패 또는 skip 사유 변경이 있으면 해소하기 전까지 중단한다.

2026-09-04 분석 도구·테스트 추가 후 freeze-prep 재검사에서는 `git diff --check`,
lint, build가 모두 통과했고 unittest는 **755 tests OK, 6 skipped, 실패 0**이었다.
Vite는 1,735 modules를 변환했다. 이는 아직 commit/tag 전 결과이며, staging 뒤
`git diff --cached --check`와 exact staged-path 대조를 한 번 더 수행한다.

사용자 승인 뒤에만 승인된 정확한 파일 목록을 stage하고 freeze한다.

```bash
git add -- <사용자가 승인한 정확한 파일 목록>
git diff --cached --check
git diff --cached --stat
git commit -m "chore: freeze PNU final evaluation code"
git tag -a pnu-eval-code-freeze-20260904-v1 -m "PNU final evaluation code freeze"
test -z "$(git status --porcelain)"
git rev-parse HEAD
```

`git add -A`와 `git add .`은 쓰지 않는다. Push도 별도 요청 없이는 하지 않는다.
Freeze 뒤 코드, holdout, sign-off, 세 index의 SHA-256과 commit을 기록한다.

## 8. Step 3 — 평가 selector와 호출 계획 동결

아래 selector는 답변을 보기 전에 만든다. Core 27, Challenge 9, 영역별
`multi_evidence` 9개 oracle selector, 영역별 `single_fact` 9개 Judge-stability
selector 중 하나라도 어긋나면 중단한다. 두 9개 selector의 목적을 섞지 않는다.

```bash
jq -r 'select(.split == "holdout-core") | .id' "$HOLDOUT" \
  > "$RUN_ROOT/meta/core.ids"
jq -r 'select(.split == "holdout-challenge") | .id' "$HOLDOUT" \
  > "$RUN_ROOT/meta/challenge.ids"
jq -r 'select(.split == "holdout-core" and .difficulty_type == "multi_evidence") | .id' \
  "$HOLDOUT" > "$RUN_ROOT/meta/oracle.ids"
jq -r 'select(.split == "holdout-core" and .difficulty_type == "single_fact") | .id' \
  "$HOLDOUT" > "$RUN_ROOT/meta/stability.ids"

test "$(wc -l < "$RUN_ROOT/meta/core.ids" | tr -d ' ')" = 27
test "$(wc -l < "$RUN_ROOT/meta/challenge.ids" | tr -d ' ')" = 9
test "$(wc -l < "$RUN_ROOT/meta/oracle.ids" | tr -d ' ')" = 9
test "$(wc -l < "$RUN_ROOT/meta/stability.ids" | tr -d ' ')" = 9
test "$(jq -s '[.[] | select(.split == "holdout-core" and .difficulty_type == "multi_evidence") | .category] | unique | length' "$HOLDOUT")" = 9
test "$(jq -s '[.[] | select(.split == "holdout-core" and .difficulty_type == "single_fact") | .category] | unique | length' "$HOLDOUT")" = 9

CORE_IDS=$(paste -sd, "$RUN_ROOT/meta/core.ids")
CHALLENGE_IDS=$(paste -sd, "$RUN_ROOT/meta/challenge.ids")
ORACLE_IDS=$(paste -sd, "$RUN_ROOT/meta/oracle.ids")
STABILITY_IDS=$(paste -sd, "$RUN_ROOT/meta/stability.ids")

shasum -a 256 \
  "$HOLDOUT" "$ACTIVE_SIGNOFF" "$CASCADE_INDEX" \
  "$BASELINE_INDEX" "$CHALLENGER_INDEX" \
  "$PRICING_SNAPSHOT" \
  scripts/search_api.py scripts/evaluate_service_answers.py \
  scripts/run_final_retrieval_schedule.py \
  scripts/run_final_generation_schedule.py \
  scripts/evaluate_oracle_context_answers.py \
  scripts/judge_service_answers.py scripts/build_answer_review_packet.py \
  scripts/analyze_holdout_retrieval.py \
  scripts/analyze_judge_human_calibration.py \
  scripts/analyze_final_generation_gfc.py \
  scripts/build_evidence_manifest.py \
  > "$RUN_ROOT/meta/input-sha256.txt"
```

두 정본 schedule은 clean freeze commit에서 답변을 보기 전에 한 번만 만든다.
Schedule 파일은 기존 경로를 덮어쓰지 않는다. Generation runner가 seed `20260914`로
영역 층화 AB/BA 41/40을 직접 계산하므로 별도 AWK schedule을 만들지 않는다.

```bash
test -z "$(git status --porcelain)"
EXPECTED_GIT_COMMIT=$(git rev-parse HEAD)
CASCADE_INDEX_SHA=$(shasum -a 256 "$CASCADE_INDEX" | awk '{print $1}')

test ! -e "$RETRIEVAL_SCHEDULE"
python3 -B scripts/run_final_retrieval_schedule.py \
  --create \
  --schedule "$RETRIEVAL_SCHEDULE" \
  --cases "$HOLDOUT" \
  --experiment-id "$EXPERIMENT_ID" \
  --output-root "$RUN_ROOT" \
  --expected-git-commit "$EXPECTED_GIT_COMMIT" \
  --expected-source-manifest-sha256 "$SOURCE_MANIFEST_SHA" \
  --par-b-api-base http://127.0.0.1:8100 \
  --par-b-index "$BASELINE_INDEX" \
  --par-b-revision "$BASELINE_CORPUS_REVISION" \
  --par-ch-api-base http://127.0.0.1:8100 \
  --par-ch-index "$CHALLENGER_INDEX" \
  --par-ch-revision "$CHALLENGER_CORPUS_REVISION" \
  --c0-api-base http://127.0.0.1:8101 \
  --c0-index "$CASCADE_INDEX" \
  --c0-revision "$CORPUS_REVISION" \
  --c1-api-base http://127.0.0.1:8100 \
  --c1-index "$CASCADE_INDEX" \
  --c1-revision "$CORPUS_REVISION"

test ! -e "$GENERATION_SCHEDULE"
python3 -B scripts/run_final_generation_schedule.py \
  --create \
  --schedule "$GENERATION_SCHEDULE" \
  --cases "$HOLDOUT" \
  --experiment-id "$EXPERIMENT_ID" \
  --output-root "$RUN_ROOT" \
  --expected-corpus-revision "$CORPUS_REVISION" \
  --expected-git-commit "$EXPECTED_GIT_COMMIT" \
  --expected-index-sha256 "$CASCADE_INDEX_SHA" \
  --expected-source-manifest-sha256 "$SOURCE_MANIFEST_SHA" \
  --model gemini-3.5-flash-lite \
  --c0-api-base http://127.0.0.1:8101 \
  --c1-api-base http://127.0.0.1:8100

python3 -B scripts/run_final_retrieval_schedule.py \
  --check --schedule "$RETRIEVAL_SCHEDULE"
python3 -B scripts/run_final_generation_schedule.py \
  --check --schedule "$GENERATION_SCHEDULE"
```

Challenge는 C1만 run 1→3 순서로 실행한다. Oracle은 영역별 `multi_evidence`
9개 ID를 C1 frozen generator에 gold context로 1회 전달한다. Judge stability는
별도 `single_fact` 9개를 C0/C1 양쪽에서 사용한다. Schedule 생성 뒤에는 holdout,
sign-off, index, code 또는 schedule을 변경하지 않는다.

## 9. Step 4 — C0/C1 서버 시작과 health gate

C1은 tuning ON, C0은 tuning OFF다. 두 서버는 같은 freeze commit, index,
generator 환경을 사용한다. 서로 다른 터미널에서 실행한다.

### Terminal A: C1, tuning ON, port 8100

```bash
RAG_EVAL_TRACE=1 \
RAG_GENERATION_MAX_CONTEXT_CHARS=24000 \
RAG_GEMINI_MAX_OUTPUT_TOKENS=900 \
RAG_GEMINI_MODEL=gemini-3.5-flash-lite \
RAG_GEMINI_FALLBACK_MODELS=gemini-3.5-flash-lite \
python3 -B scripts/search_api.py \
  --host 127.0.0.1 \
  --port 8100 \
  --index "$CASCADE_INDEX" \
  --profile-index baseline="$BASELINE_INDEX" \
  --profile-index challenger="$CHALLENGER_INDEX" \
  --default-parser-profile cascade \
  --context-chunks-per-document 2 \
  --env-file .env
```

### Terminal B: C0, tuning OFF, port 8101

```bash
RAG_EVAL_TRACE=1 \
RAG_GENERATION_MAX_CONTEXT_CHARS=24000 \
RAG_GEMINI_MAX_OUTPUT_TOKENS=900 \
RAG_GEMINI_MODEL=gemini-3.5-flash-lite \
RAG_GEMINI_FALLBACK_MODELS=gemini-3.5-flash-lite \
python3 -B scripts/search_api.py \
  --host 127.0.0.1 \
  --port 8101 \
  --index "$CASCADE_INDEX" \
  --profile-index baseline="$BASELINE_INDEX" \
  --profile-index challenger="$CHALLENGER_INDEX" \
  --default-parser-profile cascade \
  --no-retrieval-tuning \
  --context-chunks-per-document 2 \
  --env-file .env
```

세 번째 터미널에서 양쪽 health를 fail-closed로 확인한다.

```bash
curl -fsS http://127.0.0.1:8100/health | jq -e \
  --arg baseline_rev "$BASELINE_CORPUS_REVISION" \
  --arg challenger_rev "$CHALLENGER_CORPUS_REVISION" \
  --arg cascade_rev "$CORPUS_REVISION" '
    .ready == true and
    .default_parser_profile == "cascade" and
    .service_config.retrieval_tuning == true and
    .service_config.context_chunks_per_document == 2 and
    .service_config.evaluation_trace_enabled == true and
    any(.parser_profiles[];
      .id == "baseline" and .ready == true and
      .corpus_revision == $baseline_rev and
      any(.retrieval_modes[]; .id == "bm25" and .ready == true)) and
    any(.parser_profiles[];
      .id == "challenger" and .ready == true and
      .corpus_revision == $challenger_rev and
      any(.retrieval_modes[]; .id == "bm25" and .ready == true)) and
    any(.parser_profiles[];
      .id == "cascade" and .ready == true and
      .corpus_revision == $cascade_rev and
      any(.retrieval_modes[]; .id == "bm25" and .ready == true))
  '

curl -fsS http://127.0.0.1:8101/health | jq -e \
  --arg baseline_rev "$BASELINE_CORPUS_REVISION" \
  --arg challenger_rev "$CHALLENGER_CORPUS_REVISION" \
  --arg cascade_rev "$CORPUS_REVISION" '
    .ready == true and
    .default_parser_profile == "cascade" and
    .service_config.retrieval_tuning == false and
    .service_config.context_chunks_per_document == 2 and
    .service_config.evaluation_trace_enabled == true and
    any(.parser_profiles[];
      .id == "baseline" and .ready == true and
      .corpus_revision == $baseline_rev and
      any(.retrieval_modes[]; .id == "bm25" and .ready == true)) and
    any(.parser_profiles[];
      .id == "challenger" and .ready == true and
      .corpus_revision == $challenger_rev and
      any(.retrieval_modes[]; .id == "bm25" and .ready == true)) and
    any(.parser_profiles[];
      .id == "cascade" and .ready == true and
      .corpus_revision == $cascade_rev and
      any(.retrieval_modes[]; .id == "bm25" and .ready == true))
  '
```

둘 중 하나라도 실패하면 collector를 실행하지 않는다. 서버를 재시작한 경우 health
gate를 다시 실행하고 재시작 시각을 log에 남긴다.

## 10. Step 5 — retrieval-only append-only run

정본 retrieval runner는 PAR-B/PAR-CH/C0/C1을 case-major로 한 slot씩 실행한다.
먼저 동일한 4-gate를 적용하되 서비스 호출이 없는 dry-run을 한다.

```bash
python3 -B scripts/run_final_retrieval_schedule.py \
  --run \
  --dry-run \
  --schedule "$RETRIEVAL_SCHEDULE" \
  --dev-manifest "$DEV_MANIFEST" \
  --signoff "$ACTIVE_SIGNOFF"
```

Dry-run 결과가 `completed_call_count=0`, `total_call_count=108`,
`next_call_order=1`이고 위 NO-GO 조건이 모두 해소된 뒤, 사용자가 실제 final
수집을 명시 승인했을 때만 다음 live 명령을 실행한다.

```bash
python3 -B scripts/run_final_retrieval_schedule.py \
  --run \
  --schedule "$RETRIEVAL_SCHEDULE" \
  --dev-manifest "$DEV_MANIFEST" \
  --signoff "$ACTIVE_SIGNOFF" \
  --authorize-final-collection

python3 -B scripts/run_final_retrieval_schedule.py \
  --check --schedule "$RETRIEVAL_SCHEDULE"
```

Live runner는 `.final-retrieval-run.lock`을 `O_EXCL`로 획득하고, 각 answer마다
정확한 `slot_started → slot_completed` WAL과 전역 `call_order` prefix를 검사한다.
각 slot 전후에는 holdout·DEV·index·sign-off와 그 packet·A/B 원본의 byte binding을
전체 재구성해 exact-match를 요구한다.
다른 runner가 lock을 보유하거나 crash가 남긴 stale lock, non-empty error artifact,
uncertain/poisoned WAL이 있으면 중단한다. Lock을 조용히 지우거나 direct collector로
건너뛰지 않는다. 완료 audit은 `complete=true`, `completed_call_count=108`이어야 한다.

그 다음 네 artifact의 byte SHA를 명시하여 정본 retrieval 통계를 한 번 생성한다.
JSON·CSV를 먼저 내구성 있게 게시하고 completion manifest를 마지막 권위 artifact로
게시하므로, 세 경로 중 하나라도 이미 있으면 새 분석 ID/경로를 사용한다.

```bash
PAR_B="$RUN_ROOT/retrieval/par-b.answers.jsonl"
PAR_CH="$RUN_ROOT/retrieval/par-ch.answers.jsonl"
RET_C0="$RUN_ROOT/retrieval/c0.answers.jsonl"
RET_C1="$RUN_ROOT/retrieval/c1.answers.jsonl"
RET_JSON="$RUN_ROOT/stats/final-retrieval.json"
RET_CSV="$RUN_ROOT/stats/final-retrieval-cases.csv"
RET_COMPLETE="$RUN_ROOT/stats/final-retrieval-completion.json"

HOLDOUT_SHA=$(shasum -a 256 "$HOLDOUT" | awk '{print $1}')
PAR_B_SHA=$(shasum -a 256 "$PAR_B" | awk '{print $1}')
PAR_CH_SHA=$(shasum -a 256 "$PAR_CH" | awk '{print $1}')
RET_C0_SHA=$(shasum -a 256 "$RET_C0" | awk '{print $1}')
RET_C1_SHA=$(shasum -a 256 "$RET_C1" | awk '{print $1}')

test ! -e "$RET_JSON"
test ! -e "$RET_CSV"
test ! -e "$RET_COMPLETE"
python3 -B scripts/analyze_holdout_retrieval.py \
  --cases "$HOLDOUT" \
  --cases-sha256 "$HOLDOUT_SHA" \
  --schedule "$RETRIEVAL_SCHEDULE" \
  --par-b "$PAR_B" \
  --par-b-sha256 "$PAR_B_SHA" \
  --par-ch "$PAR_CH" \
  --par-ch-sha256 "$PAR_CH_SHA" \
  --c0 "$RET_C0" \
  --c0-sha256 "$RET_C0_SHA" \
  --c1 "$RET_C1" \
  --c1-sha256 "$RET_C1_SHA" \
  --completion-manifest "$RET_COMPLETE" \
  --json-out "$RET_JSON" \
  --csv-out "$RET_CSV" \
  --bootstrap 10000 \
  --seed 20260914 \
  --expected-experiment-id "$EXPERIMENT_ID" \
  --expected-generation-run-id retrieval-once
```

Collector의 legacy `retrieval_hit`을 Source Hit/MRR로 보고하지 않는다. 최종 수치는
`$RET_COMPLETE`가 존재하고 입력 schedule/answer/WAL 재감사가 모두 PASS한
`$RET_JSON`에서만 인용한다.

## 11. Step 6 — generation append-only run

Direct generation의 공통 pin은 다음과 같다.

```text
provider=frontier
model=gemini-3.5-flash-lite
parser_profile=cascade
retrieval_mode=bm25
institution=null (`--institution none`)
context_k=8
max context chars=24000
max output tokens=900
temperature/seed=unsupported, not sent
```

Direct 189 slot은 frozen generation schedule 외의 경로로 실행하지 않는다. 먼저
동일한 holdout/sign-off/corpus/Git gate를 적용하는 dry-run으로 다음 slot과 기존
artifact prefix를 확인한다.

```bash
python3 -B scripts/run_final_generation_schedule.py \
  --run \
  --dry-run \
  --schedule "$GENERATION_SCHEDULE" \
  --dev-manifest "$DEV_MANIFEST" \
  --corpus-index "$CASCADE_INDEX" \
  --signoff "$ACTIVE_SIGNOFF"
```

Dry-run 결과가 `completed_call_count=0`, `total_call_count=189`,
`next_call_order=1`이고 위 NO-GO 조건이 모두 해소된 뒤, 사용자가 실제 외부 생성을
명시 승인했을 때만 live run을 실행한다.

```bash
python3 -B scripts/run_final_generation_schedule.py \
  --run \
  --schedule "$GENERATION_SCHEDULE" \
  --dev-manifest "$DEV_MANIFEST" \
  --corpus-index "$CASCADE_INDEX" \
  --signoff "$ACTIVE_SIGNOFF" \
  --authorize-final-collection

python3 -B scripts/run_final_generation_schedule.py \
  --check --schedule "$GENERATION_SCHEDULE"
```

Live runner는 `.final-generation-run.lock`을 `O_EXCL`로 획득하고 모든 호출 전후에
현재 holdout/sign-off/packet/A·B 원본/index/source-manifest byte, clean Git commit,
schedule과 server startup identity를 재검사한다. 각 slot은 fsync된 정확한
`slot_started → slot_completed` WAL을 가져야 하며, 전역 `call_order` prefix에서
한 칸만 증가해야 한다. 정상 종료 시 lock은 제거되지만 crash가 남긴 stale lock,
`slot_started`만 있는 uncertain slot, `slot_poisoned`는 자동 재실행하지 않는다.

필요 lane과 예상 record는 다음과 같다.

| Lane | Port | Tuning | Runs | Cases/run | 합계 |
|---|---:|---|---:|---:|---:|
| `c0-core` | 8101 | OFF | 3 | 27 | 81 |
| `c1-core` | 8100 | ON | 3 | 27 | 81 |
| `c1-challenge` | 8100 | ON | 3 | 9 | 27 |
| `c1-oracle-context` | retrieval 우회 | N/A | 1 | 9 | 9 |

각 direct artifact는 schedule이 지정한 27개 또는 9개 논리 slot을 정확히 가진다.
성공 row는 frozen frontier/model/config와 request attempt provenance를 가져야 한다.
최대 세 번의 retryable transport attempt 뒤에도 끝나지 않은 slot은 terminal
`service_error` row와 정확히 한 error sidecar로 종결한다. 이 row도 189개 논리
slot에는 포함하지만 가짜 answer text나 Judge score를 갖지 않는다. Control mismatch,
response contract 위반, uncertain/poisoned WAL은 전체 run을 중단한다. 출력 품질을
이유로 새 답변을 뽑지 않는다.

Oracle lane은 `/chat` 서버가 아니라 같은 generator/postprocessor를 in-process로
호출한다. Secret이 이미 안전하게 주입된 shell에서 다음 명령을 실행한다. 실제
secret 값은 명령행이나 log에 넣지 않는다.

```bash
ORACLE_OUT="$RUN_ROOT/generation/c1-oracle-context-run1.answers.jsonl"
test ! -e "$ORACLE_OUT"

python3 -B scripts/evaluate_oracle_context_answers.py \
  --cases "$HOLDOUT" \
  --dev-manifest "$DEV_MANIFEST" \
  --corpus-index "$CASCADE_INDEX" \
  --signoff "$ACTIVE_SIGNOFF" \
  --out "$ORACLE_OUT" \
  --experiment-id "$EXPERIMENT_ID" \
  --condition-id c1-oracle-context \
  --generation-run-id run1 \
  --provider frontier \
  --model gemini-3.5-flash-lite \
  --selection-difficulty-type multi_evidence \
  --answer-style standard \
  --max-context-chars 24000 \
  --max-output-tokens 900 \
  --deadline-seconds 45 \
  --sleep 2
```

이 CLI는 final filename과 schema·DEV 분리·corpus·사람 2인 sign-off의 4개 gate를
자체 재검사하며, 기존 output이면 호출 전에 실패한다. 종료 코드와 별도로 다음을
검사한다.

```bash
HOLDOUT_SHA=$(shasum -a 256 "$HOLDOUT" | awk '{print $1}')
SIGNOFF_SHA=$(shasum -a 256 "$ACTIVE_SIGNOFF" | awk '{print $1}')
CORPUS_SHA=$(shasum -a 256 "$CASCADE_INDEX" | awk '{print $1}')

test "$(wc -l < "$ORACLE_OUT" | tr -d ' ')" = 9
diff -u \
  <(LC_ALL=C sort "$RUN_ROOT/meta/oracle.ids") \
  <(jq -r '.case_id' "$ORACLE_OUT" | LC_ALL=C sort)

jq -s -e \
  --arg experiment "$EXPERIMENT_ID" \
  --arg holdout_sha "$HOLDOUT_SHA" \
  --arg signoff_sha "$SIGNOFF_SHA" \
  --arg corpus_sha "$CORPUS_SHA" '
  length == 9 and
  ([.[].answer_id] | unique | length) == 9 and
  ([.[].case_id] | unique | length) == 9 and
  ([.[].category] | unique | length) == 9 and
  all(.[];
    .experiment_id == $experiment and
    .condition_id == "c1-oracle-context" and
    .generation_run_id == "run1" and
    .difficulty_type == "multi_evidence" and
    .collector_config.cases_file_sha256 == $holdout_sha and
    .collector_config.signoff_file_sha256 == $signoff_sha and
    .collector_config.corpus_index_sha256 == $corpus_sha and
    .collector_config.selection_rule.split == "holdout-core" and
    .collector_config.selection_rule.difficulty_type == "multi_evidence" and
    (.collector_config.selection_rule.one_per_category | length) == 9 and
    (.collector_config.selected_case_ids | length) == 9 and
    .generation.requested == "frontier" and
    .generation.used == "frontier" and
    .generation.model == "gemini-3.5-flash-lite" and
    .generation.fallback_reason == null and
    .request.context_source == "gold_evidence_options" and
    .request.max_context_chars == 24000 and
    .request.max_output_tokens == 900 and
    .retrieval.mode == "oracle_gold_evidence" and
    .retrieval.retrieval_bypassed == true and
    .retrieval.service_performance_eligible == false and
    .oracle_diagnostic.diagnostic_only == true and
    .oracle_diagnostic.retrieval_bypassed == true and
    .oracle_diagnostic.service_performance_eligible == false and
    .oracle_diagnostic.aggregation_instruction == "exclude_from_service_performance" and
    .oracle_diagnostic.sampling_parameters.temperature.status == "not_sent" and
    .oracle_diagnostic.sampling_parameters.seed.status == "unsupported_not_sent" and
    (.oracle_diagnostic.oracle_contexts_sha256 | length) == 64 and
    (.oracle_diagnostic.gold_evidence_provenance_sha256 | length) == 64 and
    (.answer | type == "string" and length > 0) and
    (.answer_sha256 | type == "string" and length == 64) and
    (.record_sha256 | type == "string" and length == 64))
  ' "$ORACLE_OUT"
```

Oracle batch가 중간 실패하면 partial 파일을 삭제·수정·append하지 않는다. 그 파일은
실패 근거로 보존하고 새 `generation_run_id`와 새 output 경로로 9개 전체를 다시
실행한다. 성공한 9개도 서비스 headline이나 C0/C1 paired 통계에 섞지 않는다.

## 12. Step 7 — Judge v11

Judge v11의 결정론적 guard는 세 가지뿐이다. (1) 답할 수 있는 질문에 최종 답변이
명시적으로 회피하면 score 0·GFC false로 확정하고, (2) `supported/partial` claim의
`answer_quote`가 최종 답변에서 연속 부분 문자열로 확인되지 않으면 `missing`으로
강등하며, (3) GFC true가 score·claim·미지지 사실·모순·citation 세부 판정과
자기모순이면 GFC false와 score≤1로 확정한다. 이 규칙들은 credit을 제거하거나
유지할 뿐 점수 또는 GFC를 올리는 규칙은 없다. 원본 모델 응답과 guard 적용 전
필드는 artifact에 함께 보존한다.

먼저 모든 answer artifact를 API 호출 없이 검증한다. Partial artifact인 현재
형식에서는 `--allow-partial`이 필요하다.

```bash
python3 -B scripts/judge_service_answers.py \
  --answers "$RUN_ROOT/generation/c0-core-run1.answers.jsonl" \
  --cases "$HOLDOUT" \
  --out "$RUN_ROOT/judge/c0-core-run1-judge-v11-r1.jsonl" \
  --judge-run-id judge-v11-r1 \
  --judge-model gemini-3.1-flash-lite \
  --max-output-tokens 1600 \
  --expected-experiment-id "$EXPERIMENT_ID" \
  --expected-condition-id c0 \
  --expected-generation-run-id run1 \
  --allow-partial \
  --validate-only
```

출력의 `validated_answers`, answer SHA, `judge_config_sha256`를 log에 보존한다.
그 다음에만 같은 인자에서 `--validate-only`를 제거해 외부 Judge를 실행한다.

```bash
test ! -e "$RUN_ROOT/judge/c0-core-run1-judge-v11-r1.jsonl"
python3 -B scripts/judge_service_answers.py \
  --answers "$RUN_ROOT/generation/c0-core-run1.answers.jsonl" \
  --cases "$HOLDOUT" \
  --out "$RUN_ROOT/judge/c0-core-run1-judge-v11-r1.jsonl" \
  --judge-run-id judge-v11-r1 \
  --judge-model gemini-3.1-flash-lite \
  --max-output-tokens 1600 \
  --timeout 60 \
  --retries 4 \
  --sleep 2 \
  --expected-experiment-id "$EXPERIMENT_ID" \
  --expected-condition-id c0 \
  --expected-generation-run-id run1 \
  --allow-partial
```

이 패턴으로 성공한 generation answer를 정확히 한 번씩 Judge한다. Terminal
generation error가 없을 때 최초 판정은 198개다. 새 answer artifact마다
`--answers`, expected condition/run, output path를 바꾼다. 같은 논리적 첫 판정은
`judge-run-id=judge-v11-r1`을 유지한다. Terminal error slot에는 가짜 answer를
만들거나 Judge 호출을 채워 넣지 않는다.

각 Judge 파일은 종료 코드와 별도로 generation artifact의 Judge-eligible 성공 slot과
정확히 같은 수인지 검사한다. Terminal service-error slot은 자동 제외하며 숫자를
27로 하드코딩하지 않는다.

```bash
ANSWERS="$RUN_ROOT/generation/c0-core-run1.answers.jsonl"
JUDGMENTS="$RUN_ROOT/judge/c0-core-run1-judge-v11-r1.jsonl"
EXPECTED_ELIGIBLE=$(jq -s '[.[] | select(.slot_outcome == "answer")] | length' "$ANSWERS")
test "$(wc -l < "$JUDGMENTS" | tr -d ' ')" = "$EXPECTED_ELIGIBLE"
jq -s -e --argjson expected "$EXPECTED_ELIGIBLE" '
  length == $expected and
  ([.[].judgment_id] | unique | length) == $expected and
  all(.[];
    .error == null and
    .judge_config.rubric_version == "pnu-grounded-fully-correct-v11" and
    .judge_config.max_output_tokens == 1600 and
    (.judge.score == 0 or .judge.score == 1 or .judge.score == 2) and
    (.judge.grounded_fully_correct | type == "boolean") and
    (.answer_sha256 | length == 64))
  ' "$JUDGMENTS"
```

고정 stability set은 generation run 1의 9개 `single_fact` 질문을 C0/C1 양쪽에서
택한 18개 답변이다. 이 18개만 `judge-v11-r2`, `judge-v11-r3`로 두 번 더 판정해
최대 36회를 만든다. Partial Judge는 답변 artifact SHA, 답변 순서, 선택 answer ID를
고정한 selection manifest가 반드시 필요하다.

```bash
C0_STABILITY_SELECTION="$RUN_ROOT/meta/c0-run1-stability-selection.json"
C1_STABILITY_SELECTION="$RUN_ROOT/meta/c1-run1-stability-selection.json"

python3 -B scripts/build_judge_repeat_selection.py \
  --answers "$RUN_ROOT/generation/c0-core-run1.answers.jsonl" \
  --case-ids "$STABILITY_IDS" \
  --out "$C0_STABILITY_SELECTION"
python3 -B scripts/build_judge_repeat_selection.py \
  --answers "$RUN_ROOT/generation/c1-core-run1.answers.jsonl" \
  --case-ids "$STABILITY_IDS" \
  --out "$C1_STABILITY_SELECTION"

# C0 judge-v11-r2 예시. r3와 C1도 새 output/judge-run-id로 같은 방식으로 실행한다.
python3 -B scripts/judge_service_answers.py \
  --answers "$RUN_ROOT/generation/c0-core-run1.answers.jsonl" \
  --cases "$HOLDOUT" \
  --out "$RUN_ROOT/judge/c0-core-run1-judge-v11-r2.jsonl" \
  --judge-run-id judge-v11-r2 \
  --judge-model gemini-3.1-flash-lite \
  --max-output-tokens 1600 \
  --timeout 60 \
  --retries 4 \
  --sleep 2 \
  --expected-experiment-id "$EXPERIMENT_ID" \
  --expected-condition-id c0 \
  --expected-generation-run-id run1 \
  --only "$STABILITY_IDS" \
  --allow-partial \
  --selection-manifest "$C0_STABILITY_SELECTION"
```

선택된 답변이 terminal이면 대체 표본을 뽑지 않고 Judge 호출도 하지 않는다.
`summarize_judge_repeats.py --selection-manifest`에는 selection-bound 추가 판정
r2/r3 두 artifact만 전달한다. 최초 definitive r1은 별도 정본 판정으로 유지한다.
이 안정성 결과는 주 GFC의 generation n=3과 별도 진단으로만 보고한다. Judge API
error를 재판정할 때도 기존 파일을 수정하지 않고 새 `judge_run_id`와 새 파일을 쓴다.

## 13. Step 8 — condition-blind 사람 packet, A/B, adjudication

사람 검토는 run 1의 C0 Core, C1 Core, C1 Challenge에서 성공한 답변만 사용한다.
Terminal slot은 blind item에서 제외하지만 private mapping에 자동 GFC=0 대상과 전체
논리 slot 수로 남는다. 아래 다섯 output이 모두 없을 때 한 번 생성한다.

```bash
PACKET="$RUN_ROOT/human/blind-review.md"
MAPPING="$RUN_ROOT/human/private-mapping.json"
LABEL_A="$RUN_ROOT/human/reviewer-a.jsonl"
LABEL_B="$RUN_ROOT/human/reviewer-b.jsonl"
ADJ="$RUN_ROOT/human/adjudication.jsonl"

test ! -e "$PACKET"
test ! -e "$MAPPING"
test ! -e "$LABEL_A"
test ! -e "$LABEL_B"
test ! -e "$ADJ"

python3 -B scripts/build_answer_review_packet.py \
  --cases "$HOLDOUT" \
  --answers "$RUN_ROOT/generation/c0-core-run1.answers.jsonl" \
  --answers "$RUN_ROOT/generation/c1-core-run1.answers.jsonl" \
  --answers "$RUN_ROOT/generation/c1-challenge-run1.answers.jsonl" \
  --out "$PACKET" \
  --mapping-out "$MAPPING" \
  --labels-a-out "$LABEL_A" \
  --labels-b-out "$LABEL_B" \
  --adjudication-out "$ADJ" \
  --seed 20260914

python3 -B scripts/build_answer_review_packet.py \
  --cases "$HOLDOUT" \
  --answers "$RUN_ROOT/generation/c0-core-run1.answers.jsonl" \
  --answers "$RUN_ROOT/generation/c1-core-run1.answers.jsonl" \
  --answers "$RUN_ROOT/generation/c1-challenge-run1.answers.jsonl" \
  --out "$PACKET" \
  --mapping-out "$MAPPING" \
  --labels-a-out "$LABEL_A" \
  --labels-b-out "$LABEL_B" \
  --adjudication-out "$ADJ" \
  --seed 20260914 \
  --check
```

`--check`는 템플릿을 사람이 편집하기 **전**에만 수행한다. 편집 후에는 placeholder
원본과 달라지는 것이 정상이다. 생성기는 packet/label template 네 개를 먼저
불변 게시하고 private mapping JSON을 마지막 권위 artifact로 게시한다.

- Reviewer A에게 packet과 A 파일만 전달한다.
- Reviewer B에게 packet과 B 파일만 전달한다.
- Private mapping과 상대 reviewer 파일은 독립 평가 종료 전 공개하지 않는다.
- 두 파일을 회수·hash한 뒤에만 불일치를 adjudicate한다.
- Adjudication은 불일치 사례만이 아니라 **성공한 run 1 blind item 전부** 채운다.
  Terminal slot에는 사람 label을 만들지 않는다. Terminal이 없을 때만 63개다.
- `score`, `grounded_fully_correct`, `uncertain`은 placeholder/null로 남아 있으면
  안 된다. Challenge type에 따라 `correct_abstention` 또는
  `injection_obedience`도 채운다.

## 14. Step 9 — Judge-human calibration gate

Calibration에는 사람에게 보여 준 성공한 run 1 answer의 첫 Judge 판정만 넣는다.
Terminal slot에는 Judge/사람 label을 만들지 않는다. Output이 없을 때 한 번 생성한다.

```bash
CAL_JSON="$RUN_ROOT/stats/judge-human-calibration.json"
CAL_CSV="$RUN_ROOT/stats/judge-human-calibration.csv"
test ! -e "$CAL_JSON"
test ! -e "$CAL_CSV"

python3 -B scripts/analyze_judge_human_calibration.py \
  --judgments \
    "$RUN_ROOT/judge/c0-core-run1-judge-v11-r1.jsonl" \
    "$RUN_ROOT/judge/c1-core-run1-judge-v11-r1.jsonl" \
    "$RUN_ROOT/judge/c1-challenge-run1-judge-v11-r1.jsonl" \
  --human-labels "$LABEL_A" "$LABEL_B" "$ADJ" \
  --json-out "$CAL_JSON" \
  --csv-out "$CAL_CSV"

jq '.summary.protocol_gate' "$CAL_JSON"
```

Terminal이 없을 때 기대값은 answer 63, 독립 rating 126, adjudication 63, Core
gate answer 54다. 네 기준(raw agreement ≥ 0.80, balanced accuracy ≥ 0.80,
Cohen's kappa ≥ 0.60, macro-F1 ≥ 0.75) 중 하나라도 실패하거나 kappa가 undefined면
threshold gate FAIL이다. 또한 run 1에 terminal slot이 하나라도 있으면 threshold
결과와 별개로 B4의 **effective headline gate가 FAIL**이다.

FAIL이면 3-run Judge 결과를 exploratory로 내리고, 성공한 run 1의 adjudicated human
label에 terminal 자동 GFC=0을 더한 C0/C1 Core 27쌍을 headline fallback으로 쓴다.
이 분기는 `analyze_final_generation_gfc.py`가 원본 calibration 입력을 재해시해
자동 적용한다. 사람 label을 본 뒤 Judge prompt나 threshold를 수정하면 calibration은
무효이며 새 calibration 절차가 필요하다.

## 15. Step 10 — GFC 질문 단위 paired 통계

Gate PASS 뒤의 주 추정량은 다음 하나다.

```text
각 case의 question_gfc(condition) = generation run 1/2/3 GFC 성공 수 / 3
paired_effect = 27개 case에서 question_gfc(C1) - question_gfc(C0)의 평균
```

반드시 질문 27개를 재표집한 paired bootstrap 10,000회 95% CI, paired sign-flip,
2/3 majority의 exact McNemar를 보고한다. 유효 표본은 `n=27`이다. 162 generation
출력이나 234 Judge 호출을 n으로 쓰지 않는다.

정본 분석기는 여섯 generation artifact와 여섯 최초 Judge artifact를 run 1/2/3
순서로 받는다. Calibration threshold가 PASS이고 run 1 terminal이 없을 때는
`--human-labels`를 넘기지 않는다. 그 외에는 calibration artifact에 hash로 결합된
adjudication 파일을 넘겨 사람 fallback을 활성화한다.

```bash
GFC_JSON="$RUN_ROOT/stats/final-generation-gfc.json"
GFC_CSV="$RUN_ROOT/stats/final-generation-gfc-cases.csv"

RUN1_TERMINAL_COUNT=$(jq -s \
  '[.[] | select(.slot_outcome == "service_error")] | length' \
  "$RUN_ROOT/generation/c0-core-run1.answers.jsonl" \
  "$RUN_ROOT/generation/c1-core-run1.answers.jsonl")

if jq -e '.summary.protocol_gate.passed == true' "$CAL_JSON" >/dev/null \
  && test "$RUN1_TERMINAL_COUNT" = 0; then
  GFC_HUMAN_ARGS=()
else
  GFC_HUMAN_ARGS=(--human-labels "$ADJ")
fi

test ! -e "$GFC_JSON"
test ! -e "$GFC_CSV"
python3 -B scripts/analyze_final_generation_gfc.py \
  --cases "$HOLDOUT" \
  --calibration "$CAL_JSON" \
  --c0-answers \
    "$RUN_ROOT/generation/c0-core-run1.answers.jsonl" \
    "$RUN_ROOT/generation/c0-core-run2.answers.jsonl" \
    "$RUN_ROOT/generation/c0-core-run3.answers.jsonl" \
  --c0-judgments \
    "$RUN_ROOT/judge/c0-core-run1-judge-v11-r1.jsonl" \
    "$RUN_ROOT/judge/c0-core-run2-judge-v11-r1.jsonl" \
    "$RUN_ROOT/judge/c0-core-run3-judge-v11-r1.jsonl" \
  --c1-answers \
    "$RUN_ROOT/generation/c1-core-run1.answers.jsonl" \
    "$RUN_ROOT/generation/c1-core-run2.answers.jsonl" \
    "$RUN_ROOT/generation/c1-core-run3.answers.jsonl" \
  --c1-judgments \
    "$RUN_ROOT/judge/c1-core-run1-judge-v11-r1.jsonl" \
    "$RUN_ROOT/judge/c1-core-run2-judge-v11-r1.jsonl" \
    "$RUN_ROOT/judge/c1-core-run3-judge-v11-r1.jsonl" \
  "${GFC_HUMAN_ARGS[@]}" \
  --json-out "$GFC_JSON" \
  --csv-out "$GFC_CSV" \
  --bootstrap 10000 \
  --sign-flip-iterations 100000 \
  --seed 20260914 \
  --expected-core-count 27
```

도구는 CSV를 먼저 불변 게시하고 JSON을 마지막 권위 artifact로 게시한다. 원본
answer/judgment/calibration/human-label hash를 다시 열어 검증하고 다음을 fail-closed로
거부해야 한다.

- C0/C1 case/family set 불일치
- generation run 1/2/3 누락·중복
- answer와 judgment의 SHA/condition/run 불일치
- terminal 계약을 어긴 answer, terminal을 Judge한 입력, error-bearing judgment
- 같은 answer의 Judge 재채점을 generation 반복으로 오인한 입력
- Core 이외 case가 headline 통계에 섞인 입력

보고서에는 `$GFC_JSON.headline.source`가 실제로
`llm_judge_three_independent_generation_runs`인지
`adjudicated_human_generation_run1_fallback`인지 함께 기록한다. 이 JSON/CSV가 아직
생성되지 않았으므로 현재 final 생성 성능 수치는 주장하지 않는다.

## 16. Step 11 — manifest와 result freeze

원본·사람 label·통계는 먼저 Git에서 제외된 `$RUN_ROOT` 아래에 보존한다. Code
worktree가 여전히 freeze commit과 같은지 확인한다.

```bash
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$(git rev-list -n 1 pnu-eval-code-freeze-20260904-v1)"
```

`build_evidence_manifest.py`는 다음을 검사한다.

- `--expect-jsonl PATH=COUNT`: record 수와 고유 ID
- `--require-judge-scores PATH`: score 0/1/2, error 없음
- answer↔judgment의 experiment/condition/run/case/answer SHA 연결
- `--selection-manifest PATH`: partial Judge group의 answer SHA·순서·선택 ID 연결
- `blind_item_id`/`answer_id` 기반 human-label 고유성
- `--require-clean`: manifest 작성 직전 clean worktree
- 기존 output, symlink input/output, input-output alias 거부와 fsync된 immutable publish

아래는 한 artifact pair의 문법 예다. 실제 final 명령에는 retrieval 4개,
generation 10개, Judge 최대 14개, 사람 label 3개, calibration/GFC 결과와 selector,
holdout, sign-off를 모두 나열한다.

```bash
MANIFEST="$RUN_ROOT/stats/manifest.json"
RUN1_C0_JUDGE="$RUN_ROOT/judge/c0-core-run1-judge-v11-r1.jsonl"
RUN1_C0_JUDGE_COUNT=$(wc -l < "$RUN1_C0_JUDGE" | tr -d ' ')
test ! -e "$MANIFEST"

python3 -B scripts/build_evidence_manifest.py \
  --experiment-id "$EXPERIMENT_ID" \
  --output "$MANIFEST" \
  --artifact "$HOLDOUT" \
  --artifact "$ACTIVE_SIGNOFF" \
  --artifact "$PRICING_SNAPSHOT" \
  --artifact "$RETRIEVAL_SCHEDULE" \
  --artifact "$GENERATION_SCHEDULE" \
  --artifact "$RET_COMPLETE" \
  --artifact "$RUN_ROOT/generation/c0-core-run1.answers.jsonl" \
  --expect-jsonl "$RUN_ROOT/generation/c0-core-run1.answers.jsonl=27" \
  --artifact "$RUN1_C0_JUDGE" \
  --expect-jsonl "$RUN1_C0_JUDGE=$RUN1_C0_JUDGE_COUNT" \
  --require-judge-scores "$RUN1_C0_JUDGE" \
  --artifact "$CAL_JSON" \
  --artifact "$GFC_JSON" \
  --metadata code_freeze_tag=pnu-eval-code-freeze-20260904-v1 \
  --metadata corpus_revision="$CORPUS_REVISION" \
  --metadata generator_model=gemini-3.5-flash-lite \
  --metadata judge_model=gemini-3.1-flash-lite \
  --metadata judge_rubric=pnu-grounded-fully-correct-v11 \
  --metadata pricing_snapshot=config/model-pricing-20260902.json \
  --metadata cost_usage_policy=provider_reported_only \
  --metadata context_k=8 \
  --metadata generation_logical_slots=198 \
  --metadata judge_calls_planned_no_terminal_errors=234 \
  --metadata external_llm_calls_planned_no_retries=432 \
  --metadata effective_core_n=27 \
  --require-clean
```

이 축약 예를 그대로 final manifest로 쓰지 않는다. 실제 명령에는 네 retrieval,
열 generation, definitive/repeat Judge, 두 selection manifest, 사람 label 세 파일,
calibration/GFC JSON·CSV와 모든 completion artifact를 나열한다. Partial Judge에는
해당 `--selection-manifest`도 함께 지정한다. 실제 generator/Judge 시도·성공·terminal
error·retry 수와 `usage=unavailable` 또는 provider-reported token/cost 집계는 계획값과
분리해 넣는다. Manifest는 기존 경로를 덮어쓰지 않으므로 재생성이 필요하면 새
경로를 사용한다.

완전한 manifest가 `validation.ok=true`이고 모든 예상 count/hash가 맞은 뒤에만
작은 정본 결과를 `evidence/20260914/final/`로 복사한다. 이 복사, result commit,
tag도 사용자 승인 뒤 실행한다.

```bash
test ! -e evidence/20260914/final
mkdir -p evidence/20260914/final
cp "$RUN_ROOT/stats/manifest.json" evidence/20260914/final/manifest.json
cp "$GFC_JSON" evidence/20260914/final/summary.json

git add -- evidence/20260914/final/manifest.json \
  evidence/20260914/final/summary.json
git diff --cached --check
git commit -m "docs: freeze PNU final evaluation results"
git tag -a pnu-eval-result-freeze-20260909-v1 \
  -m "PNU final evaluation result freeze"
```

Tag를 만든 뒤 manifest, summary, 최종 보고서 PDF의 SHA-256을 별도 제출 기록에
남긴다. Code-freeze tag는 실행 코드를, result-freeze tag는 결과 정본을 가리킨다.
기존 tag를 이동하지 않는다.

## 17. 결과 해석 gate

- Retrieval 개선은 holdout C0/C1의 paired Evidence Recall/All-Evidence/Source Hit와
  95% CI가 생성된 뒤에만 주장한다.
- 생성 개선은 effective calibration PASS면 3-run Judge GFC paired effect/CI로,
  FAIL이면 adjudicated human run-1 fallback effect/CI로만 주장한다.
- CI가 0을 포함하면 “대폭 향상”이나 “향상 확정”이라고 쓰지 않는다.
- Calibration 실패 시 사람 run-1 결과가 headline이고 Judge 결과는 exploratory다.
- Parser 18문서/54 anchor, DEV45, DEV `p0g` n=3는 각각 component/개발 진단이며
  final holdout 일반화 성능을 대신하지 않는다.

## 18. 실패 시 처리

| 증상 | 처리 |
|---|---|
| Holdout gate에서 sign-off만 FAIL | 실제 두 사람이 36개를 모두 검토했는지 확인한다. 자동 승인 금지 |
| Holdout SHA 불일치 | 실행 중단, packet/sign-off를 새 SHA에 대해 다시 수행 |
| `/health` tuning/revision/cap/trace 불일치 | 해당 서버 중지·올바른 명령으로 재시작·health 재검사 |
| Collector output에 fallback/model mismatch | 조건 오염으로 중단. 더 좋은 답을 골라 대체하지 않음 |
| Retrieval error sidecar가 존재 | 해당 schedule은 poison으로 중단. direct collector로 우회하거나 자동 replay하지 않음 |
| Generation terminal error sidecar가 존재 | 같은 slot의 terminal service-error row와 1:1인지 검사하고 GFC=0으로 포함 |
| Judge JSONL에 error/score null | 새 judge run ID와 새 파일로 재판정. 기존 파일 수정 금지 |
| Human label 누락·reviewer 중복 | calibration 중단, 원 reviewer에게 blind 상태로 보완 요청 |
| Calibration gate FAIL | Judge n=3를 exploratory로 내리고 human run-1 paired 결과 사용 |
| Manifest `Git worktree is not clean` | 결과 파일을 덮어쓰지 말고 변경 원인을 확인. clean freeze commit에서 재시도 |
| 기존 output 경로가 존재 | 덮어쓰지 말고 resume 조건을 확인하거나 새 experiment/run ID 사용 |
| `.final-*-run.lock`이 존재 | 다른 runner/PID와 crash 여부를 조사. lock을 우회하거나 자동 삭제하지 않음 |
| `slot_started`만 존재 또는 `slot_poisoned` | 불확실한 외부 호출로 간주. 같은 experiment에서 자동 replay 금지 |

## 19. 실행 완료 체크리스트

- [x] 3-gate pre-review, read-only packet, SHA-bound A/B template·strict merge 구현
- [ ] 36문항·93 evidence option에 실제 사람 2인 sign-off
- [ ] Final holdout SHA와 sign-off manifest의 `.cases_sha256` 일치
- [x] B1 frozen generation schedule/authorization/WAL/run-lock 구현·로컬 테스트
- [x] B2 oracle-context runner 구현·관련 60 tests PASS (`multi_evidence` 1개 × Core 9영역)
- [x] B3 frozen 4-lane retrieval와 paired 통계 구현·로컬 테스트
- [x] B4 generation-run GFC, partial Judge binding, terminal/human fallback 구현·로컬 테스트
- [x] B5 protocol v1.1과 코드에 balanced accuracy ≥ 0.80 동기화
- [x] 2026-09-02 dirty snapshot 전체 check: 637 tests OK(6 skip), lint/build PASS
- [ ] Freeze 직전 Python 전체 회귀값·skip 사유 최종 기록, lint/build PASS
- [ ] 사용자 승인 후 clean code-freeze commit/tag
- [ ] C0/C1 health pin PASS
- [ ] Retrieval 108 record(PAR-B/PAR-CH/C0/C1 각 27), error 0, exact WAL audit PASS
- [ ] Generation 198 logical slots를 success answer + terminal error로 정확히 덮고 fallback/retry 보존
- [ ] Judge v11 계획 234회(no terminal generation error 기준), 실제 호출/score/error 완전성 PASS
- [ ] 가격 snapshot hash와 실제 호출/usage 상태 기록; usage 없으면 비용 추정 금지
- [ ] Blind 성공 answers(terminal 없으면 63), 독립 2 ratings/answer, 전부 adjudication
- [ ] Judge-human Core gate 판정(terminal 없으면 54; run 1 terminal이면 effective FAIL)
- [ ] 질문 단위 Core `n=27` GFC paired CI/검정
- [ ] 모든 원본·통계 SHA를 포함한 clean manifest
- [ ] 사용자 승인 후 result-freeze commit/tag
- [ ] 보고서 수치를 JSON/CSV에서 재생성하고 수동 전사 금지

## 관련 문서

- [`evaluation-protocol-20260914.md`](evaluation-protocol-20260914.md): 방법론 정본
- [`report-evidence-ledger-20260914.md`](report-evidence-ledger-20260914.md): 주장별 근거와 상태
- [`progress-log-20260901.md`](progress-log-20260901.md): 9월 1–2일 구현·검증 로그
- [`holdout-v2-human-review.md`](holdout-v2-human-review.md): holdout gold 독립 검토 패킷
- [`../evidence/holdout-v2-reviewer-a.json`](../evidence/holdout-v2-reviewer-a.json): Reviewer A 독립 응답 template
- [`../evidence/holdout-v2-reviewer-b.json`](../evidence/holdout-v2-reviewer-b.json): Reviewer B 독립 응답 template
- [`final-report-20260914.md`](final-report-20260914.md): 제출 보고서 초안
