# 2026-09-01 성능 개선·평가 작업 진행 로그

상태: **재개됨 / 평가 진행 중**
최종 갱신: 2026-09-03
브랜치: `feat/role-based-answers`
최종 보고서 마감: 2026-09-14
커밋: 생성하지 않음. 기존 사용자 변경을 포함한 dirty worktree를 그대로 보존함.

이 문서는 최초 중단 시점의 기록에 2026-09-01 재개 후 확인된 결과를 이어서
반영한다.

## 이번 작업의 목표

부산대학교 문서 챗봇의 실제 서비스 경로를 기준으로 검색 성능과 생성 성능을 분리해 개선하고, DEV 튜닝과 비공개 holdout 최종 평가를 구분하는 재현 가능한 평가 체계를 만드는 작업이다.

## 현재까지 완료된 내용

### 1. 검색 C0/C1 정의와 DEV45 비교

- C0: 같은 Cascade 코퍼스·BM25·생성/후처리 코드를 사용하되 서비스 검색 튜닝만 OFF.
- C1: 검색 튜닝 ON. 문서당 context chunk cap은 두 조건 모두 2로 고정.
- 최신 DEV45 결과:

| 지표 | C0 | C1 | 변화 |
|---|---:|---:|---:|
| Document Hit@5 | 29/45 (0.644) | 40/45 (0.889) | +11문항 |
| MRR | 0.494 | 0.717 | +0.223 |
| Any Gold Chunk@8 | 23/45 (0.511) | 28/45 (0.622) | +5문항 |
| All Gold Chunks@8 | 10/45 (0.222) | 18/45 (0.400) | +8문항 |
| Mean Gold Recall@8 | 0.359 | 0.511 | +0.152 |

- 분석 산출물:
  - `processed/eval/dev45-p0c-20260901/c0-vs-c1-retrieval.json`
  - `processed/eval/dev45-p0c-20260901/c0-vs-c1-retrieval.csv`
  - `processed/eval/dev45-p0c-20260901/c0-vs-c1-retrieval.html`
- 지연시간은 서로 다른 실행 시점의 단일 측정이므로 성능 향상 근거로 사용하지 않는다. 동일 세션 반복 측정이 필요하다.
- 위 수치는 DEV 결과다. 비공개 holdout 전에는 “대폭 성능 향상”이나 일반화 성능으로 표현하지 않는다.

### 1.1 DEV45 전체 8조건 retrieval matrix

- 2026-09-02에 실제 `/chat` 경로를 `extractive` provider로 고정해 외부 LLM
  호출 없이 8조건×45문항=360개 검색 요청을 완료했다. 오류는 0건이었다.
- Baseline/Challenger/Cascade parser의 BM25 4조건과 Cascade의 KURE·Snowflake
  Dense/Hybrid 4조건을 같은 `top_k=8`, 문서당 context cap 2로 비교했다.

| 조건 | Source Hit@5 | MRR@5 | Any Gold@8 | All Gold@8 | Gold Recall@8 |
|---|---:|---:|---:|---:|---:|
| PAR-B | .867 | .726 | N/A | N/A | N/A |
| PAR-CH | .889 | .714 | N/A | N/A | N/A |
| C0 | .644 | .494 | .511 | .222 | .359 |
| C1 | .889 | .717 | .622 | .378 | .500 |
| D-K | .822 | .596 | .600 | .333 | .481 |
| H-K | .756 | .544 | .444 | .200 | .304 |
| D-S | .800 | .616 | .644 | .356 | .500 |
| H-S | .778 | .560 | .467 | .222 | .326 |

- DEV gold chunk ID는 Cascade 전용이어서 parser 두 조건은 Source 지표만
  보고한다. D-S는 Any Gold@8만 C1보다 1문항 높았고 Source Hit@5와 All
  Gold@8은 낮았다. 현재 DEV에서는 C1을 대체할 근거가 없으며 Hybrid 자체의
  일반적 열위로 해석하지 않는다.
- 원본 full trace 603,510,647 bytes는 그대로 보존했다. 중첩 answer/citation/trace
  payload를 제외하되 원본·record SHA, control, 단계별 rank/score/text SHA와 final
  source를 남긴 compact derivative는 31,600,532 bytes로 94.76% 작다. 8조건의
  document/gold/unique-document/latency 지표가 원본과 동일함을 재계산했다.
- 산출물과 한계는 `docs/dev45-retrieval-matrix-20260902.md` 및
  `processed/eval/dev45-matrix-20260902/matrix-summary.json`에 기록했다.

### 2. 적용한 검색 개선

- `BIDV`, `TOPIK` 같은 희소 영문 식별자의 정확한 제목/파일명 일치 부스트.
- 일반 제목 문서의 본문 rescue는 강한 질의어가 3개 이상이고 85% 이상 일치할 때만 허용.
- 절차/신청 질의에서 등록금 회의록 같은 구조적 노이즈 문서를 강등하되 의사결정 질의는 제외.
- 질의와 문서의 학년도 불일치는 다른 부스트보다 우선해 감점.
- 절차·시간·다중 질문에 대해 같은 문서의 인접 answer-bearing chunk를 C1에서만 보완.
  - 등록금 휴학 FAQ: 반환 chunk와 이월 chunk 결합.
  - BIDV 안내: 시작 절차 chunk와 후속 절차 chunk 결합.
- 이 개선으로 핵심 등록 DEV 3문항은 C1에서 각각 필요한 두 근거 chunk를 모두 회수했다.

### 3. 생성 프롬프트와 근거 후처리 개선

- 여러 하위 질문 중 근거가 있는 항목은 답하고, 근거가 없는 항목만 구분해 알 수 없다고 답하도록 프롬프트를 변경.
- 이월/반환처럼 서로 다른 결과를 한 문장에 섞지 않도록 생성 규칙을 추가.
- 생성 후 각 문장을 검색 근거에 귀속하고, 지원되지 않는 문장만 제거하는 경로를 강화.
- 수정한 주요 오귀속 사례:
  - `수납은행: A·B·C`를 “납부 가능한 은행”으로 표현한 안전한 범주형 paraphrase.
  - `BIDV 등록금 납부 방법` 문서가 BIDV 채널 사용 가능성을 직접 함의하는 경우.
  - 분할납부 완납 의무를 “그래야 반환 가능”으로 잘못 확대하는 문장 차단.
  - 제목의 재학생/학기 스코프와 일정 행의 날짜를 결합하되 다른 일정 행의 날짜는 섞지 않음.
  - 한 행에 본등록 기간과 고지서 출력일이 함께 있을 때 시작/종료 날짜를 혼동하지 않음.
- 최신 관련 회귀 테스트: `tests.test_api_hardening` + `tests.test_rag_generators`, **142개 통과**.
- 출처 표시는 생성 모델이 붙이는 것이 아니다. 생성 후 문장별 근거 검증이 끝난 뒤 서버가 citation 객체와 표시를 붙인다.

### 4. 생성 preflight 상태

- 모델: `gemini-3.1-flash-lite`로 고정.
- DEV 등록 3문항을 C0/C1 실제 `/chat` 경로에서 여러 차례 점검함.
- 과거 `p0f` 실제 서비스 artifact는 다음과 같다.
  - `processed/eval/preflight-20260901/generation-p0f/dev-smoke-c0-run1.answers.jsonl`
  - `processed/eval/preflight-20260901/generation-p0f/dev-smoke-c1-run1.answers.jsonl`
- 단, 위 `p0f` 수집 후 기준선 공정성을 위한 날짜/은행 귀속 버그를 추가 수정했다. 따라서 **p0f는 현재 코드의 최종 생성 점수 산출물로 사용하면 안 된다.**
- p0f의 raw draft를 현재 코드로 offline replay한 방향성:
  - C0 reg01: 본등록 날짜·수납은행 답변 가능.
  - C0 reg02: 반환은 답하지만 복학 시 이월 근거는 회수하지 못함.
  - C0 reg03: BIDV 절차 근거가 없어 올바르게 abstain.
  - C1 reg01: 본등록 날짜·수납은행을 간결하게 답함.
  - C1 reg02: 이월과 반환을 모두 답함.
  - C1 reg03: BIDV 가능 여부와 앱 절차를 답함.
- 재개 후 현재 코드로 `p0g` C0/C1 artifact를 수집했다. 두 조건 모두 생성 API
  오류는 0건이었다.
  - `processed/eval/preflight-20260901/generation-p0g/dev-smoke-c0-run1.answers.jsonl`
  - `processed/eval/preflight-20260901/generation-p0g/dev-smoke-c1-run1.answers.jsonl`

| DEV 3문항 진단 지표 | C0 | C1 |
|---|---:|---:|
| Retrieval hit | 2/3 | 3/3 |
| MRR | 0.250 | 0.778 |
| Any Gold @5 / @8 | 2/3 / 2/3 | 3/3 / 3/3 |
| All Gold @5 / @8 | 0/3 / 0/3 | 3/3 / 3/3 |
| Mean Gold Recall | 0.333 | 1.000 |
| Generation API error | 0 | 0 |

- 최종 유효 Judge v8을 조건별 3회 반복해 모든 문항에서 3/3 동일 판정을 얻었다.

| Judge v8 DEV 3문항 | C0 | C1 |
|---|---:|---:|
| 문항별 점수 (`reg01`, `reg02`, `reg03`) | `[2, 1, 0]` | `[2, 1, 2]` |
| 평균 점수 | 1.000 | 1.667 |
| GFC | 1/3 | 2/3 |
| 문항별 반복 일치 | 모두 3/3 | 모두 3/3 |

- 차이는 `svc_reg_03`의 BIDV 검색 근거가 C1에서 복구된 데서 발생했다.
  `svc_reg_02`는 두 조건 모두 복학 시 이월 세부를 누락해 1점이었다.
- v6/v7 실패·calibration artifact는 quota 및 quote-format 문제 때문에 유효 결과에서
  제외한다. v8은 18.8–24.3KB compact input과 exact match 및 line/list-marker
  canonical quote validation을 사용했다.
- 생성기와 Judge가 같은 모델 계열이므로 self-preference 가능성이 남고 사람
  calibration은 아직 완료하지 않았다.
- 3문항은 튜닝에 사용한 DEV의 표적 preflight일 뿐이며 holdout 최종 성능이나
  일반화 성능으로 사용할 수 없다. 질문 표본 수가 3이라 CI나 생성 성능 headline도
  산출하지 않는다.
- Judge 반복 요약을 질문 단위로 다시 묶는 `scripts/analyze_gfc_pairs.py`를
  추가했다. 원본 answer/judgment SHA와 바인딩을 재검증한 DEV 3문항 결과는
  C0 GFC 0.333, C1 0.667, 차이 +0.333이지만 paired bootstrap 95% CI는
  `[0.000, 1.000]`, exact sign-flip과 McNemar는 모두 `p=1.0`이다. 따라서
  점 추정치의 방향만 확인됐고 통계적으로 성능 향상을 확정할 근거는 아니다.
  - `processed/eval/preflight-20260901/judge-p0g-v8/gfc-paired-analysis.json`
  - `processed/eval/preflight-20260901/judge-p0g-v8/gfc-paired-cases.csv`

### 5. 비공개 holdout v2 초안과 누수 방지

- 초안: `config/pnu-service-answer-holdout-v2.draft.jsonl`
- 현재 SHA-256: `2fbedad63531491d3c069a5de6fe51fff623f7e571b9170c6a9aa7452e2d51f9`
- 구성:
  - Core 27: 9개 카테고리 × `single_fact`, `multi_evidence`, `structure_sensitive`.
  - Challenge 9: unanswerable 3, scope/version ambiguity 3, prompt injection 3.
  - 총 36문항, evidence option 93개.
- 통과한 게이트:
  - 스키마: PASS.
  - DEV 문서 family/SHA/title/URL 누수: PASS.
  - canonical 코퍼스와 원본 파일 SHA 및 quote 검증: PASS.
- 독립 AI 내용 검토에서 지적된 12건은 수정 완료했고 schema·DEV 누수·corpus
  검증을 모두 다시 통과했다.
- 실제 2인 검수 서명은 의도적으로 생성하지 않았다. 따라서 전체 preflight는 signoff 게이트만 FAIL인 것이 정상이다.
- signoff gate는 exact cases/packet/A·B response SHA, 동일한 두 reviewer roster,
  36개 판정의 원본 응답 일치를 다시 검증한다. Validator 테스트 16개와 A/B
  template·merge 테스트 18개를 통과했다.
- 읽기 전용 검토 패킷 `docs/holdout-v2-human-review.md`와 SHA-bound A/B 응답
  `evidence/holdout-v2-reviewer-{a,b}.json`을 생성했다. 두 응답은 현재 각각
  36개 `PENDING`, 144개 check `null`, 독립 검토 확인 `false`다.
- 실제 2인 human signoff 전에는 파일을 최종 holdout으로 동결하거나 실행하지
  않는다.
- 최종 답변이 수집된 뒤 사람 평가를 바로 시작할 수 있도록
  `scripts/build_answer_review_packet.py`를 추가했다. 사람용 Markdown과 Reviewer
  A/B·합의 라벨에서는 condition/model/provider 및 자동 support 판정을 숨기고,
  별도 비공개 mapping에만 보존한다. 미작성 라벨은 분석 단계에서 fail-closed된다.
- `scripts/analyze_judge_human_calibration.py`는 두 독립 라벨과 합의 라벨을 직접
  읽어 human-human agreement, quadratic weighted kappa, Judge-human confusion
  matrix·balanced accuracy·Cohen's kappa·macro-F1·false-pass rate와 protocol
  gate를 계산한다. 사람 라벨을 보기 전인 2026-09-02에 protocol v1.1로
  balanced accuracy ≥ 0.80을 gate에 추가했으며 이후 threshold는 고정한다.
- gold 근거를 검색 없이 같은 generator/postprocessor에 넣는 oracle-context
  진단 수집기 `scripts/evaluate_oracle_context_answers.py`를 구현했다. Core 9개
  영역의 `multi_evidence` 1문항씩을 사전 선택하고, required claim의 검증된
  evidence quote만 주입한다. 실제 9개 생성은 사람 2인 signoff 뒤에만 실행하며
  `diagnostic_only=true`, `service_performance_eligible=false`로 집계에서 제외한다.

### 6. 파서 3종 source-bound 원자 근거 감사

- 표적 문서 18개(HWP/HWPX 6, Digital PDF 6, OCR/Table PDF 6)에 문서당
  3개씩 총 54개 원자 anchor를 두고 Baseline·Challenger·Cascade의 보존 여부를
  기계 평가했다.
- 공통 source manifest SHA-256:
  `1fa7e0f2f5d03a2a1249584d91f7232c9b2cb4642b39115bf3c9e6a871845a2f`

| Profile | Anchor 보존 | 3/3 보존 문서 | HWP/HWPX | Digital PDF | OCR/Table PDF |
|---|---:|---:|---:|---:|---:|
| Baseline | 48/54 | 14/18 | 18/18 | 17/18 | 13/18 |
| Challenger | 54/54 | 18/18 | 18/18 | 18/18 | 18/18 |
| Cascade | 54/54 | 18/18 | 18/18 | 18/18 | 18/18 |

- 평가셋 SHA-256:
  `5371353a52b36df8685d4892d8138ba1eaf691fb93a2fd335ced21e279208720`
- JSON 결과 SHA-256:
  `9fabcd7525ba22dea1758a55418546e25627d2d7413faf7cccea1efc0dbf2755`
- CSV 결과 SHA-256:
  `bd98a7469b33970af2c018a8e0af7a06a004fa3e366dd9957bdc638762d7de81`
- 이 결과는 실패 모드를 섞은 표적 18문서의 결정적, 기계적 source-bound 진단이다.
  Cascade 산출물을 canonical binding으로 사용했으므로 full-corpus 파서 우월성,
  RAG 성능 향상 또는 54개 원문 화면의 독립 2인 육안검수 결과로 해석하지 않는다.

### 7. LLM-as-a-Judge 설계 결정과 이론 근거

- 최종 주 평가는 원자적 rubric 기반 pointwise 평가로 한다.
- pairwise는 보조 분석으로만 사용하며 A/B와 B/A 순서를 모두 돌린다.
- Judge 1회 결과를 확정값으로 쓰지 않고 같은 답변을 3회 평가해 다수결과 3/3·2/3 일치율을 기록한다.
- 정답성, 검색 근거성, citation support/correctness, citation completeness를 분리한다.
- Gemini 생성 + Gemini Judge의 자기선호 가능성을 명시하고 사람 평가로 보정한다.
- 사람 검토: generation run1의 C0 Core 27 + C1 Core 27 + C1 Challenge 9,
  총 63답변을 2인이 condition-blind로 독립 평가하고 전부 adjudicate한다.
- 통계 단위는 질문이며 반복 Judge 호출을 추가 표본으로 세지 않는다.
- 보고 지표: Judge-사람 confusion matrix, agreement/macro-F1, false-pass rate, Krippendorff's alpha, paired bootstrap 95% CI, exact McNemar test.
- 핵심 참고 문헌:
  - G-Eval, EMNLP 2023: https://aclanthology.org/2023.emnlp-main.153/
  - MT-Bench/Chatbot Arena, NeurIPS 2023: https://proceedings.neurips.cc/paper_files/paper/2023/file/91f18a1287b398d378ef22505bf41832-Paper-Datasets_and_Benchmarks.pdf
  - Prometheus, ICLR 2024: https://proceedings.iclr.cc/paper_files/paper/2024/hash/803485352e61e3ebf41221e4776c9fd4-Abstract-Conference.html
  - LLM-RUBRIC, ACL 2024: https://aclanthology.org/2024.acl-long.745/
  - RAGAS, EACL 2024: https://aclanthology.org/2024.eacl-demo.16/
  - ARES, NAACL 2024: https://aclanthology.org/2024.naacl-long.20/
  - RAGChecker, NeurIPS 2024: https://proceedings.neurips.cc/paper_files/paper/2024/hash/27245589131d17368cccdfa990cbf16e-Abstract-Datasets_and_Benchmarks_Track.html
  - ALCE, EMNLP 2023: https://aclanthology.org/2023.emnlp-main.398/
  - ContextualJudgeBench, ACL 2025: https://aclanthology.org/2025.acl-long.470/
  - Rating Roulette, Findings of EMNLP 2025: https://aclanthology.org/2025.findings-emnlp.1361/
  - JudgeDeceiver, CCS 2024: https://doi.org/10.1145/3658644.3690291
  - TREC 2025 RAG Track: https://trec.nist.gov/pubs/trec34/papers/Overview_rag.pdf

### 8. 9월 2일 final 실행·분석 경로 완성

- B1 생성 실행기는 `scripts/run_final_generation_schedule.py`로 구현했다. 직접
  서비스 호출은 Core `27×2조건×3회=162`와 Challenge C1 `9×3=27`, 합계
  189 logical slot이며 oracle 9개는 별도 진단으로 유지한다. Core는 고정 seed
  `20260914`의 category-stratified AB/BA `41/40` 순서다.
- 매 logical slot 전후로 현재 holdout·DEV manifest·signoff·검토 packet·Reviewer
  A/B 원본 응답·index byte hash, 공통 source-manifest SHA, clean local Git commit,
  서버 startup commit/clean, provider/model/request config를 다시 검증한다. 두
  runner는 `O_EXCL` 배타 run lock과 `slot_started → slot_completed` write-ahead
  log를 사용하며, crash 뒤 stale lock이나 started-only slot은 자동 재생하지 않는다.
- 기술 retry는 최대 3회이고 408/429/5xx 또는 명시된 network error만 허용한다.
  전송 재시도 소진과 유효 HTTP 응답의 빈/비문자 answer는 terminal service error로
  보존한다. Terminal slot은 Judge 호출 대상에서 제외하되 GFC=0으로 남아 survivor
  bias를 만들지 않는다.
- B3 검색 실행기 `scripts/run_final_retrieval_schedule.py`는 Core 27문항을
  PAR-B/PAR-CH/C0/C1 네 lane, 총 108 slot의 case-major 순서로 고정한다. C0/C1은
  동일 Cascade index를 강제하며 generation은 `extractive`, model은 `null`이다.
  `scripts/analyze_holdout_retrieval.py`는 Source Hit@1/3/5, Evidence Recall과
  All-Evidence@5/8, MRR@50, Candidate Recall@50, p50/p95, family bootstrap,
  exact McNemar와 parser 비교 Holm 보정을 계산한다.
- 합성 108-slot/216-WAL 영구 E2E 회귀에서 schedule audit `complete=108`, raw
  BM25 60개 중 `[:50]` 평가, four-lane 분석과 immutable completion manifest를
  끝까지 확인했다. 실제 holdout이나 외부 LLM은 호출하지 않았다.
- B4 `scripts/analyze_final_generation_gfc.py`는 27개 질문을 유효 표본으로 유지한
  채 조건별 서로 다른 generation run 3개의 GFC를 결합하고 family bootstrap
  10,000회, sign-flip, 2/3 majority exact McNemar를 계산한다. Judge-human gate가
  실패하면 adjudicated human run1을 headline으로 사용하는 fallback도 구현했다.
- Judge stability는 run1 C0/C1 Core에서 고정 9개씩만 선택하고 terminal을 다른
  답변으로 대체하지 않으며, 같은 답변에 추가 Judge를 정확히 2회 수행한다.
  selection·answer·judgment hash와 immutable/no-clobber 출력을 검증한다.
- 사람 holdout handoff는 `validate_service_holdout.py --pre-review`의 세 기계 gate,
  읽기 전용 no-clobber packet, 별도 A/B 응답, strict merge로 분리했다. Merge된
  sign-off만 보는 것이 아니라 packet과 두 원본 응답 파일의 현재 SHA 및 모든
  case 판정을 final gate에서 다시 대조한다. 독립 공격 감사에서 발견한 cases
  다중-read TOCTOU, 중복 JSON key, case/packet alias, 빈 packet, parent symlink swap,
  repo 밖 provenance를 fail-closed로 막고 회귀 테스트를 추가했다.
- 전체 `bun run check`를 로컬 fixture socket이 허용된 환경에서 재실행해
  **637 tests OK(6 skip)**, ESLint, TypeScript, Vite production build PASS를
  확인했다. Compileall, `git diff --check`, final CLI `--help` 검사도 PASS다.
- 현재 draft validator는 schema·DEV·corpus `93/93`만 PASS하고 실제 2인 signoff가
  없어 의도대로 전체 FAIL이다. Final holdout·외부 API 호출은 0회이며, 실제 실행은
  사람 2인 signoff, final filename, 사용자 승인에 따른 clean freeze 뒤에만 한다.

9월 2일 final-tooling working snapshot SHA-256:

| 파일 | SHA-256 |
|---|---|
| `run_final_generation_schedule.py` | `d483f8b837db3c5171ee07448654c7807ab3e084fe64cd0757287a8721ca2c7a` |
| `run_final_retrieval_schedule.py` | `d6c5edcc9dd73e8f7229eae3ff2b5ede7f596fb52a00364111d4ae08a9b11332` |
| `analyze_holdout_retrieval.py` | `c448c5f37dfa71c0d61a81606c361379f0e4f03bed17fbcb38e832e75ba1da28` |
| `analyze_final_generation_gfc.py` | `4c71a46604a338e87b3f7da881f910271f9ccfd6a6fe2ea61fb4bc3d559d7afb` |
| `service_eval_artifacts.py` | `1b83fdd2004db18b29fd5a5d5bbff5abf9d877a434de0f22ad247e1b1902ff9d` |
| `immutable_outputs.py` | `b4ba8143957043207de09f40e7f209ac5dacf9c419fec13e373fbbe0502e32b3` |
| `build_holdout_signoff.py` | `8833d4db9a1e9a9c06a7d82eca6e6336d64c87552455426917b9013dae33f3b3` |
| `build_holdout_review_packet.py` | `f90500e54ce2d47d42f203b01d36e0b9da067a3f7f486236b0acf170e026ce06` |
| `validate_service_holdout.py` | `0b04713007212f3cb5cbf0c464179e9faaa3bdd7dc3ebdcee58eead4924f90b0` |

### 9. 9월 2일 후처리 오삭제 수정과 multi-facet 검색 보강

#### 9.1 생성 후처리 원인과 수정

- `svc_reg_02`의 저장 LLM 초안에는 다음의 올바른 문장이 있었지만, 기존
  semantic attribution이 이를 `semantic_relation_mismatch`로 삭제했다.
  - 복학 시 별도 등록 절차 없이 납부 처리된다.
  - 학생지원시스템의 납부확인 메뉴에서 `수기분 0원`으로 확인할 수 있다.
- 원인은 claim의 `확인할 수 있습니다`를 permission 관계로 해석하면서도, 공식
  근거의 `학생지원시스템 → 등록 → 납부확인 ... 수기분 0원 확인` 같은 명사형
  UI 안내는 같은 관계로 인식하지 못한 비대칭이었다.
- 공식 UI 경로, 동일 조회 동작, 동일 범위 용어와 critical value가 함께 있는
  경우에만 허용하는 좁은 bridge를 추가했다. 명시적 불가 문구와 다른 메뉴·다른
  대상의 `확인`은 계속 거부한다.
- 현재 코드로 저장 초안을 replay하면 `svc_reg_02`의 support claim은 3개에서
  4개로 복원된다. 근거에 없던 "분할납부자는 잔여 등록금을 완납해야 반환 절차를
  진행할 수 있다"는 확대 문장은 계속 제거된다.

동일 초안·동일 최종 context를 고정한 raw/current 비교를 위해
`scripts/project_raw_draft_answers.py`와
`scripts/analyze_postprocessor_pairs.py`를 추가했다. 두 projection은 원본·답변·
초안·context·후처리 코드 SHA에 묶인 immutable 진단 artifact이며 외부 모델을
호출하지 않는다. GFC, citation 또는 end-to-end 서비스 성능으로 집계하는 것을
명시적으로 금지한다.

| DEV 등록 3문항 결정론적 진단 | 결과 |
|---|---:|
| UTF-8 exact answer 변화 | 3/3 (bullet·공백 포맷 포함) |
| 정규화 문장 내용 변화 | 1/3 (`svc_reg_02`) |
| raw → final 문장 수 | 12 → 11 |
| 유지 / 삭제 / 추가 | 11 / 1 / 0 |
| critical value 유지 / 손실 / 추가 | 7 / 0 / 0 |
| 표준 회피 답변 | 0/3 |

- 분석 결과:
  `processed/eval/preflight-20260902/postprocessor-ab/dev-smoke-c1-postprocessor-pairs-v2.analysis.json`
- 결과 SHA-256:
  `ad765dd0dc1d0877b5f6171180e439a4226cfe42dcc04ca9d8a8b46dbceb4a1b`
- 이 단계는 문장 유지·삭제를 측정한 결정론적 진단이다. raw/current의 품질 점수
  비교에는 동일 설정의 별도 LLM Judge가 필요하다. 저장 답변과 근거 context를
  외부 Gemini에 전송하려면 사용자의 명시적 승인이 필요해 이번 실행에서는
  수행하지 않았다.

#### 9.2 명시적 multi-facet 검색 보강

- 문서당 context cap은 2로 유지했다. cap을 3 또는 4로 전역 확대하면 DEV45의
  31문항 context가 바뀌는 반면 이득은 소수 문항에 그쳐 채택하지 않았다.
- 질문이 날짜·금액·자격·방법 중 둘 이상을 명시적으로 요구하고 현재 같은 문서의
  두 chunk가 한 facet만 중복 제공할 때에만, 중복 chunk를 누락 facet의 인접 sibling
  으로 교체한다.
- 질문과 sibling의 학년도·학기가 충돌하면 교체하지 않는다. 단일사실 질의와 기존
  temporal/procedure completion은 그대로 유지한다.

실제 로컬 `POST /chat` 경로를 `extractive` provider로 고정해 DEV45를 다시
수집했다. 외부 LLM 호출은 0회였고 error row도 0개였다.

| DEV45 C1 검색 지표 | 수정 전 | 수정 후 | 변화 |
|---|---:|---:|---:|
| Source Hit@5 | 40/45 (.889) | 40/45 (.889) | 0 |
| MRR@5 | .717407 | .717407 | 0 |
| Any Gold Chunk@8 | 28/45 (.622) | 28/45 (.622) | 0 |
| All Gold Chunks@8 | 17/45 (.378) | 18/45 (.400) | +1문항 |
| Mean Gold Recall@8 | .500000 | .511111 | +.011111 |
| Mean unique documents | 6.0222 | 6.0222 | 0 |

- 최종 context ID가 바뀐 문항은 2/45이고 나머지 43/45는 동일했다.
  - `svc_sch_02`: 중복 금액 chunk를 신청기간 chunk로 바꿔 exact gold recall이
    .5에서 1.0, All Gold가 false에서 true로 개선됐다.
  - `svc_sch_05`: 질문과 무관한 중복수령 chunk를 신청자격 chunk로 바꿨다.
    DEV gold에 질문이 요구하지 않은 중복수령 chunk도 포함돼 있어 legacy exact
    recall은 2/3으로 같지만, 실제 질문이 요구한 자격+접수기간 두 facet은 모두
    확보했다.
- 지연시간은 서로 다른 단일 로컬 실행의 값이므로 개선 근거로 사용하지 않는다.
- 실제 서비스 경로 artifact:
  `processed/eval/dev45-c1-multifacet-20260902-v2/c1.answers.jsonl`
  (SHA-256
  `5e106047c4f9549ae29a639c7ad73062f04c218f65c4043283dc0c08e4807a0e`)
- compact artifact:
  `processed/eval/dev45-c1-multifacet-20260902-v2/compact/c1.retrieval.jsonl`
  (SHA-256
  `e17cf54a9597c5573a43fc5b5e54db8cec1ee7f8a7145eac595676eaee11174c`)
- 전후 비교 JSON/CSV/HTML:
  `processed/eval/dev45-c1-multifacet-20260902-v2/old-c1-vs-multifacet.*`
- Historical 생성 `+0.0296`의 단계별 원인 분석:
  `docs/dev45-generation-root-cause-20260902.md`
- 첫 실행 디렉터리 `processed/eval/dev45-c1-multifacet-20260902/`는 기존 시각화
  서버가 사용 중인 8765 포트와 충돌해 수집 전에 중단된 실패 기록이다. 재실행은
  이를 덮어쓰지 않고 새 `-v2` 디렉터리와 18765 포트를 사용했다.

### 10. 9월 3일 표적 생성 재검증과 평가 정합성 수정

- query-specific facet completion과 scholarship sibling 선택을 보강한 DEV45
  C1 재수집에서 Source Hit@5는 `40/45`로 같았고 MRR은 `.717407`에서
  `.728519`, All Gold@8은 `18/45`에서 `19/45`, Mean Gold Recall@8은
  `.511111`에서 `.522222`로 소폭 개선됐다. DEV 진단이며 일반화 성능 주장이
  아니다.
- 표적 5문항을 3회 외부 생성한 뒤 동일 초안·동일 context를 현재 후처리기로
  replay했다. 날짜·학기·UI 경로·행 단위 관계 검증을 고친 v7에서 55개 claim 중
  47개가 유지됐고, `1 ・ 4차`처럼 마지막 항목에만 단위가 붙는 병렬 회차 표기를
  해석하도록 고친 v8에서는 50/55가 유지됐다. 이어 장학 표의
  `국내 학사 8학기, 국내외 석사 4학기, 국내외 박사 6학기까지 지원 가능`에서
  마지막 상한을 병렬 항목 전체에 적용하고, 이 문맥에서만 `학부`를 `학사`와
  동등하게 처리한 v10은 51/55를 유지했다. 세 run의 `svc_reg_06`은 모두
  2/3에서 3/3으로 복원됐고 `svc_sch_05`의 학기 제한도 세 run 모두 보존됐다.
  중간 v9의 회귀 결과 49/55도 덮어쓰지 않고 보존했다. 이 replay는 결정론적
  후처리 진단이며 GFC나 end-to-end 성능으로 집계하지 않는다.
- `svc_reg_06` 원문을 재검수한 결과, 2026학년도 2학기 재학생 등록금 납부계획과
  같은 게시물 FAQ가 모두 분할 `1·4차` 학자금대출 불가를 명시했다. 기존 DEV
  gold가 일반 학자금대출 기본계획의 `1회차`만 필수로 둔 것은 세부 분할납부
  규정과 불일치하므로, 현재 DEV gold는 등록금 납부계획의 `1·4차`로 수정하고
  지급 실행·등록 완료 절차는 대출 기본계획의 근거를 결합하도록 바꿨다.
- 기존 DEV Judge는 flat `evidence[]` 전부를 필수 claim으로 변환해 질문하지 않은
  거주자격·중복수령·희망과목담기까지 누락 감점했다. DEV45 전체 질문-근거를
  재검수한 현재 파일은 45문항, evidence 140개이며 105개는 필수, 30문항의
  35개는 `required_for_answer=false` 참고 근거다. 빠져 있던 이자지원액·두 번째
  학사일정·장학 신청 경로·학생지도영역 근거도 보강했다. 새 Judge rubric은 v10,
  input projection은
  `compact-observed-v2`다.
- 개정 전 DEV45는 byte-identical
  `config/pnu-service-answer-eval-v1.jsonl`로 보존했다(SHA-256
  `3c3e19d6c5524218b2f2cb1600c1c64abd80e1783ce7b235dbb80109723a3be6`).
  개정 DEV45의 현재 SHA-256은
  `961c3079bb6d0e2298be4ecf0dde9ea5f7ed79695494dd35e6d12510ffc95ffe`다.
- 사전실험 재개 성공 row의 `collection_attempt_number=2`는 정상 메타데이터였다.
  최종 one-shot 전용 `=1` 검증이 사전실험에도 적용되어 Judge가 거부한 것이
  원인이므로, final authorization이 있는 artifact만 `=1`을 요구하고 preflight는
  양의 정수를 허용하도록 수정했다. 기존 run2 5개 답변은 v1 cases로
  `--validate-only`를 통과했다.
- 평가 기준을 바꾼 뒤 과거 v9 Judge 점수를 새 v10 점수처럼 재해석하지 않는다.
  다음 외부 비교는 개정 DEV와 v10을 동결한 새 run으로만 수행한다.

### 11. 9월 3일 required-evidence 기준 검색 재검증과 표적 보강

- 검색 A/B 분석기 두 개가 optional evidence까지 분모에 포함하던 오류를 고쳐
  `required_for_answer=false`를 제외했다. exact chunk ID뿐 아니라 동일 문서의
  중복·재청킹을 구분하기 위해, 필수 quote 전체가 한 반환 chunk에 정규화 exact로
  존재하는 경우만 인정하는 보수적 companion metric
  `required_evidence_item_exact_chunk_or_normalized_quote_v2`도 추가했다. fuzzy나
  semantic match는 사용하지 않는다.
- 실제 Cascade index에 대해 DEV validator를 다시 실행해 45문항, error 0을
  확인했다. 서로 다른 공식 문서를 결합해야 하는 질문을 허용하되, 적어도 한 필수
  근거는 대표 source title/host와 일치해야 한다. 공유 문서 경고 6개는 role pair와
  동일 공지 재사용에 관한 예상된 경고다.
- 명시적 multi-facet 질의에서 같은 공식 문서의 답변 chunk가 멀리 떨어지는 세
  원인을 표적으로 보강했다.
  - 수료후연구생 질의: 자격 chunk를 seed로 삼아 3~6칸 떨어진 신청방법·금액
    sibling을 찾는다.
  - 이자지원 질의: `소득분위`와 `통장/계좌 입금`을 별도 facet으로 보고 FAQ의
    소득 제한 `#0010`과 원리금 상환계좌 `#0012`를 함께 반환한다.
  - AI학업장려대출 질의: `대상학과`, `금리`, `한도`를 분리해 금리·한도 chunk
    `#0014`를 보존하고 중복 한도 chunk `#0185`만 대상학과 `#0006`으로 교체한다.
- marginal candidate 선택 뒤 루프 종료가 선택된 후보가 아니라 마지막 검사
  후보의 임시 facet을 참조하던 버그도 수정했다. 이 때문에 정답 문서가 이미 모든
  facet을 채운 뒤 무관한 두 번째 문서까지 확장될 수 있었다. 현재는 선택 후보의
  실제 `added_required`로 종료하며 회귀 테스트가 이를 고정한다.

동일한 current DEV45와 metric 정의로 9월 2일 C1 저장 실행을 다시 계산한 결과는
다음과 같다. 이는 DEV 검색 튜닝 결과이며 holdout·생성·일반화 성능이 아니다.

| DEV45 Cascade+BM25, k=8 | 9월 2일 C1 | 현재 | 변화 |
|---|---:|---:|---:|
| Source Hit@5 | 40/45 (.889) | 40/45 (.889) | 0 |
| Document MRR@5 | .702593 | .713704 | +.011111 |
| Any required gold chunk | 28/45 (.622) | 30/45 (.667) | +2문항 |
| All required gold chunks | 21/45 (.467) | 27/45 (.600) | +6문항 |
| Mean required gold chunk recall | .544444 | .625926 | +.081481 |
| All required evidence items | 21/45 (.467) | 29/45 (.644) | +8문항 |
| Mean required evidence recall | .562963 | .659259 | +.096296 |

- All-required-evidence 개선 문항은 `svc_sch_01`, `svc_sch_02`, `svc_sch_03`,
  `svc_sch_05`, `svc_sch_06`, `svc_grad_03`과 두 role duplicate이며, 손실 문항은
  없다. 저장 source 배열은 13/45에서 달라졌으므로 이 비교는 좁은 최종 패치 하나의
  영향 범위가 아니라 9월 2일 이후 누적 검색 변경의 결과다.
- 직전 destination-facet 실행과 현재 실행을 비교하면 source가 바뀐 것은
  `svc_sch_01` 1/45뿐이고 All-required-evidence@8은 28/45→29/45, 손실은 0이다.
- 단일 실행 p50은 9월 2일 287.5ms, 현재 345.2ms였지만, 실행 시점과 시스템
  부하가 통제되지 않아 코드 latency 효과로 해석하지 않는다. 최종 지연시간 주장은
  warm-up 뒤 교차 순서 반복 측정으로만 한다.
- 최신 45문항 실제 서비스 경로 artifact:
  `processed/eval/preflight-20260903/retrieval-department-facet-v3/c1.answers.jsonl`
  (SHA-256
  `7df3657b2f9b3b6e51fa44575e0bb12b41c989c1f112d039388f191da9cb9b58`)
- 누적 비교 JSON/CSV/HTML:
  `processed/eval/preflight-20260903/retrieval-department-facet-v3/dev45-vs-current.*`
- 직전 패치 비교 JSON/CSV/HTML:
  `processed/eval/preflight-20260903/retrieval-department-facet-v3/before-vs-after.*`
- 외부 LLM 호출은 0회다. 저장 초안 replay의 factual claim 51/51 support는
  후처리 진단으로만 유지하며, 새 검색 context를 사용한 end-to-end GFC는 별도의
  승인된 generation/Judge run 전까지 미확정이다.

### 12. 제출기한·예산표 검색 및 예산 답변 생성 보강

- 문서 제출기한 질의에서 `[별표 2-1]` 같은 표 번호와 질문과 무관한
  `수강신청 10일 전까지`를 제출기한으로 오인하던 문제를 수정했다. 질문에 나온
  문서 객체(성적표·논문 등), 제출 동작, 기한 표현, 실제 날짜가 같은 행에 있는
  경우만 deadline 근거로 인정한다.
- `학위논문` 질의가 문서 제목의 `학위청구논문`을 같은 주제로 인식하도록 하고,
  학년도 문서의 다음 달력연도 일정(예: 2025학년도 후기 심사의 2026년 일정)을
  허용했다. `svc_grad_02`, `svc_grad_04`는 직전 실행 대비 exact gold와 모든 필수
  근거를 새로 확보했고 손실은 없었다.
- `전체 예산 규모`와 `영역별 비중`을 별도 필수 facet으로 추가했다. 합계·예산액과
  교육/연구/학생지도 영역별 구성비가 실제로 있는 표만 답변 근거로 인정하며,
  질문 연도와 충돌하는 과거 예산표는 sibling 탐색 앵커에서 제외한다.
- 예산표가 검색된 뒤에도 추출형 생성기가 주변 공고 문구만 출력하던 문제를
  수정했다. 현재 `svc_core_02`와 역할 변형은 모두 `30,405,200`, 교육
  `6,502,600(21.39%)`, 연구 `16,936,679(55.70%)`, 학생지도
  `6,965,921(22.91%)`를 직접 답하고, 두 생성 문장 모두 정답 표 chunk
  `doc_f2b85b3ef83ed3f4084f293d:cascade#0023`에 귀속된다.

현재 DEV45 누적 검색 결과는 다음과 같다. 양쪽 모두 current DEV45와 같은
보수적 required-evidence 산식으로 다시 계산했다.

| DEV45 Cascade+BM25, k=8 | 9월 2일 C1 | 현재 | 변화 |
|---|---:|---:|---:|
| Source Hit@5 | 40/45 (.889) | 40/45 (.889) | 0 |
| Document MRR@5 | .7026 | .7137 | +.0111 |
| Any required gold chunk | 28/45 (.622) | 34/45 (.756) | +6문항 |
| All required gold chunks | 21/45 (.467) | 31/45 (.689) | +10문항 |
| Mean required gold chunk recall | .5444 | .7148 | +.1704 |
| All required evidence items | 21/45 (.467) | 33/45 (.733) | +12문항 |
| Mean required evidence recall | .5630 | .7481 | +.1852 |

- All-required-evidence 개선은 12문항, 손실은 0문항이다. 누적 source 배열 변경은
  15/45이므로 DEV 튜닝 결과이며 일반화 성능으로 해석하지 않는다.
- 직전 제출기한 실행 대비 예산 패치에서 source가 바뀐 것은
  `svc_core_02`, `svc_core_02_role_researcher` 2/45뿐이고, 두 문항 모두
  All-required-evidence@8을 새로 충족했으며 손실은 없다.
- 최신 실제 서비스 경로 artifact:
  `processed/eval/preflight-20260903/retrieval-budget-breakdown-v5/c1.answers.jsonl`
  (SHA-256
  `2c77fc95e88b32c28f6a6d7204d8e7a2b1709f272e62835ab90cb42fce085334`)
- 직전 패치 비교:
  `processed/eval/preflight-20260903/retrieval-budget-breakdown-v5/before-vs-after.*`
- 9월 2일 누적 비교:
  `processed/eval/preflight-20260903/retrieval-budget-breakdown-v5/dev45-matrix-vs-current.*`
- 관련 회귀 테스트 281개, DEV validator 45문항 error 0을 확인했다. 외부 LLM
  호출은 0회다. 위 예산 생성 결과는 결정론적 추출형 표적 검증이며, 전체
  end-to-end GFC/Judge 성능은 승인된 최종 실행 전까지 미확정이다.

## 재개 후 검증 상태

- PASS: API/생성 관련 회귀 테스트 142개.
- PASS: BM25 검색 개선 집중 테스트 33개(구현 작업 당시 결과).
- PASS: holdout validator 22개와 독립 signoff handoff 19개, 합계 41개 집중 테스트.
- PASS: 수정된 holdout 36문항/93 evidence option의 schema·DEV·corpus gate.
- PASS: 최신 전체 Python 회귀 656개 실행, OK(6개 skip), ESLint·TypeScript·Vite build.
- PASS: 9월 3일 평가 정합성·후처리·검색 A/B 관련 집중 Python 회귀 281개, OK.
- PASS: 현재 코드 기준 C0/C1 `p0g` 생성 artifact 수집, 양쪽 오류 0건.
- PASS: Judge v8 C0/C1 각 3회 반복과 aggregate 완료, 모든 문항 3/3 일치.
- PASS: 질문 단위 GFC paired bootstrap·sign-flip·McNemar 분석기와 DEV3 smoke.
- PASS: condition-blind 답변 검토 패킷·라벨 템플릿 생성기와 Judge-human
  calibration 분석기의 합성/DEV smoke. 실제 사람 라벨은 생성하지 않음.
- PASS: DEV45 8조건 retrieval matrix 360요청, 오류 0, 외부 LLM 0회와 compact
  derivative 원본 지표 동등성 검증.
- PASS: oracle-context 수집기 mock/extractive 검증. 실제 holdout·외부 생성은
  signoff 전이라 실행하지 않음.
- PASS: B1 189-slot generation schedule, B3 108-slot retrieval schedule/analyzer,
  B4 generation-run GFC·Judge repeat·human fallback과 immutable/WAL/run-lock 계약.
- 미실행: 36문항 holdout 실제 평가.
- 미완료: 2인 holdout 검수·서명.
- 완료: 파서 3종 18문서/54 anchor 기계적 source-bound 평가와 결과 hash 확인.
- 미완료: 파서 anchor 54개의 독립 2인 원문 화면 육안검수.

## 2026-09-04 재개 검증 상태 (Codex 세션 중단 → Claude 인계)

Codex 세션이 9월 3일 21:50 기록(12절, 예산표 검색·생성 보강) 직후 비정상
종료되어 작업을 인계받았다. 마지막 코드 수정(`scripts/search_api.py`,
`tests/test_api_hardening.py`, 21:44)은 12절 기록과 artifact
(`retrieval-budget-breakdown-v5`, 21:47)로 마감되어 있어 미완성 편집은 없다.

인계 직후 현재 worktree에서 다시 실행한 검증:

- PASS: `python3 -m unittest discover -s tests` **695 tests OK (6 skip)**
  (9월 2일 기록 637/6에서 증가. freeze 시 manifest metadata에 이 값으로 기록)
- PASS: `git diff --check`, `bun run lint`(ESLint 출력 0), `bun run build`
  (tsc -b + vite build)
- PASS: holdout draft SHA = canonical `2fbedad6…2d51f9` (draft 미변경)
- PASS: Step 1 6.1 pre-review gate (schema·dev·corpus, 36 = 27 + 9,
  verified_evidence_option_count 93)
- PASS: Step 1 6.2 read-only 검사 — active packet이 draft와 일치, A/B 빈
  template 36문항 current. PACKET_SHA `0c14120e…1d11b1`
- 미변경: Reviewer A/B 응답은 여전히 빈 template(`reviewer_id` placeholder,
  36문항 `PENDING`, `independent_review_confirmed=false`). **Step 1 사람
  검수가 유일한 차단 요인**이며 Step 2 이후는 전부 이 뒤에 온다.
- 미수행(승인 대기): final holdout byte-copy, code-freeze commit/tag
  (`pnu-eval-code-freeze-20260904-v1`). 런북 7절 규정대로 사용자 명시 승인 뒤에만
  수행한다.

일정 대비: 계획상 9/2~3 몫이던 holdout 교차 검수가 9/4 현재 미착수다. 9/5
retrieval one-shot·9/6 generation을 지키려면 사람 검수와 sign-off를 9/4~9/5
중에 끝내야 한다. 절단 순서상 사람 검수는 절단 불가 항목이다.

### 2026-09-04 00시 — Codex 마지막 계획(DEV45 현재 코드 생성) 재개와 preflight 결함 2건

Codex 세션 기록(guardian 평가 로그)에서 마지막 계획 행동을 복원했다: 포트 18800에
`RAG_EVAL_TRACE=1` Cascade 서버(문서당 2청크)를 띄우고
`evaluate_service_answers.py`로 **DEV45 C1 run1**을 `gemini-3.5-flash-lite`로
생성해 `processed/eval/preflight-20260903/dev45-generation-current-v1/`에
저장하는 단계였다. Codex의 안전 검사기가 "부산대 문서 컨텍스트의 Gemini 외부
전송에 대한 명시적 승인 없음"으로 이 호출을 거부한 직후 세션이 종료됐고,
디렉터리만 빈 채로 남았다. 사용자가 이 단계부터 이어가도록 지시했다.

같은 명령을 실행하자 preflight가 두 번 실패했다. 둘 다 Codex가 관측하지 못한
결함이다(호출이 차단됐으므로).

1. **모델별 생성 한도 대조 결함(수정)**: `validate_health_controls`는
   `service_config.generation.models[<model>]`에서 max_context_chars 등을
   읽지만, 호출부가 `expected_generation_model`을 최종(holdout) 실행에서만
   넘겨 DEV 실행에서는 상위 객체를 대조 → 항상 `got None` mismatch.
   호출부를 "모델별 한도 검사를 요청한 경우에도 모델명을 넘긴다"로 수정하고
   회귀 테스트(`test_validate_health_controls_reads_per_model_generation_caps`)
   추가. 서버 `/health`는 이미 올바른 값(24000/900/샘플링 없음)을 냈다.
2. **인덱스 SHA 핀의 clean-worktree 요구**: `--expected-index-sha256`를 주면
   `freeze.startup_worktree_clean=true`까지 요구한다. 작업 트리는 규칙상
   미커밋(dirty)이므로 DEV 실행에서는 이 핀을 생략했다. 인덱스 정체성은
   `--expected-corpus-revision`과 `--expected-source-manifest-sha256`로
   고정되며, 서버 `/health` freeze 블록이 index_sha256 `a4c1ca63…`을 보고하므로
   artifact 메타데이터로 남는다. 최종 holdout 실행은 code-freeze 뒤 원래
   핀 전체로 수행한다.

실행 구성: C1 = 포트 18800(튜닝 ON), C0 = 포트 18801(`--no-retrieval-tuning`),
둘 다 `RAG_EVAL_TRACE=1`, Cascade allow-suspect 인덱스, 문서당 2청크,
`--context-k 8`, provider frontier, model gemini-3.5-flash-lite, sleep 2.

**C1 run1 수집 완료 (2026-09-04 00:4x)**:
`processed/eval/preflight-20260903/dev45-generation-current-v1/c1-run1.answers.jsonl`
— 45/45 answer, slot_outcome 전부 `answer`, 생성 모델 45건 모두
`gemini-3.5-flash-lite`. 수집 중 Gemini 3.5의 일시 오류로 서비스가 3.1로
자동 대체한 응답 6건은 collector의 control mismatch로 거부·sidecar 기록 후
resume에서 3.5로 재수집했다(`c1-run1.answers.errors.jsonl` 6행, 최종 답변에
3.1 응답 없음). 검색 진단: hit@5 40/45(.889), MRR .714, gold-chunk@8
any 34/45 · all 31/45 · mean recall .715 (9월 3일 12절 수치와 일치).
Judge 입력 검증(`--validate-only`): validated 45, eligible 45, terminal error 0,
answers_sha256 `1383561b…3dabd6`, judge_config_sha256 `48e32b0b…b119a7`.
Judge v8 r1(gemini-3.1-flash-lite, 1600 tokens)은
`.../dev45-generation-current-v1/judge/c1-run1-judge-v8-r1.jsonl`로 실행.

운영 메모: resume는 collector config 해시가 같아야 한다(`--sleep` 등 인자를
바꾸면 "existing output has a different collector config"로 거부). 일시 오류
대응은 인자 변경이 아니라 동일 명령의 제한 재시도 루프로 한다.

**C0 run1 수집 완료 (2026-09-04 01시 전후)**:
`.../dev45-generation-current-v1/c0-run1.answers.jsonl` — 45/45 answer, 생성
모델 45건 모두 `gemini-3.5-flash-lite`, 포트 18801(`--no-retrieval-tuning`).
일시 오류 sidecar 2행, resume로 전건 3.5 재수집. 검색 진단: hit@5
29/45(.644), MRR .489, gold-chunk@8 any 24/45 · all 12/45 · mean recall .400
(8월 31일 기준선 검색 수치와 일치). Judge v8 r1은
`.../judge/c0-run1-judge-v8-r1.jsonl`로 실행.

DEV45 현재 코드 C0/C1 검색 대비 (같은 인덱스·같은 생성 설정, 차이는 tuning
ON/OFF뿐):

| DEV45 k=8 | C0 (OFF) | C1 (ON) |
|---|---:|---:|
| Source Hit@5 | 29/45 (.644) | 40/45 (.889) |
| Document MRR@5 | .489 | .714 |
| Any required gold chunk @8 | 24/45 | 34/45 |
| All required gold chunks @8 | 12/45 | 31/45 |
| Mean required gold chunk recall @8 | .400 | .715 |

### 2026-09-04 01시 — Judge 첫 실행에서 드러난 오류 3가족과 결정론적 guard 확장 (rubric v11)

C1 run1 Judge(현재 코드 rubric v10, run id는 실수로 `judge-v8-r1`로 명명)
45행 중 오류 행 3건, C0 run1 Judge(20행에서 중단) 오류 행 3건. 분류:

| 가족 | 사례 | 원인 | 재시도로 해결? |
|---|---|---|---|
| 503 UNAVAILABLE | C1 core_02_role, grad_03_role / C0 grad_02 | judge 스트림 2개 동시 실행 중 3.1 과부하, 4회 재시도 소진 | 예 (단일 스트림·재시도 증가) |
| GFC 자기모순 | C1 grad_03 / C0 reg_06 | judge가 GFC=true인데 claim_checks에 미지지 claim 포함 → validate ValueError, 온도 0이라 4회 동일 | **아니오** |
| 인용문 계약 위반 | C0 acad_03 | supported claim의 answer_quote가 final_answer의 연속 부분 문자열 아님 → 설계상 재시도 불가 종단 오류 | **아니오** |

요약기(`summarize_judge_repeats`)는 오류 행이 하나라도 있으면 파일을 거부하고,
런북 18절은 "새 run id로 재판정"만 규정하므로, 결정론적 두 가족은 조건 전체를
집계 불능으로 만든다. 기존 guard(`answerable-clear-refusal-v1`)와 같은
원칙 — **judge 자신의 출력과 관측된 final_answer에서 증명되는 모순만, 점수를
낮추는 방향으로만 교정** — 으로 `apply_deterministic_judge_guards`를 확장했다:

- `unquotable_supported_claim_forces_missing`: supported/partial claim의
  answer_quote가 final_answer의 연속 부분 문자열이 아니면 그 claim을
  missing으로 강등 (검증 불가한 지지는 인정하지 않음)
- `gfc_contradicted_by_own_checks_forces_gfc_false`: GFC=true가 score<2·미지지
  사실·모순·부분 인용·필수 claim 미지지와 공존하면 세부 판정이 우선하여
  GFC=false, score≤1
- 원본 필드는 `deterministic_guard.original_fields`에, 원문 응답은
  `raw_judge_response`에 그대로 보존. guard version
  `answerable-clear-refusal-v1+output-consistency-v1`
- judge config 버전을 `pnu-grounded-fully-correct-v11`로 상향 (prompt·스키마
  불변, 후처리 일관성 규칙 추가). 런북의 v8 표기(이미 v10과 불일치)도 v11로 갱신
- 회귀 테스트 3건 추가 (강등·GFC 확정·무변경 경로), rubric 핀 테스트 갱신

판정 원칙: v11은 C0/C1 양 조건에 동일하게 적용되며 점수를 올리는 방향이 없다.
v10 이전 결과(p0g 등)와 v11 결과를 섞어 해석하지 않는다. 오류 행이 있던
`judge/c?-run1-judge-v8-r1.jsonl`은 실패 이력으로 보존한다.

### 2026-09-04 02시 — DEV45 현재 코드 C0/C1 생성·Judge v11·쌍 분석 완료

Judge v11 r1(gemini-3.1-flash-lite, 1600 tokens, 단일 스트림, sleep 3,
retries 6): C1·C0 각 45/45 판정, **오류 행 0**, 완전성 검사(judge 수 = 대상 수,
judgment_id 유일, rubric v11, 점수 0~2, GFC boolean) 양쪽 PASS. guard 적용
C1 9건(명시적 회피 7, GFC 자기모순 2), C0 17건(명시적 회피 13, GFC 자기모순 2,
인용문 강등 2). 산출물: `.../judge/c{0,1}-run1-judge-v11-r1.jsonl`,
`c{0,1}-summary-v11.json`, `gfc-paired-analysis-v11.json`, `gfc-paired-cases-v11.csv`.

| DEV45 현재 코드 (생성 gemini-3.5-flash-lite, Judge v11 r1) | C0 (OFF) | C1 (ON) |
|---|---:|---:|
| GFC (majority) | 12/45 (.267) | 18/45 (.400) |
| 평균 점수 (0~2) | .867 | 1.200 |
| 검색 hit@5 | .644 | .889 |
| 필수 gold 청크 전부 포함 @8 | 12/45 | 31/45 |

질문 단위 쌍 분석 (n=45, judge repeat 1): Δ GFC = **+.1333**, paired bootstrap
95% CI **[-.0444, +.3111]**, sign-flip p = .238, exact McNemar p = .238.
majority both/C0-only/C1-only/neither = 6/6/12/21. **점 추정치는 향상됐으나
통계적 근거는 충분하지 않다** (CI가 0 포함) — DEV 튜닝 결과이며 holdout·
일반화 성능이 아니다. 검색은 hit@5 +.245·gold 청크 +19문항으로 크게 올랐지만
GFC로의 전이는 부분적이다: C1에서 검색은 맞았는데 GFC가 아닌 문항이 다수이고,
C0-only GFC 6문항은 튜닝이 컨텍스트 구성을 바꿔 잃은 사례로 오류 분석 대상.

Judge 반복(r2·r3)은 이번 DEV 진단에서는 수행하지 않았다(안정성 측정은 holdout
final에서 수행). 서버 18800/18801 종료.

### 2026-09-04 추가 승인 — DEV45 현재 코드 생성 n=3·Judge v11 완료

사용자의 Gemini 외부 전송 승인을 받은 뒤 같은 C0/C1 설정으로 서로 다른 생성
run2·run3을 추가 수집했다. C0/C1 각각 45문항×3회, 총 270개 답변은 모두
`gemini-3.5-flash-lite`를 사용했으며 최종 answer artifact에는 fallback, 빈 답변,
terminal error가 없다. run1 수집 중 모델 fallback으로 거부한 8개 시도는 error
sidecar에만 남고 동일 collector config로 재수집했으며, run2·run3에는 error
sidecar가 생기지 않았다. 여섯 generation artifact 모두 동일 service code SHA
`90997857…f0e6`, index SHA `a4c1ca63…74c31`, source manifest SHA
`1fa7e0f2…845a2`를 기록한다.

각 답변을 `gemini-3.1-flash-lite`, temperature 0, 1600 tokens,
`pnu-grounded-fully-correct-v11`로 정확히 한 번씩 판정했다. 270/270 judgment가
오류 없이 완료됐고 judge config SHA는 모두 `c165059d…18fce`다. v11의
결정론적 하향 guard는 78/270에 적용됐다(C0 46, C1 32). 명시적 회피를 0점으로
고정한 경우 61건, judge 자체 세부 판정과 GFC의 모순을 false로 고정한 경우
13건, 답변의 연속 인용문이 아닌 지지 claim을 강등한 경우 5건이며 점수를 올린
규칙은 없다.

| DEV45 현재 코드, 독립 생성 n=3 | C0 (OFF) | C1 (ON) | C1-C0 |
|---|---:|---:|---:|
| 0–2점 run 평균 | .867 / .933 / .844 | 1.200 / 1.222 / 1.222 | — |
| 문항별 3-run 평균의 평균 | .8815 | 1.2148 | **+.3333** |
| 0점 응답 | 52/135 | 29/135 | -23 |
| 2점 응답 | 36/135 | 58/135 | +22 |
| run별 GFC | 12 / 13 / 11 | 18 / 20 / 20 | — |
| 2/3 majority GFC | 13/45 (.289) | 20/45 (.444) | **+.1556** |

0–2점 문항 평균은 C1 개선/동률/악화가 19/15/11이고, 기본–역할 변형을 같은
family로 묶은 paired cluster bootstrap 10,000회의 차이 95% CI는
**[+.0889, +.5887]**로 0을 포함하지 않았다. 따라서 이 DEV45와 이 Judge
설정에서는 평균 점수 개선이 확인된다.

더 엄격한 2/3 majority GFC는 both/C0-only/C1-only/neither가 9/4/11/21이고,
차이의 family-cluster bootstrap 95% CI는 **[-.0217, +.3201]**, exact sign-flip과
McNemar의 양측 p값은 모두 **.1185**다. 즉 완전정답률의 점 추정치는
15.6%p 상승했지만 통계적 근거는 아직 충분하지 않다. 생성기와 Judge가 같은
Gemini 계열이고 사람 calibration이 없으며, 반복 튜닝한 DEV45이므로 이 결과를
holdout·일반화 성능 또는 객관적 정답률로 표현하지 않는다. 같은 고정 답변을
Judge가 반복 판정하는 stability run도 아직 수행하지 않았다.

정본 집계는
`processed/eval/preflight-20260903/dev45-generation-current-v1/judge/dev45-3run-service-ab-v11.json`
(SHA-256 `dd3d61d8…92db8`)과 companion CSV(SHA-256
`8e41edfb…0353d`)다. 45문항의 C0/C1 3회 답변을 나란히 보는 self-contained
review HTML(SHA-256 `d9f78dd2…27853`)도 같은 디렉터리에 보존했다. 여섯
answer와 여섯 judgment 원본은 같은 디렉터리에 보존했다.

### 2026-09-04 후처리 관계 오판 수정 — 저장 초안 270개 projection·변경 9개 재평가

DEV45 n=3 실패를 답변 단위로 추적한 결과 `svc_reg_04`와 `svc_reg_05`는 C1이
필수 gold chunk를 회수하고 생성기도 정답 문장을 작성했지만, deterministic
attribution guard가 `semantic_relation_mismatch`로 삭제한 후처리 false negative였다.
재현된 원인은 (1) 과거형 연결어 `되었고` 미분리, (2) `등록금`을 두 번째 필수
subject anchor로 취급, (3) `학부 및 대학원 모두 동결` 한 claim과 문서의 두 개
대상별 행을 결합하지 못한 점이다.

`scripts/search_api.py`에는 과거형 절 분리, 구체 subject가 있을 때만 일반
`등록금` object를 relation scope에서 제외, 모든 병렬 subject에 같은 연산자의
근거 행이 있고 충돌 행은 없을 때만 허용하는 coordinated-direction 검증을
추가했다. 수정 전 신규 positive 회귀 3/3 실패, 수정 후 3/3 통과했으며, 대상
누락·반대 연산자 negative 회귀도 거부했다. API hardening 160/160, 전체 unittest
discovery 703 OK(6 skip), `git diff --check` PASS다.

저장된 270개 `sanitized_draft`와 당시 full final contexts를 공식 projection
도구로 다시 처리했으며 모델을 호출하지 않았다. 최종 답변 텍스트는 정확히
9/270만 바뀌었다: C0 `svc_reg_05` 3개, C1 `svc_reg_04` 3개와 `svc_reg_05`
3개다. 나머지 261개는 기존 v11 Judge input hash와 새 input hash가 261/261
일치했다. 변경된 9개만 동일 Judge v11 설정으로 재평가한 결과 모두 2점, 오류
0이었다. 이전→신규 점수는 C0 reg05 `1/1/1→2/2/2`, C1 reg04
`0/0/0→2/2/2`, C1 reg05 `1/0/0→2/2/2`다.

261개 동일 입력의 기존 판정을 재사용한 **score-only 진단**에서 C0 평균은
.8815→.9037, C1은 1.2148→1.2963, C1-C0는 +.3333→**+.3926**으로 변했다.
질문 단위 개선/동률/악화는 20/15/10, family-cluster bootstrap 10,000회
95% CI는 **[+.1429,+.6519]**다. 다만 projection artifact는 명시적으로
service/GFC/citation 성능 주장에 부적격하므로, 이 수치를 새 공식 E2E 또는
holdout 성능으로 쓰지 않는다. 상세 원인·전후 답변·해시는
`processed/eval/preflight-20260903/dev45-generation-current-v1/postprocessor-direction-fix-v1/README.md`에
보존했다.

### 2026-09-04 추가 후처리·단일 숫자 facet·생성 점검표 보강

후처리 오류와 생성 누락을 다시 분리했다. 저장 초안에는 정답이 있었지만 최종
답변에서 사라진 `svc_sch_02`는 같은 표 행의 날짜/시각 분리와 두 지원분야 금액
귀속 문제였고, `svc_adm_02`는 날짜가 붙은 `고지서출력` 행과 바로 다음 UI 경로
행을 결합하지 못한 문제였다. 같은 chunk 안에서 동일 행동·대상·정확한 날짜가
모두 맞을 때만 인접 행을 결합하도록 수정했다. `납부확인` claim을 무관한
`확인` 메뉴가 지지하는 오탐 회귀가 발견되어, claim의 대상 stem이 source의
복합어 안에 들어가는 단방향 일치만 허용했다.

이전 direction-fix-v1 projection과 비교하면 C1 135개 중 6개만 바뀌었다:
`svc_sch_02` 3회, `svc_adm_02` 2회, `svc_grad_05` 1회다. 129/135는 동일하다.
명시적으로 승인된 이 6개 답변과 context만 동일 Judge v11로 판정했으며 오류는
0건이었다. `svc_sch_02` 3회는 모두 `1→2`, `svc_adm_02`는 `1→2` 1회와
`1→1` 1회, `svc_grad_05`는 `1→1`이었다. 뒤의 두 1점은 저장 초안 자체가 각각
전액장학 예외와 이수 기준을 쓰지 않은 생성 누락이라 후처리만으로 복구할 수 없다.

v1 판정 중 입력이 동일한 129개를 재사용해 누적한 score-only 진단에서 C0은
.9037로 그대로이고 C1은 1.2963→1.3259, C1-C0는 +.3926→**+.4222**로
증가했다. 질문 단위 개선/동률/악화는 20/16/9, family-cluster bootstrap
10,000회 95% CI는 **[+.1812,+.6742]**다. 이는 동일 저장 초안 projection이라
fresh E2E/GFC/citation/holdout 성능 주장은 아니다. 상세 판정·해시·집계는
`.../postprocessor-grounding-fix-v4/README.md`와 같은 디렉터리의
`score-only-diagnostic-v4.json`에 기록했다.

0점군의 별도 원인도 확인했다. `svc_core_01`과 역할 변형은 정답 문서가 검색
1위였지만 문서당 cap 2가 `#0004`, `#0015`를 선택해 실제 교원 연간 한도표
`#0017`을 놓쳤다. 단일 숫자 facet도 강한 동일 문서 seed에서만 확장하고,
거리보다 질문 대상(`교원` 대 `직원·조교`) 일치를 먼저 보도록 수정했다.
금액과 같은 행의 `납부금액 10%`는 유지하되, 무관한 `실적급 50%`는 금액
facet으로 보지 않는다. 저장된 C1 DEV45 후보 풀 전체 재선택에서 바뀐 문항은
두 교연비 문항뿐이고 필수 evidence micro recall은 47/71(.662)에서
49/71(.690)로 +.0282, 감소 문항은 0이었다. 이는 DEV 저장 후보 재생이며 새
서비스 검색·생성 성능이 아니다. 원천은
`evidence/20260914/single-numeric-facet-dev45-replay.json`이다.

마지막으로 생성 자체가 필수 정보를 쓰지 않은 `svc_grad_05`, `svc_emp_03`,
`svc_adm_02`를 위해 질문 의도별 누락 점검표를 추가했다. 대체·면제 질문에는
이수 기준, 직무체험 질문에는 실제 활동 방식, 등록금 납부 질문에는 고지서 출력
시점과 미납·전액장학 예외를 요구한다. 반대로 모든 절차 질문에 대상·기간·서류·
문의처를 일괄 요구하던 규칙은 제거해 질문하지 않은 `확인할 수 없음` 문장이
답변 줄 수를 소모하지 않게 했다. 프롬프트 테스트 20/20, API hardening
164/164, 전체 discovery 709 OK(6 skip), py_compile과 `git diff --check`가
통과했다. 프롬프트 효과는 fresh generation 전에는 성능 향상으로 표현하지 않는다.

### 2026-09-04 표적 fresh generation 5건과 semantic guard 재수정

알려진 실패 5문항만 최신 C1 서비스로 1회씩 재생성했다. Generator는
`gemini-3.5-flash-lite`, Judge v11은 `gemini-3.1-flash-lite`였고 각 5/5,
오류·fallback 0이다. 이전 세 C1 run에서 다섯 문항 평균은 매번 .600이었고 이번
표적 run은 1.600, GFC 3/5였다. `svc_core_01`과 역할 변형은 `0→2`,
`svc_adm_02`는 `1→2`로 바뀌었고 `svc_grad_05`, `svc_emp_03`은 1점이었다.
의도적으로 실패군만 골랐고 n=1이므로 이 +1.000을 전체 성능으로 일반화하지 않는다.

trace를 보면 두 1점의 생성 초안에는 필수 사실이 실제로 모두 있었다. grad05의
가중 70점 기준과 emp03의 프로그램명·대상·VOD·실무과제·현직자 피드백을
deterministic guard가 `semantic_relation_mismatch`로 삭제했다. 원인은 뒤쪽
`이수할 수 있다`가 앞쪽 comparator modality까지 오염시킨 점, `모집대상`의
`학과(부)` 괄호를 제외 목록으로 오인한 점, 공고의 `직무체험`·`직무 확인`
명사형 사실을 생성기의 자연스러운 가능 표현과 연결하지 못한 점이다.

명시적 부정을 우선 거부하면서 정확 scope 75%·최소 4개 anchor와 동일 comparator를
요구하는 좁은 bridge로 수정했다. 신규 positive 회귀는 수정 전 5/5 실패, 수정 후
5/5 통과했고 잘못된 점수·활동 불가·제외 대상 negative도 거부한다. 실제 저장
초안 replay는 두 문항만 바꾸고 나머지 3/5는 동일하다. API 169/169, 전체
714 OK(6 skip), py_compile과 diff check가 통과했다. 별도 승인한 Judge v11 partial
run에서 변경 두 문항은 모두 Gemini 3.1 Flash Lite 기준 2점·GFC true·citation
support full이었고 오류·fallback은 0이다. 따라서 동일 저장 초안 5건의 진단
projection은 fresh pre-fix 1.600·GFC 3/5에서 2.000·5/5가 됐다. 다만 알려진
실패군의 저장 초안 재처리이므로 fresh E2E나 일반화 성능으로 주장하지 않는다. 원천은
`processed/eval/preflight-20260904/targeted-generation-prompt-retrieval-v1/README.md`다.

### 2026-09-04 의도 기반 검색 확장과 분리 청크 완결

남은 DEV 검색 실패를 전수 확인한 결과, 정답 문서 부재보다 사용자 표현과 공식
문서 용어의 불일치가 주원인이었다. 검색 튜닝 lane에만 학생증·증명서 위탁,
입학 모집인원 변경, D-2 단체접수, 외국인 학부 신입학, 교환학생 선발,
하계방학 자격증 과정, 장애학생 학습지원, 제3자 인권신고의 좁은 의도 확장을
추가했다. 확장어는 문서 유형·항목명만 사용하며 정답 숫자·기관명·판정값은
주입하지 않는다. 긴 구어체가 FTS 32-term 예산을 먼저 소모하지 않도록 확장어를
앞에 배치했다.

교환학생의 선발 규모와 일정, 제3자 신고의 가능 여부·피해자 의사·인적사항처럼
같은 문서의 떨어진 청크가 함께 필요한 경우에는 질문 facet을 각각 분리하고,
강한 동일 문서 seed에서만 zero-marginal 청크를 교체했다. 일반 외국인 학부
입학 질문에는 일반 모집요강을 우선하고, 명시적 대학원 문서는 뒤로 보내되
사용자가 특수 트랙을 직접 지정하면 원래 BM25 순서를 유지한다. 생성 prompt에도
같은 누락 점검표를 추가했으나, 실제 모델 재생성 전까지 생성 성능으로 표현하지
않는다.

실제 `POST /chat` DEV45 재수집에서 C1은 Source Hit@5 45/45(1.000), MRR
.8137, Any required evidence@8 45/45(1.000), All required evidence@8
40/45(.889), mean required-evidence recall@8 .9407을 기록했다. 직전 확장 전
C1의 40/45, .7137, 34/45, 33/45, .7481 대비 각각 +5문항, +.1000,
+11문항, +7문항, +.1926이다. 추적한 모든 지표의 loss는 0건이었다. 다만
반복 확인한 DEV45의 표적 튜닝 결과이므로 holdout·일반화·LLM 생성 성능이 아니다.
전체 unittest discovery는 720 OK(6 skip), py_compile과 `git diff --check`가
통과했다. 원천과 해시는
`processed/eval/preflight-20260904/dev45-retrieval-query-expansion-v5/README.md`에
고정했다.

### 2026-09-04 의도 기반 검색의 표적 fresh 생성·Judge 9건

이전 DEV45 생성에서 실패했던 9문항을 명시적으로 고른 뒤 현재 C1 서비스로
각 1회 새로 생성하고 Judge v11로 각 1회 판정했다. Generator는
`gemini-3.5-flash-lite`, Judge는 `gemini-3.1-flash-lite`였으며 생성·판정
각 9/9가 한 번의 시도로 끝나 오류·retry·fallback은 없었다. 검색은 9/9 hit,
MRR .944였고 exact required gold chunk는 @8 any 9/9, all 7/9였다.

| 표적 9문항 | 이전 C1 run1 | 이전 run2 | 이전 run3 | fresh run |
|---|---:|---:|---:|---:|
| Judge 평균(0–2) | .5556 | .4444 | .5556 | 1.5556 |
| GFC | 1/9 | 0/9 | 1/9 | 6/9 |

`svc_core_03`, `svc_adm_03`, `svc_intl_01`, `svc_emp_01`, `svc_sup_01`,
`svc_sup_03`은 2점·GFC를 받았다. `svc_intl_02`와 `svc_intl_03`은 1점,
`svc_intl_03_role_free`는 0점이었다. 알려진 실패군을 개선 예상 방향으로
선택한 n=1 진단이므로 이 차이를 전체 DEV 또는 일반화 성능으로 쓰지 않는다.
원천은
`processed/eval/preflight-20260904/targeted-generation-intent-retrieval-v5/README.md`다.

### 2026-09-04 잔여 국제 문항 원인 수정과 DEV45 v6

`svc_intl_02`의 raw draft에는 1차 온라인 지원 시작·종료 날짜와 시각,
스마트학생정보시스템 경로가 모두 있었지만 semantic guard가 요일 괄호와 축약된
종료일을 별도 관계로 오인해 문장을 삭제했다. 요일을 날짜-시각 atom에 포함하고
표의 `1차`/`2차` scope를 다음 행까지 유지하되 다른 회차의 날짜를 빌리지 못하게
수정했다. 동일 저장초안 replay에서는 이 답변 하나만 복원되고 나머지 8개는
byte-identical하다. 모델이나 Judge를 다시 부르지 않았으므로 post-fix 점수는 없다.

D-2 단체접수 두 문항은 생성 문제가 발생하기 전에 검색 context가 불완전했다.
예약·정의 `#0000`, 1차 일정 `#0006`, D-2 연장 서류·현금 60,000원 `#0014`가
한 PDF에 떨어져 있어 전역 문서 cap 2로는 세 근거를 동시에 제공할 수 없었다.
전역 cap은 2로 유지하고, D-2 단체접수의 예약·일정·수수료/서류를 함께 묻는
질문에만 effective cap 3을 기록해 세 청크를 선택하도록 했다. 외국인 유학생
역할 설명에도 `국제·비자`를 포함했다.

fresh retrieval-only DEV45 v6에서 Hit@5 45/45와 MRR .8137은 유지되고,
All required evidence@8은 40/45(.889)→42/45(.933), mean recall은
.9407→.9704로 증가했다. 개선 문항은 D-2 기본·역할 두 개뿐이며 추적한 loss는
모두 0이다. 확장 전과 누적 비교하면 Hit@5 40/45→45/45,
All required evidence@8 33/45→42/45, mean .7481→.9704다. 이는 반복 튜닝한
DEV 검색 결과이고 D-2 extractive 출력은 생성 품질 지표가 아니다. 수정 뒤
fresh LLM 생성·Judge는 아직 수행하지 않았다.

### 2026-09-04 자부담금 지원 3-part 문맥과 DEV45 v7

v6의 미완전 required-evidence 세 문항을 다시 분류했다. 교연비 기본·역할 두
문항은 질문의 답인 교원 합계 `1,800만원+α` 표 `#0017`을 이미 회수하며, 앞선
fresh generation에서 둘 다 2점·GFC였다. 빠진 `#0016`은 “예산 범위 내에서
개인별 연간 지급한도액 설정”이라는 배경문이다. 이 단계에서 세 번째 청크를
추가하거나 required label을 바꾸면 답변 품질보다 DEV 숫자를 최적화하게 되므로
annotation은 그대로 두고 사람 검토 대상으로 남겼다.

반면 `svc_sup_02`는 이전 C1 생성 3회가 1/1/0점이었고, 제출서류와 접수 경로
누락이 직접 감점 사유였다. 공식 안내문은 지원 대상·전액지원 `#0002`, 선납부 후
증빙 확인 절차 `#0010`, 보급결정 통지문·재학증명서·납부 증빙·통장사본 및
구글폼 경로 `#0007`로 갈라져 있었다. 정보통신보조기기 자부담금 지원 질문에만
세 독립 facet과 effective cap 3을 적용하고, 일반 보조기기 질문은 cap 2를
유지했다. 생성 점검표도 지원범위·선납부 절차·제출서류·접수 경로로 맞췄다.

실제 서비스 한 건에서 exact gold가 1/2→2/2가 됐고, 이어서 수행한 fresh
retrieval-only DEV45 v7에서는 v6 대비 이 한 문항만 바뀌었다. Hit@5 45/45와
MRR .8137은 유지됐으며 All required evidence@8은 42/45(.933)→43/45(.956),
mean recall은 .9704→.9778로 증가했다. 모든 tracked loss는 0이다. 확장 전과
누적하면 All required evidence@8 33/45→43/45, mean .7481→.9778이다.
전체 unittest discovery는 725 OK(6 skip)다. 외부 생성·Judge는 호출하지 않았고,
원천과 해시는
`processed/eval/preflight-20260904/dev45-retrieval-query-expansion-v7/README.md`에
기록했다.

Judge 수집 전 검증 과정에서는 DEV 표적 실행의 `max_attempts=1`도 최종 holdout과
같이 3회로 강제하던 검증 결함을 발견했다. 최종 실행은 계속 정확히 3회를 요구하고,
비최종 DEV 성공 record만 양의 고정 시도 수를 허용하도록 분리했으며 회귀 테스트를
추가했다. 전체 unittest discovery 723 OK(6 skip), py_compile과
`git diff --check`가 통과했다. v6 원천은
`processed/eval/preflight-20260904/dev45-retrieval-query-expansion-v6/README.md`다.

### 2026-09-04 표적 4문항 fresh 생성·Judge와 D-2 후처리 재분해

사용자 승인 범위를 Gemini 3.5 Flash Lite 생성 4회와 Gemini 3.1 Flash Lite
Judge v11 4회, 각 최대 1시도·fallback 없음으로 고정했다. 생성 서버의 허용
모델도 3.5 하나로 제한했다. `svc_intl_02`, `svc_intl_03`, `svc_sup_02`,
`svc_intl_03_role_free` 모두 첫 시도에 완료됐고 오류·retry·fallback은 0이었다.
검색 hit, MRR, exact required gold all@5와 all@8은 모두 4/4였다.

Judge 점수는 순서대로 `2,1,2,1`, 평균 1.500, GFC 2/4였다. 직전 동일 4문항의
`2,1,1,2`와 평균·GFC가 같고, 자부담금 지원은 신청기간과 선납부 절차를 모두
포함해 1→2가 됐지만 역할 D-2가 2→1로 내려가 상쇄됐다. 따라서 이 실행은
대폭 성능 향상을 지지하지 않는다. 알려진 실패군 n=4의 단일 확률 표본이며
holdout·일반화 수치도 아니다.

trace를 분해하자 D-2 검색 근거 세 청크는 모두 들어왔고 raw generation도 생각보다
완전했다. 기본 문항 raw에는 정확한 1차 기간·시간과 회차별 대상이 있었지만
`9시 20분` 대 `9:20`의 표기 차이 및 세 표 행 요약 때문에 후처리에서 삭제됐다.
역할 문항 raw에는 서비스 정의와 60,000원·현금·만원권·정부초청장학생 면제까지
있었지만 permission semantic guard가 둘을 삭제했다. 기본 문항 raw 자체에는
수수료 납부방식이 없었다.

후속 로컬 수정은 D-2 점검표를 5개 독립 항목으로 나누고, 근거 값 대신
“확인하세요”라고 쓰지 못하게 하며, 답변 상한을 7줄로 넓혔다. 후처리는 다수의
정확한 시간값을 가진 표기 변환, 필수 사전예약이 명시된 D-2 서비스 정의,
현금·만원권·장학증서 면제 행, 1·2차 대 3차 대상을 각각 좁게 허용했다. 서비스
중단·현금 불가·만원권 미만·다른 행사 일정은 음성 회귀로 거부한다. 같은 4개
저장 초안/context를 재투영한 결과 19/19 문장과 39/39 critical value를 보존하고
거부 claim은 0이었다. 이는 외부 호출 없는 same-draft 진단이므로 새 Judge 점수로
사용하지 않는다. 전체 unittest discovery는 735 OK(6 skip), py_compile과
`git diff --check`가 통과했다. 원천과 해시는
`processed/eval/preflight-20260904/targeted-generation-postfix-v8/README.md`에
고정했다.

### 2026-09-04 DEV45 C1 전체 저장초안 후처리 감사와 표적 수정

직전 표적 4문항을 넘어 기존 DEV45 C1 생성 3회, 총 135개 저장 `draft_answer`를
전수 재투영했다. 초기 현재 코드 projection은 run별 47/48/48개 문장을 삭제했고
핵심값 11/34/17개를 잃었다. 모델이 원래 출력한 회피문 제거가 다수였지만,
이전 Judge 실패 사유와 대조하자 정답 문장도 반복적으로 삭제되고 있었다.

원인은 단위가 한 번만 적힌 수량 범위, 명사형 `지원자격/대출대상`, 예산표의
`예산 규모/비중` 동의 표현, `참여혜택` 번호 행, 부서별 연락처 행, 특강 운영과
자격증 취득 지원의 명사형 서술이었다. 각 패턴은 같은 행·가까운 표제·동일 숫자와
강한 용어 일치를 요구하도록 좁게 허용하고, 잘못된 범위·연도·비율·제외 대상·
지급 불가·다른 부서·운영 중단을 음성 회귀로 고정했다. 서로 다른 근거 블록의
대상·자격, 신청 경로, 비용을 한 문장으로 합쳐 후처리되는 문제는 validator를
느슨하게 하지 않고 생성 prompt에서 줄을 분리하도록 했다.

같은 저장 초안·context의 최종 재투영에서 답변 15/135가 바뀌고, source-backed
문장 16개가 추가 보존됐으며, 핵심값 손실은 합계 62→44로 18개 감소했다. 변경된
15개 record 중 13개는 기존 Judge 2점 미만 문항이며 당시 실패 사유가 복원된
누락 사실을 직접 지목했다. 이는 수정 대상을 설명하는 진단이지 새 점수나
13건의 성능 개선 판정은 아니다. fresh 생성·Judge와 holdout은 실행하지 않았다.
API 201 tests, generator 20 tests, 전체 discovery 747 tests OK(6 skip)다. 원천,
case 목록, 해시와 해석 제한은
`processed/eval/preflight-20260904/dev45-postprocessor-capability-fix-v1/README.md`에
고정했다.

### 2026-09-04 튜닝 동결

이 시점부터 `scripts/search_api.py`, `scripts/bm25_search.py`,
`scripts/rag/generators.py`의 검색 규칙·질의 정규화·생성 prompt·점검표·후처리
규칙을 변경하지 않는다. 이후 작업은 분석 도구·테스트·문서에 한정하며, 새 결함은
고치지 않고 `동결 후 발견`로 기록한다. `/health`의 `startup_code_sha256`과 같은
계산인 `scripts/search_api.py` 파일 SHA는
`9a8865db361d6d3a044d26fc57698f56840540ed573a5f37bb50ef684c1cf4a5`다.

| 동결 대상 | 줄 수 | SHA-256 |
|---|---:|---|
| `scripts/search_api.py` | 8,325 | `9a8865db361d6d3a044d26fc57698f56840540ed573a5f37bb50ef684c1cf4a5` |
| `scripts/bm25_search.py` | 1,273 | `6c474dc83a4f27ab3172de5b839f731e9954206b1d3cb1748dc08f3848cb1ad7` |
| `scripts/rag/generators.py` | 1,294 | `67cccfd602b808929c226fcbb659635495e50e05103592664020c6c3a725166b` |
| Cascade index | — | `a4c1ca6318cb14721866a6512f3dfeacdca58c62ecff98ea7b81df2e75c74c31` |
| Baseline index | — | `a993f00d222177adf668fa7069869f0a79576166fba9234662c37a7b657b94e7` |
| Challenger index | — | `6c1aab850a0b1ac572de9123eee8acfb21007afffaeb6b6e66c7dffc6875ece4` |

동결 전 품질 게이트는 `git diff --check`, `bun run lint`, `bun run build`가 모두
exit 0이었다. unittest는 제한 샌드박스에서 fixture용 `127.0.0.1` bind가 막혀
23개 `PermissionError`가 발생했으나, 외부 API 호출 없이 동일 명령을 로컬 소켓이
허용된 환경에서 다시 실행해 **747 tests OK, 6 skipped, 실패 0**을 확인했다.
negative-path 테스트가 출력하는 `stale`, `error`, `usage` 문구는 예상된 stderr다.
Vite build는 1,735 modules를 변환하고 성공했다.

기계판독 스냅샷은
`evidence/20260914/tuning-freeze-snapshot-20260904.json`이며 SHA-256은
`a34911e6166ead6640f19259be51c873d8e770d9f14525c36037c95627a78106`이다.
당시 HEAD는 `0a45cd5f66e843abe83eeb16abafd8acede1cc4d`, worktree status entry는
96개였다. 이는 서비스 튜닝 동결 스냅샷이며, 분석 도구와 문서 정비가 끝난 뒤
최종 코드 freeze 품질 게이트와 승인용 파일 목록을 다시 고정한다. 외부 LLM 호출은
0회였고 holdout 내용은 열지 않았다.

### 2026-09-04 규칙 인벤토리 고정

동결된 서비스 코드 3개에서 검색 결과·생성 지시·최종 claim 채택을 바꾸는
production decision rule을 전수 분류해 `docs/rule-inventory-20260904.md`를
작성했다. 단순 URL·보안·입력 형식·lexical parser는 범위에서 제외하고,
`Rule ID ↔ 의사결정 함수 또는 함께 움직이는 정규식 그룹`을 1:1 대응시켰다.

기계 검증 결과는 총 68행·고유 Rule ID 68개, 일반 21개, 표적 47개다. 표적
규칙의 DEV 근거 ID는 role 변형을 포함해 중복 제거 37개이며, 표 행에서 추출한
ID 집합과 문서 집계 목록이 정확히 일치했다. `rg`로 질의 확장·facet sibling·
후처리 bridge·후보 강등·생성 점검표의 production 함수가 인벤토리 범위에 있는지
확인했다. 후속 analyzer와 일치하도록 검증 명령의 Rule ID 패턴을 보정한 최종
산출물 SHA-256은
`c20fbafd8a460514f475c0a5779c6704162546beb76f322f74335b090fc6a491`이다.
서비스 코드는 변경하지 않았고, 외부 호출 0회·holdout 내용 접근 0회다.

### 2026-09-04 동결 후 규칙 발동 분석기 준비

`scripts/analyze_rule_activation.py`와 합성 trace 단위 테스트 4개를 추가했다.
분석기는 answer JSONL만 읽고 서비스 모듈을 import하거나 API를 호출하지 않는다.
인벤토리의 68개 Rule ID를 record마다 모두 출력하며, trace 직접 증거는
`observed_active/inactive`, 질의·artifact 추정은 `trigger_possible`, 식별자가
없으면 `trace_absent`로 분리한다. JSON은 규칙별·조건별 발동률과 문항 상세를,
CSV는 문항별 규칙 목록을 제공한다.

DEV45 C1 3회 135개로 검증한 v3 정본은 인벤토리 68개·표적 근거 ID 37개를 모두
읽었다. 표적 규칙의 trace상 관측 case는 29개이며 의도된 표적 24개와 spillover
5개로 나뉜다. 발동 가능 case는 40개로, 의도된 37개를 모두 포착하고
`svc_acad_01`, `svc_reg_01_role_student`, `svc_reg_06` 세 문항이 spillover였다.
관측 24/37의 차이는 9월 3일 artifact가 일부 9월 4일 규칙보다 오래됐고 후보
강등·세부 후처리 bridge가 rule ID를 trace에 남기지 않기 때문이다. 이는 실제
발동으로 보정하지 않고 그대로 `trace_absent`로 남겼다.

첫 v1은 `QRY-NORM-*` 두 ID를 누락해 66개만 처리했고, v2는 68개를 복구했으나
spillover 집계 전 버전이다. 둘 다 덮어쓰지 않았고 v3만 정본으로 지정했다.
단위 테스트는 4 tests OK다. 산출물과 SHA는 다음과 같다.

- `processed/eval/post-freeze-analysis-20260904/rule-activation-v3/dev45-c1-n3-rule-activation.json`:
  `f55241e7cce50bf8b8747fe85222916374e573523ce1673b63793d1968333a1e`
- 같은 디렉터리 `dev45-c1-n3-rule-activation.csv`:
  `6c49b3f70b2b468283408ba1093fe71c4df6768f11d0bd344593e184b4bebdc7`
- 같은 디렉터리 `README.md`:
  `f990b81035c972f46eebfdfd653d9519ead826d68df8e371d2aaf09189bc0920`
- `scripts/analyze_rule_activation.py`:
  `0eaef1e7ee2df68367580f4c57f02b009d5cf7031419d5a35b597bcd8c613b96`
- `tests/test_analyze_rule_activation.py`:
  `e1749d8bf85380ed0010572b0ad4b7e4baefc7b31aff06f6d06dbde27a8ef1f8`

서비스 튜닝 코드 변경, 외부 호출, holdout 내용 접근은 모두 0회다.

### 2026-09-04 DEV45 생성 실패 분해 도구 준비

`scripts/analyze_generation_failures.py`와 합성 answer/Judge 단위 테스트 4개를
추가했다. 분석기는 answer와 같은 `answer_id`의 Judge v11 판정을 SHA·case·조건·
run까지 검증해 결합하고, 검색 hit × 필수 근거 all@8 × 0~2점/GFC 교차표를
조건별·run별로 출력한다. 비GFC 원인은 다중 레이블과 상호배타적인 주원인 분할을
함께 제공하므로 중복 원인 수의 합을 실패 건수로 오해하지 않게 했다.

DEV45 C0/C1 각 3회, 총 270개 기존 판정을 새 v2 경로에서 분석했다. C0는 평균
0.881481·GFC 36/135·비GFC 99, C1은 평균 1.214815·GFC 58/135·비GFC 77로
정본 `judge/dev45-3run-service-ab-v11.json`의 평균·점수 분포·GFC 수와 모두
일치했다. 지시서의 확인점인 C1 run1도 GFC 18/45·비GFC 27이며, 다중 레이블
기준 명시적 회피 guard 7, 필수 claim 누락 19, 부적절 회피 16, 모순 1,
부분 인용 1을 재현했다. 단위 테스트는 **4 tests OK, skip 0, 실패 0**이었다.

- `processed/eval/post-freeze-analysis-20260904/generation-failures-v2/dev45-c0-c1-n3-generation-failures.json`:
  `5a83fadf4b8ae275c8721187f5e47fb3a7bbe5906a93909680de33d9171e58e8`
- 같은 디렉터리 `dev45-c0-c1-n3-generation-failures.csv`:
  `8ec985d7459918aef2e850f4b41e212de7541c1b6057fcf6ec8c11ba1b652e04`
- 같은 디렉터리 `README.md`:
  `83c3ca0b0b35035b5f938273fa6218eabedd14cc6f0a2bab44bbfcabf2d0b830`
- `scripts/analyze_generation_failures.py`:
  `29c9a9e411fd23ae48a30020e57b98d1d3252451d9db890ae7e3144eb268ae8a`
- `tests/test_analyze_generation_failures.py`:
  `8e6b66cbb99bf5ad30961173fb8301c3723bb175f6366dce7a255d20236c2c6a`

v1은 run별 요약을 넣기 전 최초 출력이므로 덮어쓰지 않고 보존했다. v2를 정본으로
지정한다. 서비스 튜닝 코드 변경, 외부 호출, holdout 내용 접근은 모두 0회다.

### 2026-09-04 보고서·런북 정비

`docs/final-report-20260914.md`에서 서비스 headline을 BM25 단일 lane으로 바로잡고
dense/hybrid를 연구비·탐색 실험과 분리했다. 4장에는 일반 규칙 21개와 표적 규칙
47개(표적 DEV ID 37개)를 구분했고, 6장에는 사전 고정 주지표가 GFC이며 0–2점
평균은 보조 지표라고 명시했다. 8장에는 표적 규칙 과적합과 holdout 발동률 검증,
서비스 코드 줄 수 증가, DEV45·Core27의 검정력, 동일 모델 계열 자기 선호와 사람
calibration, suspect 약 11%·base64·연도별 사본 누적 문제를 반영했다. 기존 성능
수치는 바꾸지 않고 정본 artifact와 이 log를 각주로 연결했다.

`docs/final-eval-runbook-20260914.md`에는 Judge v11의 감점 전용 guard 3종,
dirty DEV에서 `--expected-index-sha256`/Git pin을 생략할 때도 corpus revision과
source manifest를 유지하는 규칙, resume 시 `--sleep` 등을 포함한 collector config
hash를 바꾸지 않는 규칙을 추가했다. 2026-09-04 freeze-prep 품질 gate도 기록했다.

- `docs/final-report-20260914.md`:
  `cd9ea0e49a1ff431b009f50aed7cc0ae56e2271beee55709c16085b77b7ce977`
- `docs/final-eval-runbook-20260914.md`:
  `9e2e075c5c67522ada7ede32cf907ce6f38416adb7d297183ee752fd4e8891f6`

### 2026-09-04 freeze 커밋 승인 목록 준비

`git status --short --untracked-files=all` 130개 항목과 새 승인 문서를 대조해
`docs/freeze-file-list-20260904.md`를 작성했다. 최종 status 131개 중 holdout draft,
검토 packet, Reviewer A/B 응답 4개는 내용·해시를 확인하지 않고 보류했으며, 나머지
127개를 수정 9·신규 scripts 37·tests 32·config 6·docs 15·evidence 28로 분류했다.
문서 표의 경로 집합은 status에서 보류 4개를 뺀 집합과 정확히 일치했다
(`missing=[]`, `extra=[]`).

`.env`와 `processed/`의 gitignore를 확인했고, embargo 4개를 제외한 경로에 대해
실제 값 형태 API key 패턴은 파일명 매치 0건이었다. 5 MB 초과 신규 단일 파일은
없고, 가장 큰 cap2/cap4 JSONL은 각각 약 3.88/3.79 MB의 재현 evidence로 표시했다.
승인 목록 SHA-256은
`0f8bc1bf878a275943217c9c8c71a1dfa02d39cf24794fa4015ecbbe7e809b44`다.

커밋 메시지 초안은 `chore: freeze PNU final evaluation code`, 태그 초안은
`pnu-eval-code-freeze-20260904-v1`이다. **승인 대기** 상태이며 staging·commit·
tag·push는 실행하지 않았다.

### 2026-09-04 최종 freeze-prep 품질 gate

분석 도구와 문서까지 포함한 현재 worktree에서 `git diff --check`,
`python3 -B -m unittest discover -s tests -p 'test_*.py'`, `bun run lint`,
`bun run build`를 다시 실행했다. 네 명령은 모두 통과했고 unittest는
**755 tests OK, 6 skipped, 실패 0**, Vite build는 1,735 modules였다. 테스트 중
보이는 `stale`, `error`, `usage`는 fail-closed 음성 경로가 의도적으로 출력한
stderr다. 동결 서비스 3개 SHA는 최초 snapshot과 다시 정확히 일치했다.

### 2026-09-04 DEV45 Judge v11 안정성 반복 계획 — 승인 대기

외부 호출 전 계획만 고정했다. C0/C1 run1의 기존 답변은 합계 90개다. 각 답변을
`judge-v11-r2`와 `judge-v11-r3`로 두 번 추가 판정하므로 필요한 **추가 성공 판정은
90회가 아니라 180회**다(2조건 × 45답변 × 2반복). `--sleep 3 --retries 6`의
단일 스트림으로 실행하며, 이 CLI에서 `--retries 6`은 최초 요청을 포함한 최대
6 attempts이므로 극단적 상한은 1,080 HTTP attempts다. 기존 r1이나 answer를
수정하지 않고 네 개의 새 judgment JSONL에 기록한 뒤, 조건별로
`summarize_judge_repeats.py`에 r1/r2/r3를 전달해 score·GFC 일치율을 집계한다.

비용·무료 tier 여부와 무관하게 현재까지 외부 LLM 호출은 0회다. 사용자의 별도
명시 승인 전에는 실행하지 않는다.

### 2026-09-04 동결 후 C2 quote-bound 생성 실험 준비

DEV45 추가 규칙을 만들지 않고 생성 성능 병목을 조사했다. frozen C1의 기존 answer
artifact를 현재 postprocessor로 재생한 결과 unsupported claim 127개 중 83개는
model abstention, 나머지 44개 중 40개는 `semantic_relation_mismatch`, 3개는
`critical_value_mismatch`, 1개는 `low_lexical_overlap`이었다. 반면 C1 3회에서는
필수 근거 all@8이 충족된 93개 중 37개가 비GFC였다. 즉 검색 결과가 있어도 자유
서술 문장과 사후 휴리스틱 근거 추론 사이에서 올바른 claim을 버리거나 잘못된 출처를
연결하는 것이 현재 상한의 주된 원인이다. 이 발견으로 frozen 서비스 코드나 DEV
규칙을 수정하지 않았다.

별도 실험 lane인 `scripts/rag/grounded_claims_v2.py`를 추가했다. 모델이 claim마다
`source_number`와 원문에서 복사한 연속 `quote`를 함께 출력하도록 계약하고,
application 쪽에서 JSON shape, 출처 번호, normalized exact quote 포함, claim 숫자의
인용문 포함, 핵심 lexical anchor, 가능/불가·동결/인상 등 관계 방향을 deterministic
fail-closed로 검사한다. 일부 claim만 근거가 있으면 그 claim은 보존하고 나머지만
명시적으로 답변 불가 처리한다. 기존 `scripts/search_api.py`,
`scripts/bm25_search.py`, `scripts/rag/generators.py`에는 연결하거나 수정하지 않았다.

`scripts/evaluate_grounded_claims_v2.py`는 frozen C1 answer의
`evaluation_trace.retrieval_stages.final_contexts`를 그대로 재사용한다. 입력 answer
identity와 파일 SHA를 전후 검증하고, `holdout`/`draft` 경로와 symlink를 거부하며,
한 Gemini model만 사용하고 fallback하지 않는다. 요청은
`responseMimeType=application/json`과 `responseJsonSchema`로 고정하되 schema 준수와
별도로 위 application 검증을 수행한다. dry-run은 credential을 읽거나 파일을 쓰지
않고, live 실행은 정확한 승인 문구가 없으면 첫 호출 전에 종료한다. 정상·실패
artifact 모두 새 경로에만 원자적으로 게시하며 기존 artifact는 덮어쓰지 않는다.

DEV45 C1 run1 정본으로 dry-run한 결과 **45문항·예정 성공 호출 45회·실제 외부 호출
0회**였고, 45개 prompt hash와 source artifact SHA
`1383561bb04ba242c9dd039d26c821abfd53a36de8e7d18fa268e2293e3dabd6`를
검증했다. 비교 기준은 같은 C1 run1 Judge v11 r1의 **평균 1.2000, GFC 18/45**다.
C2 성능 수치는 아직 생성·Judge를 호출하지 않았으므로 존재하지 않으며, 개선됐다고
주장하지 않는다. 검증에는 C2 생성 45회와 Judge 45회, 합계 90 successful calls가
필요하다. 이는 앞 절의 Judge 안정성 반복 180회와 별도이며 둘 다 승인 대기다.

추가한 코드·테스트 SHA-256은 다음과 같다.

- `scripts/rag/grounded_claims_v2.py`:
  `07e82e7e285cf19f67e78e7244ec8f4d27378e1f7d50e48cc961ce612c47048e`
- `tests/test_grounded_claims_v2.py`:
  `9f4bbb23bc66d01fea5a6495b671a1ffa20ff8704c1c26e8322fc19891e601c2`
- `scripts/evaluate_grounded_claims_v2.py`:
  `6622fbc5f79e44a409ae075dfa88df4c09f4f3088eb70d9373f3749ff4d4b68c`
- `tests/test_evaluate_grounded_claims_v2.py`:
  `7c579bfba8624c696cdc61dc899fc8612fedacd16c6858c02a6e02a3b8bbd2e9`

표적 테스트는 **16 tests OK**, 전체 suite는 로컬 fixture 소켓이 허용된 환경에서
**771 tests OK, 6 skipped, 실패 0**이었다. 제한 샌드박스의 첫 전체 실행에서는
소켓 bind가 필요한 기존 테스트 23개가 `PermissionError`였고, 동일 명령을 승인된
비샌드박스 환경에서 재실행해 코드 회귀가 아님을 확인했다. `git diff --check`,
`bun run lint`, `bun run build`도 통과했고 build는 1,735 modules였다. 외부 LLM
호출과 holdout 내용 접근은 모두 0회다. C2 파일은 최초 C1 freeze 목록 작성 뒤 생긴
post-freeze 실험 파일이지만, 사용자의 2026-09-04 git merge 승인에 따라 승인 목록의
별도 C2 addendum으로 포함했다.

### 2026-09-04 GitHub 반영과 C2 v3 생성기 동결

사용자가 로컬 merge가 아니라 GitHub 반영을 뜻한다고 명확히 했고 push 및 외부
API 호출을 승인했다. 승인된 freeze 목록만 exact path로 stage한 commit
`8d1f29a`(`chore: freeze PNU final evaluation code`)를 기존 `origin/main`과
병합한 `731b7a0`까지 `origin/main`에 push했다. holdout draft·검토 packet·
Reviewer A/B 파일 4개는 stage하지 않았다. tag는 요청받지 않아 만들지 않았다.

C2 최초 live run은 세 번째 응답의 같은 `source_number` 복수 quote를 local
verifier가 중복으로 잘못 거부해 중단됐다. 정상 2행 partial과 error 기록은
덮어쓰지 않고 각각 다음 경로에 보존했다.

- `processed/eval/preflight-20260904/dev45-grounded-claims-c2-v1/c2-run1.answers.jsonl.partial.jsonl`:
  `37b1c821075d0343b636100de8dc1388af40363d9647190a049c4ba3f1028eca`
- 같은 디렉터리 `c2-run1.answers.jsonl.error.json`:
  `9337407f68fb002f2b0d93e0b4533f642ed2d0b28095511ffa618395093c333e`

중복 출처 번호를 허용하되 서로 다른 quote identity를 요구하도록 C2 parser를
고쳤다. `c2-run1b`는 45/45 응답, 수용 claim 75·거부 44로 완료됐고 answer SHA는
`f088b752db2d2dbd881c6a4ca8c586f1be705aef7ffdf5c05a651453668f9b6a`다.
최초 summary가 dry-run 계획의 `external_calls=0`을 잘못 계승한 결함은 기존
summary를 수정하지 않고 실제 45회 호출을 기록한
`c2-run1b.calls-audit-v1.json`
(`b66fe04208a97be2154228827aad1ada3c2e108bd08a432891a99a5b3ee3445d`)을
추가하고 collector의 후속 summary 계산만 고쳤다.

이 답변의 첫 Judge v11은 평균 .9778·GFC 14/45로 C1보다 낮았다. 44개 local
거부를 분해하자 critical numeric 31, anchor 7, relation 6이었고 `10시`와
`10:00`, 점 표기 날짜, 제목에만 있는 학년도, 한국어 복합어, `로그인하여` 안의
`인하`, `선택하고`와 선택사항 혼동 같은 일반 false negative가 확인됐다. 특정
DEV 정답값을 규칙에 넣지 않고 시간·날짜 정규화, metadata scope, 복합어 anchor,
관계어 경계만 일반화해 수정했다. 같은 raw response를 외부 호출 0회로 재검증한
v3는 수용 claim 116·거부 3이며 다음 경로가 정본이다.

- `processed/eval/preflight-20260904/dev45-grounded-claims-c2-v3/c2-run1b-reproject-v3.answers.jsonl`:
  `047c2898001c266c8c2b0248963dc484ff3a4aeefb49abf85dca803961950836`
- 같은 디렉터리 `judge/c2-run1b-reproject-v3-judge-v11-r1.jsonl`:
  `13ebc1767288cc3b4aa066832df8acfd0b5782fe40e6e78f62c1c3460ad1e56a`

v3 run1은 평균 1.4222·GFC 28/45였고 C1 run1보다 평균 +.2222, GFC +10문항이었다.
GFC paired gain/loss는 11/1, exact McNemar p=.00635였지만 이는 verifier 수정에
사용한 같은 DEV raw response의 재투영 결과라 confirmatory 성능으로 보지 않았다.

다음 네 파일만 다시 검증·stage해 commit
`85709b5`(`fix: harden quote-bound claim verification`)로 동결하고
`origin/main`에 push했다.

- `scripts/evaluate_grounded_claims_v2.py`:
  `86bf30dab92e02970475db2566c4982f41e7552e54db45738127c060028e972b`
- `scripts/rag/grounded_claims_v2.py`:
  `65779d9dca5ba11233d23f8082a87f605def143d3b96941ca1d2b118b149aa13`
- `tests/test_evaluate_grounded_claims_v2.py`:
  `cfb92b740a1c6bc6a4129f6fa196f9c05d54dddad76f6139ad412d90f7d3ecc1`
- `tests/test_grounded_claims_v2.py`:
  `fb1aba6f1363f77d52e7b2dd97aa80b9b5ae57f2f769cde0e5fbe7b22914e253`

동결 직전 `git diff --check`, 전체 unittest, `bun run lint`, `bun run build`는
모두 통과했다. unittest는 **779 tests OK, 6 skipped, 실패 0**, Vite build는
1,735 modules였다. frozen 서비스 파일 3개와 holdout 파일은 변경하지 않았다.

### 2026-09-04 C2 v3 동결·독립 n=3 평가

동결 commit `85709b5`에서 C1 run2/run3의 기존 retrieval trace를 입력으로 C2
run2/run3를 각각 단일 스트림으로 생성했다. 명령은
`evaluate_grounded_claims_v2.py`에 model `gemini-3.5-flash-lite`,
`--max-output-tokens 1200 --timeout 180 --retries 3 --sleep 3`과 정확한 승인
문구를 사용했다. run2는 45/45·수용/거부 claim 120/7, run3는
45/45·103/7이었고 terminal error는 없었다.

- `processed/eval/preflight-20260904/dev45-grounded-claims-c2-v3/c2-run2.answers.jsonl`:
  `5f3b4e67b01c36c63ef02a44e89bd5cdaed34bd8e814735a7f88b63fbdcb9e7a`
- 같은 파일의 `.summary.json`:
  `3652d14b99235b209f0bd9c355c75e7e1477a8c63761c5ef0df4462ccfd3979d`
- 같은 디렉터리 `c2-run3.answers.jsonl`:
  `8f13c280e14442d912eb745097cb5e06d3711d28b8126c7e9606b68661c95c85`
- 같은 파일의 `.summary.json`:
  `76189061b1ce03303a75c7fff1c75b1d4f045cf092ccd77bceae477eb9b113ee`

각 answer를 `judge_service_answers.py`의 model `gemini-3.1-flash-lite`,
Judge v11, `--max-output-tokens 1600 --timeout 180 --retries 6 --sleep 3`으로 한
번씩 판정했다. validate-only는 두 run 모두 45/45 eligible, service error 0,
judge config SHA `c165059daa5903b74d9b7fe260856419b697721ecc1e869a701aa0c85dc18fce`로
통과했다. run2는 평균 1.3556·GFC 25/45, run3는 1.3111·GFC 23/45였다.

- `processed/eval/preflight-20260904/dev45-grounded-claims-c2-v3/judge/c2-run2-judge-v11-r1.jsonl`:
  `285dc828d9f85221767364d9c64c570f64de8149274fbad572f3042344dec24f`
- 같은 디렉터리 `c2-run3-judge-v11-r1.jsonl`:
  `0726deed223ab1dd90d95c6830673273d39ad070fee612a7922afa859096a7f5`

v3 n=3에 직접 사용한 외부 호출은 generation 135회 모두 성공(run1b 45,
run2 45, run3 45), Judge 139 HTTP attempts 중 성공 135·일시적 503 4회였다.
개발 중 중단된 최초 generation 3회와 낮은 점수를 확인한 최초 run1b Judge
47 attempts(성공 45·503 2)를 포함하면 C2 전체 작업은 324 HTTP attempts,
성공 응답 318·일시적 실패 6이다. fallback은 없었고 사용량 token metadata가
없는 호출의 비용은 추정하지 않는다.

`analyze_service_ab.py`로 질문별 3-run 평균을 비교하면 C1 1.2148에서 C2 v3
1.3630으로 +.1481이었다. paired 개선/동률/악화는 15/21/9이고 100,000회
family-cluster bootstrap 95% CI는 `[-.0000,+.3116]`으로 0을 포함한다.
검색 trace를 재사용했으므로 두 조건의 Hit@5 .889, MRR .714,
RequiredGoldChunkRecall@5 .693, @8 .715는 정확히 같다.

새 `analyze_generation_gfc_repeats.py`는 세 generation을 135개의 독립 표본으로
세지 않고 각 질문에서 strict 2/3 majority 하나를 만든다. C1 run별 GFC는
18/20/20, C2는 28/25/23이고 majority는 **20/45(.4444) → 26/45(.5778)**,
차이 +6문항·+.1333이다. paired gain/loss 9/3, family-cluster bootstrap
100,000회 95% CI `[.0000,+.2826]`, exact McNemar 양측 p=.145996이므로
개선 방향은 관측됐지만 통계적 확정이나 “대폭 향상”으로 표현하지 않는다.

- `processed/eval/preflight-20260904/dev45-grounded-claims-c2-v3/analysis/c1-vs-c2-v3-3run-ab.json`:
  `db29150f2d7b93ef18085664cd6df7d7ee1b1b9910293b847df78d3b1dd2d21d`
- 같은 디렉터리 `c1-vs-c2-v3-3run-ab.csv`:
  `751fbb3935ce2c683aa342ba5ab300c0d86357cdb5418287430038352c76b7d0`
- 같은 디렉터리 `c1-vs-c2-v3-3run-majority-gfc.json`:
  `aff4456e070c4818f4c5a92c027d9f822701a43871d0a60d851046bd0d685955`
- 같은 디렉터리 `c1-vs-c2-v3-3run-majority-gfc.csv`:
  `651561a2e754436e63287eacd1105402bf7f5fbfa4873e2d691f71cb4051841b`

majority 비GFC 19문항 중 13문항은 필수 근거가 context@8에 모두 없었고,
6문항은 필수 근거가 모두 있었는데도 실패했다. 즉 동결 뒤 남은 병목은 우선 검색
13건, 생성·Judge 6건으로 분리된다. run별 실패 분해 JSON SHA는 run1
`78cd24adbb7732a9432b2d56de0e1c7a9986f8100d14dc44f20f3d92e927b16a`,
run2 `452ce5c3e99eb877c1d539fff255aa71456fb9dfbef852e64ca976fe80f7cfda`,
run3 `e3fba8c705eb49833bffaa92191a035c53f629a591de9f80706ca976feb6640c`다.
코드를 더 튜닝하지 않고 이 항목을 **동결 후 발견**으로 기록한다.

분석기와 단위 테스트 SHA는 각각
`6b41f1d2d97a2ad0799714369cb2b2d602c079900c3c8bcedd09850ba381b1de`,
`03d539aec67a16485ce2fdc4350bec8c71aaaf990fc745f700fc1cc5cdc3636d`이고
표적 테스트 5개가 통과했다. 보고서·런북 SHA는 각각
`55a764564eb252b3194b8ed128b246a8357b18d8ea2b4a876caaae71955d957d`,
`e26cd936e509e8c3fc43ca2f958b8cc758b27d9728b0b7157c68356641871508`다.
최종 `git diff --check`, 전체 unittest, lint, build는 모두 통과했고 unittest는
**784 tests OK, 6 skipped, 실패 0**, build는 1,735 modules였다. holdout 내용
접근·수정과 frozen 서비스 규칙 변경은 모두 0회다.

### 2026-09-04 C1/C2 v3 육안 비교판

기존 `build_service_ab_review.py`가 조건명을 `Baseline`/`검색 튜닝`으로
하드코딩해, 검색 context가 동일한 C1/C2 생성 비교를 검색 개선처럼 보이게 하는
표시 결함을 브라우저 렌더링에서 발견했다. summary의 `labels.a/b`를 검증해 제목·
카드·승패 기준에 사용하고, 신뢰구간 경고도 실제 CI의 0 포함 여부로 표시하도록
일반화했다. 기존 잘못 표시된 HTML은 수정하지 않고 v2 새 경로를 생성했다.

- `processed/eval/preflight-20260904/dev45-grounded-claims-c2-v3/analysis/c1-vs-c2-v3-3run-review-v2.html`:
  `90d5dd8500f1119c2180d185ed0ced1d120dae1fb0e2ad37638a09a41bebe927`
- `scripts/build_service_ab_review.py`:
  `040f1c6319a86aeb20f39bb7eed13189745f934f6dc17055c2cf974b775974d7`
- `tests/test_build_service_ab_review.py`:
  `661f7b983d336b3c52f87233b60ec209b5d2fef85a02f1bdc5a186c4eaf3e93d`

합성 label 테스트 2개가 통과했고, Chrome에서 `C1 → C2-v3`, 평균
`1.215 → 1.363`, 승/무/패 `15/21/9`, CI가 0을 포함한다는 경고, 45/45문항이
표시되는 것을 직접 확인했다. 로컬 `127.0.0.1:8765`에서 비교판을 열어 두었다.
비교판 경로를 부록에 추가한 최종 보고서 SHA는
`9a2ceca5355945f3c1a6d2d7563aa4740771345bb2df007ebf21307254b4ca56`이다.
비교판 코드까지 포함한 최종 gate는 `git diff --check`, 전체 unittest, lint,
build가 모두 통과했고 **786 tests OK, 6 skipped, 실패 0**, Vite 1,735 modules였다.

DEV 결과를 production 채택으로 오해하지 않도록
`docs/c2-grounded-claims-decision-20260904.md`를 작성했다. C2의 변경점, n=3
점수·GFC·검정, 검색 13/생성 6의 잔여 병목, run1 확인 편향, 사람 signoff·
calibration을 포함한 채택 gate를 한 장으로 정리했다. 문서 SHA는
`8f58020a6a7080bb8abc10776d49b360efb9ca03a52231942da96e8267e821c6`다.
이를 부록에 연결한 최종 보고서의 최신 SHA는
`7d82c9abee42e46b21f35300f4ebbc5b0ef9f98ab68d947641e9f98da9be2070`이다.

### 2026-09-04 C0/C1 Judge v11 안정성 반복 권한 상태

C0/C1 run1 각 45개 answer는 validate-only에서 45/45 eligible, terminal service
error 0, Judge config SHA
`c165059daa5903b74d9b7fe260856419b697721ecc1e869a701aa0c85dc18fce`로
통과했고 r2/r3 네 output 경로가 모두 미존재임을 확인했다. 그러나 live C0 r2의
첫 요청 전에 권한 검토가 C2 전송 승인과 C0/C1 payload 승인을 별개로 판정해
중단했다. **외부 전송과 새 judgment 파일은 0건**이며 우회하지 않았다.

재개하려면 기존 DEV45 C0/C1 run1 답변·PNU 검색 context·평가 rubric을 Google
Gemini 3.1 Flash Lite로 전송해 r2/r3 판정 180개를 만드는 작업에 대한 명시적
승인이 필요하다. 승인 뒤에도 모델을 3.5로 섞지 않고
`--max-output-tokens 1600 --timeout 180 --retries 6 --sleep 3` 단일 스트림을
유지한다.

### 2026-09-04 C0/C1 Judge v11 안정성 반복 완료

직전 응답에서 C0/C1 `run1`의 고정 답변·검색 context·rubric을 Gemini 3.1 Flash
Lite로 전송하는 대상, 새 r2/r3 경로, 성공 판정 180회를 명시한 뒤 사용자가
`계속 해줘`로 실행을 승인했다. `judge_service_answers.py`를 조건별·반복별 단일
스트림으로 네 번 실행했으며 설정은 model `gemini-3.1-flash-lite`,
`--max-output-tokens 1600 --timeout 180 --retries 6 --sleep 3`이었다. 모델
fallback이나 3.5 혼용은 없었다. 네 파일 모두 45행·case 45개·오류 행 0으로
완료됐다.

| 새 Judge artifact | 평균 | GFC | HTTP attempts (200/503) | SHA-256 |
|---|---:|---:|---:|---|
| `judge/c0-run1-judge-v11-r2.jsonl` | .8667 | 12/45 | 58 (45/13) | `bd7a4b1844078bbfc5839ffe79c906c1ff958b88be2034790bec9b2e27cec47d` |
| `judge/c0-run1-judge-v11-r3.jsonl` | .8667 | 12/45 | 53 (45/8) | `2be78f774aae60ff30db65c5531c8f67da65d2690c613e62a2fed747ab02ddce` |
| `judge/c1-run1-judge-v11-r2.jsonl` | 1.2000 | 18/45 | 59 (45/14) | `008ab98323c033c794fee22615058d97de28ece5c2eb15e5534214031a167eb6` |
| `judge/c1-run1-judge-v11-r3.jsonl` | 1.2000 | 18/45 | 57 (45/12) | `b3f59e6039b3d077c99123bc51879abf08005ca42e2ffd56679fbe7782bfa98a` |

이번 추가 호출은 HTTP 227 attempts 중 성공 180·일시적 503 47회였다. 기존 r1까지
합치면 352 attempts 중 성공 270·503 82회다. 재시도 뒤 terminal error는 없었다.
모든 입력은 동일 answer SHA(C0 `bf228e5e…fc8`, C1 `1383561b…bd6`)와 Judge
config SHA `c165059d…8fce`에 결속됐다.

`summarize_judge_repeats.py`에 조건별 r1/r2/r3를 입력한 결과 C0와 C1 모두
**점수 45/45·GFC 45/45가 세 반복에서 완전 일치**했다. 반복은 표본 수를 늘리지
않으며 유효 n은 조건별 질문 45개다. 고정 `run1`의 다수결 GFC는 C0
12/45(.2667), C1 18/45(.4000), 차이 +6문항(+.1333)이었다. 질문 단위 paired
bootstrap 10,000회 95% CI는 `[-.0444,+.3111]`, exact paired sign-flip과
McNemar 양측 p는 모두 `.2378845`; both/C0-only/C1-only/neither는
6/6/12/21이다. 즉 Judge 재현성은 높지만 사람 기준 타당도나 C1의 통계적 우월을
입증하지 않는다.

정본 산출물과 SHA-256은 다음과 같다. 모든 경로의 공통 prefix는
`processed/eval/preflight-20260903/dev45-generation-current-v1/judge/stability-20260904/`다.

- `c0-run1-judge-v11-r1-r3.json`: `f5fc357a11c4dcc74e001f8023fdcedd2e33d40491530944e92628c594c56904`
- `c0-run1-judge-v11-r1-r3.csv`: `a1cdd9cc1424d9c16a989a5a99031fcb69d43ccede2cdfde82cf352027aa1e44`
- `c1-run1-judge-v11-r1-r3.json`: `37fb57a519912056d347dfeff466c65d030f23d9bf2e866e72775f5f1b432c13`
- `c1-run1-judge-v11-r1-r3.csv`: `e2044d9af7e208741893b065c594b29c93887593b16a1a6cc418195904b336bb`
- `c0-vs-c1-run1-majority-gfc.json`: `e649da8b3f6a4436e6c4d2b2d7a57734e11f02b904907b4f70ffc9da508c8875`
- `c0-vs-c1-run1-majority-gfc.csv`: `efb0058ed062febd19e51a5ad46f7a4dffba09cb3ef34f888f36bb339dc659e0`

실행한 분석 명령은 조건별
`summarize_judge_repeats.py --answers ... --judgments <r1> <r2> <r3>`와
`analyze_gfc_pairs.py --c0-summary ... --c1-summary ...`다. 분석기는 참조 answer와
judgment의 SHA, record hash, case·condition 결속을 원본에서 다시 검증해 PASS했다.
기존 artifact를 수정하지 않았고 holdout 파일을 읽거나 수정하지 않았다.

보고서의 Judge 안정성 결과·한계·재현 산출물 목록을 갱신했다.
`docs/final-report-20260914.md` SHA-256은
`ce9d0c780b9c1d6c83ffdd2425a7d9558abb727fb176f33f4a9a65597877e57a`다.
최종 gate에서 `git diff --check`, `bun run lint`, `bun run build`가 통과했고
Vite는 1,735 modules를 build했다. 제한 샌드박스의 첫 unittest는 로컬 fixture
socket bind가 금지돼 기존 소켓 테스트 23개가 `PermissionError`였으며, 동일 명령을
소켓 허용 환경에서 재실행해 **786 tests OK, 6 skipped, 실패 0**을 확인했다.

### 2026-09-04 Shadow60 추가 질문지 동결·검색 일반화 진단

사람 holdout 검수는 미루되 기존 DEV45를 다시 튜닝하지 않기 위해, source-disjoint
합성 진단 세트 Shadow60을 만들었다. 구성은 simple 24, multi 18, role variant 8,
challenge 10(unanswerable 4, scope/version ambiguity 3, prompt injection 3)이다.
core 42문항은 42개의 서로 다른 source document를 사용한다. DEV source manifest와
document id·source SHA·연도 제거 제목·canonical URL을 모두 대조했다. 최초 후보의
숨은 DEV source-family 중복 6건을 preflight에서 발견해 **점수 확인 전에** 교체했고,
최종 DEV source overlap은 0이다. 이후 질문과 gold는 SHA
`0906e6c8de40040915e668e6cf61639306657e6f94f41e5bcdf62197a6d90754`로
고정했으며 결과를 보고 수정하지 않았다.

생성·검증 명령은 `build_shadow_testset.py`, `validate_shadow_testset.py --report
evidence/20260914/shadow60-v1-preflight.json`이었고, C0/C1 local service에 대해
`collect_service_answers.py`를 `provider=extractive`, `context-k=8`,
`eval-trace`로 각 60문항 실행했다. C0는 port 18810의
`--no-retrieval-tuning`, C1은 port 18811의 현재 tuning이며 외부 API 호출은 0회다.
두 조건 모두 60/60 성공·error 0이다. `analyze_shadow_retrieval.py
--bootstrap 10000 --seed 20260904`로 paired family-cluster bootstrap과 exact
McNemar를 계산했다.

| core 42 | C0 | C1 | 차이 |
|---|---:|---:|---:|
| source hit@5 | 40/42 | 38/42 | -2 |
| source hit@8 | 41/42 | 39/42 | -2 |
| all required evidence@5 | 21/42 | 20/42 | -1 |
| all required evidence@8 | 22/42 | 21/42 | -1 |
| evidence recall@8 | .6071 | .5833 | -.0238 |

source hit@8 차이의 cluster bootstrap 95% CI는 `[-.1190, 0]`, McNemar
p=`.5`; all evidence@8은 CI `[-.1190,+.0476]`, p=`1.0`이다. role variant
8문항은 source hit@8 8/8, all evidence@8 7/8로 두 조건이 같았다. 전체 answerable
53문항의 all evidence@8은 C0 29/53에서 C1 28/53으로, C1 1건 개선·2건
악화·50건 동일이었다. 따라서 이 진단 세트에서는 C1의 양의 일반화 효과가
관측되지 않았고, 차이도 통계적으로 확정되지 않았다.

**동결 후 발견:** 악화 2건은 모두 연도 불일치 후보 강등의 false positive였다.
`shadow_sch_01`은 HTML 식별자 `203839`를 연도 `2038`로 오인해 정답 chunk의
rank가 1→71로 밀렸다. `shadow_grad_02`는 파일명/title이 2025지만 실제 표의
일정이 2026이라 정답 chunk가 rank 2→52로 밀렸다. 개선 1건
`shadow_sup_02`는 facet sibling completion이 일정 본문 chunk를 보충한 경우다.
동결 원칙에 따라 production 검색 코드는 고치지 않았다.

또한 source hit@8이 97.62%인데 strict all evidence@8이 52.38%인 C0 결과는
남은 병목이 source 발견보다 문서 내부 chunk 선택·context 완성에 가깝다는 것을
보인다. 단, atomic gold가 선택한 하나의 full chunk를 엄격히 맞추므로 같은 source의
다른 충분한 chunk도 실패할 수 있다는 측정 한계가 있다. 점수 확인 뒤 gold를
완화하지 않았다.

정본 SHA-256은 다음과 같다.

- blueprint: `9e0298f6e60fcef1fdca55e37d870cdb5cd7e9cd1a2591750dfa7879d05adbe7`
- frozen cases: `0906e6c8de40040915e668e6cf61639306657e6f94f41e5bcdf62197a6d90754`
- builder: `dcd9efac85b9e9b60034524664b5925bdf252d813bb2c6914537502f64d3f57c`
- validator: `a1b18b4d62d760ca16d8dc4326c48c745594061b2f0a0b544b1a55c7e639b28b`
- analyzer: `0415abcf3d3a0d5e0e3f6120df1dd4a8a23b4a6d94c7d2685f0c79799552d34e`
- tests: `test_shadow_testset.py`
  `d88b313baa31e9e9d2836e982e22571956365d6e948193bc3c566ebdfe5c1730`,
  `test_analyze_shadow_retrieval.py`
  `f7bb60e36e6a8041510ccd4134ee24f704ec5c865c3052cc29acca93322fdbdb`
- preflight report: `a16116b42ab14bb93a36be0971e2f44eba378b3fe05fcdcf5a0618e96fa4f93a`
- C0 answers: `6fd0a8d7f594ccf0f1bf0293ca1991885d13f58c13dc953f48535a9fe8cd1b97`
- C1 answers: `164980441dbb90244517db1a6e55086c0dedc6ca1f630014a8841d387795177d`
- analysis JSON: `dda7cbef89ba7e2f51ae01ee49810385af8a157a851db8bd69743d4eaee907b5`
- analysis CSV: `0619a6acdf040d62a384ee9340434424b07e9b3c635175a5661004b6c82e23c4`
- 설명 문서: `docs/shadow60-evaluation-20260904.md`,
  `e4db98d9e585a65445a52e79bc8c0feba3a964965e1ab128c11f871fd20c129a`

`git diff --check`, 전체 unittest, `bun run lint`, `bun run build`가 모두
통과했다. unittest는 **793 tests OK, 6 skipped, 실패 0**, Vite build는 1,735
modules였고 표적 Shadow 테스트는 7개 모두 통과했다. frozen production service
파일 3개와 금지된 holdout 파일은 읽거나 수정하지 않았다.

### 2026-09-04 Shadow60 Gemini 3.5 생성 가용성 실패

사용자에게 전송 대상과 내용을 명시해 “Shadow60 60개 질문과 검색된 부산대학교
문서 context를 Google Gemini API `gemini-3.5-flash-lite`로 전송, C1 3회·최대
180개 생성” 승인을 요청했고, 사용자가 `해줘`로 승인했다.

첫 port 18811 수집은 첫 문항에서 1회 extractive fallback, 1회
`gemini-3.1-flash-lite` fallback이 발생했다. collector가 각각 provider/model
control mismatch로 거부했다. 모델 혼용 가능성을 없애기 위해 production 코드는
건드리지 않고 환경 설정상 허용 모델을 3.5 하나로 고정한 port 18812 서버를 새로
띄웠다. 같은 frozen 생성 한도(24,000 context chars, 900 output tokens, sampling
parameter 없음)와 C1 검색 설정에서 동일 문항을 두 번 재개했으나 두 번 모두
extractive fallback으로 거부됐다.

마지막 진단 요청의 원본 응답에서 `generation.requested=frontier`,
`used=extractive`, `fallback_reason=gemini:timeout`, 3.5 attempt elapsed 30,192ms를
확인했다. 진단 응답도 평가 answer로 채택하지 않았다. collector service request는
4회였고 유효 answer record는 **0개**다. 기존/신규 답변을 덮어쓰지 않았고
3.1·extractive 결과를 3.5 평가에 섞지 않았다.

- `c1-run1.answers.errors.jsonl`: 2행,
  SHA `df00b4c885ea33da3e7b1d5ba88ed7db25fc9c2c07dcabf35d9739d279f639b7`
- `c1-run1b.answers.errors.jsonl`: 2행,
  SHA `4e6eca27d7011cb23baddee937a301e1f05811f82f85a7f54247c8db1767f956`
- compact evidence `evidence/20260914/shadow60-gemini35-availability-20260904.json`:
  SHA `2eeed566f3b38c26c32420316ef1020a78958332c346168e57dd989993a0d3db`

이는 평가 대상 성능 실패가 아니라 현 시점 3.5 호출 가용성 실패다. 3.1로 바꾸면
DEV45의 3.5 생성 결과와 직접 비교할 수 없으므로 자동 전환하지 않고 승인 대기로
남긴다. frozen service 코드와 Shadow60 질문/gold는 수정하지 않았다.

## 다음 우선순위

1. 현재 cases/packet SHA에 대해 사람 2인이 읽기 전용
   `docs/holdout-v2-human-review.md`를 보고 각자
   `evidence/holdout-v2-reviewer-{a,b}.json`에 독립 판정한 뒤 strict merge로
   signoff manifest를 만든다.
2. signoff 뒤 holdout을 최종명과 SHA로 동결하고 frozen schedule대로 final
   retrieval·generation·oracle 진단을 one-shot 실행한다.
3. 동결 답변으로 condition-blind 패킷을 생성해 사람 2인 평가·합의 판정을 받고,
   Judge v11의 same-model self-preference와 사람 기준 타당도를 calibration한다.
4. 준비된 분석기로 질문 단위 paired bootstrap CI, sign-flip과 exact McNemar를
   산출하고 최종 보고서 headline은 이 holdout 및 사람 검증 결과로만 갱신한다.
5. 강한 파서 품질 주장이 필요하면 54개 anchor의 독립 2인 원문 화면 육안검수를
   별도 signoff로 남긴다.

## 주의사항

- 현재 worktree는 작업 전부터 dirty였고 기존 변경이 섞여 있다. 커밋, reset, checkout, 대량 정리를 임의로 하지 않는다.
- `processed/eval/preflight-20260901/generation-p0*`는 실패·개선 이력을 보존하는 artifact다. 덮어쓰거나 삭제하지 않는다.
- `p0f` 이후 현재 코드가 변경됐으므로 p0f 답변을 현재 최종 결과라고 부르지 않는다.
- 검색 DEV와 생성 3문항 preflight를 holdout 최종 성능으로 표현하지 않는다.
- CI가 0을 포함하면 “점 추정치는 향상됐으나 통계적 근거는 충분하지 않다”고 쓴다.
- `p0g` v8 결과는 보존하되 n=3 DEV smoke이고 사람 calibration 전이므로 생성 성능
  headline, CI 또는 일반화 주장에 사용하지 않는다. v6/v7 artifact는 실패·보정
  이력으로만 보존한다.
