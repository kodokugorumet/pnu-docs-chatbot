# 2026-09-14 최종 보고서 근거 대장

작성 시작: 2026-08-31
제출 마감: 2026-09-14
권장 코드 동결일: **2026-09-04** · 수치·artifact 동결일: **2026-09-09**
보고서 정본: `docs/final-report-20260914.md`

이 문서는 최종 보고서의 모든 수치와 주장을 원천 산출물에 연결하기 위한
단일 근거 대장이다. `확정`으로 표시되지 않은 수치는 초록·결론·발표자료의
최종 주장으로 사용하지 않는다.

## 1. 상태 규칙

| 상태 | 의미 | 보고서 사용 규칙 |
|---|---|---|
| 확정 | 동일 설정 재현과 산출물 확인 완료 | 본문·초록·결론에 사용 가능 |
| 조건부 | 원천은 있으나 최신성·비교 공정성·단일 정본 등 보완 필요 | 범위와 한계를 붙여 본문에만 사용 |
| 진행 중 | 평가가 완주되지 않았거나 반복 수가 부족함 | 중간 진행표에만 사용 |
| 보류 | 구현됐지만 제출 핵심성과로 쓰기 어려움 | 향후 과제 또는 부록으로 이동 |
| 미착수 | 제출 전 반드시 생성해야 하는 산출물 | 수치 인용 금지 |

## 2. 핵심 주장과 증거

| ID | 보고서 주장 | 현재 수치 | 상태 | 원천 산출물 | 남은 확인 |
|---|---|---:|---|---|---|
| C01 | 부산대 공식 웹 코퍼스를 구축했다 | 크롤 완료 10,977행 | 확정 | `downloads/pnu-web-crawl/summary.json` | 구형 전체 수집과 최종 exact-host 정책 차이 서술 |
| C02 | 관련 문서를 선별·중복 제거했다 | 2,260문서, 473,678,602 bytes, 중복 118행 | 확정 | `processed/curation/20260725-pnu-curated-v5/summary.json` | 최종 manifest SHA-256을 부록에 고정 |
| C03 | cascade 파서로 다형식 문서를 구조화했다 | 2,247문서 파싱, 44,591청크, 표 7,813개 | 확정 | `processed/runs/20260725-pnu-curated-cascade-v5/cascade/parse_summary.json` | 인덱스 44,520청크와의 71개 차이를 gate 제거로 설명 |
| C04 | 초기 24문항 검색 비교와 운영 조건을 근거로 cascade를 서비스 profile로 채택했다 | Baseline 23/24, Challenger 24/24, Cascade 24/24 | 확정 | `processed/eval/20260728-pnu-parser-profile-search-comparison.csv` | 24문항 문서 retrieval 운영 결정이며 파서 추출 품질이나 full-corpus 우월성 근거가 아님 |
| C05 | 초기 데모 검색이 24문항을 통과했다 | Hit@5 24/24, MRR 0.8486 | 확정 | `processed/eval/20260728-pnu-cascade-demo-ranking.json` | “최종 답변 정확도”가 아님을 본문에 한정 |
| C06 | 연구비 트랙에서 단계별 검색·생성 개선을 검증했다 | context recall 0.4522→0.5266, 생성 0.81→1.15 | 조건부 | `docs/report-20260806-retrieval-optimization.md`, grant worklog D24–D33 | metric 정의·모델·날짜를 본문 표에 함께 기록 |
| C07 | 확보 가능한 50문항에서 경쟁 서비스보다 높았다 | 우리 1.607, 상대 1.440 | 조건부 | `processed/eval/20260811-gen-anchor14-run{1,2,3}.jsonl`, `processed/eval/20260806-rival-judged.jsonl` | 상대 재수집, 양방향 blind A/B, 단일 정본 결과 생성 |
| C08 | 역할 기반 답변 요구를 구현했다 | 프리셋·자유입력·안전 매핑 구현 | 확정 | commit `0a45cd5`, `scripts/rag/role_router.py`, `tests/test_role_router.py` | “검색 품질 개선”이 아니라 요구 구현·무해성으로 표현 |
| C09 | 서비스 평가셋과 기준선을 구축했다 | 45문항, Hit@5 29/45=0.644, MRR 0.494 | 확정 | `config/pnu-service-answer-eval.jsonl`, `processed/eval/20260831-svc-retrieval-smoke.jsonl` | 평가셋 구축 절차·개발셋 한계 서술 |
| C10 | 서비스 기준선의 gold chunk coverage를 측정했다 | Any@5 21/45=0.467, Any@8 23/45=0.511, Recall@5 0.315, Recall@8 0.359 | 확정 | `evidence/20260914/service-ab-summary-metric-corrected-20260901.json` | DEV의 flat chunk gold이며 atomic evidence 지표가 아님 |
| C11 | 서비스 기준선 생성 품질의 historical snapshot을 반복 측정했다 | run별 0.733/0.667/0.756, 문항 평균 0.7185 | 조건부 | `evidence/20260914/20260831-svc-gen-baseline-run{1,2,3}.jsonl`, `evidence/20260914/interim-service-n3-manifest.json` | raw 집계는 보존됐지만 exact producing-code snapshot은 없음; 0–2점 judge 평균이며 정답률이 아님 |
| C12 | 대화형 접미어·연도·상용구 문서 문제를 완화했다 | Hit@5 40/45=0.889, MRR 0.723, Any-Gold-Chunk@5 27/45=0.600, @8 28/45=0.622, 신규 문서 적중 11·상실 0 | 확정 | `evidence/20260914/service-ab-summary-metric-corrected-20260901.json` | 45문항 개발셋 검색 성과로 범위 한정 |
| C13 | 검색 튜닝 뒤 end-to-end 생성 평균이 소폭 상승했다 | 0.7185→0.7481, +0.0296, 10승·29무·6패; 0점 61→52이나 2점 23→18 | 조건부 | `evidence/20260914/service-ab-summary-metric-corrected-20260901.json`, `docs/dev45-generation-root-cause-20260902.md` | 실제 gold 신규 유입군 +11점이 기존 gold 보유군 -11점에 상쇄됨. cluster bootstrap 95% CI [-0.1407, 0.1951]로 0 포함. 확정적 생성 개선 주장 금지 |
| C14 | 2층 폴백은 기존 벤치마크를 해치지 않는다 | 53/53 결과 불변 | 보류 | `grant-fallback-layer-20260811.sqlite`, worklog D50 | 실서비스 통합은 일정 지연 시 컷 |
| C15 | 최종 동결 전 전체 회귀를 통과한다 | 2026-09-02 `bun run check`: 656 tests OK(6 skip), ESLint·TypeScript·Vite build PASS | 진행 중 | `docs/progress-log-20260901.md` | dirty 작업 snapshot 결과이며 clean final commit에서 전체 check와 데모 smoke 재실행 필요 |
| C16 | 날짜·짧은 사실 후처리 손실을 수정했다 | 공식 기준답안 offline A/B: critical-value 보존 0.7426→1.0000, 개선 20·동일 25·악화 0 | 조건부 | `evidence/20260914/postprocessor-offline-ab.json` | splitter 단위 안전성만 증명. 외부 API 승인을 받은 실제 생성 n=3 필요 |
| C17 | 문서당 청크 cap=4 탐색 결과를 검토했다 | Hit/MRR 불변, GoldChunkRecall@8 0.456→0.500, unique docs 6.13→5.04 | 보류·미채택 | `evidence/20260914/cap2-vs-cap4-retrieval-dev45.json` | 단일 latency run이고 DEV45 중 31문항 context가 바뀌어 영향 범위가 큼. 최종 서비스는 cap=2 유지 |
| C18 | claim별 citation과 고위험 관계 반전 veto를 구현했다 | 합성 adversarial 77건 joint 0.4156→1.0000, false positive 43→0 | 조건부 | `evidence/20260914/grounding-guard-ablation.json`, `scripts/search_api.py` | permission·comparator·direction·temporal·modality 표적 component microbenchmark이며 실제 생성·일반화 성능이 아님 |
| C19 | 서비스 기준선 miss의 후보 단계 원인을 추적했다 | miss 16건 중 15건은 후보 pool에 존재, 1건은 FTS 후보에도 없음 | 조건부 | `docs/worklog-20260831-service-track.md`의 “S7. 미적중 16문항 전수 진단” | 별도 기계판독 JSON 진단 산출물은 아직 없으므로 worklog 근거로만 사용 |
| C20 | 표적 18문서에서 세 파서의 원자 근거 보존을 기계 진단했다 | Baseline 48/54·3/3 문서 14/18, Challenger 54/54·18/18, Cascade 54/54·18/18 | 확정 | `processed/eval/20260901-parser-audit-v1.json`, `processed/eval/20260901-parser-audit-v1.csv`, `docs/parser-audit-20260914.md` | 기계적 source-bound 표적 진단에 한정. full-corpus 우월·RAG 향상 주장은 금지하며 54-anchor 독립 2인 원문 육안검수는 별도 필요 |
| C21 | holdout v2 AI 검토 수정과 기계 gate를 완료했다 | SHA `2fbedad63531491d3c069a5de6fe51fff623f7e571b9170c6a9aa7452e2d51f9`, 36문항·93 evidence option, schema·DEV·corpus pre-review PASS | 진행 중 | `config/pnu-service-answer-holdout-v2.draft.jsonl`, `docs/holdout-v2-human-review.md` | A/B 응답은 모두 PENDING. 실제 사람 2인 독립 검토와 signoff 필요 |
| C22 | 현재 코드의 DEV 3문항 C0/C1 `/chat` preflight artifact를 수집했다 | C0 hit 2/3·MRR .250·Any@5/8 2/3·All@5/8 0/3·recall .333; C1 hit 3/3·MRR .778·Any/All@5/8 3/3·recall 1.000; 양쪽 generation error 0 | 진행 중 | `processed/eval/preflight-20260901/generation-p0g/dev-smoke-c0-run1.answers.jsonl`, `processed/eval/preflight-20260901/generation-p0g/dev-smoke-c1-run1.answers.jsonl` | 3개 DEV 진단을 holdout·일반화 headline으로 사용 금지 |
| C23 | 고정 `p0g` 답변을 Judge v8로 3회 반복 판정했다 | C0 `[2,1,0]`, 평균 1.000, GFC 1/3; C1 `[2,1,2]`, 평균 1.667, GFC 2/3; 전 문항 양 조건 3/3 일치 | 진행 중 | `processed/eval/preflight-20260901/judge-p0g-v8/c0-summary.json`, `processed/eval/preflight-20260901/judge-p0g-v8/c1-summary.json` | n=3 DEV smoke라 CI·headline 금지. same-model self-preference와 사람 calibration 미완 |
| C24 | 반복 Judge GFC를 질문 단위 paired 통계로 분석한다 | C0 .333, C1 .667, Δ +.333; bootstrap 95% CI [0,1], sign-flip p=1, McNemar p=1 | 조건부 | `processed/eval/preflight-20260901/judge-p0g-v8/gfc-paired-analysis.json`, `processed/eval/preflight-20260901/judge-p0g-v8/gfc-paired-cases.csv` | n=3 DEV 방향성 진단이며 유의한 개선이나 일반화 근거가 아님 |
| C25 | condition-blind 사람 평가와 Judge-human calibration 경로를 준비했다 | 자동 판정·조건·모델 누출 없는 packet/mapping/독립 A·B/합의 label 생성, agreement·kappa·macro-F1·false-pass gate 구현 | 진행 중 | `scripts/build_answer_review_packet.py`, `scripts/analyze_judge_human_calibration.py` | 실제 final 답변과 사람 라벨·합의는 아직 없음 |
| C26 | DEV45에서 parser BM25와 Cascade Dense/Hybrid 8조건을 같은 서비스 경로로 비교했다 | 8×45=360, 오류·외부 LLM 0; C1 Hit@5 .889/MRR .717/All-Gold@8 .378, D-S Any-Gold@8 .644이나 Hit@5 .800/All .356 | 조건부 | `processed/eval/dev45-matrix-20260902/matrix-summary.json`, `docs/dev45-retrieval-matrix-20260902.md` | DEV exploratory이며 holdout·생성·일반화 성능이 아님. parser lane chunk gold 비교 금지 |
| C27 | 검색 실패와 생성 실패를 분리하는 oracle-context 수집 경로를 준비했다 | Core 9영역 `multi_evidence` 1개씩, draft 기준 9문항·39 claims·38 dedup contexts | 진행 중 | `scripts/evaluate_oracle_context_answers.py`, `tests/test_evaluate_oracle_context_answers.py` | 실제 9개는 final holdout 2인 signoff와 동결 뒤에만 생성; 서비스 성능 집계에서 제외 |
| C28 | final 검색·생성 호출을 사전 고정 schedule로만 실행한다 | 검색 Core 27×4=108, 직접 생성 Core 162+Challenge 27=189; case-major/AB-BA, exact WAL, 배타 run-lock, local/server Git·index·source와 packet·A/B 원본 pin | 진행 중 | `scripts/run_final_retrieval_schedule.py`, `scripts/run_final_generation_schedule.py`, 관련 tests | 실행 도구만 합성 검증. 2인 signoff·clean freeze 전 실제 holdout/API 호출 금지 |
| C29 | final retrieval headline 분석 경로를 검증했다 | 합성 108 answer+216 WAL에서 audit complete=108, four-lane 분석·immutable completion manifest PASS | 진행 중 | `scripts/analyze_holdout_retrieval.py`, `tests/test_analyze_holdout_retrieval.py` | 합성 fixture 결과이며 실제 holdout 수치가 아님 |
| C30 | 서로 다른 생성 3회의 질문 단위 GFC와 Judge calibration fallback을 구현했다 | 유효 단위 n=27, family bootstrap 10k·sign-flip·2/3 McNemar; terminal GFC=0; fixed 18-answer Judge repeat | 진행 중 | `scripts/analyze_final_generation_gfc.py`, `scripts/final_generation_slots.py`, `scripts/build_judge_repeat_selection.py`, 관련 tests | 합성 fixture만 검증. 실제 answer/Judge/사람 label은 미생성 |
| C31 | 사람 holdout sign-off는 독립 원본에 추적 가능하다 | A/B 별도 36-record 응답, one-snapshot strict cases parse, cases/packet/response SHA binding, alias·symlink 차단, strict merge와 원본 재검증; 관련 41 tests PASS | 구현 완료·사람 판정 대기 | `scripts/build_holdout_signoff.py`, `scripts/validate_service_holdout.py`, `evidence/holdout-v2-reviewer-a.json`, `evidence/holdout-v2-reviewer-b.json` | 자동 true 없음; 실제 두 reviewer가 작성하기 전 signoff 생성 불가 |
| C32 | 같은 저장 초안에서 후처리의 문장 유지·삭제 효과를 분리 측정한다 | DEV 등록 3문항 중 normalized 내용 변화 1/3; raw 12문장→final 11문장, 유지 11·삭제 1·추가 0, critical value 손실 0 | 조건부 | `processed/eval/preflight-20260902/postprocessor-ab/dev-smoke-c1-postprocessor-pairs-v2.analysis.json`, `scripts/project_raw_draft_answers.py`, `scripts/analyze_postprocessor_pairs.py` | 동일 초안·context의 결정론적 진단일 뿐 품질 점수가 아님. 별도 Judge 전후 평가는 외부 데이터 전송 승인 뒤 수행 |
| C33 | 전역 context 확대 없이 명시적 multi-facet 질의의 누락 근거를 보강했다 | DEV45 C1 All-Gold@8 17/45→18/45, recall .500→.511; Hit@5·MRR·Any@8 불변, context 변경 2/45 | 조건부 | `processed/eval/dev45-c1-multifacet-20260902-v2/matrix-summary.json`, `processed/eval/dev45-c1-multifacet-20260902-v2/old-c1-vs-multifacet.json` | DEV 튜닝 결과이며 일반화·생성 향상 근거가 아님. `svc_sch_05` legacy gold에는 질문 비필수 근거가 있어 exact recall은 그대로임 |
| C34 | 질문에 필수인 근거만으로 DEV45를 재검증하고 원거리·입금처·대상학과·제출기한·예산표 facet 검색을 표적 보강했다 | 9월 2일 C1→현재: Hit@5 40/45 유지, MRR .7026→.7137, All-required-gold@8 21/45→31/45, All-required-evidence@8 21/45→33/45; evidence 개선 12·손실 0 | 조건부 | `processed/eval/preflight-20260903/retrieval-budget-breakdown-v5/dev45-matrix-vs-current.json`, `processed/eval/preflight-20260903/retrieval-budget-breakdown-v5/c1.answers.jsonl`, `docs/progress-log-20260901.md` §12 | current DEV45로 양쪽을 재계산한 누적 DEV 튜닝 결과다. 저장 source 변경은 15/45이며 holdout·생성 향상 주장이 아님. latency는 통제 반복 전까지 인용 금지 |
| C35 | 예산표 검색 이득이 실제 추출형 답변 수치로 이어지도록 표 열 해석과 표 단위 critical-value 귀속을 보강했다 | `svc_core_02` 및 역할 변형 2/2가 합계와 3개 영역 예산·구성비를 직접 출력하고 두 claim 모두 정답 chunk 귀속 PASS | 조건부 | `processed/eval/preflight-20260903/retrieval-budget-breakdown-v4/target.answers.jsonl`, `processed/eval/preflight-20260903/retrieval-budget-breakdown-v5/c1.answers.jsonl`, `tests/test_api_hardening.py` | 2문항 결정론적 표적 검증이며 전체 생성 GFC·LLM Judge 개선 주장이 아님 |
| C36 | 현재 코드에서 검색 개선의 생성 전달 효과를 DEV45 C0/C1 독립 생성 3회로 재측정했다 | Judge 평균 .8815→1.2148, Δ+.3333, cluster bootstrap 95% CI [+.0889,+.5887], 개선/동률/악화 19/15/11; 2/3 majority GFC 13/45→20/45, Δ+.1556, CI [-.0217,+.3201], McNemar p=.1185 | 조건부 | `processed/eval/preflight-20260903/dev45-generation-current-v1/{c0,c1}-run{1,2,3}.answers.jsonl`, `processed/eval/preflight-20260903/dev45-generation-current-v1/judge/{c0,c1}-run{1,2,3}-judge-v11-r1.jsonl`, `processed/eval/preflight-20260903/dev45-generation-current-v1/judge/dev45-3run-service-ab-v11.{json,csv}`, `processed/eval/preflight-20260903/dev45-generation-current-v1/judge/dev45-3run-service-ab-v11-review.html` | 생성·Judge 각 270/270 오류 0이나 반복 튜닝한 DEV이며 same-family Judge, 사람 calibration·고정 답변 Judge 반복·holdout이 없음. 평균 점수 개선만 해당 조건에서 지지되며 완전정답률·일반화 개선 확정 금지 |
| C37 | 등록금 복합 관계문의 deterministic 후처리 false negative를 수정하고 저장 초안으로 효과를 분리 검증했다 | 답변 변경 9/270, 비변경 Judge input SHA 261/261 동일, 변경 9개 Judge v11 점수 모두 2; score-only 진단 C0 .9037, C1 1.2963, Δ+.3926, cluster bootstrap 95% CI [+.1429,+.6519] | 조건부·진단 | `processed/eval/preflight-20260903/dev45-generation-current-v1/postprocessor-direction-fix-v1/README.md`, 같은 디렉터리의 6개 projection과 `judge/` 6개 partial judgment, `scripts/search_api.py`, `tests/test_api_hardening.py` | 같은 저장 초안·context의 postprocessor projection이며 service/GFC/citation 성능 주장에 부적격. fresh frozen E2E와 holdout으로 최종 확인 필요 |
| C38 | 표 행 날짜·시각/분야별 최대액과 인접한 날짜-UI 경로의 후처리 false negative를 좁게 수정했다 | direction-fix-v1 대비 C1 답변 변경 6/135, 비변경 129/135; 승인 범위 Judge v11 6/6 오류 0, 4개 `1→2`·2개 `1→1`; 누적 score-only C0 .9037, C1 1.3259, Δ+.4222, W/T/L 20/16/9, cluster bootstrap 95% CI [+.1812,+.6742]; 전체 709 tests OK(6 skip) | 조건부·진단 | `processed/eval/preflight-20260903/dev45-generation-current-v1/postprocessor-grounding-fix-v4/README.md`, 같은 디렉터리의 `score-only-diagnostic-v4.json`, projection 3개와 `judge/` partial judgment 3개, `scripts/search_api.py`, `tests/test_api_hardening.py` | 같은 저장 초안·context의 postprocessor projection이며 service/GFC/citation/holdout/일반화 성능 주장에 부적격. 1점 잔여 2개는 생성 누락이며 fresh generation 필요 |
| C39 | 단일 숫자 질문도 강한 동일 문서 seed에서 질문 대상이 같은 금액 행으로 보완했다 | 저장 C1 DEV45 후보 재생에서 변경 2/45, required-evidence micro 47/71(.662)→49/71(.690), 개선 2·감소 0 | 조건부·검색 진단 | `evidence/20260914/single-numeric-facet-dev45-replay.json`, `scripts/search_api.py`, `tests/test_api_hardening.py` | DEV 저장 candidate pool 재선택이며 fresh retrieval·generation·holdout·일반화 성능이 아님 |
| C40 | 생성 누락 3유형의 질문별 점검표를 보강하고 무관한 절차 항목 강제를 제거했다 | 대체강좌 이수기준·직무체험 방식·등록금 고지서/예외 prompt 회귀와 무관 항목 음성 회귀 PASS; generator tests 20/20 | 구현·생성 검증 대기 | `scripts/rag/generators.py`, `tests/test_rag_generators.py`, `docs/progress-log-20260901.md` | 실제 모델 재생성 전이므로 답변 점수 또는 GFC 개선 주장 금지 |
| C41 | 단일 숫자 검색과 질문별 생성 prompt를 알려진 실패 5문항에서 실제 서비스로 표적 검증했다 | 생성 5/5·Judge v11 5/5 오류/fallback 0; 이전 세 C1 run의 선택 문항 평균 .600·GFC 0/5 대비 fresh n=1 평균 1.600·GFC 3/5; 교연비 기본·역할 `0→2`, 등록금 `1→2` | 조건부·표적 DEV | `processed/eval/preflight-20260904/targeted-generation-prompt-retrieval-v1/README.md`, 같은 디렉터리의 answer와 `judge/` judgment | 알려진 실패군을 의도적으로 선택한 n=5×1이므로 전체 DEV·holdout·일반화 성능 추정 불가 |
| C42 | 대체강좌·직무체험의 완전한 raw draft를 삭제한 semantic guard false negative를 수정했다 | 신규 positive 5/5 수정 전 FAIL→수정 후 PASS; wrong threshold·활동 불가·제외대상 negative 거부; 저장 5답변 replay에서 2개만 복원·3개 동일; 변경 2개 Judge v11 모두 2점·GFC true·citation full·오류/fallback 0; 5건 진단값 1.600·3/5→2.000·5/5; API 169/169, 전체 714 OK(6 skip) | 조건부·진단 | `processed/eval/preflight-20260904/targeted-generation-prompt-retrieval-v1/postprocessor-semantic-fix-v2/c1-run1.current-postprocessed.jsonl`, 같은 디렉터리의 `judge/c1-run1-judge-v11-semantic-fix-v2-r1.jsonl`, 상위 `README.md`, `scripts/search_api.py`, `tests/test_api_hardening.py` | 같은 draft/context와 알려진 실패군의 diagnostic projection이므로 fresh E2E·전체 DEV·holdout·일반화 성능 주장 금지 |
| C43 | 구어체 서비스 질문을 공식 문서 어휘로 좁게 확장하고 분리된 필수 근거를 동일 문서에서 완결했다 | 확장 전→v5: Hit@5 40/45→45/45, MRR .7137→.8137, Any required evidence@8 34/45→45/45, All required evidence@8 33/45→40/45, mean recall .7481→.9407; 지표별 loss 0; 전체 720 OK(6 skip) | 조건부·DEV 검색 | `processed/eval/preflight-20260904/dev45-retrieval-query-expansion-v5/README.md`, `vs-pre-expansion.{json,csv,html}`, `c1.answers.jsonl`, `scripts/search_api.py`, `scripts/bm25_search.py`, `scripts/rag/generators.py` | 반복 열람한 DEV45의 의도별 튜닝이며 외부 LLM은 호출하지 않음. fresh targeted generation·Judge와 동결 holdout 전까지 생성·일반화 headline 금지 |
| C44 | 의도별 검색 보강이 실제 답변까지 전달되는지 알려진 실패 9문항에서 fresh 생성·Judge로 확인했다 | 생성·Judge 각 9/9, 오류·retry·fallback 0; 이전 동일 9문항 C1 세 run 평균 .5556/.4444/.5556·GFC 1/0/1건 대비 fresh n=1 평균 1.5556·GFC 6/9 | 조건부·표적 DEV | `processed/eval/preflight-20260904/targeted-generation-intent-retrieval-v5/README.md`, `c1-run1.answers.jsonl`, `judge/c1-run1-judge-v11-r1.jsonl` | 알려진 실패군을 의도적으로 고른 n=9×1이다. 전체 DEV·holdout·일반화 성능 추정이나 대폭 향상 headline에 사용 금지 |
| C45 | 교환 일정 semantic guard와 D-2 동일문서 3-facet 병목을 원인별로 수정했다 | 교환 일정 저장초안 replay에서 1/9만 복원·8/9 동일(재Judge 없음); D-2 기본·역할 exact required evidence 1/3→3/3, DEV45 v5→v6 All-required-evidence@8 40/45→42/45, mean .9407→.9704, tracked loss 0; 전체 723 OK(6 skip) | 조건부·검색/후처리 진단 | `processed/eval/preflight-20260904/targeted-generation-intent-retrieval-v5/README.md`, `processed/eval/preflight-20260904/dev45-retrieval-query-expansion-v6/README.md`, `scripts/search_api.py`, `scripts/rag/role_router.py` | 수정 뒤 fresh LLM 생성·Judge를 수행하지 않았다. D-2 extractive 답변과 same-draft projection을 생성 성능으로 해석 금지; holdout 미접촉 |
| C46 | 자부담금 지원의 지원범위·선납부 절차·제출서류/구글폼 3-part 문맥을 좁게 완결했다 | `svc_sup_02` exact gold 1/2→2/2; DEV45 v6→v7 All-required-evidence@8 42/45→43/45, mean .9704→.9778; 변경·개선 1건, tracked loss 0; 전체 725 OK(6 skip) | 조건부·DEV 검색 | `processed/eval/preflight-20260904/dev45-retrieval-query-expansion-v7/README.md`, `vs-v6.{json,csv,html}`, `c1.answers.jsonl`, `scripts/search_api.py`, `scripts/rag/generators.py` | 수정 뒤 fresh 생성·Judge 없음. 남은 2건은 같은 교연비 family의 배경문 annotation이고 실제 최대액 표와 기존 GFC는 확보되어 숫자 맞추기 튜닝을 중단함 |
| C47 | 검색·보조기기 절차 수정 뒤 알려진 실패 4문항을 fresh 생성·Judge로 재검증했다 | 생성 4/4·Judge 4/4 단일 시도, 오류·retry·fallback 0; retrieval hit/MRR/All-Gold@5·@8 모두 1.000; Judge 평균 1.500·GFC 2/4로 직전 4문항과 동일, `svc_sup_02` 1→2이나 D-2 역할 2→1이 상쇄 | 조건부·표적 DEV | `processed/eval/preflight-20260904/targeted-generation-postfix-v8/README.md`, `c1-run1.answers.jsonl`, `judge/c1-run1-judge-v11-r1.jsonl` | 알려진 실패군 n=4×1의 확률적 단일 표본이다. 대폭 향상·전체 DEV·holdout·일반화 주장 금지 |
| C48 | fresh D-2 raw draft의 정상 문장을 삭제하던 시간 표기·예약 서비스·수수료/면제·다중 회차 후처리를 좁게 수정했다 | 같은 4개 저장 draft/context 재투영에서 19/19 문장·39/39 critical value 유지, 거부 claim 0; 신규 양성/음성 회귀 포함 전체 735 OK(6 skip) | 조건부·후처리 진단 | `processed/eval/preflight-20260904/targeted-generation-postfix-v8/postprocessor-d2-fix-v3/c1-run1.current-postprocessed.jsonl`, `c1-run1.analysis.json`, `scripts/search_api.py`, `tests/test_api_hardening.py` | 외부 생성·Judge 없는 same-draft projection이다. 특히 기본 D-2 raw에는 수수료가 없었으므로 강화 prompt의 fresh 검증 전 생성 개선으로 해석 금지 |
| C49 | DEV45 C1 저장 초안 전수 감사로 반복되는 후처리 false negative 6유형을 좁게 수정했다 | 135개 중 답변 15개 변경, source-backed 문장 16개 추가 보존, 핵심값 손실 62→44(-18); 변경 15개 중 기존 2점 미만·해당 누락을 Judge가 지목한 record 13개; 전체 747 OK(6 skip) | 조건부·후처리 진단 | `processed/eval/preflight-20260904/dev45-postprocessor-capability-fix-v1/README.md`, 같은 디렉터리의 3개 projection·analysis, `scripts/search_api.py`, `scripts/rag/generators.py`, 관련 tests | 같은 저장 draft/context의 결정론적 projection이며 새 Judge·fresh E2E 없음. 13개를 점수 개선으로 세지 않고, citation/GFC/holdout/일반화 성능 주장 금지 |

2026-09-01 평가 분리·trace·후처리·grounding 수정 뒤 `bun run check`를 다시
실행해 시스템 Python 3.9 테스트 348건 실행(342 pass·선택 의존성 6 skip),
ESLint·TypeScript·Vite production build 통과를 확인했다. 로그는
`evidence/20260914/local-full-check-20260901.log`에 보존했다. 이는 dirty working
tree snapshot의 결과이며 최종 동결 커밋의 C15를 대신하지 않는다. 최종 게이트는
고정 의존성 환경과 clean commit에서 같은 명령 및 데모 smoke를 다시 실행한다.
후속 final schedule·retrieval E2E·generation/Judge/human fallback까지 반영한
2026-09-02 전체 검사는 656 tests OK(6 skip), ESLint·TypeScript·Vite build
PASS다. 다만 dirty 작업 snapshot이므로 최종 동결 커밋의 전체 check와 데모
smoke는 별도다.

2026-09-01 재개 후 승인된 Gemini 호출로 현재 코드의 `p0g` DEV 등록 3문항
C0/C1 answer artifact를 수집하고 Judge v8 유효 3회 반복을 완료했다. v6/v7
실패·calibration artifact는 quota·quote-format 문제로 무효이며 v8은
18.8–24.3KB compact input, exact match와 line/list-marker canonical quote
validation을 사용한다. 그래도 n=3 DEV smoke이고 동일 모델 계열의 생성·판정에
따른 self-preference와 사람 calibration 미완 문제가 남으므로 headline이나
일반화 근거로 쓰지 않으며 2026-08-31 historical n=3과도 구분한다.

## 3. 수치 표현 사전

다음 용어는 최종 보고서 전체에서 동일하게 쓴다.

| 수치 | 정확한 표현 | 금지 표현 |
|---|---|---|
| Hit@5 | “상위 5개 결과 중 하나 이상의 정답 문서가 포함된 질의 비율” | “답변 정확도”, “정답률” |
| Any-Gold-Chunk@k (DEV) | “상위 k개에 지정 gold chunk ID가 하나 이상 포함된 질의 비율” | “Atomic Evidence Recall”, “사실성”, “완전 정답률” |
| GoldChunkRecall@k (DEV) | “질의별 지정 gold chunk ID 회수율의 평균” | “모든 필수 사실 회수율” |
| Parser anchor recall | “표적 18문서 source-bound anchor의 기계적 문자열 보존율” | “전체 코퍼스 파서 정확도”, “RAG 성능 향상” |
| Evidence Recall@k (holdout v2) | “필수 atomic claim 중 profile-independent evidence option이 검색된 비율” | “답변 정확도” |
| MRR | “첫 관련 결과의 역순위 평균” | “전체 관련 문서 순위 품질” |
| 0–2 judge 평균 | “고정 rubric과 judge 설정에서의 답변 품질 평균” | “정답률”, “객관적 정확도” |
| RAGAS Answer Correctness | “사실·의미 유사도를 결합한 연속형 evaluator score” | “퍼센트 정확도” |
| 우리 1.607 vs 상대 1.440 | “확보50·해당 judge·수집일 기준 비교 점수” | “항상 더 우수”, “일반적 성능 우위” |

## 4. 서비스 튜닝 채택 게이트

검색 튜닝은 아래 조건을 모두 만족할 때만 최종 서비스 구성으로 채택한다.

1. 검색 스모크에서 기준선 대비 Hit@5와 MRR이 하락하지 않는다.
2. 기준선 적중 문항을 잃지 않거나, 잃은 문항이 수동 검토상 명백한 gold 오류다.
3. 동일 45문항 생성 n=3에서 평균과 문항별 paired 결과를 확인한다.
4. 평균이 같거나 낮으면 검색 개선만 별도 성과로 보고하고, 생성 경로 채택은
   과잉 회피·컨텍스트 순서·프롬프트 진단 후 결정한다.
5. 채택 설정에서 테스트와 데모 smoke를 다시 수행한다.

현재 판정: 검색 지표는 게이트 1·2를 통과했고 생성 n=3 평균도 하락하지 않아
서비스 검색 조정은 **잠정 채택**한다. 다만 생성 차이의 신뢰구간이 0을 포함하고
6문항이 악화됐으므로 “최종 답변 성능의 확정적 개선”으로 주장하지 않는다.
날짜·짧은 사실 후처리 수정의 실제 생성 표적 n=3을 완료한 뒤 서비스 구성을
최종 확정한다.

## 5. 최종 평가 프로토콜

### 5.1 검색

- 동일 질의·동일 코퍼스·동일 top-k에서 설정을 비교한다.
- Hit@1·3·5, MRR, DEV GoldChunk coverage@5/@8, 질의당 지연시간을 보존한다.
- 전체 평균과 함께 카테고리별 수치, 신규 적중, 상실 적중을 제시한다.
- 개선 전후 차이는 paired bootstrap 95% 신뢰구간을 산출한다.

### 5.2 생성

- 기준선과 후보 설정을 각각 3회 실행한다.
- 검색 결과, 컨텍스트 순서, 생성 답변, 출처, 모델·temperature·prompt 버전을
  JSONL에 함께 보존한다.
- 평균뿐 아니라 표준편차, 문항별 개선/동률/악화 수, 검색 적중×답변 점수
  교차표를 제시한다.
- 서로 다른 생성 답변의 3회 평가는 judge 자체의 신뢰도 검증이 아니다.

### 5.3 Judge 신뢰도와 경쟁 서비스 비교

- 고정된 동일 답변 표본을 judge가 3회 재채점하여 평균·표준편차·일치율을
  산출한다.
- 현재 `p0g` DEV 3문항 v8에서는 양 조건의 모든 문항이 3/3 일치했다. C0 점수는
  `[2,1,0]`, C1은 `[2,1,2]`이며 차이는 BIDV `svc_reg_03` 검색 복구에서 발생했다.
  `svc_reg_02`는 두 조건 모두 이월 세부를 누락해 1점이다.
- 경쟁 서비스 비교에서는 제품명을 숨기고 A/B와 B/A 순서를 모두 평가한다.
- 양방향 판정이 모순이면 tie 또는 사람 재검토로 처리한다.
- 최신성, 검색 miss, 검색 hit-생성 실패, 경계 점수 사례를 층화 표본으로
  사람이 검토한다.

### 5.4 개발셋과 holdout

현재 서비스 45문항은 원인 진단과 튜닝에 반복 사용됐으므로 **개발셋**이다.
이 점수는 해당 평가셋에서의 개선 효과이며 일반화 성능으로 표현하지 않는다.
Final holdout 36문항(Core 27 + Challenge 9)은 정본 프로토콜상 필수이나 현재
AI 검토의 12개 지적을 수정한 draft 36문항과 evidence option 93개가 작성됐다.
SHA는 `2fbedad63531491d3c069a5de6fe51fff623f7e571b9170c6a9aa7452e2d51f9`이며
schema·DEV·corpus pre-review gate는 PASS다. 읽기 전용 사람 검토 패킷과
SHA-bound A/B 응답 template은 준비됐지만 둘 다 아직 `PENDING`이므로 signoff는
없다. 2인 human signoff와 최종 파일 동결·실행은 아직 미완료다.

## 6. 제출 필수 산출물 체크리스트

- [x] 서비스 tuned 생성 n=3 완주 및 검색 튜닝 잠정 채택
- [x] 현재 코드 DEV45 C0/C1 생성 270개·Judge v11 판정 270개와 strict A/B 집계
- [x] 알려진 실패 9문항 의도별 검색 보강 fresh 생성·Judge와 잔여 실패 원인 분해
- [x] D-2 3-facet 보강 뒤 DEV45 retrieval-only 무손실 회귀(v6)
- [x] 자부담금 지원 3-part 보강 뒤 DEV45 retrieval-only 무손실 회귀(v7)
- [x] 신규 holdout 36문항(Core 27 + Challenge 9) AI 검토 12건 수정·기계 gate·사람 검토 패킷 생성
- [ ] 현재 cases SHA에 대한 사람 2인 독립 검토·signoff·최종 파일 동결
- [x] parser audit 18문서/54 anchor 기계적 source-bound 평가와 산출물 hash 확인
- [x] 질문 단위 GFC paired bootstrap·sign-flip·McNemar 분석기와 DEV3 smoke
- [x] condition-blind 답변 검토 packet·독립 A/B·합의 label 및 Judge-human calibration 분석기
- [x] DEV 8조건 retrieval matrix 360요청과 metric-complete compact derivative
- [x] oracle-context 진단 수집기와 fail-closed signoff/draft guard
- [ ] parser audit 54 anchor 독립 2인 원문 화면 육안검수(강한 파서 품질 주장 시)
- [ ] HOLDOUT 4조건 retrieval one-shot
- [ ] generation 198개와 structured judge 234회 완주
- [ ] 사람 2인의 63개 출력 blind 평가와 adjudication
- [ ] 잔여 검색 miss와 hit-0점 문항의 원인 표
- [ ] 경쟁 서비스 최신 재수집 또는 날짜가 명시된 한계 문구
- [ ] 경쟁 비교 양방향 blind judge와 사람 표본검증
- [ ] 최종 수치 JSON/CSV 정본
- [ ] 평가 산출물 SHA-256 manifest
- [ ] 요구사항 원문과 구현 대응표
- [ ] 최종 전체 회귀 로그
- [ ] 최종 아키텍처 그림 또는 기존 그림의 범위 한정
- [ ] 최종 보고서 Markdown
- [ ] PDF 렌더링과 페이지별 시각 검수
- [ ] 필요 시 발표자료와 데모 리허설 체크리스트

최종 manifest는 `scripts/build_evidence_manifest.py`로 만든다. Answer JSONL은
`--expect-jsonl`로 행 수·고유 ID를 검증하고, judgment JSONL은 별도
`--expect-jsonl`과 필요한 score 검증을 적용한다. Manifest는 answer–judgment
ID/hash와 실험 조건의 exact coverage를 교차 검증하며 생성·judge 모델, 실행
명령, index·config hash는 metadata와 artifact 항목으로 기록한다.
진행 중 파일은 manifest에 포함하지 않는다. 재채점은 새 `judge_run_id`와 새
judgment JSONL에 append하며 answer 원본은 수정하지 않고 SHA-256으로 연결한다.

## 7. 일정과 컷라인

| 날짜 | 종료 조건 |
|---|---|
| 9/1 | evaluator 오류 수정, 설정·trace·gold schema 구현 |
| 9/2–9/3 | holdout 36문항과 parser audit 18문서 작성·교차 검수·hash 동결 |
| 9/4 | DEV 8조건·parser audit·smoke 완료, clean code-freeze tag |
| 9/5 | HOLDOUT 4조건 retrieval one-shot, generation 시작 |
| 9/6 | generation 198개와 structured judge 234회 완료 |
| 9/7 | 사람 2인 blind 평가 완료 |
| 9/8 | adjudication·통계·실패 분석, 장애 복구 buffer |
| 9/9 | final manifest, 수치·artifact result-freeze tag |
| 9/10–9/11 | 퇴고, 인용·번호·재현 부록 검토 |
| 9/12 | PDF 생성 및 페이지별 시각 검수 |
| 9/13 | 최종 교정·제출 파일 검증·데모 리허설 |
| 9/14 | 제출 버퍼 |

지연 시 컷 순서는 신규 모델·파서 실험 → 2층 폴백 실서비스 통합 → 잔여
5문항 추가 튜닝 → 과잉 회피 추가 튜닝 순이다. Holdout one-shot, 생성·Judge,
사람 검증, 전체 회귀, 요구사항 대응표, final manifest와 최종 PDF는 컷하지 않는다.
