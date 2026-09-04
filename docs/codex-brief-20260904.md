# Codex 작업 지시서 — 2026-09-04 (튜닝 동결 → freeze 준비 → 사후 분석 준비 → 보고서 정비)

## 0. 현재 상태 (사실, 재확인하지 말고 전제로 삼을 것)

- DEV45 현재 코드, 독립 생성 n=3, Judge v11: 0~2점 평균 C0 .8815 → C1 1.2148
  (family-cluster bootstrap 95% CI [+.0889, +.5887]). 2/3 majority GFC 13/45 → 20/45
  (CI [-.0217, +.3201], p=.1185). 정본: `processed/eval/preflight-20260903/dev45-generation-current-v1/judge/dev45-3run-service-ab-v11.json`
- 이후 의도 기반 확장·후처리 수정으로 DEV45 검색은 hit@5 45/45, All required evidence@8 43/45까지 올랐으나, 이는 남아 있던 DEV 실패 문항을 겨냥한 표적 규칙의 결과다.
- 서비스 코드 규모: `scripts/search_api.py` 3,087 → 8,101줄, `bm25_search.py` 963 → 1,273줄, `generators.py` 1,053 → 1,292줄. 증가분의 상당수가 특정 DEV 문항(학생증·증명서 위탁, 교환학생, 자부담금, D-2 단체접수, 하계방학 자격증, 제3자 인권신고, 모집인원 변경)에 대응하는 정규식·점검표·후처리 허용 규칙이다.
- 사람 2인 holdout 검수는 미착수(`evidence/holdout-v2-reviewer-{a,b}.json` 36문항 전부 PENDING). signoff 없음. 코드 freeze 없음. 미커밋 파일 95개. 전체 unittest 735 OK(6 skip).

## 1. 원칙 (위반 금지)

1. **오늘부터 검색 규칙·생성 prompt·점검표·후처리 규칙을 추가·수정하지 않는다.** DEV45 점수를 올리기 위한 어떤 코드 변경도 금지한다. 발견한 결함은 코드를 고치지 말고 `docs/progress-log-20260901.md`에 "동결 후 발견" 항목으로만 기록한다.
2. `config/pnu-service-answer-holdout-v2.draft.jsonl`과 검토 패킷·A/B 응답 파일을 읽거나 수정하지 않는다. holdout 질문 텍스트를 어떤 분석에도 투입하지 않는다 (holdout 실행이 끝난 뒤의 사후 분석만 허용).
3. 커밋·태그·푸시·브랜치 조작은 사용자 명시 승인 뒤에만 한다. `git add -A`·`git add .` 금지.
4. 외부 LLM 호출(Gemini 생성·Judge)은 사용자 명시 승인 뒤에만 한다. 아래 작업은 전부 API 호출 0회로 수행 가능하다.
5. 기존 산출물(`*.answers.jsonl`, `*judge*.jsonl`, summary, README)을 수정·삭제하지 않는다. 새 산출물은 새 경로에 쓴다.
6. 각 작업이 끝나면 progress-log에 날짜·산출물 경로·SHA-256·테스트 결과를 기록한다.

## 2. 작업 (순서대로)

### 작업 1 — 튜닝 동결 선언과 동결 스냅샷 검증
- progress-log에 "2026-09-04 튜닝 동결" 절을 추가하고, 동결 시점의 코드 SHA(`/health`의 `startup_code_sha256` 방식과 동일한 계산)와 인덱스 3종 SHA를 기록한다.
- 다음을 실행해 결과(테스트 수·skip 수·경고)를 그대로 기록한다:
  `git diff --check`, `python3 -B -m unittest discover -s tests -p 'test_*.py'`, `bun run lint`, `bun run build`.
- 수용 기준: 네 명령 모두 통과, 결과 수치가 progress-log와 런북 7절의 freeze 기록 자리에 반영됨.

### 작업 2 — freeze 커밋 준비 (커밋은 하지 않는다)
- `git status --short`를 기반으로 **승인용 파일 목록 초안**을 `docs/freeze-file-list-20260904.md`에 작성한다. 분류: 수정 파일 / 신규 scripts / 신규 tests / 신규 config / 신규 docs / `evidence/`. `processed/`와 `.env`는 gitignore로 제외됨을 명시한다.
- 목록에 "포함 근거 한 줄"을 파일별로 적는다. 포함하면 안 되는 파일(임시 파일, 비밀, 대용량 바이너리)이 있으면 별도 표시한다.
- 커밋 메시지 초안 `chore: freeze PNU final evaluation code` 와 태그명 `pnu-eval-code-freeze-20260904-v1`을 함께 적는다.
- 수용 기준: 사용자가 목록을 보고 승인만 하면 런북 7절 명령을 그대로 실행할 수 있는 상태.

### 작업 3 — 규칙 인벤토리 문서화 (코드 변경 없음)
- `docs/rule-inventory-20260904.md`를 작성한다. 대상: `scripts/search_api.py`, `scripts/bm25_search.py`, `scripts/rag/generators.py`의 검색 확장·질의 정규화·후보 강등·facet 분리·생성 점검표·후처리 허용/거부 규칙 전부.
- 각 규칙을 표로 정리한다: 규칙 이름 / 위치(파일:행) / 트리거 조건(정규식·키워드) / 분류(**일반 규칙** = 원리 기반, 특정 문항 없이 설명 가능 vs **표적 규칙** = 특정 DEV 문항의 실패를 고치기 위해 도입) / 도입 근거 DEV 문항 id / 도입일 / progress-log 절 번호.
- 마지막에 집계: 일반 규칙 수, 표적 규칙 수, 표적 규칙이 겨냥한 DEV 문항 수(중복 제거).
- 수용 기준: 인벤토리의 행 수가 코드의 규칙 상수·정규식 정의와 1:1로 대응하고, 누락 여부를 `grep`으로 검증한 명령을 문서 끝에 기록.

### 작업 4 — holdout 사후 분석 도구 준비 (실행은 holdout 완료 뒤)
- `scripts/analyze_rule_activation.py`를 작성한다. 입력: answer artifact(`*.answers.jsonl`, eval trace 포함). 출력: 문항별로 어떤 규칙이 발동했는지(검색 확장어 삽입 여부, 강등 규칙 적용 여부, 점검표 항목 삽입 여부, 후처리 허용/거부 규칙 적용 여부)와 조건별 발동률 집계(JSON+CSV).
- 규칙 식별은 작업 3의 인벤토리 이름을 그대로 쓴다. trace에 규칙 식별자가 남지 않는 규칙은 "trace 부재"로 표시하고, 코드를 고쳐 trace를 추가하지 않는다(동결). 대신 질의 텍스트에 트리거 정규식을 적용해 "발동 가능"을 별도 열로 계산한다.
- **검증은 DEV45 산출물로만** 한다: `processed/eval/preflight-20260903/dev45-generation-current-v1/c1-run1.answers.jsonl` 등. holdout 파일은 절대 입력하지 않는다.
- 단위 테스트를 추가한다(합성 trace 3~5건).
- 수용 기준: DEV45 C1에서 표적 규칙 발동 문항 수가 인벤토리의 "겨냥 문항 수"와 일치하거나 차이 사유가 설명됨.

### 작업 5 — 생성 실패 분해 집계 도구 (실행은 DEV45로 검증, holdout은 완료 뒤)
- `scripts/analyze_generation_failures.py`를 작성한다. 입력: answer artifact + Judge v11 judgment artifact. 출력(조건별): 검색 적중·필수 근거 포함 여부 × Judge 결과의 교차표, 비GFC 문항의 원인 분포(명시적 회피 guard, 부적절 회피, 필수 claim 누락, 미지지 사실, 모순, 부분 인용), guard 규칙별 적용 수.
- DEV45 v11 산출물(`judge/c{0,1}-run1-judge-v11-r1.jsonl` 및 n=3 판정)로 실행해 표를 만들고 progress-log에 붙인다. 참고: v11 r1 기준 C1 비GFC 27건 중 회피 guard 7, 필수 claim 누락 19, 부적절 회피 16.
- 수용 기준: 집계가 `summarize_judge_repeats` 결과(GFC 수, 평균)와 일치.

### 작업 6 — 보고서·런북 정비 (수치 변경 없음)
- `docs/final-report-20260914.md`:
  - 초록의 기여 (2) "sparse와 dense 신호를 결합한 근거 검색"을 headline 조건(BM25 단일 lane, Dense/Hybrid는 exploratory에서 열세)과 일치하도록 고친다.
  - 4장에 "일반 규칙"과 "표적 규칙"을 구분한 소절을 추가하고, 작업 3의 집계 수치를 인용한다.
  - 8장(한계)에 다음을 명시: 표적 규칙에 의한 DEV 과적합 위험과 holdout 발동률로 검증할 계획, 코드 복잡도 증가(줄 수), 표본 크기(DEV 45·holdout core 27)에 따른 검정력 한계, 생성기·Judge 동일 계열의 자기 선호 가능성과 사람 calibration 계획, 코퍼스 구조 결함(suspect 11%, base64 유출, 연도별 사본 누적) 미해결.
  - 6장 결과에 0~2점 평균을 보조 지표로 병기하되, 사전 고정 주지표는 GFC임을 문장으로 명시한다.
- `docs/final-eval-runbook-20260914.md`:
  - Judge v11 guard 규칙 3종(명시적 회피, 인용 불가 claim 강등, GFC 자기모순 확정)과 "점수를 올리는 규칙 없음"을 Step 7에 한 단락으로 추가한다.
  - DEV 실행 주의: `--expected-index-sha256`는 clean worktree를 요구하므로 freeze 전 DEV 실행에서는 생략하고 corpus revision·source manifest 핀으로 대체한다는 문장을 5절 또는 11절에 추가한다.
  - resume 시 collector config 해시가 같아야 하므로 `--sleep` 등 인자를 바꾸지 말라는 주의를 11절에 추가한다.
- 수용 기준: 문서 변경만, 수치 인용은 전부 progress-log의 정본 산출물 경로를 각주로 가리킴.

### 작업 7 — (선택, 사용자 승인 시에만) DEV45 Judge v11 안정성 반복
- 사용자가 외부 호출을 승인한 경우에만 `judge-v11-r2`, `judge-v11-r3`를 C0/C1 run1 answer artifact에 대해 단일 스트림(`--sleep 3 --retries 6`)으로 수행하고 `summarize_judge_repeats`로 판정 일치율을 집계한다. 승인 전에는 실행 계획과 예상 호출 수(90회)만 기록한다.

## 3. 보고 형식

- 작업마다 progress-log에 절을 추가: 수행 명령, 산출물 경로와 SHA-256, 테스트 결과(총 수·skip·실패), 발견 사항.
- 승인이 필요한 항목(작업 2의 커밋, 작업 7의 외부 호출)은 실행하지 말고 "승인 대기"로 표시한 뒤 사용자에게 질문한다.
- 작업 3~5에서 코드 결함을 발견하면 고치지 말고 "동결 후 발견" 목록에 기록한다.
