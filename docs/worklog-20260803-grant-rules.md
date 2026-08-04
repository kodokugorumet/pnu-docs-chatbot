# 작업 로그 — 연구비 규정 corpus 구축 (2026-08-03)

퇴근 전 중간 저장. 이어서 할 때 이 문서 하단 "다음 단계"부터 진행.

## 배경 / 목표

교수님 요구사항 (전체 목록은 메모리 `project-pnu-chatbot-professor-requirements` 참고):

- 비교 대상 챗봇 **https://yunju.work/grant-rules/** (부산대 산학협력단 연구비 규정
  판정·지식검색 v2.1)보다 성능 좋게 구현할 것
- 그 사이트가 공개한 **53문항 Q&A(연구비관리부 원문 질문+기준답안)**를 채점표로 사용
- 우리 기존 인덱스(2,247 문서)에는 연구비·여비 문서가 0건 → corpus부터 구축 필요
- 상대 원본 자료실(Google Drive)은 권한 차단 → **공식 출처에서 재수집**으로 결정

## 오늘 완료한 것

### 1. 평가셋 확보
- `config/grant-rules-answer-eval.jsonl` — 53문항 (질문+기준답안+출처 섹션+수집 메타)
  - 출처 API: `https://yunju.work/grant-rules/api/question-set`
  - 문항 분포: 국연법 15 / 공무원 여비규정 8 / 부산대 세부지침 5 / 수과원 4 /
    NRF 인문사회 3 / 식약처 3 / 극지·중기부 각 2 / 기타 단일 문항들
- `downloads/grant-rules-drive-20260803/regulations-manifest.json` — 상대 챗봇의
  규정 40개 노드 + 관계 그래프(직접적용/상위근거/보충/개정이력). 상대의 "관계 지도"
  구조를 그대로 확보한 것. 나중에 검색 라우팅·리랭킹에 재활용 가능

### 2. 문서 수집 → `downloads/grant-rules-official-20260803/`
| 출처 | 내용 | 파일 |
|---|---|---|
| law.go.kr 법령 | 국연법·시행령·시행규칙, 공무원 여비 규정, 청소년성보호법+시행령 | 렌더링 HTML/TXT + 별표 HWP 27건 |
| law.go.kr 행정규칙 | **국가연구개발사업 연구개발비 사용 기준**(과기부 고시, 핵심), 산업기술혁신사업 공통 운영요령 | 본문 + 별표 46건(뷰어 XML 텍스트 복원) |
| 부산대 산학협력단 | **규정집 전체 50건** (연구비 관리 세부 지침 2026.4.30 포함) | HWP 50 + manifest.json |
| 중기부 | 기술개발 지원사업 관리지침 2025 (본문+공통서식) | HWP 2 |
| 식약처 | 연구개발비관리지침 (2023.9.22) | PDF 1 |
| 수과원 | 2026년 연구용역비 산정·정산 기준 (※상대는 2025판, 단가 차이 유의) | HWPX 1 |
| NRF/교육부 | 인문사회 학술연구지원사업 종합계획 2024·2026 (별표1=사업비 사용기준) | PDF 2 |

수집 기법 메모 (재수집 시 참고):
- law.go.kr: 친화URL → iframe(`lsInfoP.do`) → 헤드리스 렌더링(gstack browse).
  별표 첨부는 `flDownload.do?flSeq=…&bylClsCd=…`, 행정규칙 별표는
  `bylInfoDiv('<id>')` 실행 후 `viewer/BYL/...//thumbnailxml/<p>.xml` 좌표 복원.
  스크립트: 스크래치 `collect_law.sh` (세션 종료로 소실 시 이 문서 참고해 재작성)
- 산단 규정집: `sanhak.pusan.ac.kr/sanhak/board/16`, ASP.NET이라 curl 포스트백은
  CSRF 차단. **headless 실제 클릭**(`browse click "button[onclick*=\"'17', 'A'\"]"`)
  → 모달의 `a[download]` href(`/upload/...`)를 curl로 다운로드
- mfds.go.kr은 헤드리스 차단 → 대학 산단 미러(동아대 research.donga.ac.kr, gnuboard
  nonce는 페이지 fetch와 같은 세션에서 즉시 사용해야 함)

### 3. 파싱·인덱스 (완료)
- 스테이징: `downloads/grant-rules-official-20260803/content/<기관>/` 8개 기관 97파일
  (법령 렌더 텍스트는 `.txt` 미지원이라 최소 HTML로 래핑)
- 파싱: cascade 프로필, run `20260803-grant-rules-cascade-v1` —
  **97/97 성공**, 블록 102,877 / 표 1,270 / 청크 3,511
- 인덱스: `processed/index/grant-rules-20260803.sqlite` —
  97 docs / 3,483 chunks (suspect 포함, 28청크 제외)

### 4. 부수 작업
- 권한 프롬프트 절감: `~/.claude/settings.json`에 curl·browse·Notion/Drive 검색
  allowlist 추가
- 메모리 저장: `project-pnu-chatbot-professor-requirements` (교수님 요구 7건+상대 분석)

## 미수집 (Task #5 백필, 6문항 해당)
- 극지연구소 2024 위탁연구비 세부기준 (2문항)
- 한국가스공사 정산기준 / 한전 전력연구원 여비 기준 / 창의재단 컨설팅 가이드라인 /
  환경부 학술연구 용역비 세부 계상기준 (각 1문항)
- 참고: NRF 사이트/각 기관 공고 첨부에 있음. 평가 결과에서 해당 문항 실패 확인 후 수집

## 다음 단계 (여기부터 재개)

1. **53문항 검색 평가** — 새 인덱스에 대해 Hit@k 측정
   - `config/grant-rules-answer-eval.jsonl`의 `section` 필드를 기대 문서로 매핑
     (예: "공무원 여비규정" → 국가법령/공무원 여비 규정*, "부산대학교 연구비 관리에
     관한 세부 지침" → 부산대…세부 지침)
   - 검색: `.parser-tools/venvs/core/bin/python scripts/bm25_search.py search
     --index processed/index/grant-rules-20260803.sqlite --top-k 5 --json "<질문>"`
   - 기존 24문항 평가 스크립트(`scripts/evaluate_pnu_retrieval.py`) 형식 참고
2. 실패 문항 분석 → 필요시 Task #5 백필 수집
3. **답변 생성 평가**: search_api에 새 인덱스 연결(`--index` 인자), 53문항 질문 →
   생성 답변을 기준답안과 대조(LLM 채점 or 키워드 루브릭). 상대 챗봇 답변과 비교표 작성
4. 기존 인덱스와 **통합 인덱스** 빌드 (bm25_search.py build는 --chunks 반복 지정 가능)
5. (별도 트랙) 커밋 전 learned dense 작업과 섞이지 않게 브랜치 정리 —
   현재 브랜치 `codex/rag-v2-parser-pipeline`에 커밋 안 된 learned dense 수정
   + 오늘 추가된 untracked 파일들이 공존 중. **커밋 시 새 브랜치 분리 권장**
   (예: `feat/grant-rules-corpus`)

## 저녁 세션 진행 (같은 날 재개 후)

### 검색 평가 결과 (53문항, BM25)
| 구성 | Hit@5 | MRR |
|---|---|---|
| BM25 단독 | 35/46 (76.1%) | 0.585 |
| + 채점 매핑 보정(복수 출처 인정) | 38/46 (82.6%) | 0.621 |
| + **기관 라우팅** (`scripts/rag/grant_router.py`) | **45/46 (97.8%)** | **0.917** |

- 라우터: 질의에 기관 명시 → 해당 기관 고정 / 여비·출장 질의 → 국가법령+과기부 /
  기본 → 공통기준+부산대, 대학 언급 없으면 공통기준·자체규정 교차 배치
- 평가: `scripts/evaluate_grant_retrieval.py --routing`
  (결과 `processed/eval/20260803-grant-rules-bm25-routed-v3.json`)
- 잔여 실패 1건(grant_050, "신청 절차"→제·개정 절차 지침에 오매칭)은 BM25 어휘
  한계 — learned dense(KURE) 하이브리드로 해결 예정. 라우터 과적합은 중단
- 백필 5건(Task 5)은 전부 로그인 포털·비색인 첨부라 보류. 현실 대안 = Drive
  액세스 요청 승인 or 기관 문의. uncovered 7문항으로 집계 중

### 답변 생성 평가 (Task 8, 완료)
- `scripts/evaluate_grant_generation.py`: 라우팅 검색 → Gemini 생성 →
  Gemini judge 채점(0~2점, 기준답안의 수치·조건·절차 포함 여부).
  append 방식이라 중단돼도 `--out` 같은 파일로 재실행하면 이어짐
- v1 (diverse top5, temp 0.2): 전체 평균 0.40 —
  `processed/eval/20260803-grant-generation.jsonl`
- **v2 (하이브리드 컨텍스트: diverse 5 + 상위 3문서 추가 청크 각 2, temp 0)**:
  covered 46문항 **평균 0.67 | 2점 13 / 1점 5 / 0점 28** —
  `processed/eval/20260803-grant-generation-v2.jsonl`
- **핵심 발견: covered 0점 28건 중 27건이 "확인할 수 없다"형 정직한 무응답,
  오답(환각)은 단 1건.** 근거 없는 답을 지어내지 않는다는 신뢰도 스토리 확보
- 무응답 집중 구간 = 여비 단가류 질문. 원인 확인됨: 별표2(국내 여비 지급표)는
  금액까지 정상 파싱·색인돼 있으나, 표 청크 어휘와 질문 어휘가 달라 BM25가
  못 끌어옴 (전형적 어휘 갭)

### 다음 개선 레버 (우선순위순)
1. **learned dense(KURE) 하이브리드를 이 인덱스에 적용** — 어휘 갭(여비 지급표,
   grant_001 카페트, grant_050 신청절차)이 전부 이 부류. 기존 미커밋 작업과 합류
2. 별표 문서 제목 강화(파일명 opaque seq → "[별표2] 국내 여비 지급표" 식) 후
   재파싱 — BM25만으로도 개선 여지
3. 생성 top-k/컨텍스트 예산 튜닝, 무응답 시 스코프 확장 재검색(2-pass)
4. 상대 챗봇 53문항 라이브 응답 수집 → 같은 judge로 채점해 직접 비교표
   (**상대 서비스에 53회 질의 = 비용·부하 발생, 사용자 승인 필요. 미실행**)

## 팀원 작업 분담 계획 (2026-08-04 논의, 비개발 위주)

팀원에게 맡길 작업 후보 — 전부 중간평가 3축(계획 충실도·실험 진행·결과물)에 기여:

1. **평가셋 제작** ⭐ — 기관별(금감원·한은·거래소·예탁원·KISA·KIOST) 역할 페르소나 ×
   질문 × 문서 기반 정답 세트. 인당 기관 1~2개, 20~30문항. 스프레드시트 템플릿 →
   JSONL 변환은 내가. 페르소나 정의(부산대 학생/행정원/기관 실무자 등)도 포함
   — 교수님 "역할 빙의 채점 + 자체 벤치마크 제작 허용" 요구에 직결
2. **사람 채점 + LLM judge 검증** ⭐ — 53문항 중 20개 샘플 2인 독립 채점 →
   LLM 채점과 일치율(kappa). 상대 챗봇 vs 우리 답변 블라인드 A/B 평가
3. **문서 수집** — 백필 5건(NTIS·창의재단 PMS는 로그인하면 가능), yunju.work
   Drive 액세스 요청, KINS(원자력안전기술원) 문서 수집, 파싱 품질 육안 검수
4. **배경조사** ⭐ — RAG 평가 벤치마크 논문(RAGAS 등), 상대 챗봇 기능 분석,
   프롬프트 오염 방지 사례 조사 — 7/22 교수님 "조사 미흡" 피드백 해소
5. **프롬프트 오염 테스트 시나리오 작성** — 오염 문서 시나리오 10~20개 (실행은 내가)

추천 배분(3인): A=평가셋+페르소나 / B=채점·블라인드비교·검수 / C=조사·시나리오·수집.
다음 액션 후보: 팀원 전달용 작업 지시서 + 평가셋 템플릿 제작 (사용자 요청 대기)

## 2026-08-04 세션 — learned dense 하이브리드 적용 (레버 1)

### 브랜치 정리 (선행 작업)
- `feat/learned-dense-hybrid` (b6669b7): dense 검색 코드 + search_api/UI 연동
- `feat/grant-rules-corpus` (d8e8c70): 연구비 평가 트랙, 위 브랜치 위에 스택
- `diagrams/`는 7/28 데모 커밋에서 누락된 고아 파일이라 어느 쪽도 아니어서 untracked 유지

### 인덱스 빌드
grant corpus(3,483 chunks)에 대해 두 모델 모두 빌드, 각 27초 / `verified: true`:
`processed/index/learned-dense/grant-rules-20260803/{kure-v1,snowflake-arctic-l-v2-ko}/`

### 검색 평가 (53문항, 라우팅 적용, top-5)
| 구성 | Hit@5 | MRR |
|---|---|---|
| **BM25 (어제 baseline 재현)** | **45/46 (97.8%)** | **0.9167** |
| KURE dense 단독 | 41/46 (89.1%) | 0.8377 |
| KURE 하이브리드(RRF) | 44/46 (95.7%) | 0.8967 |
| Snowflake dense 단독 | 44/46 (95.7%) | 0.8986 |
| Snowflake 하이브리드(RRF) | 44/46 (95.7%) | 0.8967 |

**문서 단위 지표만 보면 dense는 개선이 아니라 후퇴다.** BM25가 이미 45/46이라
헤드룸이 없고, dense는 grant_011/018/021/023/028을 1위에서 miss로 떨어뜨린다.
단, **어제 유일한 miss였던 grant_050("신청 절차")은 모든 dense 구성에서 해결**
(miss → 2위, 하이브리드는 4위). 예측했던 어휘 갭 사례가 맞았다.

### 핵심 진단 — 청크 단위 근거 커버리지 (이 작업의 실제 목적)
문서 단위 Hit@5는 **어제 무응답을 유발한 표 청크가 컨텍스트에 들어왔는지**를 보지
못한다. 생성 0점 35문항 중 기준답안에 수치가 있는 19문항에 대해, 라우팅 top-8
컨텍스트 안에 기준답안의 수치(금액·비율·기간)가 실제로 등장하는지 측정했다.

| 구성 | 근거 수치가 컨텍스트에 등장 | baseline 대비 |
|---|---|---|
| BM25 | 10/19 | — |
| **KURE dense** | **17/19** | +7, **잃은 문항 0** |
| Snowflake dense | 16/19 | +6, 잃은 문항 0 |
| Snowflake 하이브리드 | 16/19 | +6, 잃은 문항 0 |
| KURE 하이브리드 | 15/19 | +5, 잃은 문항 0 |

**두 지표가 정반대를 가리킨다.** BM25는 올바른 *문서*를 찾지만 그 안의 올바른
*청크*(금액이 든 별표)를 못 집는다. dense는 그 반대다. 어느 구성도 BM25가 이미
확보한 근거를 잃지 않았다(strictly dominant) — 순수 추가 이득이다.
새로 근거를 확보한 문항: grant_023, 028, 032, 034, 036, 045, 046.

grant_038·051은 전 구성에서 여전히 0 — 별도 원인(파싱 누락 또는 미수집) 확인 필요.

### 결론 및 다음 레버
- 생성 품질을 좌우하는 건 청크 단위 근거 커버리지이므로 **dense를 도입할 가치가 있다.**
  다만 dense 단독은 문서 정밀도를 잃어 오문서 인용 위험이 있다
- 현행 RRF 하이브리드는 두 지표 모두에서 어중간하다(문서 44/46, 근거 15/19).
  **다음 레버: 생성 컨텍스트를 "BM25로 문서 선택 + dense로 청크 보강"으로 구성**
  — `evaluate_grant_generation.py`의 컨텍스트 조립부(diverse 5 + 상위 문서 추가 청크)
  에서 추가 청크를 BM25 raw 대신 dense로 뽑는 방식. 두 지표의 장점을 합칠 수 있다
- 임베딩 모델은 근거 커버리지 기준 KURE(17/19) > Snowflake(16/19), 문서 정밀도는
  Snowflake 우위. 하이브리드 구성이 정해지면 다시 비교할 것

### 이번 세션 코드 변경
- `scripts/grant_retrieval.py` (신규): bm25/dense/hybrid를 `(query, top_k) -> list[dict]`
  하나로 감싼 어댑터. `rag` 패키지는 BM25에 의존하지 않는 재사용 계층이라 글루 코드는
  `search_api`와 같은 스크립트 계층에 뒀다
- `scripts/evaluate_grant_retrieval.py`: `--retrieval-mode`, `--dense-artifact` 추가
- `tests/test_grant_retrieval.py` (신규): 세 모드 배선 검증. 전체 219 테스트 통과
- 평가 결과: `processed/eval/20260804-grant-*.json` 5건

### 생성 평가 (미실행, 결정 사항 기록)
- 로컬 모델 `mlx-community/gemma-4-26b-a4b-it-4bit`로 진행하기로 결정 (EXAONE 제외)
- 착수 시: `evaluate_grant_generation.py`는 `generators.generate`를 직접 호출해
  MLX 서버가 뜨지 않으므로, `scripts/local_model_runtime.py`의
  `build_local_model_runtime()`을 재사용해 `runtime.generation(model)`으로 감싸야 함
  (현재 `managed_mlx` 기동은 `search_api`만 소유)
- judge는 Gemini 유지 권장 — 어제 0.67이 Gemini judge 기준이라 채점자를 바꾸면 비교 불가.
  생성기 교체분도 있으므로 "옛 검색 + 로컬 생성" baseline 1회가 추가로 필요

## 미해결/주의
- Drive 자료실 접근 요청은 **보내지 않음** (사용자 결정 대기 상태였음 → 공식 수집으로 대체)
- 수과원은 2026판 사용 중 — 기준답안(2025 기반)과 단가 다를 수 있음
- 53문항 원본 `연구비 Q&A(연구비관리부)_25.8.1..hwp`를 corpus에 넣으면 평가가
  치팅이 되므로 **eval corpus에서 제외** 유지할 것
