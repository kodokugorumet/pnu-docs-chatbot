# 부산대학교 문서 RAG 최종 평가 프로토콜 v1.2

- 프로토콜 동결일: 2026-08-31
- 사전 개정: 2026-09-02, holdout 공개·사람 라벨링 전에 Judge-human gate에
  balanced accuracy를 추가했다. 이후 threshold 변경은 금지한다.
- DEV 정합성 개정: 2026-09-03, holdout 공개 전에 기존 DEV45를 byte-identical
  `pnu-service-answer-eval-v1.jsonl`로 보존하고, 질문에 직접 필요한 근거와
  참고 근거를 분리했다. 이 변경은 DEV 진단에만 적용하며 holdout 계약은
  변경하지 않는다.
- 최종 결과 동결일: 2026-09-09
- 보고서 제출일: 2026-09-14
- 적용 범위: 부산대학교 문서 기반 실제 `POST /chat` 서비스
- 상태: **평가 설계 확정, 실행 지원 부분 구현**. Generation collector, 생성과
  분리된 structured Judge, eval trace, health fail-closed, answer–judgment hash
  연결, DEV retrieval 비교기는 구현하고 로컬 검증했다. 표적 parser audit
  18문서/54 anchor, holdout draft 36문항/93 evidence option의 기계 gate와 사람
  검토 패킷, DEV `p0g` 외부 생성 및 Judge v8 조건별 3회 반복, 질문 단위 GFC
  통계와 condition-blind 답변 검토·Judge-human calibration 도구까지 완료했다.
  DEV45 전체 retrieval matrix와 oracle-context 수집 경로도 구현·검증했다.
  holdout one-shot과 실제 사람 평가·signoff는 미실행이다.
  구현상 오류 수정은 허용하지만, holdout을 연 뒤 질문·gold·비교 조건·주지표를
  바꾸지 않는다.
- 외부 실행 상태: **2026-09-02 현재 DEV preflight만 실행**. `p0g` 등록 3문항은
  최종 프로토콜의 `gemini-3.5-flash-lite`가 아니라 명시적으로 고정한
  `gemini-3.1-flash-lite`로 C0/C1 답변을 수집하고 Judge v8을 각각 3회 실행했다.
  이는 2026-08-31 historical n=3 및 향후 final holdout 실행과 구분하며, 정본의
  최종 생성 model 조건을 변경하지 않는다.

이 문서는 최종 보고서의 평가 방법론 정본이다. 다른 문서와 충돌하면 이
프로토콜을 우선한다. 기존 연구비 53문항 트랙은 별도 연구 부록이며, 부산대학교
학생 서비스의 headline 성능으로 사용하지 않는다.

## 1. 평가가 답할 질문

| ID | 연구 질문 | 평가 단계 |
|---|---|---|
| RQ1 | 세 파서가 공식 문서의 핵심 근거를 얼마나 보존하고 검색 가능하게 만드는가? | Parser audit, retrieval-only |
| RQ2 | BM25 서비스 튜닝이 같은 Cascade corpus에서 실제 근거 도달률과 순위를 개선하는가? | C0/C1 retrieval |
| RQ3 | 검색 개선이 최종 답변의 완전성·정확성으로 이어지는가? | C0/C1 generation |
| RQ4 | 답변의 각 사실과 인용이 실제 검색 문맥으로 지지되며, 근거가 없을 때 올바르게 회피하는가? | Claim, citation, challenge |
| RQ5 | 최종 서비스의 지연시간·비용·fallback·오류율은 얼마인가? | Operational evaluation |

검색, 생성, grounding, 운영 성능을 한 개의 “정확도”로 합치지 않는다.

## 2. 비교 조건과 명칭

`Baseline parser`와 서비스 A/B의 `baseline`을 혼동하지 않도록 서비스 조건은
다음처럼 C0/C1으로만 부른다.

| 항목 | C0 — untuned control | C1 — tuned service |
|---|---|---|
| Corpus/index | PNU 2026-07-25 curated Cascade v5 | C0와 동일 |
| Parser profile | `cascade` | `cascade` |
| Retrieval | `bm25` | `bm25` |
| Service tuning | OFF | ON |
| Generation context | `top_k=8` | `top_k=8` |
| Generator | 같은 provider·정확한 model version | C0와 동일 |
| Prompt/postprocessor | 같은 frozen code | C0와 동일 |

정본 인덱스는
`processed/index/pnu-20260725-curated-cascade-v5-allow-suspect.sqlite`다.
현재 manifest 기준 SHA-256은
`a4c1ca6318cb14721866a6512f3dfeacdca58c62ecff98ea7b81df2e75c74c31`이다.

최종 생성 실행은 요청에 아래 값을 명시하며 서버 기본값에 의존하지 않는다.

```text
parser_profile=cascade
retrieval_mode=bm25
top_k=8
provider=frontier
model=gemini-3.5-flash-lite
institution=null
```

Fallback은 실제 서비스의 일부이므로 숨기지 않고 결과에 포함한다. 다만 각 출력에
실제 사용 provider/model, fallback 사유와 시도 순서를 저장하고, 서로 다른 모델의
결과를 하나의 모델 성능처럼 표현하지 않는다.

## 3. 평가 데이터

### 3.1 DEV — 기존 45문항

`config/pnu-service-answer-eval.jsonl`의 45문항은 개발셋으로 고정한다. 9월 3일
질문-근거 정합성 검수 전 원본은 `config/pnu-service-answer-eval-v1.jsonl`
(SHA-256 `3c3e19d6c5524218b2f2cb1600c1c64abd80e1783ce7b235dbb80109723a3be6`)
로 보존한다.

- 검색 규칙 진단과 exploratory ablation에만 사용한다.
- 이미 전체 miss를 보고 튜닝했으므로 일반화 성능으로 주장하지 않는다.
- holdout 실행 전까지만 수정·선택 근거로 사용할 수 있다.
- 기존 24문항 파서 검색셋은 제목 중심 `parser pilot`으로만 보고한다.
- DEV flat evidence의 `required_for_answer=false`는 공식 근거이지만 질문이 직접
  요구하지 않는 참고 사실이라는 뜻이다. 이 항목의 누락은 답변 완전성이나
  gold-chunk recall을 낮추지 않으며, 답변이 실제로 언급했다면 여전히 retrieved
  context에 의해 지지되어야 한다.

### 3.2 FINAL HOLDOUT — 총 36문항

최종 holdout은 결과를 보기 전에 새로 작성하고 SHA-256을 동결한다.

#### Core 27문항

현재 서비스의 9개 영역에서 정확히 3문항씩 구성한다.

```text
academic, admissions, core, employment, graduation,
international, registration, scholarship, student_support
```

각 영역의 세 문항은 다음 난이도를 하나씩 갖는다.

1. 단일 문서·단일 핵심 사실
2. 복수 조건 또는 복수 근거를 결합해야 하는 질문
3. 표·날짜·금액·학기·자격조건 등 구조 보존이 중요한 질문

자연스러운 의역·약어·구어체를 포함하되 오타만으로 난이도를 만드는 문항은
최대 3개로 제한한다. 최소 6개 문항에는 `pnu-student`, `pnu-staff`,
`pnu-researcher` 중 하나의 역할을 명시한다. 역할별 성능은 표본이 작으므로
주지표가 아니라 진단 지표로만 사용한다.

#### Challenge 9문항

- 공식 corpus에 충분한 답이 없는 질문: 3
- 범위·적용 연도·규정 버전이 모호하거나 충돌하는 질문: 3
- 질문 또는 역할 입력을 통한 instruction/prompt injection 시도: 3

Challenge는 Core 주지표에 섞지 않고 correct abstention, 잘못된 단정,
injection obedience를 별도로 보고한다. 검색 문서 내부의 prompt injection은
통제된 fixture가 필요한 별도 보안 실험으로 남기고 이번 headline 결과에는
포함하지 않는다.

#### 누수 방지

- DEV와 질문 문장뿐 아니라 원본 `source document family`도 겹치지 않는다.
- Core 27개는 holdout 내부에서도 서로 다른 `family_id`를 가져야 한다. 역할만
  바꾼 같은 질문이나 같은 공지 template을 독립 문항으로 세지 않는다.
- 연도만 다른 동일 공지 template은 같은 family로 본다.
- URL, 첨부 SHA-256, 연도를 제거한 정규화 제목으로 중복을 검사한다.
- holdout 결과를 한 번이라도 확인한 뒤 규칙이나 prompt를 바꾸면 해당
  holdout은 DEV로 강등하고 새 holdout을 만들어야 한다.

### 3.3 Profile-independent gold

Cascade chunk ID를 정답으로 사용하지 않는다. 문항은 최소 다음 필드를 가진다.

```json
{
  "id": "holdout_registration_01",
  "split": "holdout-core",
  "family_id": "...",
  "source_document_family_ids": ["..."],
  "source_sha256s": ["..."],
  "normalized_source_titles": ["..."],
  "category": "registration",
  "query": "...",
  "role": null,
  "answerable": true,
  "required_claims": [
    {
      "claim_id": "c1",
      "description": "...",
      "critical_values": ["..."],
      "evidence_options": [
        {
          "document_id": "...",
          "source_document_family_id": "...",
          "source_sha256": "...",
          "source_url": "...",
          "source_title": "...",
          "normalized_title_without_year": "...",
          "quote": "..."
        }
      ]
    }
  ],
  "optional_claims": [],
  "forbidden_claims": [],
  "expected_behavior": "answer"
}
```

각 필수 claim은 하나 이상의 대체 가능한 공식 근거를 가질 수 있다. Gold 작성자와
검수자는 source URL 또는 원본 첨부를 열어 날짜·금액·조건을 확인한다.
답변 postprocessor는 최대 8개 claim을 기록한다. 다만 Core의 required claim은
평가자 간 경계 합의와 사람 검수 부담을 통제하기 위해 2~5개로 제한한다. 5개를
초과해야 답할 수 있는 질문은 이번 holdout에 넣지 않는다.

파서 간 evidence matcher는 holdout을 보기 전에 다음 순서로 동결한다.

1. `document_id`가 gold evidence option과 일치해야 한다.
2. Unicode NFKC, 대소문자, 연속 공백, 레이아웃 구분자와 문장부호를
   정규화한 quote의 exact substring을 먼저 검사한다.
3. Exact match가 아니면 모든 `critical_values`와 최소 2개의 의미 anchor가
   일치하고, 정렬된 후보 window와 quote의
   `difflib.SequenceMatcher(autojunk=False).ratio()`가 0.85 이상일 때 fuzzy
   match로 인정한다.
4. 표 evidence는 순서가 아니라 지정 header-value/row 관계와 모든 critical
   value가 보존됐는지 검사한다.
5. 경계 사례는 parser/condition 이름을 숨긴 사람이 판정하고 그 사유를 남긴다.

Threshold와 anchor는 DEV에서만 확정하며 holdout 결과를 본 뒤 바꾸지 않는다.

### 3.4 Parser audit — 18문서

DEV 문서 중 holdout과 겹치지 않는 다음 18개를 층화 표집한다. Parser audit
결과를 먼저 본 것이 holdout 문서 선택이나 코드 수정으로 이어지는 누수를 막기
위해 Core holdout 문서에서는 뽑지 않는다.

- HWP/HWPX: 6
- 디지털 PDF: 6
- 스캔·OCR·표 중심 PDF: 6

문서당 heading/본문, 날짜·금액·자격조건, 표 cell 등 3개 anchor를 지정한다.
각 anchor는 `anchor_type=text|critical_value|table_row`, `must_contain`,
`same_unit=document|chunk|table_row`를 명시한다.
따라서 54개 gold anchor를 세 파서에서 비교한다. 기존 24문항 검색 평가는 이
감사의 과거 pilot으로만 남긴다.

### 3.5 Holdout preflight gate

Holdout hash를 동결하기 전에 validator가 다음을 모두 통과해야 한다.

- Core가 27개이고 9개 영역별 정확히 3개
- Challenge가 유형별 3개, 총 9개
- Core의 ID, 질문, `family_id`, source document family가 내부에서 중복 0
- DEV와 Core의 source document family 중복 0
- 모든 answerable required claim에 critical value와 evidence option이 존재
- 모든 evidence option의 URL, document/content SHA와 원문 snapshot이 존재
- 공식 원문에서 gold claim과 quote를 사람이 교차 확인
- answerable/challenge label을 두 사람이 확인

Preflight 결과와 cases SHA-256을 holdout 최초 실행 전에 manifest에 넣는다.

## 4. 실험 순서와 실행량

전체 `3 parser × 5 retrieval mode × generation n=3` factorial은 수행하지
않는다. 세 profile 모두 KURE·Snowflake artifact는 존재하지만, 예상
생성·judge 요청량이 지나치게 크고 어떤 변경이 효과를 냈는지 분리하기 어렵다.
Dense/Hybrid 검색 방식 효과는 최종 서비스 parser인 Cascade로 범위를 한정한다.

### Phase A — Parser quality

18문서 × 3파서를 오프라인 비교한다.

- parse success
- `pass/suspect/hard_fail`
- 54개 evidence anchor preservation
- heading/reading order preservation
- 날짜·금액·자격조건 preservation
- table cell/row preservation

### Phase B — DEV retrieval ablation

기존 45 DEV에서 파서 비교 3개, Cascade tuning 인과 비교 1개, Cascade
Dense/Hybrid 4개를 합쳐 총 8개 고유 조건을 retrieval-only로 실행한다.

| ID | Parser | Retrieval | Service tuning | 용도 |
|---|---|---|---|---|
| PAR-B | Baseline | BM25 | ON | 동일 검색기에서 파서 비교 |
| PAR-CH | Challenger | BM25 | ON | 동일 검색기에서 파서 비교 |
| C1 | Cascade | BM25 | ON | 현재 서비스 |
| C0 | Cascade | BM25 | OFF | 인과 비교 control |
| D-K | Cascade | KURE Dense | 해당 없음 | exploratory |
| H-K | Cascade | KURE Hybrid | BM25 service tuning 미적용 | exploratory |
| D-S | Cascade | Snowflake Dense | 해당 없음 | exploratory |
| H-S | Cascade | Snowflake Hybrid | BM25 service tuning 미적용 | exploratory |

총 45 × 8 = 360회이며 외부 생성 API를 사용하지 않는다. Baseline과
Challenger의 tuning OFF 셀은 파서 효과와 Cascade tuning 효과를 분리하는 데
필요하지 않아 제외한다. Dense/Hybrid와
Cross-encoder 결과는 holdout의 C0/C1 headline 조건을 바꾸지 않는다.
Cross-encoder는 이번 필수 범위에서 제외한다.

기존 DEV evidence가 Cascade chunk ID이므로 DEV의 세 파서 비교에는 Source
Hit/MRR만 공정 지표로 사용한다. 파서 간 atomic Evidence Recall은 profile-
independent gold가 있는 Parser audit과 HOLDOUT에서만 계산한다. Cascade 내부
검색모드끼리는 기존 evidence quote를 진단용으로 사용할 수 있다. 이때 DEV의
flat `evidence[]`는 atomic claim이 아니므로 `required_for_answer=true`인 항목만
분모로 삼고 지표명을 `Any-Required-Gold-Chunk@k`,
`All-Required-Gold-Chunks@k`, `RequiredGoldChunkRecall@k`로 제한한다.

### Phase C — HOLDOUT retrieval

Core 27문항에서 사전에 고정한 네 조건만 한 번 실행한다.

```text
Baseline + tuned BM25
Challenger + tuned BM25
C0: Cascade + untuned BM25
C1: Cascade + tuned BM25
```

Cascade C1이 파서 비교와 검색 튜닝 비교에 공통으로 사용되므로 총 108회다.
검색은 결정적이므로 반복하지 않는다. 현재 `/chat`에서는 `top_k`가 생성 문맥
수까지 결정하고 기본 최대값도 20이므로, 한 요청을 `top_k=50`으로 바꿔 두
지표를 섞지 않는다. 다음 snapshot을 분리한다.

```text
preselection candidate ranking@50 → candidate recall과 MRR@50
actual /chat final contexts@8     → Evidence Recall/All-Evidence@8과 생성
```

현재 `/search`는 role·temporal neighbor·dedupe가 빠진 다른 pipeline이므로
`/search@50`을 `/chat` 최종 순위처럼 사용하지 않는다. Eval trace에 `/chat`
내부 preselection 후보 50개를 별도로 저장하거나, 동일 내부 함수를 부르는
retrieval-only harness를 구현한다. 실제 생성 문맥은 언제나 top 8로 고정한다.

### Phase D — HOLDOUT generation

| 실험 | 계산 | 생성 출력 수 |
|---|---:|---:|
| C0/C1 실제 `/chat`, Core | 27 × 2 × 3 | 162 |
| C1 실제 `/chat`, Challenge | 9 × 3 | 27 |
| C1 generator + gold/oracle context, 영역별 Core 1개 | 9 × 1 | 9 |
| 합계 |  | 198 |

Oracle 실험은 retrieval을 우회하는 generator 진단이며 실제 서비스 성능에
합산하지 않는다. Challenge는 final C1의 안전성 평가이므로 C0에서 반복하지
않는다. 3회 반복은 질문 내부 변동성을 보기 위한 것이며 출력 수를 독립 표본
수로 주장하지 않는다.

생성 설정은 `max_output_tokens=900`, 최대 context 24,000자로 고정한다.
`gemini-3.5-flash-lite`가 temperature/seed를 지원하지 않는 현재 adapter에서는
해당 parameter를 보내지 않고 각각 `unsupported/not_sent`로 기록한다. 지원되는
다른 모델로 바뀌면 모델 변경 자체가 새 experiment이므로 C0/C1을 모두 다시
수집한다.

Provider 시간대 변화가 조건 효과와 섞이지 않도록 Core는 `case → run →
condition` 순으로 실행하고 seed `20260914`로 C0/C1의 AB/BA 순서를 균형 있게
교차한다. UTC timestamp와 실제 호출 순서를 저장한다.

Retry는 timeout, network error, HTTP 408/429/5xx에만 최대 2회 허용한다. 모든
최초 시도와 retry를 보존하며 낮은 점수나 빈약한 답변을 이유로 재실행하지 않는다.
Retry 이후에도 error, timeout 또는 빈 답변이면 GFC=0과 service error로 포함하고
대체 출력을 뽑지 않는다. Provider 전체 장애가 양 조건 합산 출력의 10%를 넘고
health check로 조건과 무관한 장애임이 확인된 경우에만 전체 paired batch를
`operational-outage`로 무효화하고 새 experiment ID로 양 조건을 모두 재실행한다.
무효 batch도 삭제하지 않고 reliability evidence로 남긴다.

## 5. 지표 정의

### 5.1 Parser

- Parse success rate
- Evidence preservation recall: 54개 anchor 중 정규화된 근거가 보존된 비율
- Critical-value preservation: 날짜·금액·시간·자격조건 보존율
- Table preservation: 지정 cell/row의 값과 관계가 보존된 비율
- Quality-gate decision 분포와 문서 형식별 실패율

청크 수나 파싱 성공 문서 수만으로 추출 정확도라고 부르지 않는다.
54개 anchor는 18개 문서 안에 nested되어 있으므로 parser 신뢰구간과 비교의
resampling 단위는 anchor 54개가 아니라 문서 18개다.

### 5.2 Retrieval

- Source Hit@1/3/5
- Preselection MRR@50: `/chat` 내부 최종선택 전 후보에서 첫 정답 source가
  top 50에 없으면 0
- Preselection Candidate Recall@50
- Evidence Recall@5/@8: 필수 atomic claim 중 top-k에서 근거를 찾은 비율
- All-Evidence@5/@8: 모든 필수 claim 근거가 top-k에 있는 문항 비율
- 신규 적중·상실 적중 문항
- retrieval p50/p95 latency

`@5`는 정확히 첫 5개만, `@8`은 생성기에 실제 전달된 8개만 검사한다. 기존
전체 context를 검사한 수치를 `Evidence Hit@5`라고 부르지 않는다. 실제 생성
가능성 진단의 우선 지표는 Evidence Recall@8과 All-Evidence@8이다.
Preselection @50과 final-context @8은 서로 다른 단계의 지표로 표와 이름을
분리한다.

### 5.3 Generation과 grounding

최종 시스템의 주지표는 Core 27문항의
**Grounded Full-Correct Rate(GFC)** 다. 출력 하나가 아래를 모두 만족할 때만
1, 아니면 0이다.

1. 모든 required atomic claim을 정확히 포함한다.
2. 답변의 factual claim이 공식 gold와 모순되지 않는다.
3. 모든 factual claim이 실제 retrieved context로 지지된다.
4. 각 citation이 해당 claim을 지지하는 passage를 가리킨다.
5. 잘못된 날짜·금액·조건이나 근거 없는 추가 사실이 없다.
6. 답변 가능한 질문을 과도하게 거절하지 않는다.

Judge와 사람은 서버가 미리 나눈 최대 8개 claim만 신뢰하지 않고 최종 답변
전체를 다시 atomic factual claims로 분해한다. 각 factual claim에는 최소 한 개의
실제 supporting citation이 있어야 한다. 붙은 citation 중 claim과 모순되는 것이
하나라도 있으면 GFC는 실패다. 단순히 관련이 약한 추가 citation은 Citation
Correctness를 낮추지만, 최소 한 개가 정확히 지지하고 오도·모순하지 않으면 그것
하나만으로 GFC를 실패시키지는 않는다.

부지표는 다음과 같다.

- Atomic Claim Recall
- Supported-Claim Precision
- Unsupported/Contradictory Critical Claim Rate
- Citation Correctness
- Citation Completeness
- 기존 0/1/2 correctness·completeness 점수와 분포
- Over-refusal Rate
- Challenge Correct Abstention Rate
- Injection Obedience Count — 목표 0건

Correctness와 faithfulness를 구분한다. 정답과 우연히 일치해도 검색 문맥이
지지하지 않으면 GFC가 아니다.

서비스의 규칙 기반 claim attribution guard는 GFC Judge를 대신하지 않는다.
`config/pnu-grounding-adversarial-eval.jsonl`의 합성·공지형 77건은
permission, comparator, direction, temporal boundary, scope와 modality의
명백한 반전을 막는 component 회귀 게이트로만 사용한다. 이 microbenchmark의
1.0을 실제 답변 정확도나 일반화 성능으로 보고하지 않는다. 이는 일반 NLI가
아니며 의무·필요 관계(예: `제출해야 한다`와 `제출할 필요가 없다`)는 현재
표적 규칙의 범위 밖이다.

### 5.4 운영 지표

- 실제 client 기준 end-to-end p50/p95
- retrieval, generation, postprocessing 단계별 시간
- 입력·출력 token과 추정 비용
- provider/model별 시도, fallback, retry, timeout, error rate
- 응답에 남은 claim 수와 후처리 삭제 claim 수

현재 저장된 provider attempt 시간만으로 end-to-end latency라고 부르지 않는다.

## 6. LLM judge와 사람 평가

### 6.1 Structured pointwise judge

Judge는 질문·기준답안·최종 답변만 보지 않고 다음 입력을 모두 받는다.

- 질문과 역할
- required atomic claims와 critical values
- 공식 gold evidence
- 실제 retrieved contexts와 순서
- 최종 claims, citations, cited answer
- answerable/challenge label

시스템 이름 C0/C1은 숨긴다. 출력 JSON에는 claim별 포함·정확성·근거 지지,
citation support, contradiction, unsupported fact, abstention 적절성,
`uncertain`을 따로 기록한다. 추가 사실은 “기준답과 모순되지 않는다”는 이유만으로
통과시키지 않고 실제 검색 근거가 있어야 한다.

Judge 모델·버전·temperature·prompt SHA-256을 고정한다. 현재 기본 judge를
사용할 경우 `gemini-3.1-flash-lite`, temperature 0.0으로 기록한다. 생성과
judge는 별도 프로세스로 실행한다. 먼저 198개 답변을 immutable raw artifact로
저장한 다음에만 judge를 실행한다. 첫 채점은 198개 출력에 한 번씩 수행하고,
층화한 고정 답변 18개는 동일 judge로 두 번 더
채점해 반복 안정성을 측정한다. 재채점은 원본 JSONL을 덮어쓰지 않고 새 파일에
append한다. 필수 judge 호출은 최초 198회와 반복 36회, 총 234회다.
답변과 판정은 다음 고유 key로 연결한다.

```text
answer_id     = SHA256(experiment_id + condition_id + generation_run_id + case_id)
judgment_id   = SHA256(experiment_id + answer_id + judge_run_id)
answer_sha256 = immutable final answer hash
```

Judge run별 별도 JSONL을 사용하고 manifest가 `answer_id` 중복과
`judgment_id` 누락을 검사한다.

필수 외부 LLM 호출은 generator 198회 + judge 234회 = 432회다. 기술적 retry는
이 수에 포함하지 않으며 별도로 센다.

### 6.2 사람 검수

C0/C1 Core의 사전에 지정한 첫 번째 반복 54개 출력(27문항 × 2조건)과 C1
Challenge 첫 반복 9개, 총 63개 출력을 두 사람이 조건명을 보지 않고 독립
평가한다. 본평가 전에 DEV 답변 8개로 atomic claim 경계, entailment, citation
support, 부분정답, 과도한 거절과 critical error rubric을 함께 calibration한다.
이 8개는 본평가 agreement에 포함하지 않는다. 본평가 불일치는 합의 판정하며
원래 두 label과 최종 adjudication을 모두 남긴다.
본평가 workload는 63출력 × 2인 = 126 ratings다.

- 사람 대 사람: binary raw agreement·Cohen's kappa, 0/1/2 weighted kappa
- Judge 대 adjudicated human: accuracy, balanced accuracy, Cohen's kappa,
  macro-F1, confusion matrix
- Challenge: correct abstention과 injection obedience agreement를 Core와 별도 계산

LLM judge 결과를 headline n=3 집계에 사용하기 위한 gate는 **Core 54개에서
Judge와 adjudicated human을 비교한 값**으로 판정한다. 기준은 raw agreement
≥ 0.80, balanced accuracy ≥ 0.80, Cohen's kappa ≥ 0.60, macro-F1 ≥ 0.75다.
Balanced accuracy는 adjudicated-human GFC 통과·실패가 모두 존재할 때만
정의하며, 한 class라도 없으면 gate를 통과시키지 않는다. Human label을 보고
judge prompt를 수정하면 이 calibration은 무효이며 새 calibration 표본이
필요하다.

Gate를 통과하면 3회 judge GFC를 질문별로 집계한다. 하나라도 충족하지 못하면
LLM judge n=3 결과는 exploratory로 낮추고, 사람 판정 run 1의 C0/C1 paired
GFC(n=27)를 주 결과로 제시해 exact McNemar와 paired proportion CI를 계산한다.
Judge의 모든 `uncertain`, critical-value 불일치, injection 성공 사례는 표본과
무관하게 사람이 확인한다.

다른 모델 계열 judge 및 A/B·B/A pairwise 평가는 시간이 남을 때의 보조 분석이며
주지표를 대신하지 않는다.

## 7. 통계 분석

Judge-human gate를 통과한 경우 Core 각 질문과 조건에서 먼저 다음 값을 만든다.

```text
question_gfc = 3회 중 GFC 성공 횟수 / 3
paired_effect = mean(question_gfc_C1 - question_gfc_C0)
```

- 분석 단위는 출력 162개가 아니라 질문 family 27개다.
- GFC 평균 차이는 질문 family paired bootstrap 10,000회 95% CI와 paired
  sign-flip permutation을 보고한다.
- 민감도 분석은 3회 중 2회 이상 GFC인 문항을 성공으로 보고 exact McNemar
  test를 사용한다.
- Source Hit과 All-Evidence 같은 paired binary 지표는 exact McNemar를 쓴다.
- MRR과 Evidence Recall 차이는 paired bootstrap 95% CI를 보고한다.
- 세 파서와 Dense/Hybrid 다중 비교는 exploratory로 표기한다. 확증적으로
  해석해야 한다면 Holm 보정을 적용한다.
- 평균·절대 차이·95% CI를 항상 함께 제시한다. CI가 0을 포함하면
  “향상 경향” 또는 “불확실”로 쓰고 “성능 향상 확정”이라고 쓰지 않는다.

Gate 실패 시에는 사람 판정 run 1의 27개 paired binary 결과만 주 분석으로
사용한다. 이때 exact McNemar와 paired proportion difference의 exact 또는
bootstrap 95% CI를 보고하고, judge 기반 3회 분석은 exploratory 부록으로
내린다.

표본 27개로 작은 차이를 확증하기 어렵다는 사실은 연구 한계로 명시한다.

## 8. 재현성 및 실행 규칙

최종 실행 전에 다음을 모두 만족해야 한다.

1. Holdout 실행 전 Git worktree가 clean인 code-freeze commit/tag를 기록한다.
2. corpus/index, DEV, holdout, prompt, evaluator의 SHA-256을 기록한다.
3. 서버 실행 명령과 `parser_profile/retrieval_mode/top_k/provider/model`을
   모든 요청 record에 반복 저장한다.
4. raw model draft, postprocessed answer, claims, citations, retrieved contexts와
   순서, role mapping, generation attempts를 보존한다.
5. 모델이 반환하면 token usage를 저장하고, 없으면 `unavailable`로 명시한다.
6. API·judge 오류도 record로 남기며 성공한 결과로 조용히 대체하지 않는다.
7. 재실행은 새 generation run 파일에, rejudge는 새 judge run 파일에 쓰고 기존
   answer/judgment 결과를 수정하지 않는다.
8. 모든 실행은 expected ID, 중복, 오류, model 혼합을 검사한 뒤 manifest를
   `--require-clean` 조건으로 생성한다.
9. 결과 표와 보고서는 JSONL에서 스크립트로 생성하며 숫자를 수동 전사하지
   않는다.

Code-freeze 직전에는 holdout이 아닌 DEV 3문항으로 양 서버와 generation/judge
smoke를 수행한다. Index·tuning·provider/model 확인, raw draft·contexts·claims·
citations·prompt hash 저장, 조건 내부 retrieval 결정성, fail-closed, append-only를
전부 확인한 뒤 clean tag를 만든다. 9월 9일에는 코드 tag와 구분되는
result-freeze tag로 최종 artifact와 manifest를 고정한다.

### 8.1 실행 토폴로지와 fail-closed 검사

BM25 tuning은 요청 옵션이 아니라 서버 시작 설정이므로 동일 commit과 index로
두 인스턴스를 사용한다.

```text
127.0.0.1:8100 = tuning ON
127.0.0.1:8101 = --no-retrieval-tuning
```

두 서버 모두 PNU Cascade 정본을 `--index`로, Baseline과 Challenger 인덱스를
각각 `--profile-index`로 명시한다. 서버 시작 전후 `/health`에서 index revision,
profile mapping, service tuning, 문서당 청크 cap, eval-trace 활성화를 확인한다.
이 필드는 2026-09-01 구현됐으며 collector가 기대값과 다르면 시작 전에
실패한다. 인자 없는
`search_api.py` 기본 인덱스는 사용하지 않는다.

Evaluator는 최소 다음 인자를 받아 요청 body와 결과 record에 저장해야 한다.

```text
--experiment-id --condition-id --run-id
--parser-profile --retrieval-mode
--context-k 8 --provider --model
--expected-corpus-revision REVISION
--expected-retrieval-tuning on|off
--expected-context-chunks-per-document N
--eval-trace
```

최종 서비스 요청은 `institution=null`로 고정해 현재 UI 기본 경로와 역할
라우팅을 그대로 통과시킨다. `--profile-index`는 반환 chunk·corpus 검증과 전체
context 복원에, `--run-dir`은 parse success·quality decision·selected parser와
reading-order 진단에 사용한다. SQLite index만으로 parser 품질을 추정하지 않는다.

각 응답에서 요청/반환 parser profile과 retrieval mode, 기대 tuning, corpus
revision, chunk가 속한 profile index를 검사한다. 하나라도 다르면 조용히 계속하지
않고 해당 run을 실패 처리한다. `context-top-k=8`과 지표의 `@5`는 독립적으로
계산하며, 지표를 맞추기 위해 실제 생성 문맥 수를 바꾸지 않는다.

`RAG_EVAL_TRACE`는 2026-09-01 구현했다. 평가에서만
`RAG_EVAL_TRACE=1`을 사용해 raw draft, 정확한 generation prompt와 hash,
provider request-config hash, 단계별 전체 context를 저장한다.
일반 서비스 응답에는 이 필드를 노출하지 않는다. Raw draft와 최종 답변의 차이로
generator 실패와 postprocessor 삭제를 분리한다.

### 8.2 필수 구현 기능과 현재 상태

새 파일 개수는 완료 조건이 아니다. 기존 스크립트를 확장하거나 공용 모듈을
만들어 다음 기능을 충족하면 된다.

| 기능 | 상태 | 현재 근거 또는 남은 일 |
|---|---|---|
| v2 holdout/gold 검증, NFKC·fuzzy·table evidence matcher | 구현·기계 gate 완료 | draft 36문항·93 evidence option, schema·DEV·corpus pre-review PASS; SHA `2fbedad63531491d3c069a5de6fe51fff623f7e571b9170c6a9aa7452e2d51f9`, 사람 판정은 미실행 |
| 독립 2인 holdout sign-off handoff | 구현·로컬 검증 완료 | read-only packet과 SHA-bound A/B 응답, strict merge, 동일 2인 roster 및 packet/response 원본 재검증; 실제 응답은 PENDING |
| 세 profile의 document/anchor 보존 평가 | 구현·표적 audit 완료 | 18문서·54 anchor 평가 완료; 독립 2인 원문 육안검수는 미완료 |
| profile×BM25와 Cascade mode retrieval matrix·paired 변화 | 구현·DEV 실행·final 합성 검증 완료 | DEV 8조건×45=360과 final Core 27×4=108 scheduler/analyzer를 검증; 합성 108-slot/216-WAL E2E 완료, 실제 holdout 결과 아님 |
| 명시 설정과 전체 `/chat` trace를 저장하는 generation collector | 구현·DEV 외부 preflight·final 합성 검증 완료 | `p0g` C0/C1 각 3문항 수집; final 189-slot runner는 model/source/index/Git/WAL/run-lock을 고정하며 실제 holdout은 미실행 |
| generation과 분리된 structured claim/citation Judge | 구현·DEV v8 반복 완료 | C0/C1 각 3회, 모든 문항 3/3 일치; same-model 사람 calibration은 미완료 |
| 질문 단위 GFC paired 통계 | 구현·DEV·final 합성 검증 완료 | generation run 3개를 질문 `n=27` 내부 반복으로 결합하고 terminal GFC=0, calibration 실패 human fallback, paired bootstrap/sign-flip/majority McNemar를 검증; final raw는 미생성 |
| condition-blind 사람 평가와 Judge-human calibration | 구현·로컬 검증 | blind mapping·Reviewer A/B·합의 템플릿과 agreement/balanced accuracy/kappa/macro-F1/false-pass gate 구현; 실제 사람 라벨은 미생성 |
| retrieved/oracle context 진단 | 구현·로컬 검증 | 같은 generator/postprocessor에 gold quote만 주입하는 Core 9영역 수집기; 실제 9개는 signoff 뒤 미생성 |
| 조건·model·prompt·corpus·case ID final manifest | 구현·로컬 검증 | answer–judgment·partial-selection ID/hash와 no-clobber authority-last publication 검증; 최종 raw 미생성 |
| immutable raw summary·visual review builder | 구현·DEV·합성 검증 완료 | terminal-aware Judge summary, holdout gold 검토 패킷, condition-blind 답변 검토 패킷 구현; final 답변 기반 packet은 미생성 |

위 표는 구현·실행 현황만 갱신한다. 최종 생성 조건의
`model=gemini-3.5-flash-lite`와 holdout one-shot 설계는 그대로 유지하며,
`gemini-3.1-flash-lite` `p0g`는 n=3 DEV preflight로만 해석한다.

비용 계산은 가격을 코드에 하드코딩하지 않고 평가일과 출처를 기록한
`config/model-pricing-YYYYMMDD.json` snapshot을 사용한다. Provider가 usage를
반환하지 않으면 비용을 추측하지 않고 호출 수와 `usage=unavailable`만 보고한다.

필수 저장 필드는 다음과 같다.

```text
experiment_id, condition_id, generation_run_id, judge_run_id,
answer_id, judgment_id, answer_sha256, case_id, family_id,
git_commit, dirty, corpus_sha256, index_sha256, cases_sha256,
parser_profile, retrieval_mode, retrieval_tuning, institution, top_k,
provider_requested, provider_used, model_requested, model_used,
temperature, seed, max_output_tokens, max_context_chars,
prompt_sha256, postprocessor_revision, utc_timestamp, call_order,
preselection_candidates, retrieved_contexts, raw_draft, final_answer, claims, citations,
retrieval_ms, generation_ms, postprocess_ms, end_to_end_ms,
input_tokens, output_tokens, estimated_cost, fallback, retries, error
```

## 9. 주장 판정 규칙

- 검색 개선: holdout의 paired Evidence Recall/All-Evidence/Source Hit 차이와
  95% CI를 근거로 쓴다.
- 생성 개선: 사람 검증 gate를 통과한 GFC paired effect와 CI를 근거로 쓴다.
- CI가 0을 포함하면 “관찰된 증가” 또는 “결론 불확실”로 제한한다.
- 검색이 개선되고 생성이 불확실하면 두 결론을 그대로 분리해 쓴다.
- Prompt injection을 실제로 따르거나 잘못된 날짜·금액·자격조건을 단정한
  사례는 평균에 묻지 않고 critical failure로 별도 공개한다.
- DEV, parser pilot, grant track 수치는 모두 해당 범위를 이름에 붙인다.

## 10. 이번 제출에서 제외하는 범위

- 3파서 × 5검색모드 전체 생성 factorial
- Baseline/Challenger Dense/Hybrid의 generation 비교와 전체 factorial
- Cross-encoder hyperparameter sweep과 생성 모델 fine-tuning
- 현재 API가 history를 전송하지 않는 상태의 멀티턴 성능 주장
- 대규모 사용자 연구와 동시접속 load test
- 새 exact-host 전체 재크롤을 평가 완료의 선행조건으로 두는 것
- 연구비 53문항 결과를 PNU 서비스 headline으로 사용하는 것
- holdout 결과를 본 뒤 같은 holdout에 재튜닝하는 것

## 11. 필요한 산출물

```text
config/pnu-service-answer-eval.jsonl
config/pnu-service-answer-holdout-v2.jsonl
config/pnu-parser-audit-v2.jsonl
processed/eval/final-20260909/retrieval/*.jsonl
processed/eval/final-20260909/generation/*.jsonl
processed/eval/final-20260909/judge/*.jsonl
processed/eval/final-20260909/human/*.jsonl
evidence/20260914/final/summary.json
evidence/20260914/final/manifest.json
evidence/20260914/final/visual-review.html
config/pnu-grounding-adversarial-eval.jsonl
evidence/20260914/grounding-guard-ablation.json
evidence/20260914/grounding-guard-ablation.csv
evidence/20260914/cap2-vs-cap4-retrieval-dev45.html
```

기존 형식의 DEV45는 flat evidence 구조를 유지하되 질문 직접 요구 여부만
`required_for_answer`로 구분한다. 개정 전 byte-identical 원본은
`config/pnu-service-answer-eval-v1.jsonl`에 보존한다. 45개 전체를 holdout v2의
atomic-claim schema로 이관하는 일은 필수 범위에서 제외한다.
Profile-independent atomic gold는 신규 `holdout-v2`와 parser audit에 필수
적용한다.

## 12. 일정과 완료 조건

| 날짜 | 완료 조건 |
|---|---|
| 9/1 | evaluator의 @5 오류, 명시적 요청 설정, trace 저장, gold schema 구현 |
| 9/2–3 | Core 27 + Challenge 9, Parser audit 18문서 작성·교차 검수·hash 동결 |
| 9/4 | DEV 8조건·parser audit·smoke 완료, clean code-freeze tag |
| 9/5 | HOLDOUT 4조건 retrieval one-shot, generation 시작 |
| 9/6 | generation 198개와 structured judge 234회 완료 |
| 9/7 | 사람 2인 blind 평가 완료 |
| 9/8 | adjudication·통계·실패 분석, 장애 복구 buffer |
| 9/9 | final manifest, 수치·artifact result-freeze tag |
| 9/10–13 | 보고서 표·그래프·한계·재현 명령 작성 및 검수 |

평가 단계 완료의 정의는 단순히 스크립트가 끝난 것이 아니다. 데이터와 설정이
동결되고, 필수 record가 누락 없이 수집되며, 사람 검증을 포함한 통계표와 manifest가
clean commit에서 재생성되어야 완료다.
