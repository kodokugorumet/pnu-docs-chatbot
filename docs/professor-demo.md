# 교수님 데모 실행 가이드

이 문서는 부산대학교 공문서 RAG 챗봇의 교수님 시연용 기준 구성을
재현하기 위한 실행 순서입니다. 데모에서는 검증되지 않은 실험 구성을 섞지
않고, 24개 검색 평가 질문을 모두 통과한 BM25 단일 검색 경로를 사용합니다.

## 데모 기준 상태

- 문서: 2,247개
- 검색 chunk: 44,520개
- 검색 평가: Hit@5 24/24(100%)
- MRR: 0.84861111
- BM25 인덱스:
  `processed/index/pnu-20260725-curated-cascade-v5-allow-suspect.sqlite`
- 평가 결과:
  `processed/eval/20260728-pnu-cascade-demo-ranking.json`

세 파서 프로필을 같은 gate 조건으로 비교한 결과는 다음과 같습니다.

| 검색 파싱 버전 | 문서 | chunk | Hit@5 | MRR |
|---|---:|---:|---:|---:|
| Baseline | 2,243 | 51,161 | 23/24 | 0.8854 |
| Challenger | 2,245 | 43,566 | 24/24 | 0.8417 |
| Cascade | 2,247 | 44,520 | 24/24 | 0.8486 |

UI에서 질문마다 세 버전을 바꿔 검색할 수 있으며, 답변 아래 배지에 실제로
사용한 버전이 표시됩니다. 기본값은 Cascade입니다.

로컬 hashing Dense 인덱스도 별도로 시험했지만 같은 24개 질문에서
Hit@5가 20/24로 낮아졌습니다. 따라서 이번 데모에서는 Dense와 RRF를
의도적으로 끄고, 평가 결과가 더 좋은 BM25 문서 다양성 검색을 사용합니다.

## 파이프라인의 각 단계

### 1. 문서 파싱

PDF, HWP/HWPX, HTML, Office 문서에서 제목, 본문, 표, 페이지와 섹션 위치를
공통 Block 형식으로 꺼냅니다. `cascade` 프로필은 기본 파서의 결과가
불충분할 때 구조 파서나 OCR을 순서대로 시도합니다.

이 단계의 결과는 아직 답변이 아니라, 서로 다른 파일 형식을 같은 방식으로
검색할 수 있게 만든 구조화 데이터입니다.

### 2. 품질 검사와 corpus gate

빈 문서, 파싱 오류, 지나치게 손상된 텍스트처럼 검색에 넣으면 안 되는
문서를 차단합니다. 이번 데모 인덱스는 자동 검사에서 `pass`인 문서와,
본문은 정상이나 다국어 비율·목록 기호·일부 빈 페이지 때문에
`suspect`로 분류된 문서를 포함합니다.

파싱 실패와 hard-fail 문서는 여전히 제외됩니다. 즉, `allow-suspect`는
모든 파일을 무조건 허용하는 옵션이 아닙니다.

### 3. 청킹

긴 문서를 검색 가능한 작은 단위인 chunk로 나눕니다. 각 chunk에는 원래
문서 ID, 제목, 기관, 페이지·섹션 위치와 원문 URL을 함께 보존합니다.

이 정보가 있어 검색 결과를 답변 근거로 사용하고, 사용자가 출처를 눌렀을
때 원래 문서와 위치를 보여줄 수 있습니다.

### 4. BM25 문서 검색

질문의 핵심 단어와 문서 제목·파일명·본문이 얼마나 잘 맞는지를 SQLite
FTS5 BM25로 계산합니다. 한 긴 문서의 비슷한 chunk가 결과를 독점하지
않도록 문서 하나당 최대 두 chunk만 우선 선택합니다.

최종 검색 평가는 정답 문서가 상위 5개 안에 있는지를 측정합니다.
현재 기준은 24개 질문 모두 성공했고, 평균 역순위인 MRR은 0.8486입니다.

### 5. 답변 생성

검색된 근거만 Gemini에 전달해 사용자의 질문에 맞는 문장으로 정리합니다.
Gemini를 사용할 수 없으면 추출형 안전 응답으로 전환할 수 있습니다.

생성 모델은 새로운 사실을 찾는 검색기가 아니라, 이미 검색된 근거를 읽기
좋게 정리하는 역할만 맡습니다.

### 6. 문장별 근거 검증과 출처 표시

생성된 답변을 문장 단위 claim으로 나누고 검색 원문이 각 문장을 지지하는지
다시 확인합니다. 확인된 문장에는 `[1]`, `[2]` 같은 출처 번호를 붙이고,
오른쪽 근거 패널에서 문서명, 원문 일부와 위치를 보여줍니다.

이 구조는 답변 문장과 실제 근거를 교수님 앞에서 바로 비교하기 위한
장치입니다.

## 시연 전 점검

프로젝트 디렉터리로 이동합니다.

```bash
cd /Users/leehyunwoo/project/pnu-docs-chatbot
```

전체 단위 테스트와 프론트엔드 정적 검사·빌드를 실행합니다.

```bash
python3 -m unittest discover -s tests
bun run lint
bun run build
```

검색 평가 결과를 확인합니다.

```bash
python3 -c 'import json; p=json.load(open("processed/eval/20260728-pnu-cascade-demo-ranking.json")); print(p["indexes"][0]["metrics"])'
```

출력에서 `hits`가 24, `total_cases`가 24, `mrr`이
`0.84861111`인지 확인합니다.

## 서버 실행

터미널 1에서 API를 실행합니다. `--dense-index`에는 존재하지 않는
데모 전용 경로를 지정해 Dense가 나중에 우연히 활성화되지 않도록 합니다.

```bash
python3 scripts/search_api.py \
  --host 127.0.0.1 \
  --port 8000 \
  --env-file .env \
  --index processed/index/pnu-20260725-curated-cascade-v5-allow-suspect.sqlite \
  --profile-index baseline=processed/index/pnu-20260725-curated-baseline-v5-allow-suspect.sqlite \
  --profile-index challenger=processed/index/pnu-20260725-curated-challenger-v5-allow-suspect.sqlite \
  --default-parser-profile cascade \
  --dense-index processed/index/bm25-only.demo-disabled
```

다른 터미널에서 상태를 확인합니다.

```bash
curl -s http://127.0.0.1:8000/health
```

응답에서 다음 값을 확인합니다.

- `ready`: `true`
- `status`: `ready`
- `chunk_count`: `44520`
- Dense 단계: `disabled`
- RRF 단계: `single_lane`

터미널 2에서 프론트엔드를 실행합니다.

```bash
bun run dev -- --host 127.0.0.1
```

브라우저에서 `http://127.0.0.1:5173`을 열고 다음처럼 설정합니다.

- 답변 생성: `자동 선택`
- 검색 파싱 버전: `Cascade · 품질 기반 선택`
- 검색 기관: `부산대학교`
- 검색 근거 범위: `균형 8개`

Gemini 연결이 불안하면 API를 종료하고 다음처럼 추출형 fallback을
우선하도록 다시 실행할 수 있습니다.

```bash
RAG_AUTO_PROVIDER_ORDER=extractive python3 scripts/search_api.py \
  --host 127.0.0.1 \
  --port 8000 \
  --env-file .env \
  --index processed/index/pnu-20260725-curated-cascade-v5-allow-suspect.sqlite \
  --profile-index baseline=processed/index/pnu-20260725-curated-baseline-v5-allow-suspect.sqlite \
  --profile-index challenger=processed/index/pnu-20260725-curated-challenger-v5-allow-suspect.sqlite \
  --default-parser-profile cascade \
  --dense-index processed/index/bm25-only.demo-disabled
```

추출형 응답은 인터넷 없이 동작하지만 문장 품질이 낮으므로, 실제 시연 전에
Gemini 경로로 질문 하나를 보내 예열하는 편이 좋습니다.

## 안정적인 시연 질문

첫 질문은 HTML 공지 검색과 날짜·시간·진행 방식 확인에 적합합니다.

```text
2026년 4월 15일 해외취업 온라인 세미나의 시간과 진행 방식을 알려줘
```

확인할 내용:

- 2026년 4월 15일 수요일
- 오후 6시 30분
- 온라인 Zoom 진행
- 답변 문장 3개가 모두 근거 검증됨

두 번째 질문은 PDF 기반 신청 절차 검색에 적합합니다.

```text
2026학년도 2학기 수료후연구생 신청 경로와 등록금 납부 기간을 알려줘
```

확인할 내용:

- `학생지원시스템 → 학적 → 학생신청 → 수료후연구생 신청`
- 지도교수 추천서 첨부
- 근거에 납부 기간이 없으면 없다고 명시하는 안전한 응답

등록금 분할납부 날짜 질문은 원문의 날짜 표기가 chunk 경계에서 일부
잘리는 사례가 확인됐으므로 이번 시연에서는 사용하지 않습니다.

## 3분 설명 순서

### 0:00–0:30 — 문제

> 부산대학교 행정 정보는 여러 홈페이지와 PDF, HWP에 흩어져 있어서
> 학생이 정확한 문서를 찾기 어렵습니다. 이 서비스는 수집한 공문서를
> 검색하고, 검색 근거만 사용해 답변하는 RAG 챗봇입니다.

### 0:30–1:00 — 현재 데이터와 검색 성능

> 현재 2,247개 문서를 cascade 파서로 처리해 44,520개 검색 단위로
> 만들었습니다. 오늘은 재현성이 검증된 BM25 단일 검색을 사용합니다.
> 24개 수동 검증 질문에서 정답 문서를 모두 상위 5개 안에 찾았고,
> MRR은 0.8486입니다.

### 1:00–2:20 — 실제 질문과 근거 확인

질문을 입력한 뒤 답변의 출처 번호를 누르고 오른쪽 근거 패널을 엽니다.

> 답변만 보여주는 것이 아니라 어떤 문서의 어떤 내용을 사용했는지 함께
> 제공합니다. 생성된 문장도 검색 원문과 다시 비교해 지지되는 문장에만
> 출처 번호를 붙입니다.

### 2:20–3:00 — 평가 범위와 다음 단계

> 24/24는 문서 검색 평가 결과이며 모든 질문의 최종 답변 정확도를
> 의미하지는 않습니다. 다음 단계는 평가 질문을 늘리고 날짜·대상·절차의
> 답변 정확도와 출처 정확도를 별도로 측정하는 것입니다. 이후 한국어
> embedding Dense 검색도 BM25 기준선보다 실제로 개선되는 경우에만
> 하이브리드 경로에 연결하겠습니다.

## 종료

프론트엔드 터미널에서 `Ctrl+C`를 누른 뒤 API 터미널에서도 `Ctrl+C`를
누릅니다.

포트가 정리됐는지는 다음 명령으로 확인할 수 있습니다.

```bash
lsof -nP -iTCP:5173 -sTCP:LISTEN
lsof -nP -iTCP:8000 -sTCP:LISTEN
```
