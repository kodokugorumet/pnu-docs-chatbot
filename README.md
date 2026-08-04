# 공문서 RAG 챗봇

부산대학교와 주요 공공기관의 행정 문서를 검색하고, 검색된 근거를 바탕으로 답변을 생성하는 졸업과제용 RAG 챗봇입니다.

현재는 로컬 문서 파싱, 검증된 다기관 corpus gate, BM25 + Dense/RRF
하이브리드 검색, 공급자 선택형 답변 생성, React 채팅 UI까지 연결된 MVP
상태입니다.

교수님 시연에 사용하는 검증된 실행 순서와 발표 멘트는
[교수님 데모 실행 가이드](docs/professor-demo.md)에 정리되어 있습니다.

## 현재 구현 상태

- 기관별 문서 수집 데이터 정리
- 문서 파싱 및 청킹 파이프라인
- SQLite FTS5 기반 BM25 문서 다양성 검색과 선택형 로컬 hashing Dense 검색
- Dense를 사용할 때 reciprocal-rank fusion(RRF)과 hybrid reranking
- 로컬 OpenAI 호환 API, 프론티어 AI(Gemini API), 추출형 fallback 답변 생성
  - Gemini 3.1/3.5 직접 선택과 선택 모델 우선 fallback
- claim 단위 근거 검증 및 출처 번호 표시
- React + Vite + TypeScript 프론트엔드
- 답변 생성 단계 표시
  - 문서 검색
  - 답변 생성
  - 근거 검증
- 근거 패널
  - 검증
  - 문서
  - 위치

## 대상 기관

현재 로컬 데이터 기준으로 다음 기관 문서를 다룹니다.

- 금융감독원
- 부산대학교
- 한국거래소
- 한국예탁결제원
- 한국은행
- 한국인터넷진흥원(KISA)
- 한국해양과학기술원

## 부산대학교 홈페이지 크롤링

크롤러는 [config/pnu-crawl-scope.json](config/pnu-crawl-scope.json)에
명시된 정확한 호스트만 방문합니다. 중앙 학사·입학·취업·기숙사·국제·학생
지원 사이트와 확인된 단과대학·학과 사이트가 대상이며, 연구소·박물관·언론·
기록관 사이트는 범위에서 제외합니다. 이미지·동영상·압축파일·실행파일은
응답 본문을 읽기 전에 헤더에서 차단하고, 확장자가 숨겨진 generic binary
응답은 허용 문서 파일명이 확인될 때만 저장합니다.

기존 `downloads/pnu-web-crawl`은 과거의 전체 하위 도메인 규칙으로 수집한
원본이므로 그대로 보존합니다. 새 제한 규칙은 별도 출력 디렉터리에서
시작해야 합니다. 먼저 소규모 dry-run으로 확인합니다.

```bash
python3 scripts/crawl_pnu_site.py \
  --output downloads/pnu-web-crawl-scope-check \
  --dry-run \
  --max-pages 10 \
  --max-files 10 \
  --max-depth 2
```

dry-run DB에는 본문이 저장되지 않으므로 실제 수집에 재사용할 수 없습니다.
실제 수집은 새 출력 경로에서 시작합니다. 이후 그 실제 수집 명령을 중단 후
다시 실행해도 페이지·첨부·바이트 한도와 호스트별 한도는 SQLite의 누적 완료
수를 기준으로 계산됩니다.

```bash
python3 scripts/crawl_pnu_site.py \
  --output downloads/pnu-web-crawl-scoped-v1 \
  --max-depth 5 \
  --delay 1.0
```

기본 안전 한도는 HTML 4,000건, 첨부 2,000건, 합계 1.5GiB, 응답 하나당
25MiB입니다. scope 파일의 SHA-256이 SQLite에 고정되므로, 범위를 바꾼 뒤
같은 DB에 이어 붙이는 실행은 거부됩니다.

결과 구조는 다음과 같습니다.

```text
downloads/pnu-web-crawl-scoped-v1/
├─ content/부산대학교/
│  ├─ 웹페이지/            # 파서에 투입할 HTML 원문
│  └─ 첨부파일/            # PDF, HWP, HWPX, Office 문서
├─ state/
│  ├─ crawl.sqlite3        # 재개 가능한 URL 큐
│  ├─ crawl.lock           # 같은 출력 경로의 중복 실행 방지 잠금
│  ├─ pages.jsonl          # 페이지별 원 URL·제목·체크섬
│  └─ attachments.jsonl    # 첨부파일별 원 URL·저장 경로·체크섬
└─ summary.json
```

중간에 중단해도 같은 명령을 다시 실행하면 남은 URL부터 이어집니다. 현재
누적 상태는 다음 명령으로 확인할 수 있습니다.

같은 출력 경로에 두 크롤러를 동시에 실행하면 두 번째 실행은 즉시 거부됩니다.
`--status`는 DB를 읽기 전용으로 열기 때문에 처리 중인 URL을 재큐잉하지
않으며, `fetching` 상태의 복구는 기존 크롤러 잠금이 해제된 뒤 새 크롤러가
독점 잠금을 얻었을 때만 수행됩니다.

```bash
python3 scripts/crawl_pnu_site.py \
  --output downloads/pnu-web-crawl-scoped-v1 \
  --status
```

추가 사이트는 `--allow-domain`으로 정확한 호스트를 지정하고 `--seed`로
시작 URL을 추가합니다. 두 옵션 모두 반복할 수 있으며, 하위 도메인 suffix
전체를 암묵적으로 허용하지 않습니다.

파서에는 전체 `content` 디렉터리를 바로 넘기지 않습니다. SQLite 완료 행,
확장자·MIME, 실제 파일 크기와 SHA-256, exact-host 범위, 학과 공지 관련도를
검증한 zero-copy manifest를 먼저 만듭니다. 이 단계에서도 scope의 페이지·
첨부·바이트 예산과 tier별 호스트 상한을 다시 검사합니다. 예전 수집물이
호스트 상한을 넘으면 첨부 원문을 우선 보존하며, HTML의 canonical URL과
파일 SHA-256을 차례로 적용해 URL 변형 및 파일 중복을 제거합니다.

```bash
python3 scripts/curate_pnu_corpus.py \
  --crawl-root downloads/pnu-web-crawl \
  --scope-file config/pnu-crawl-scope.json \
  --output-dir processed/curation/pnu-YYYYMMDD-v1
```

완료된 기존 baseline은 재파싱하지 않고 선별 manifest에 맞는 새 run으로
파생할 수 있습니다. 원 run은 읽기 전용으로 검증되며 변경되지 않습니다.

```bash
python3 scripts/derive_curated_run.py \
  --source-run processed/runs/ORIGINAL_RUN/baseline \
  --curated-manifest processed/curation/pnu-YYYYMMDD-v1/curated-manifest.jsonl \
  --output-dir processed/runs/PNU_CURATED_BASELINE/baseline \
  --run-id PNU_CURATED_BASELINE
```

challenger와 cascade는 동일한 manifest를 authoritative input으로 각각
새로 실행합니다. `--corpus-manifest`는 `--institution`, `--extensions`,
`--limit`과 함께 사용할 수 없습니다.

```bash
.parser-tools/venvs/core/bin/python scripts/parse_pipeline.py run \
  --profile challenger \
  --input downloads/pnu-web-crawl/content \
  --corpus-manifest processed/curation/pnu-YYYYMMDD-v1/curated-manifest.jsonl \
  --output processed/runs \
  --run-id PNU_CURATED_CHALLENGER \
  --workers 1 \
  --expect-korean \
  --allow-missing

.parser-tools/venvs/core/bin/python scripts/parse_pipeline.py run \
  --profile cascade \
  --input downloads/pnu-web-crawl/content \
  --corpus-manifest processed/curation/pnu-YYYYMMDD-v1/curated-manifest.jsonl \
  --output processed/runs \
  --run-id PNU_CURATED_CASCADE \
  --workers 1 \
  --expect-korean \
  --allow-missing
```

## 프로젝트 구조

```text
.
├─ src/
│  ├─ App.tsx              # 채팅 UI
│  ├─ App.css              # 화면 스타일
│  └─ data/                # 기관별 원문 문서
├─ scripts/
│  ├─ crawl_finance_docs.py # 금융권 문서 크롤러
│  ├─ crawl_pnu_site.py    # 부산대 본문·학과·첨부파일 크롤러
│  ├─ parse_documents.py   # 문서 파싱 및 청킹
│  ├─ parse_pipeline.py    # 3개 파서 프로필 실행·진단·검증 CLI
│  ├─ document_parsing/    # 공통 Block 스키마, 어댑터, 품질·출력 계층
│  ├─ bm25_search.py       # 다기관 BM25와 로컬 Dense 인덱스
│  ├─ rag/                 # corpus gate, Dense/RRF, 생성기, 응답 계약
│  └─ search_api.py        # 로컬 RAG API 서버
├─ parser-workers/         # Java HWP/HWPX, Docling, Paddle 격리 worker
├─ requirements/           # 파서 런타임별 고정 버전 의존성
├─ config/parser-artifacts.json # 도구·모델·아티팩트 버전/체크섬
├─ processed/              # 파싱/청킹/인덱스 산출물, Git 제외
├─ logs/                   # 서버 로그, Git 제외
├─ .env.example            # Gemini API 설정 예시
└─ README.md
```

## 로컬 실행

### 1. 패키지 설치

```bash
npm install
python -m pip install -r requirements.txt
```

현재 lockfile은 `package-lock.json` 기준입니다. `pnpm`을 사용해도 동작하지만 팀 단위 재현성은 npm 기준으로 맞추는 것을 권장합니다.

### 2. 환경변수 설정

`.env.example`을 복사해서 `.env`를 만들고 Gemini API key를 입력합니다.

```bash
cp .env.example .env
```

Windows PowerShell에서는 다음 명령을 사용할 수 있습니다.

```powershell
Copy-Item .env.example .env
```

기본 모델은 다음과 같습니다.

```env
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_FALLBACK_MODELS=gemini-3.1-flash-lite
```

`.env` 파일은 Git에 포함하지 않습니다.

API 보호와 입력 제한 관련 기본값은 다음과 같습니다.

```env
RAG_ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
RAG_API_TOKEN=
RAG_MAX_TOP_K=20
RAG_MAX_REQUEST_BYTES=32768
RAG_MAX_QUESTION_CHARS=1000
RAG_SOURCE_CHARS=1800
RAG_DENSE_INDEX=processed/index/dense.sqlite
```

`RAG_API_TOKEN`을 설정하면 `/search`, `/chat` 요청에 `X-RAG-API-Key` 헤더가 필요합니다. 프론트에서 같이 쓰려면 같은 값을 `VITE_RAG_API_TOKEN`에 넣습니다. 로컬 데모에서는 비워 두면 인증 없이 동작합니다.

### 3. 로컬 모델 서버 실행(선택)

로컬 생성 경로는 OpenAI 호환 API를 사용하므로 MLX-LM과 Ollama를 같은
인터페이스로 연결할 수 있습니다. 사용할 모델은 서버가 허용한 목록 안에서
프론트엔드가 선택합니다.

```env
RAG_LOCAL_RUNTIME=managed_mlx
RAG_LOCAL_BASE_URL=http://127.0.0.1:8080/v1
RAG_LOCAL_MODEL=mlx-community/Qwen3.6-35B-A3B-4bit
RAG_LOCAL_MODELS=mlx-community/EXAONE-4.0.1-32B-MLX-Q4,mlx-community/Qwen3.6-35B-A3B-4bit,mlx-community/gemma-4-26b-a4b-it-4bit
RAG_LOCAL_API_STYLE=chat_completions
RAG_LOCAL_CHAT_TEMPLATE_ARGS={"enable_thinking":false}
```

`managed_mlx`에서는 별도의 모델 서버 명령을 실행하지 않습니다. RAG API가
첫 로컬 또는 로컬을 포함한 자동 선택 질의 직전에 `mlx_lm.server` 자식
프로세스를 실행합니다. 화면의 **메모리에서 내리기**를 누르면 앱이 직접
실행한 자식만 종료하므로 모델 메모리가 운영체제에 반환됩니다. 다음 로컬
질의에서는 서버와 선택 모델을 자동으로 다시 불러옵니다.

화면과 같은 동작을 API로 실행하려면 인증 설정을 그대로 사용해
`POST /local-model/unload`를 호출합니다. 답변 생성 중에는 안전하게
`409 local_model_busy`를 반환하며, 이미 내려간 상태에서 다시 호출해도
성공합니다.

예전에 `com.mlx-lm.server` LaunchAgent를 `KeepAlive`로 등록했다면 앱보다
먼저 8080 포트를 차지하고 모델을 다시 올립니다. 관리형 모드로 전환할 때는
한 번만 `launchctl disable gui/$(id -u)/com.mlx-lm.server`와
`launchctl bootout gui/$(id -u)/com.mlx-lm.server`를 실행해 기존 상시 실행
작업을 비활성화해야 합니다. plist 파일은 삭제되지 않습니다.

기존처럼 모델 서버를 별도 실행해야 하는 경우에는
`RAG_LOCAL_RUNTIME=external`로 설정한 뒤 `npm run local:serve`를 사용할 수
있습니다. 외부 프로세스는 이 앱이 소유하지 않으므로 화면에서 강제 종료하지
않습니다.

Ollama를 사용할 때는 `ollama list`에 표시되는 정확한 모델명으로
`RAG_LOCAL_MODEL`과 `RAG_LOCAL_MODELS`를 바꾸고 외부 관리 모드와 주소를
다음처럼 설정합니다.

```env
RAG_LOCAL_RUNTIME=external
RAG_LOCAL_BASE_URL=http://127.0.0.1:11434/v1
```

백엔드와 Ollama가 다른 PC에서 실행된다면 loopback 주소 대신 사설망 또는
VPN 주소를 사용합니다. Ollama API 포트는 공인 인터넷에 직접 노출하지
않는 것을 전제로 합니다.

### 4. RAG API 서버 실행

BM25만 사용할 때는 기존 표준 라이브러리 실행 경로를 그대로 사용할 수 있습니다.

```bash
python3 scripts/search_api.py --host 127.0.0.1 --port 8000 --env-file .env
```

KURE·Snowflake 학습형 Dense 검색을 사용할 때는 임베딩 의존성이 설치된
격리 환경으로 실행합니다. 질의 모델은 해당 검색 방식을 처음 선택할 때만
로드되며, 문서 벡터 행렬은 `processed/index/learned-dense/`에서 읽습니다.

여러 파싱 버전을 함께 등록할 때 Dense artifact는 파싱 결과별로 분리합니다.
Baseline과 Challenger는 각각 `learned-dense/<profile>/<model>/`을 사용하고,
기존 Cascade artifact의 `learned-dense/<model>/` 경로도 계속 지원합니다.

```text
processed/index/learned-dense/
├── baseline/{kure-v1,snowflake-arctic-l-v2-ko}/
├── challenger/{kure-v1,snowflake-arctic-l-v2-ko}/
├── kure-v1/                         # Cascade 호환 경로
└── snowflake-arctic-l-v2-ko/        # Cascade 호환 경로
```

```bash
python3 -m venv .parser-tools/venvs/embedding
.parser-tools/venvs/embedding/bin/pip install -r requirements/embedding-benchmark.txt

.parser-tools/venvs/embedding/bin/python scripts/search_api.py \
  --host 127.0.0.1 \
  --port 8000 \
  --env-file .env \
  --index processed/index/pnu-20260725-curated-cascade-v5-allow-suspect.sqlite \
  --profile-index baseline=processed/index/pnu-20260725-curated-baseline-v5-allow-suspect.sqlite \
  --profile-index challenger=processed/index/pnu-20260725-curated-challenger-v5-allow-suspect.sqlite \
  --default-parser-profile cascade
```

화면의 **검색 방식**에서 다음 다섯 경로를 같은 질문으로 비교할 수 있습니다.

- `bm25`: 기존 키워드 검색 기준선
- `kure_dense`, `snowflake_dense`: 학습형 Dense 코사인 검색
- `kure_hybrid`, `snowflake_hybrid`: BM25와 Dense 후보를 RRF로 결합

서버 기본 검색 방식을 바꾸려면 `.env`에 예를 들어
`RAG_RETRIEVAL_MODE=kure_hybrid`를 설정합니다. artifact의 corpus revision이나
chunk 순서가 현재 BM25 인덱스와 다르면 해당 방식은 자동 비활성화됩니다.

정상 실행 여부는 다음 주소에서 확인합니다.

```text
http://127.0.0.1:8000/health
```

### 5. 프론트엔드 실행

```bash
bun run dev -- --host localhost
```

브라우저에서 다음 주소로 접속합니다.

```text
http://localhost:5173
```

## 데이터 파이프라인

### 3개 프로필 파서 파이프라인

새 파이프라인은 모든 파서 결과를 동일한 11필드 Block 스키마로 정규화합니다.

- `baseline`: HWP/HWPX는 hwplib·hwpxlib, PDF는 PyMuPDF·pdfplumber, 스캔은 Tesseract
- `challenger`: HWP/HWPX는 unhwp, PDF는 Docling, 스캔은 PP-StructureV3 + 한국어 PP-OCRv5
- `cascade`: 문서·페이지 품질에 따라 baseline → 구조 파서/OCR → 최후 fallback을 선택
- HTML과 Office 문서는 세 프로필에서 공통 native adapter를 사용

먼저 로컬 runtime 계획과 누락 항목을 확인합니다. 기본 `prepare`는 읽기 전용이며, `--execute`를 붙여야 체크섬이 고정된 도구 다운로드와 Java worker 빌드를 수행합니다.

```bash
python3 scripts/parse_pipeline.py prepare --profile all
python3 scripts/parse_pipeline.py prepare --profile baseline --execute
python3 scripts/parse_pipeline.py doctor --profile all
```

`prepare` 결과의 `manual_actions`에는 Python 3.12 venv, Tesseract, Paddle 모델처럼 별도 준비가 필요한 항목이 표시됩니다. `doctor`는 고정 버전, 실제 import/API, 동일 venv, 모델 파일 형태를 함께 확인합니다. `doctor`가 확인한 core venv와 같은 interpreter로 실행해야 in-process PDF/Office adapter가 활성화됩니다.

Paddle scan 경로는 실행 중 모델을 받지 않습니다. 검토·고정한 각
`inference.json`, `inference.pdiparams`, `inference.yml`을 다음
디렉터리에 놓아야
`doctor`가 준비 완료로 판정합니다.

```text
.parser-tools/models/paddle/PP-StructureV3/layout_detection/
.parser-tools/models/paddle/PP-StructureV3/text_detection/
.parser-tools/models/paddle/korean_PP-OCRv5_mobile_rec/
```

이 경로는 레이아웃·텍스트 검출·한국어 인식만 로컬 모델로 실행하며,
표·수식·차트·도장 하위 모델은 비활성화합니다.

Docling은 같은 버전의 전용 환경에서 레이아웃과 TableFormer 모델을
미리 받은 경우에만 활성화됩니다.

```bash
.parser-tools/venvs/docling/bin/docling-tools models download \
  -o .parser-tools/models/docling layout tableformer
```

```bash
.parser-tools/venvs/core/bin/python scripts/parse_pipeline.py run \
  --profile all \
  --input src/data \
  --output processed/runs \
  --expect-korean
```

개발 중 누락된 외부 도구를 명시적인 `unavailable` attempt로 기록하며 일부 형식만 시험하려면 `--allow-missing`을 사용할 수 있습니다. 이 옵션은 시작 조건만 완화하며, `doctor`가 승인하지 않은 PATH 명령·Python 패키지를 우회 실행하지 않습니다. 관리형 `run`에서는 일반 `PARSER_*_CMD` 환경변수도 무시합니다(VL 수동 검토 명령 제외).

기본 입력 한도는 파일당 250 MiB, 문서당 250,000 Block입니다. 필요하면 `--max-file-mb`와 `--max-blocks`로 더 낮게 제한할 수 있습니다.

각 실행은 `processed/runs/<run-id>/<profile>/` 아래 임시 디렉터리에서 완성·검증된 뒤 원자적으로 게시됩니다. 기존 run을 덮어쓰지 않으며 `processed/current`도 변경하지 않습니다.

```text
documents.jsonl
blocks.jsonl
chunks.jsonl
attempts.jsonl
parse_report.csv
parse_summary.json
run_manifest.json
raw/
```

산출물과 raw worker snapshot의 체크섬, Block/표/청크 참조 무결성은 다음 명령으로 다시 검사할 수 있습니다.

```bash
python3 scripts/parse_pipeline.py verify-run \
  --run processed/runs/<run-id>
```

검증된 프로필의 `chunks.jsonl`은 BM25 builder와 바로 호환됩니다.
`--chunks`를 반복하면 여러 기관의 검증 결과를 하나의 collection revision으로
묶습니다. 각 chunk에는 원래 parser run revision이 그대로 남고, 중복 chunk ID나
checksum 불일치는 게시 전에 거부됩니다.

```bash
python3 scripts/bm25_search.py build \
  --chunks processed/runs/<pnu-run-id>/cascade/chunks.jsonl \
  --chunks processed/runs/<kisa-run-id>/cascade/chunks.jsonl \
  --index processed/index/bm25.sqlite \
  --require-manifest

python3 scripts/bm25_search.py build-dense \
  --source-index processed/index/bm25.sqlite \
  --index processed/index/dense.sqlite
```

기본 Dense lane은 별도 모델 다운로드 없이 재현 가능한
`local_hashing_v1` cosine baseline입니다. BM25와 Dense의 corpus revision이
다르면 API는 stale Dense를 자동으로 끄고 BM25 단일 lane으로 계속 서비스합니다.

부산대 프로필 비교는 생성 답변이 아니라 실제 curated manifest에서 확인한
24개 문서 gold를 기준으로 Hit@k와 MRR을 계산합니다. 같은 문서의 여러 chunk는
한 문서로 합쳐 평가합니다. 후보 chunk 수는 고유 문서가 충분히 확보되거나
인덱스를 소진할 때까지 적응형으로 늘어납니다. 모든 인덱스의
`source_manifest_sha256`이 동일하고 비어 있지 않은 경우에만 기본 비교를
허용하므로, 서로 다른 코퍼스의 점수를 실수로 나란히 비교하지 않습니다.
레거시 또는 혼합 코퍼스 점검이 꼭 필요할 때만
`--allow-mixed-provenance`를 명시합니다.

```bash
python3 scripts/evaluate_pnu_retrieval.py validate --expected-count 24

python3 scripts/evaluate_pnu_retrieval.py evaluate \
  --index baseline=processed/index/pnu-baseline.sqlite \
  --index challenger=processed/index/pnu-challenger.sqlite \
  --index cascade=processed/index/pnu-cascade.sqlite \
  --json-output processed/eval/pnu-profile-comparison.json \
  --csv-output processed/eval/pnu-profile-comparison.csv
```

### 기존 MVP 파서

### 문서 파싱 및 청킹

```bash
python scripts/parse_documents.py --input src/data --output processed --max-file-mb 5
```

현재 파싱 산출물은 기본적으로 다음 위치에 생성됩니다.

```text
processed/current/documents.jsonl
processed/current/chunks.jsonl
processed/current/parse_report.csv
processed/current/parse_summary.json
```

현재 로컬 처리 기준:

- 전체 파일: 2,947개
- 파싱 성공: 2,725개
- chunk 수: 42,744개
- chunk 크기: 1,800자
- chunk overlap: 250자

### BM25 인덱스 생성

```bash
python scripts/bm25_search.py build --chunks processed/current/chunks.jsonl --index processed/index/bm25.sqlite
```

검색 테스트:

```bash
python scripts/bm25_search.py search "상장폐지 공시" --institution 한국거래소
```

## API

로컬 API 서버는 다음 endpoint를 제공합니다.

- `GET /health`
- `GET /institutions`
- `GET /search?q=질문&institution=기관명&top_k=5`
- `POST /chat`

`POST /chat` 예시:

```json
{
  "question": "상장폐지 제도 개선 내용을 알려줘",
  "institution": "한국거래소",
  "top_k": 8
}
```

응답에는 답변, 근거가 붙은 답변, claim 검증 결과, 검색된 chunk 목록과
BM25/Dense/RRF/reranker 실행 trace가 포함됩니다.

## 프론트엔드 기능

- 기관별 검색 범위 선택
- 질문 입력 및 답변 생성
- 답변 생성 중 경과시간·지연·대체 모델 전환 상태 표시
- 동일 브라우저의 동시 질문 전송 차단과 진행 중 요청 취소
- 답변 내 출처 번호 배지 표시
- claim별 근거 확인 여부 표시
- 검색된 문서 chunk 미리보기
- 원문 위치 정보 표시
- 모바일 drawer 레이아웃
- 질문 길이 제한 및 요청 타임아웃 처리

## 검증 명령

```bash
bun run check
```

`bun run check`는 Python 단위 테스트, ESLint, TypeScript/Vite production build를 한 번에 실행합니다.

## 개발 방식

앞으로 기능 작업은 `main`에 직접 커밋하지 않고 작업 브랜치에서 진행합니다.

예시:

```bash
git switch -c codex/improve-retrieval-quality
```

작업 후 테스트를 통과하면 브랜치 커밋, push, main merge 순서로 정리합니다.

## 다음 작업 후보

- 학습된 한국어 embedding 모델로 hashing Dense baseline 교체
- 사용자 역할별 답변 프롬프트 분리
- RAG 평가 벤치마크 제작
- 문서별 정답 chunk 기반 retrieval 평가
- 응답 캐싱 및 Gemini rate limit 대응
- 크롤링, 파싱, 인덱싱 통합 pipeline script 추가
- API rate limit/auth 배포 정책 정리
