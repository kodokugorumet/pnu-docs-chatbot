# 공문서 RAG 챗봇

부산대학교와 주요 공공기관의 행정 문서를 검색하고, 검색된 근거를 바탕으로 답변을 생성하는 졸업과제용 RAG 챗봇입니다.

현재는 로컬 문서 파싱, 검증된 다기관 corpus gate, BM25 + Dense/RRF
하이브리드 검색, 공급자 선택형 답변 생성, React 채팅 UI까지 연결된 MVP
상태입니다.

## 현재 구현 상태

- 기관별 문서 수집 데이터 정리
- 문서 파싱 및 청킹 파이프라인
- SQLite FTS5 기반 BM25와 로컬 hashing Dense 검색
- reciprocal-rank fusion(RRF)과 lexical reranking
- local/frontier OpenAI 호환 API, Gemini, 추출형 fallback 답변 생성
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

부산대학교 메인 홈페이지에서 시작해 공개된 `*.pusan.ac.kr` 학과·기관
홈페이지 링크를 자동으로 따라갑니다. 학사, 수업, 졸업, 장학, 학생지원,
공지사항을 우선 방문하며 HTML 원문과 PDF/HWP/HWPX/Office 첨부파일을 함께
저장합니다. 로그인·관리자 페이지, 외부 도메인, 이미지·스크립트 자산은
수집하지 않습니다.

먼저 100페이지만 시험 수집합니다.

```bash
python3 scripts/crawl_pnu_site.py \
  --max-pages 100 \
  --max-files 100 \
  --max-depth 3
```

범위를 넓혀 수집하려면 다음과 같이 실행합니다.

```bash
python3 scripts/crawl_pnu_site.py \
  --max-pages 5000 \
  --max-files 3000 \
  --max-depth 6 \
  --max-pages-per-host 500 \
  --delay 1.5
```

결과는 기본적으로 `downloads/pnu-web-crawl/`에 저장됩니다.

```text
downloads/pnu-web-crawl/
├─ content/부산대학교/
│  ├─ 웹페이지/            # 파서에 투입할 HTML 원문
│  └─ 첨부파일/            # PDF, HWP, HWPX, Office 문서
├─ state/
│  ├─ crawl.sqlite3        # 재개 가능한 URL 큐
│  ├─ pages.jsonl          # 페이지별 원 URL·제목·체크섬
│  └─ attachments.jsonl    # 첨부파일별 원 URL·저장 경로·체크섬
└─ summary.json
```

중간에 중단해도 같은 명령을 다시 실행하면 남은 URL부터 이어집니다. 현재
누적 상태는 다음 명령으로 확인할 수 있습니다.

```bash
python3 scripts/crawl_pnu_site.py --status
```

이미 수집한 URL도 다시 확인하려면 `--refresh`를 붙입니다. 별도 학과
홈페이지를 시작점에 추가할 때는 `--seed`를 반복해서 지정할 수 있습니다.
기본 허용 범위는 부산대학교 공식 도메인 전체이므로 발견된 다른
`*.pusan.ac.kr` 학과·기관 사이트도 자동으로 포함됩니다.

수집 결과는 상태 파일을 제외한 `content` 디렉터리만 파서에 전달합니다.

```bash
.parser-tools/venvs/core/bin/python scripts/parse_pipeline.py run \
  --profile all \
  --input downloads/pnu-web-crawl/content \
  --output processed/runs \
  --expect-korean
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
GEMINI_MODEL=gemini-3.1-flash-lite
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

### 3. RAG API 서버 실행

```bash
python scripts/search_api.py --host 127.0.0.1 --port 8000 --env-file .env
```

정상 실행 여부는 다음 주소에서 확인합니다.

```text
http://127.0.0.1:8000/health
```

### 4. 프론트엔드 실행

```bash
npm run dev -- --host localhost
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
- 답변 생성 중 단계 표시
- 답변 내 출처 번호 배지 표시
- claim별 근거 확인 여부 표시
- 검색된 문서 chunk 미리보기
- 원문 위치 정보 표시
- 모바일 drawer 레이아웃
- 질문 길이 제한 및 요청 타임아웃 처리

## 검증 명령

```bash
npm run check
```

`npm run check`는 Python 단위 테스트, ESLint, TypeScript/Vite production build를 한 번에 실행합니다.

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
