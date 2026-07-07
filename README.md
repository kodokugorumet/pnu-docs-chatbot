# 공문서 RAG 챗봇

부산대학교와 주요 공공기관의 행정 문서를 검색하고, 검색된 근거를 바탕으로 답변을 생성하는 졸업과제용 RAG 챗봇입니다.

현재는 로컬 문서 파싱, 청킹, BM25 검색 인덱스, Gemini 기반 답변 생성, React 채팅 UI까지 연결된 MVP 상태입니다.

## 현재 구현 상태

- 기관별 문서 수집 데이터 정리
- 문서 파싱 및 청킹 파이프라인
- SQLite FTS5 기반 BM25 검색
- Gemini API 기반 답변 생성
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

## 프로젝트 구조

```text
.
├─ src/
│  ├─ App.tsx              # 채팅 UI
│  ├─ App.css              # 화면 스타일
│  └─ data/                # 기관별 원문 문서
├─ scripts/
│  ├─ crawl_finance_docs.py # 금융권 문서 크롤러
│  ├─ parse_documents.py   # 문서 파싱 및 청킹
│  ├─ bm25_search.py       # SQLite FTS5/BM25 인덱스
│  └─ search_api.py        # 로컬 RAG API 서버
├─ processed/              # 파싱/청킹/인덱스 산출물, Git 제외
├─ logs/                   # 서버 로그, Git 제외
├─ .env.example            # Gemini API 설정 예시
└─ README.md
```

## 로컬 실행

### 1. 패키지 설치

```bash
pnpm install
```

`pnpm`이 없다면 npm으로도 실행할 수 있습니다.

```bash
npm install
```

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
pnpm run dev --host localhost
```

브라우저에서 다음 주소로 접속합니다.

```text
http://localhost:5173
```

## 데이터 파이프라인

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

응답에는 답변, 근거가 붙은 답변, claim 검증 결과, 검색된 chunk 목록이 포함됩니다.

## 프론트엔드 기능

- 기관별 검색 범위 선택
- 질문 입력 및 답변 생성
- 답변 생성 중 단계 표시
- 답변 내 출처 번호 배지 표시
- claim별 근거 확인 여부 표시
- 검색된 문서 chunk 미리보기
- 원문 위치 정보 표시
- 모바일 drawer 레이아웃

## 검증 명령

```bash
pnpm run build
pnpm run lint
```

Python 스크립트 문법 검사:

```bash
python -m py_compile scripts/search_api.py scripts/bm25_search.py scripts/parse_documents.py
```

## 개발 방식

앞으로 기능 작업은 `main`에 직접 커밋하지 않고 작업 브랜치에서 진행합니다.

예시:

```bash
git switch -c codex/improve-retrieval-quality
```

작업 후 테스트를 통과하면 브랜치 커밋, push, main merge 순서로 정리합니다.

## 다음 작업 후보

- dense embedding 검색 추가
- BM25 + vector hybrid retrieval
- reranking 적용
- 사용자 역할별 답변 프롬프트 분리
- RAG 평가 벤치마크 제작
- 문서별 정답 chunk 기반 retrieval 평가
- 응답 캐싱 및 Gemini rate limit 대응
- 크롤링, 파싱, 인덱싱 통합 pipeline script 추가
