# 문서 파서 조사 로그

- 조사 기준일: 2026-07-21
- 목적: PNU 공문서 RAG의 다음 파서 실험 순서와 후보별 역할을 회의 전에 정리한다.
- 문서 성격: 후보 선정 전 조사 기록이다. 이 문서의 평가는 실제 데이터셋 벤치마크 결과가 아니다.

## 1. 조사 목적

현재 파이프라인은 파일을 읽을 수 있는지와 텍스트가 일정 길이 이상인지에는 답하지만, 검색 근거로 쓸 만한 구조를 보존했는지는 측정하지 않는다. 특히 데이터의 45.2%를 차지하는 HWP가 문단 텍스트로 평탄화되고, 표·절·페이지 위치가 chunk 메타데이터에 남지 않는다.

이번 조사의 질문은 다음과 같다.

1. HWP/HWPX에서 제어문자를 제거하면서 문단, 표, 목록, 각주와 구역을 보존할 수 있는가?
2. PDF에서 텍스트 레이어가 있는 문서는 빠르게 처리하고, 복잡한 표나 스캔 문서만 무거운 파서/OCR로 보낼 수 있는가?
3. 각 chunk에 페이지, 절, 문단, 표 위치를 남겨 답변의 Citation을 원문 위치까지 추적할 수 있는가?
4. macOS 개발 환경과 오프라인 배치 환경에서 재현 가능하고 라이선스상 배포 가능한가?

## 2. 현재 데이터와 구현 확인

### 2.1 데이터 현황

`processed/current/parse_summary.json`과 기존 데이터 조사 문서를 대조한 기준이다.

| 항목 | 수치 | 비고 |
| --- | ---: | --- |
| 전체 파일 | 2,947 | 2026-07-06 생성 산출물 기준 |
| 파싱 성공 | 2,725 | 92.5% |
| 빈 결과 | 28 | 최소 20자 미달 |
| 오류 | 30 | HWP 컨테이너 오류, 손상 PDF, XLS 의존성 등이 포함됨 |
| 건너뜀 | 53 | 미지원 확장자 |
| 보류 | 111 | `--max-file-mb 5` 기준 초과 |
| 생성 chunk | 42,744 | 1,800자, overlap 250자 |

| 형식 | 파일 수 | 전체 비율 | 현재 처리 방향 |
| --- | ---: | ---: | --- |
| PDF | 1,468 | 49.8% | PyMuPDF 우선, pypdf, 선택적 pdfplumber fallback |
| HWP | 1,331 | 45.2% | 자체 OLE/HWP 5 레코드 텍스트 추출 |
| XLSX | 67 | 2.3% | openpyxl |
| ZIP | 47 | 1.6% | 건너뜀 |
| JPG | 8 | 0.3% | 건너뜀, OCR 미구현 |
| PPTX | 7 | 0.2% | 슬라이드 XML 텍스트 추출 |
| XLS | 5 | 0.2% | pandas/xlrd |
| JSONL | 4 | 0.1% | 건너뜀 |
| HWPX | 3 | 0.1% | ZIP/XML의 모든 텍스트 평탄화 |
| PNG | 3 | 0.1% | 건너뜀, OCR 미구현 |
| DOC | 2 | 0.1% | 건너뜀 |
| MP4 | 1 | 0.0% | 건너뜀 |
| CSV | 1 | 0.0% | 건너뜀 |
| 합계 | 2,947 | 100.0% |  |

### 2.2 확인된 품질 문제

- **HWP 제어문자 오염:** 현재 `parse_hwp()`는 `HWPTAG_PARA_TEXT` payload 전체를 UTF-16LE로 디코딩한다. `clean_text()`가 C0 제어문자 일부는 지우지만, HWP 인라인 제어의 의미를 해석하거나 표·필드·문단 속성과 결합하지 않는다. 따라서 눈에 보이는 잔여 기호와 구조 손실을 함께 검사해야 한다.
- **대용량 PDF 제외:** 5MB 상한 때문에 111개가 `deferred` 상태이며, 그중 PDF가 93개다(나머지 ZIP 12, PPTX 2, HWP 2, JPG 1, XLSX 1). 보고서상 보류 사유는 전부 `over_max_file_mb:5.0`이다. 다음 실험은 파일 전체 크기 대신 페이지별 처리와 메모리 상한을 사용해야 한다.
- **Citation 위치 정보 부족:** chunk에는 기관, 파일 경로, 파일명, 확장자, 파서만 저장된다. PDF 본문에는 `[page N]` 표식이 삽입되지만 별도 메타데이터가 아니며 API 정규화 단계에서 제거된다. HWP/HWPX는 페이지·구역·문단·표 위치가 없다. 현재 출처 번호는 “어느 파일인가”까지는 가리키지만 “몇 페이지의 어느 표/문단인가”는 안정적으로 가리키지 못한다.
- **구조 품질 판정 없음:** 성공 조건은 정제 후 20자 이상이다. 표 보존율, 읽기 순서, 한글 비율, 제어문자 수, 머리말 반복, OCR 신뢰도는 기록하지 않는다.

## 3. 후보 비교 기준

비교표의 “한국어”는 한국어 전용 파싱 능력 또는 공식 한국어 OCR 모델의 존재를 뜻한다. “구조·표”는 원본의 논리 구조를 직접 돌려주는 정도이며, 아직 PNU 샘플로 검증하지 않은 평가는 후보 문서가 선언한 기능에 근거한다.

공통 평가 항목은 다음과 같다.

- 지원 형식과 암호화·손상 문서 처리 범위
- 문단, 제목, 목록, 표, 각주, 이미지, 페이지 좌표 보존
- 한국어 인식 및 HWP 고유 제어 처리
- CPU/GPU, JVM, Rust, Python, WASM 등 실행 조건
- 코드와 모델 가중치의 라이선스
- 결정적 출력, 배치 처리, 오류 격리, 버전 고정 가능성
- `page`, `section`, `paragraph`, `table`, `bbox`를 Citation 메타데이터로 내보낼 수 있는지

## 4. HWP/HWPX 후보

| 후보 | 형식 | 구조·표 보존 | 한국어 | 실행 환경 | 라이선스 | 장점 | 한계 | PNU 권장 역할 | 공식 출처 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hwplib | HWP 5.x | 높음. 문단·컨트롤·표 객체에 접근 | HWP 규격 기반 | Java 7+ / JVM | Apache-2.0 | 성숙한 읽기/쓰기 객체 모델, 텍스트 추출 모드와 예제 다수 | Python 파이프라인에 JVM 경계가 생기며 페이지 렌더링은 별도 | **HWP 주 파서 1순위** | [hwplib 저장소](https://github.com/neolord0/hwplib) |
| hwpxlib | HWPX | 높음. XML 객체를 읽고 쓸 수 있음 | HWPX/OWPML 규격 기반 | Java 7+ / JVM | Apache-2.0 | HWPX 구조를 직접 다루며 hwplib와 API 계열이 같음 | 현재 HWPX는 3개뿐이며 HWP→HWPX 변환은 별도 구성 | **HWPX 주 파서 1순위** | [hwpxlib 저장소](https://github.com/neolord0/hwpxlib) |
| unhwp | HWP 3/5, HWPX | 높음. 제목·목록·표·스타일·이미지·구역 JSON/Markdown | HWP/HWPX 전용 | Rust CLI/라이브러리, macOS·Linux·Windows, FFI | MIT | 단일 바이너리, streaming, 구조화 JSON, 표 span과 asset 추출 | 비교적 새 프로젝트라 실제 공문서 호환성과 출력 안정성 검증 필요 | **독립 fallback 및 비교 기준** | [unhwp 저장소](https://github.com/iyulab/unhwp) |
| rhwp | HWP/HWPX/HML | 매우 높음. 파싱뿐 아니라 표·수식·차트·페이지 렌더링 | HWP 전용 | Rust, CLI, WASM/Web | MIT | 페이지네이션과 SVG/PNG/PDF 출력, 시각 회귀 검증 가능 | 편집기/렌더러 범위가 커서 단순 ingestion 의존성으로는 무거움 | **렌더링 기준·시각 검증 도구**, 파서 fallback 후보 | [rhwp 저장소](https://github.com/edwardkim/rhwp) |
| Apache Tika | HWP 5 및 다수 문서 | 낮음~중간. 메타데이터와 본문 중심 | HWP 파서 포함, OCR은 별도 | Java CLI/server | Apache-2.0 | 다양한 형식의 MIME 감지와 일괄 fallback을 한 서비스로 제공 | HWP 표·페이지·세부 구조가 평탄화될 가능성이 큼 | **범용 최후 텍스트 fallback** | [Tika](https://tika.apache.org/), [HWP parser API](https://tika.apache.org/3.1.0/api/org/apache/tika/parser/hwp/package-summary.html) |

규격 구현의 기준점은 한컴이 공개한 [HWP/OWPML 형식 안내](https://www.hancom.co.kr/support/downloadCenter/hwpOwpml)와 [HWP 5.0 파일 형식 명세](https://cdn.hancom.com/link/docs/%ED%95%9C%EA%B8%80%EB%AC%B8%EC%84%9C%ED%8C%8C%EC%9D%BC%ED%98%95%EC%8B%9D_5.0_revision1.2.pdf)다. 라이브러리 간 결과가 다르면 명세의 레코드·컨트롤 정의와 원본 렌더링을 함께 확인한다.

## 5. PDF 후보

| 후보 | 형식 | 구조·표 보존 | 한국어 | 실행 환경 | 라이선스 | 장점 | 한계 | PNU 권장 역할 | 공식 출처 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PyMuPDF | PDF 등 MuPDF 지원 형식 | 중간. block/word 좌표, 표 탐지, 페이지 렌더링 | 텍스트 레이어는 언어 비의존, OCR은 Tesseract 연동 | Python/C, CPU | AGPL-3.0 또는 상용 | 빠르고 현재 이미 사용 중, 페이지 좌표와 이미지 렌더링 가능 | 읽기 순서와 복잡한 표 복원은 추가 로직 필요, 배포 라이선스 검토 필요 | **일반 PDF 주 파서와 라우터** | [PyMuPDF 문서](https://pymupdf.readthedocs.io/en/latest/), [라이선스](https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright) |
| Docling | PDF, Office, HTML, 이미지 등 | 높음. layout, reading order, table, formula, lossless JSON | OCR backend에 따름 | Python 3.10+, CPU/GPU, 로컬 실행 | 코드 MIT, 모델별 별도 | 통일된 `DoclingDocument`, 페이지·표 구조, 선택적 OCR | 모델 다운로드와 처리 시간이 단순 추출보다 큼 | **복잡 PDF 1차 구조 fallback** | [Docling 저장소](https://github.com/docling-project/docling), [기술 보고서](https://arxiv.org/abs/2408.09869) |
| MinerU | PDF, 이미지, DOCX/PPTX/XLSX | 높음. 읽기 순서·표·수식·각주·시각화 | OCR 109개 언어 선언, 한국어 샘플 검증 필요 | Python, CPU/GPU/MPS, 서비스 가능 | MinerU Open Source License(추가 조건) | 스캔/깨진 PDF 자동 감지, Markdown과 풍부한 중간 JSON | 설치·모델이 무겁고 커스텀 라이선스 조건 검토 필요 | **Docling 다음 복잡 PDF 비교 후보** | [MinerU 저장소](https://github.com/opendatalab/MinerU), [논문](https://arxiv.org/abs/2409.18839) |
| Marker | PDF, 이미지 및 일부 Office | 높음. Markdown/JSON, 표·수식·이미지, OCR/VLM 선택 | OCR 모델 성능을 샘플로 확인 | Python, GPU 권장·CPU 가능 | 코드 GPL, 모델 가중치 별도 제한 | 다양한 converter와 강제 OCR, 구조화 출력 | 코드와 모델의 배포 조건이 PNU 서비스에 까다로울 수 있음 | **품질 비교 벤치마크**, 기본 채택은 보류 | [Marker 저장소](https://github.com/datalab-to/marker) |
| pdfplumber | PDF | 중간~높음. 문자·선·사각형 좌표와 표 추출 | 텍스트 레이어는 언어 비의존 | Python, CPU | MIT | 표 탐지 설정과 시각 디버깅이 좋고 현재 fallback 의존성에 포함 | OCR이 없고 읽기 순서·무선 표는 문서별 튜닝 필요 | **선이 있는 표 전용 fallback/진단** | [pdfplumber 저장소](https://github.com/jsvine/pdfplumber) |

PDF는 전 파일에 AI 파서를 적용하지 않는다. PyMuPDF에서 페이지별 글자 수, 이미지 비율, block 순서, 표 후보를 측정하고, 저품질 페이지만 Docling 또는 MinerU로 보낸다. 이 방식이면 현재 5MB 상한을 없애면서도 큰 PDF의 처리 비용을 제어할 수 있다.

## 6. OCR/VLM 후보

| 후보 | 입력·출력 | 구조·표 | 한국어 | 실행 환경 | 라이선스 | 장점 | 한계 | PNU 권장 역할 | 공식 출처 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PP-StructureV3 | 이미지/PDF → 계층형 문서 결과 | 높음. layout·표·수식 | 텍스트 인식 모델 조합에 따름 | PaddlePaddle, CPU/GPU | Apache-2.0 | 문서 구조와 표 인식을 한 pipeline으로 제공 | 모델과 런타임이 무겁고 텍스트 PDF에는 불필요 | **스캔/복잡 페이지 OCR 1순위** | [사용 가이드](https://www.paddleocr.ai/v3.0.3/en/version3.x/pipeline_usage/PP-StructureV3.html), [PaddleOCR 저장소](https://github.com/PaddlePaddle/PaddleOCR) |
| PP-OCRv5 multilingual Korean | 이미지 → text box·confidence | 낮음. layout은 별도 | **공식 다국어 모델에 Korean 포함** | PaddlePaddle, CPU/GPU | Apache-2.0 | 한국어 모델이 명시되고 box/confidence를 받을 수 있음 | 표 의미 구조와 읽기 순서는 별도 조립 | **PP-StructureV3의 한국어 인식기** | [다국어 PP-OCRv5 문서](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html), [기술 보고서](https://arxiv.org/abs/2507.05595) |
| PaddleOCR-VL | 이미지/PDF → Markdown/구조 요소 | 높음. 표·수식·차트·문서 parsing | 다국어 VLM, 한국 공문서 별도 검증 | 0.9B VLM, GPU 권장 | PaddleOCR 코드 Apache-2.0, 모델 조건 확인 | 단일 모델로 복잡 레이아웃을 해석하고 구조화 가능 | 결정성, 속도, GPU 비용, 모델 오류를 규칙 기반 parser와 분리 평가해야 함 | **난문서 최후 VLM fallback/실험** | [PaddleOCR-VL 논문](https://arxiv.org/abs/2510.14528), [PaddleOCR 저장소](https://github.com/PaddlePaddle/PaddleOCR) |
| Tesseract | 이미지 → text/TSV/hOCR/PDF | 낮음~중간. box는 있으나 표 의미 없음 | `kor` traineddata 제공 | C++ CLI/library, CPU | Apache-2.0 | 성숙하고 가벼우며 오프라인 자동화가 쉬움 | 복잡한 표와 다단 읽기 순서는 별도 layout 분석 필요 | **가벼운 OCR baseline과 비상 fallback** | [Tesseract 저장소](https://github.com/tesseract-ocr/tesseract), [한국어 traineddata](https://github.com/tesseract-ocr/tessdata/blob/main/kor.traineddata) |

OCR은 원래 텍스트 레이어가 없거나 글자 품질 점수가 낮은 페이지에만 사용한다. OCR 결과는 원본 텍스트를 무조건 덮어쓰지 않고 `ocr_text`, `bbox`, `confidence`, `model_version`을 별도로 보존한 뒤 더 나은 결과를 선택한다.

## 7. HTML 후보

| 후보 | 구조 보존 | 한국어 | 실행 환경 | 라이선스 | 장점 | 한계 | PNU 권장 역할 | 공식 출처 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DOM 직접 파싱 | **가장 높음.** 제목·표·목록·링크·원래 selector를 그대로 설계 가능 | 언어 비의존 | 현재 Python `HTMLParser` 또는 lxml/BeautifulSoup | 구현 및 선택 라이브러리에 따름 | 공공기관별 selector와 source URL을 정확히 보존 | 사이트별 템플릿, 숨김 영역, 깨진 HTML을 유지보수해야 함 | **공식 페이지/변환 HTML 주 파서** | [WHATWG DOM 표준](https://dom.spec.whatwg.org/) |
| Trafilatura | 중간. 본문·메타데이터 중심, Markdown/XML/JSON 출력 | 언어 비의존 | Python CLI/library | Apache-2.0 | boilerplate 제거, 날짜·제목·URL 추출, batch 사용이 쉬움 | 표·코드·기관별 문서 구조를 생략할 수 있음 | **본문 추출 fallback과 DOM 결과 비교** | [Trafilatura 저장소](https://github.com/adbar/trafilatura), [사용 문서](https://trafilatura.readthedocs.io/en/latest/) |
| Mozilla Readability | 낮음~중간. article HTML과 메타데이터 | 언어 비의존 | JavaScript + DOM | Apache-2.0 | Firefox Reader View 계열의 검증된 본문 추출 | 문서 포털의 표·첨부·탐색 구조를 제거할 수 있고 입력 정화는 별도 | **뉴스/보도자료형 페이지 fallback** | [Readability 저장소](https://github.com/mozilla/readability) |

HTML에서는 다운로드 원문 URL과 게시 페이지 URL을 먼저 보존한다. 본문 추출기는 원본 DOM을 버리는 단계가 아니라 `main_content` 파생물을 만드는 단계여야 한다.

## 8. PNU 우선순위와 역할 분리

상세 조사와 실험 순서는 다음과 같이 고정한다.

| 순위 | 후보 | 먼저 답할 질문 | 통과 기준 | 예정 역할 |
| ---: | --- | --- | --- | --- |
| 1 | hwplib + hwpxlib | 1,331 HWP와 3 HWPX에서 표·문단·제어를 안정적으로 구조화하는가? | 성공률, 제어문자, 표 cell, 문단 경계, 처리 시간에서 현 파서 개선 | **주 파서** |
| 2 | unhwp | JVM 없이 같은 문서 구조를 재현하며 hwplib 실패 파일을 회수하는가? | 주 파서와 독립된 성공 집합, JSON schema 안정성 | **HWP/HWPX fallback** |
| 3 | rhwp | 파서 결과를 원본 페이지 렌더링과 자동 비교할 수 있는가? | 대표 80개에서 PNG/PDF 렌더 및 시각 diff 가능 | **렌더링 검증**, 필요 시 fallback |
| 4 | Docling | 복잡 PDF의 읽기 순서·표·페이지 위치가 개선되는가? | PyMuPDF 저품질 페이지에서 정확도 이득이 비용을 상회 | **PDF 구조 fallback** |
| 5 | MinerU | Docling이 놓친 표·수식·스캔을 회수하는가? | 고난도 샘플의 추가 이득과 라이선스 수용 가능 | **PDF 2차 fallback/비교** |
| 6 | PaddleOCR | 텍스트 레이어 없는 한국어 페이지를 구조화할 수 있는가? | 한국어 CER/WER, 표 cell 정확도, confidence 보정 | **선택적 OCR/VLM** |

### 8.1 제안 pipeline

```text
입력
├─ HWP  ── hwplib ── 품질 점수 실패 ── unhwp ── Tika 텍스트 최후 fallback
│                         └──────────── rhwp 렌더링 비교
├─ HWPX ─ hwpxlib ─ 품질 점수 실패 ── unhwp
├─ PDF  ── PyMuPDF 페이지 분석
│          ├─ 정상 text/layout ─────── 좌표 포함 block 저장
│          ├─ 복잡 layout/table ───── Docling ── MinerU 비교
│          └─ image/저품질 text ───── PP-StructureV3 + PP-OCRv5-ko
└─ HTML ── 기관별 DOM parser ─────── Trafilatura/Readability fallback
```

여기서 rhwp는 기본 텍스트 공급자가 아니라 **원본과 파싱 결과가 같은 문서를 나타내는지 확인하는 renderer**다. PaddleOCR-VL도 규칙 기반 결과를 바로 대체하지 않고, 실패 페이지에 한정한 별도 후보로 둔다.

## 9. 실험 산출물과 판정 지표

모든 후보는 같은 manifest와 block schema로 내보내 비교할 수 있어야 한다.

필수 문서 필드:

- `source_path`, `source_sha256`, `source_size_bytes`, `extension`
- `parser`, `parser_version`, `model_version`, `status`, `error`
- `page_count`, `section_count`, `paragraph_count`, `table_count`
- `extracted_chars`, `hangul_ratio`, `control_char_count`
- `elapsed_ms`, `peak_memory_mb`, `fallback_chain`

필수 block/Citation 필드:

- `block_id`, `block_type`, `text`, `reading_order`
- `page`, `section_path`, `paragraph_index`
- `table_id`, `row_index`, `column_index`, `header_context`
- `bbox`, `source_url`, `download_url`
- OCR 사용 시 `ocr`, `confidence`, `language`, `model_version`

자동 지표만으로 최종 선택하지 않는다. 부산대학교 30개, 금융감독원 30개, KISA/한국해양과학기술원 20개로 표·규정·서식·보도자료·스캔을 섞은 80개 gold sample을 만들고 다음을 함께 평가한다.

1. 파일 및 페이지 성공률
2. 사람이 표시한 문단/제목/표 cell의 보존율
3. 잘못 삽입된 문자와 제어문자 수
4. 페이지·표까지 이어지는 Citation 정확도
5. 고정 질문 30개의 top-5 relevant passage 포함률
6. 파일/페이지당 시간과 최대 메모리
7. 동일 버전 재실행 시 출력 hash 안정성

## 10. 회의에서 결정할 항목

1. Java sidecar를 허용하고 hwplib/hwpxlib를 주 파서로 실험할지
2. gold sample 80개의 정답 block과 Citation 위치를 누가 검수할지
3. 코드뿐 아니라 모델 가중치까지 포함한 라이선스 검토 담당자
4. 대용량 PDF를 페이지 단위 queue로 처리할 때 허용할 CPU/GPU 예산
5. parser 결과 JSONL과 렌더링 snapshot을 저장할 위치 및 보존 기간

## 11. 조사 시 주의사항

- 이 우선순위는 채택 결정이 아니다. 공식 기능 표시는 실제 한국 공문서 품질을 보장하지 않는다.
- 라이선스 표기는 조사 기준일의 저장소 기준 요약이며 법률 자문이 아니다. 특히 PyMuPDF, MinerU, Marker와 모델 가중치는 배포 전에 원문을 다시 검토한다.
- 버전과 모델은 실험 manifest에 고정한다. `latest` 결과끼리 비교하지 않는다.
- 원문 HWP, README, 현재 parser 코드와 dependency는 이번 조사 문서 작성 범위에서 변경하지 않는다.
