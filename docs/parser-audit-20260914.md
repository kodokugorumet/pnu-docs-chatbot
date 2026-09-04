# 파서 프로필 원자 근거 보존 감사

- 평가셋: `config/pnu-parser-audit-v1.jsonl`
- 지표 버전: `pnu.parser-atomic-anchor-v1`
- 대상: Baseline, Challenger, Cascade의 동일 curated corpus 산출물
- 범위: 파싱 산출물과 SQLite 인덱스에서 원자 근거가 보존되는지만 평가

이 평가는 검색 순위, 답변 생성, LLM Judge 성능과 분리한다. 18문서의 표적 진단
결과를 전체 코퍼스의 무작위 표본 성능이나 RAG 최종 성능으로 해석하지 않는다.

## 1. 감사 세트

| 그룹 | 문서 | 원자 근거 | 구성 |
|---|---:|---:|---|
| HWP/HWPX | 6 | 18 | HWP 5개, HWPX 1개 |
| Digital PDF | 6 | 18 | OCR parser가 선택되지 않은 PDF |
| OCR/Table PDF | 6 | 18 | PP-StructureV3 경로 4개, 표 중심 경로 2개 |
| 합계 | 18 | 54 | 문서당 정확히 3개 |

각 문서는 `document_id`, 원문 상대 경로, 원문 SHA-256, 확장자, 제목을 가진다.
각 원자 근거는 날짜, 수치, 절차, 본문, 표 관계 중 하나이며, Cascade run에서
원문 근거를 찾기 위한 `source_chunk_hint`를 가진다. 이 hint는 정답 근거 검증에만
사용하며 다른 프로필 점수 계산에는 사용하지 않는다.

## 2. 검증 gate

`validate_parser_audit.py`는 다음을 모두 fail-closed로 확인한다.

1. **Schema/balance:** 6/6/6 문서, 문서당 3개, 전역에서 54개 고유 anchor
2. **Raw source integrity:** 실제 원문 파일을 다시 SHA-256 해시하여 데이터셋과 비교
3. **Run integrity:** `run_manifest.json`에 기록된 `documents.jsonl`과
   `chunks.jsonl` 해시를 재계산
4. **Index compatibility:** run과 SQLite의 profile, run ID, source manifest SHA가 일치
5. **Canonical binding:** 54개 anchor가 지정한 Cascade run chunk와 같은 Cascade
   SQLite chunk에 모두 존재하며 문서 제목·경로·SHA·parser route가 일치

원문 HWP/PDF 바이너리 안에서 문자열을 직접 찾았다고 주장하지 않는다. 현재
`raw source integrity`는 원문 파일 자체의 동일성을, `canonical binding`은 그 원문을
파싱한 고정 run과 인덱스의 근거 존재를 보장한다. 최종 보고서에서 “사람이 원본
화면으로 54개를 이중 검수했다”고 쓰려면 별도의 reviewer sign-off가 추가로 필요하다.

## 3. 프로필 독립 비교

평가기는 동일한 anchor 문자열을 각 프로필의 전체 문서 chunk에서 다시 찾는다.
Baseline/Challenger의 chunk ID가 Cascade hint와 달라도 점수에 영향이 없다.
단위 테스트는 `:cascade#...` hint를 가진 case가 `:baseline#...` chunk에서 적중하는
경우를 명시적으로 검사한다.

문자열 비교는 NFKC, case-fold 후 문자와 숫자만 남기는 결정적 정규화를 사용한다.
따라서 공백, 탭, 표 구분자, 전각 문자의 차이는 허용하지만 숫자 변경, 철자 오류,
단어 순서 변경, 동의어 치환은 허용하지 않는다. `table_relation`에서 순서가 바뀌면
표의 행·열 관계가 보존되지 않은 것으로 판정한다.

프로필별로 다음을 별도 기록한다.

- run anchor recall
- index anchor recall
- run과 index가 모두 맞고 source identity가 같은 end-to-end anchor recall
- 문서의 3개 anchor가 모두 보존된 비율
- 형식 그룹별 anchor recall

## 4. 재현 명령

```bash
python3 -B -m unittest tests.test_parser_audit

python3 -B scripts/validate_parser_audit.py

python3 -B scripts/evaluate_parser_audit.py \
  --json-out processed/eval/20260901-parser-audit-v1.json \
  --csv-out processed/eval/20260901-parser-audit-v1.csv
```

평가기에서 다른 산출물을 비교할 때는 다음 형식을 프로필별로 한 번씩 준다.

```text
--artifact PROFILE=RUN_DIR::INDEX.sqlite
```

세 산출물이 같은 `source_manifest_sha256`를 공유하지 않으면 비교를 중단한다.

## 5. 현재 결과

공통 source manifest SHA-256은
`1fa7e0f2f5d03a2a1249584d91f7232c9b2cb4642b39115bf3c9e6a871845a2f`다.

| Profile | Anchor 보존 | Anchor recall | 3/3 문서 | HWP/HWPX | Digital PDF | OCR/Table PDF |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 48/54 | 0.8889 | 14/18 | 18/18 | 17/18 | 13/18 |
| Challenger | 54/54 | 1.0000 | 18/18 | 18/18 | 18/18 | 18/18 |
| Cascade | 54/54 | 1.0000 | 18/18 | 18/18 | 18/18 | 18/18 |

Baseline의 6개 손실은 다음과 같다.

- 등록 기간 표에서 날짜와 `본등록`의 행 관계 1개
- BIDV 앱 제목의 OCR 철자 1개
- 펜토미노 모듈·트랙·마이크로디그리 정의 3개
- 교연비 교원 지급한도 표 관계 1개

이 결과는 Challenger와 Cascade가 이 18문서에서 같은 54개 원자 근거를 모두
보존했다는 뜻이다. Cascade가 Challenger보다 우월하다는 증거는 아니며, Cascade
산출물을 canonical binding에 사용했으므로 두 프로필의 1.0은 표적 표본 선택의
영향을 받을 수 있다.

## 6. 한계와 다음 gate

- 18문서는 실패 모드를 섞은 표적 표본이며 무작위·층화 추출 표본이 아니다.
- anchor는 서비스 DEV에서 이미 근거가 확인된 문서를 주로 활용했다. 파서 진단과
  RAG DEV 결과를 섞어 하나의 성능 향상 수치로 제시하면 안 된다.
- Cascade run/index가 canonical text binding 역할을 하므로 Cascade에 유리한 선택
  편향이 있다. 이 감사는 Baseline 손실 진단에는 유효하지만 프로필 간 통계적 우열
  검정에는 부족하다.
- OCR의 CER/WER, 표 cell 단위 정밀도, 페이지 bbox, 처리 시간과 메모리는 이 v1의
  측정 범위가 아니다.
- 최종 보고서에서 강한 파서 품질 주장을 하려면 54개 원문 화면의 독립 2인 검수,
  무작위 holdout 표본, 페이지/표 cell locator를 추가해야 한다.
