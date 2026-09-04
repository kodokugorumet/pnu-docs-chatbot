# DEV45 8조건 retrieval matrix 실행 기록 — 2026-09-02

이 문서는 서비스 개발셋 45문항에서 검색 조건을 탐색한 실행 기록이다. HOLDOUT
결과나 일반화 성능이 아니며, 생성 품질·Judge 성능도 측정하지 않는다.

## 실행 통제

- 실제 `search_api.py`의 `POST /chat` 경로를 사용했다.
- 생성 provider는 모든 요청에서 in-process `extractive`로 고정해 외부 LLM 호출은
  0회다.
- 모든 조건은 최종 context `top_k=8`, 문서당 context cap 2, `eval_trace=true`다.
- PAR-B, PAR-CH, C1은 서비스 검색 튜닝 ON 서버에서 실행했다.
- C0, D-K, H-K, D-S, H-S는 검색 튜닝 OFF 서버에서 실행했다. 따라서 Hybrid의
  BM25 lane과 이후 neighbor/facet 단계에도 C1 튜닝이 섞이지 않는다.
- Baseline, Challenger, Cascade 인덱스는 모두 같은 source manifest SHA-256
  `1fa7e0f2f5d03a2a1249584d91f7232c9b2cb4642b39115bf3c9e6a871845a2f`에
  묶였다.
- KURE와 Snowflake artifact는 각각 Cascade 44,520개 chunk와 일치했고, 고정
  model revision을 로컬 캐시에서 불러왔다. 임베딩 실행환경은
  `requirements/embedding-benchmark.txt`의 버전으로 `/private/tmp`에 격리했다.
- 8조건 × 45문항 = 360개 요청이 모두 완료됐고 error row는 0개다.
- 기존 `processed/eval/dev45-p0*` C0/C1 artifact는 읽거나 덮어쓰지 않았다.

재현 실행기는 `scripts/run_dev_retrieval_matrix.py`다. 새 출력 디렉터리를 지정하면
튜닝 ON/OFF 서버를 분리 기동하고, collector control과 corpus revision을 fail-closed로
검사한 뒤 결과와 compact derivative를 만든다.

## 결과

| 조건 | Parser / Retrieval | Source Hit@5 | MRR@5 | Any-Gold-Chunk@8 | All-Gold-Chunks@8 | Mean GoldChunkRecall@8 |
|---|---|---:|---:|---:|---:|---:|
| PAR-B | Baseline / tuned BM25 | 39/45 = .867 | .726 | N/A | N/A | N/A |
| PAR-CH | Challenger / tuned BM25 | 40/45 = .889 | .714 | N/A | N/A | N/A |
| C0 | Cascade / untuned BM25 | 29/45 = .644 | .494 | 23/45 = .511 | 10/45 = .222 | .359 |
| C1 | Cascade / tuned BM25 | 40/45 = .889 | .717 | 28/45 = .622 | 17/45 = .378 | .500 |
| D-K | Cascade / KURE Dense | 37/45 = .822 | .596 | 27/45 = .600 | 15/45 = .333 | .481 |
| H-K | Cascade / KURE Hybrid | 34/45 = .756 | .544 | 20/45 = .444 | 9/45 = .200 | .304 |
| D-S | Cascade / Snowflake Dense | 36/45 = .800 | .616 | 29/45 = .644 | 16/45 = .356 | .500 |
| H-S | Cascade / Snowflake Hybrid | 35/45 = .778 | .560 | 21/45 = .467 | 10/45 = .222 | .326 |

DEV의 gold chunk ID는 Cascade 전용이므로 PAR-B/PAR-CH에는 Source Hit/MRR만
보고한다. 두 parser 조건에 Cascade chunk ID를 대입해 나오는 0은 파서 품질 지표가
아니므로 결과에서 숨겼다.

### C1 대비 질문 단위 변화

아래 `신규/상실`은 각 lane이 C1보다 성공으로 바꾼 문항 수와 C1의 성공을 잃은
문항 수다. aggregate 비율이 비슷해도 서로 다른 문항을 맞힐 수 있으므로 함께
기록한다.

| Lane − C1 | Source Hit@5 신규/상실 | Any Gold@8 신규/상실 | All Gold@8 신규/상실 |
|---|---:|---:|---:|
| C0 | 0 / 11 | 0 / 5 | 0 / 7 |
| D-K | 3 / 6 | 7 / 8 | 5 / 7 |
| H-K | 1 / 7 | 3 / 11 | 1 / 9 |
| D-S | 4 / 8 | 9 / 8 | 6 / 7 |
| H-S | 1 / 6 | 4 / 11 | 2 / 9 |

compact row에도 `case_id`, 순서가 보존된 final source ID/title, 단계별 chunk ID와
rank/score가 있으므로 이 paired 변화는 full context 본문 없이 재계산할 수 있다.

## 해석 범위

- Cascade 조건 중 C1이 Source Hit@5와 All-Gold-Chunks@8에서 가장 높다.
- D-S는 Any-Gold-Chunk@8만 C1보다 1문항 높고 Mean Recall@8은 동률이지만,
  Source Hit@5는 4문항, All-Gold-Chunks@8은 1문항 낮다. 따라서 Dense가 서비스를
  대체한다고 결론 내릴 근거가 아니다.
- 현재 Hybrid 두 조건은 대응 Dense 조건보다 낮다. 이는 고정 RRF/candidate 설정과
  untuned BM25 lane의 이 DEV 결과이며, Hybrid 검색 방식 자체가 열등하다는 증거가
  아니다.
- PAR-B/PAR-CH/C1의 Source Hit은 39/40/40으로 비슷하고 PAR-B MRR이 소폭 높다.
  이 표만으로 parser 추출 품질 우열을 주장할 수 없다. 구조·값 보존은 별도의
  18문서 parser audit 결과를 사용한다.
- C0→C1 차이는 기존 DEV45 진단과 같은 방향과 값으로 재현됐다. 그래도 같은
  DEV를 튜닝에 사용했으므로 최종 성능 주장은 사람 signoff 후 동결 HOLDOUT에서만
  판단한다.
- latency는 단일 순차 로컬 실행이다. 특히 D-K 첫 문항에는 KURE query model의
  lazy cold-load가 포함돼 mean latency가 왜곡되므로 조건 우열에 사용하지 않는다.

## 원본 크기와 compact derivative

full answer/trace 8개는 합계 603,510,647 bytes다. 큰 이유를 직렬화 크기로 나누면
다음과 같다.

| 중복 payload | 원본 전체 비중 |
|---|---:|
| claim 내부 citation/source 반복 | 29.48% |
| 최상위 citations의 source 반복 | 30.52% |
| evaluation trace 전체 | 35.76% |
| └ raw BM25 후보 본문 | 16.75% |
| └ post-retrieval 후보 본문 | 7.87% |
| └ post-neighbor 후보 본문 | 7.93% |
| └ final context 본문 | 2.09% |

원본은 provenance 확인을 위해 수정하지 않았다. 대신
`processed/eval/dev45-matrix-20260902/compact/`에 다음만 남긴 derivative를 만들었다.

- 원본 artifact SHA와 원본 record SHA
- 조건·corpus·top-k·tuning control
- 문항별 retrieval/gold 지표와 latency
- final source의 ID/title/rank/score
- 각 retrieval stage의 chunk/document ID, rank/score, 원문 `text_sha256`

답변·후처리·중첩 citation 객체, prompt, 반복 context 원문, preview/location 배열은
제외했다. 크기는 31,600,532 bytes로 94.76% 감소했다. 8개 조건 각각에 대해 compact
artifact로 다시 계산한 document/gold/unique-document/latency 지표가 원본과 완전히
같은지 실행기가 검증했고, 결과는 `compact/manifest.json`에 기록했다.

주요 산출물:

- `processed/eval/dev45-matrix-20260902/matrix-summary.json`
- `processed/eval/dev45-matrix-20260902/matrix-summary.csv`
- `processed/eval/dev45-matrix-20260902/input-audit.json`
- `processed/eval/dev45-matrix-20260902/compact/manifest.json`
