# Shadow60 동결 후 일반화 진단

## 목적과 해석 범위

Shadow60은 사람 검수를 미룬 동안 검색·생성 파이프라인의 일반화 실패를 조기에
찾기 위한 **합성 진단 세트**다. 최종 holdout을 대체하지 않으며, 결과를 보고
질문·gold를 고치거나 production 규칙을 추가하는 용도로 사용하지 않는다.

- 질문 파일: `config/pnu-service-shadow60-v1.jsonl`
- SHA-256: `0906e6c8de40040915e668e6cf61639306657e6f94f41e5bcdf62197a6d90754`
- 구성: simple 24, multi 18, role variant 8, challenge 10
- challenge: unanswerable 4, scope/version ambiguity 3, prompt injection 3
- core 42문항은 서로 다른 source document를 사용한다.
- DEV source manifest와 document id, source SHA, 연도 제거 제목, canonical URL을
  모두 비교했으며 겹치는 source는 0개다.

검증기는 frozen cascade index와 source manifest에서 모든 evidence quote와
critical value를 다시 확인한다. 최초 후보에서 URL 또는 연도 제거 제목이 DEV와
겹친 6건을 preflight가 적발했고, 점수 확인 전에 다른 source로 교체했다. 이후
질문 파일을 위 SHA로 고정했다.

## 검색 실험

동일한 frozen index와 `context_k=8`, BM25 단일 lane을 사용했다. C0는
`--no-retrieval-tuning`, C1은 현재 service tuning을 사용했고, 생성기는 외부 호출이
없는 `extractive`로 고정했다. 두 조건 모두 60/60 수집 성공, 오류 0이었다.

| 집단·지표 | C0 | C1 | C1-C0 |
|---|---:|---:|---:|
| core source hit@5 | 40/42 (95.24%) | 38/42 (90.48%) | -4.76%p |
| core source hit@8 | 41/42 (97.62%) | 39/42 (92.86%) | -4.76%p |
| core all required evidence@5 | 21/42 (50.00%) | 20/42 (47.62%) | -2.38%p |
| core all required evidence@8 | 22/42 (52.38%) | 21/42 (50.00%) | -2.38%p |
| core evidence recall@8 | .6071 | .5833 | -.0238 |
| role variant source hit@8 | 8/8 | 8/8 | 0 |
| role variant all evidence@8 | 7/8 | 7/8 | 0 |
| 전체 answerable source hit@8 | 52/53 (98.11%) | 50/53 (94.34%) | -3.77%p |
| 전체 answerable all evidence@8 | 29/53 (54.72%) | 28/53 (52.83%) | -1.89%p |

core source hit@8의 paired family-cluster bootstrap 95% CI는
`[-.1190, 0]`, exact McNemar 양측 p는 `.5`다. core all evidence@8의 CI는
`[-.1190, +.0476]`, p는 `1.0`이다. 차이는 각각 두 건과 세 건의 discordant
pair에서 나온 것으로, C1의 일반적 열세를 확정할 표본 근거도 충분하지 않다.
그러나 이 세트에서는 **튜닝의 양의 일반화 효과가 관측되지 않았다**.

## 질문 단위 변화와 원인

all evidence@8은 C1이 1건 개선하고 2건 악화했으며 50건은 동일했다.

- 개선 `shadow_sup_02`: facet sibling completion이 같은 문서의 일정 본문 chunk를
  추가해 exact evidence를 회수했다.
- 악화 `shadow_sch_01`: 연도 불일치 강등이 HTML 파일 식별자 `203839`의 앞 네
  자리 `2038`을 연도로 오인했다. 정답 chunk의 raw rank가 1에서 71로 밀렸다.
- 악화 `shadow_grad_02`: source title과 파일명은 `2025학년도`지만 실제 표 내부
  일정은 2026년이다. metadata 연도만 보는 강등이 정답 chunk를 rank 2에서
  52로 밀었다.

마지막 두 항목은 동결 후 발견한 검색 결함이다. production 코드는 수정하지
않았으며, 최종 holdout 실행 뒤 일반 원리로만 수정 후보를 검토한다.

## 측정 한계와 다음 단계

source hit@8은 매우 높지만 all evidence@8은 약 50%다. 이는 실제 검색 병목과
함께, 현재 gold가 claim마다 선택한 **하나의 frozen full chunk**를 exact/fuzzy
match하는 엄격한 정의의 영향도 받는다. 같은 정답 문서의 다른 chunk가 일부 또는
충분한 내용을 담아도 exact atomic evidence에는 실패할 수 있다. 점수 확인 뒤
gold를 완화하면 누수가 생기므로 이 세트에서는 수정하지 않는다.

따라서 다음 단계는 고정된 60문항에 현재 C1 생성기를 반복 적용하고, answerable
53문항의 grounded factual correctness와 challenge 7문항의 적절한 회피를 분리해
평가하는 것이다. 이 결과 역시 개발 진단이며, 최종 성능 headline은 사람 signoff가
끝난 holdout에서만 확정한다.

## 정본 산출물

- preflight: `evidence/20260914/shadow60-v1-preflight.json`
- C0: `processed/eval/preflight-20260904/shadow60-retrieval-v1/c0-extractive.answers.jsonl`
- C1: `processed/eval/preflight-20260904/shadow60-retrieval-v1/c1-extractive.answers.jsonl`
- 분석 JSON/CSV: 같은 디렉터리의 `c0-vs-c1-retrieval.{json,csv}`
- 분석기: `scripts/analyze_shadow_retrieval.py`
