# C2 quote-bound generation 실험 결정 메모

작성일: 2026-09-04
상태: DEV exploratory 완료, 서비스 미채택, holdout 검증 대기

## 결론

C2는 C1보다 좋은 생성 후보이지만 아직 production에 연결하지 않는다. 동일한
C1 검색 context에서 3-run 평균과 majority GFC가 모두 상승했으나, GFC 검정은
통계적으로 확정적이지 않고 verifier 개발에 DEV run1을 사용했다. 독립 holdout과
사람–Judge calibration을 통과하면 채택하고, 그렇지 않으면 C1을 유지한다.

## 무엇을 바꿨나

C1은 모델의 자유 서술을 받은 뒤 application이 어느 source가 각 문장을
지지하는지 추론한다. C2는 모델에게 claim마다 다음 두 값을 함께 내도록 요구한다.

- `source_number`: 어떤 검색 context를 사용했는지
- `quote`: 그 context에서 그대로 복사한 연속 원문

application은 출처 번호, quote 포함 여부, 숫자·시간·날짜, 핵심 주체, 가능/불가와
같은 관계 방향을 결정적으로 검사한다. 검증을 통과한 claim만 답변에 남기고,
통과하지 못한 claim은 근거 부족으로 표시한다. 검색 결과와 순서는 C1에서 그대로
재사용했으므로 실험은 생성–근거 결속 방식만 비교한다.

## DEV45 결과

| 지표 | C1 | C2 v3 | 차이 |
|---|---:|---:|---:|
| run별 GFC | 18, 20, 20 | 28, 25, 23 | — |
| 3-run 0–2점 평균 | 1.2148 | 1.3630 | +0.1481 |
| 2/3 majority GFC | 20/45 (.444) | 26/45 (.578) | +6, +.1333 |
| 평균 paired 승/무/패 | — | 15/21/9 | — |
| majority GFC gain/loss | — | 9/3 | — |

- 평균 차이 family-cluster bootstrap 95% CI: `[-.0000, +.3116]`
- majority GFC 차이 family-cluster bootstrap 95% CI: `[.0000, +.2826]`
- majority GFC exact McNemar 양측: `p=.145996`

점 추정치는 C2가 우세하지만 두 신뢰구간 모두 0을 포함한다. 따라서 “성능이
대폭 향상됐다” 또는 “향상이 확정됐다”고 쓰지 않는다.

## 남은 병목

C2 majority 비GFC 19문항은 다음처럼 나뉜다.

| 병목 | 문항 수 | 해석 |
|---|---:|---|
| 필수 근거가 context@8에 모두 없음 | 13 | 생성기만 바꿔서는 해결하기 어려운 검색 상한 |
| 필수 근거가 모두 있는데 비GFC | 6 | claim 선택·완전성·Judge 변동을 추가 분석할 영역 |

영역별로 academic 5/5, employment 3/3, scholarship 7/7은 majority GFC였지만,
international은 0/5였다. 작은 DEV category 수라 일반화하지 않으며, 특히
international과 student support의 손실을 독립 holdout에서 확인한다.

## 타당도 제한

1. C2 run1 raw response의 거부 사례를 보고 verifier의 일반 시간·날짜·복합어·
   관계 경계를 수정한 뒤 같은 응답을 v3로 재투영했다.
2. run2·run3는 commit `85709b5` 뒤 fresh 생성이지만 run1까지 포함한 n=3 전체가
   verifier 설계에 완전히 독립인 것은 아니다.
3. 생성기와 Judge가 모두 Gemini 계열이며 한국어 사람 calibration이 아직 없다.
4. DEV45는 반복적으로 사용됐고 표본이 작다. 최종 일반화 단위는 human-signoff된
   holdout Core 질문이다.

## 채택 gate

C2를 서비스에 연결하려면 다음을 모두 만족해야 한다.

1. holdout gold에 독립 평가자 2인의 signoff가 완료될 것
2. C1/C2 조건을 답변 수집 전에 고정하고 같은 retrieval context 또는 사전 고정된
   full E2E 조건에서 비교할 것
3. 질문별 독립 generation 3회와 strict majority GFC를 사용할 것
4. Judge–사람 calibration gate를 통과하거나, 실패 시 사람 run1 결과를
   headline으로 사용할 것
5. GFC 효과와 CI, gain/loss, terminal error, latency·호출량을 함께 보고할 것
6. C2가 나빠진 등록·입학·학생지원 사례를 사람이 직접 검토할 것

## 재현 산출물

- 점수 분석:
  `processed/eval/preflight-20260904/dev45-grounded-claims-c2-v3/analysis/c1-vs-c2-v3-3run-ab.json`
- majority GFC 분석:
  `processed/eval/preflight-20260904/dev45-grounded-claims-c2-v3/analysis/c1-vs-c2-v3-3run-majority-gfc.json`
- 육안 비교판:
  `processed/eval/preflight-20260904/dev45-grounded-claims-c2-v3/analysis/c1-vs-c2-v3-3run-review-v2.html`
- 전체 실행·SHA·호출 수:
  `docs/progress-log-20260901.md`의 “C2 v3 동결·독립 n=3 평가” 절

코드 기준점은 C2 verifier 동결 commit `85709b5`, 분석·보고서 commit
`294b98c`, 비교판 라벨 수정 commit `34291b6`이다.
