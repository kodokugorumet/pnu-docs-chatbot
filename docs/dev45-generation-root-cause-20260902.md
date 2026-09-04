# DEV45 historical 생성 성능 개선 폭 원인 분석 — 2026-09-02

상태: **진단 완료 / 현재 코드 재측정 필요**
대상: 2026-08-31 DEV45 baseline·tuned 생성 각 3회
범위: 저장된 historical artifact의 사후 분석. FINAL HOLDOUT 또는 현재 코드의
일반화 성능이 아니다.

## 1. 한 줄 결론

검색 튜닝으로 실제 gold chunk가 새로 들어온 5문항에서는 15개 paired 응답의
점수가 합계 11점 증가했다. 그러나 이미 gold가 있던 23문항에서 합계 11점을
잃었고, 날짜 후처리 손실과 다중 조건 누락이 겹쳐 전체 순증가는 135개 응답에서
4점뿐이었다. 이것이 평균이 0.7185에서 0.7481로 0.0296만 상승한 직접적인
산술 이유다.

## 2. 재현성과 해석 경계

- Baseline:
  `evidence/20260914/20260831-svc-gen-baseline-run{1,2,3}.jsonl`
- Tuned:
  `evidence/20260914/20260831-svc-gen-tuned-run{1,2,3}.jsonl`
- 정정 집계:
  `evidence/20260914/service-ab-summary-metric-corrected-20260901.json`
- Generator: `gemini-3.5-flash-lite`
- Inline Judge: `gemini-3.1-flash-lite`, temperature 0
- DEV cases SHA-256:
  `3c3e19d6c5524218b2f2cb1600c1c64abd80e1783ce7b235dbb80109723a3be6`
- Index SHA-256:
  `a4c1ca6318cb14721866a6512f3dfeacdca58c62ecff98ea7b81df2e75c74c31`

270/270 provider 호출은 첫 시도에 성공했으며 fallback·전송 오류는 없었다.
따라서 이번 평균 차이의 원인을 API 장애로 설명할 수 없다.

다만 manifest가 당시 코드를
`producing_retrieval_code_snapshot=unavailable-uncommitted-worktree`와
`interim-pre-postprocessor-fix`로 표시한다. 저장 JSONL에도 raw draft, claim별
판정, rejection reason, prompt·postprocessor hash가 없다. 그러므로 raw 생성
누락과 후처리 삭제를 완전한 인과관계로 분해할 수 없으며, 현재 코드 결과로
재해석해서도 안 된다.

## 3. 점수 분포가 보여 주는 현상

| 0–2점 Judge 결과 | Baseline | Tuned | 변화 |
|---|---:|---:|---:|
| 0점 | 61 | 52 | -9 |
| 1점 | 51 | 65 | +14 |
| 2점 | 23 | 18 | -5 |
| 총점 | 97/270 | 101/270 | +4 |

검색 튜닝은 완전 실패를 부분 답변으로 바꾸는 효과는 냈지만, 완전 정답이 5개
줄었다. Run-index 기준 paired 응답은 개선 20, 동일 100, 악화 15이며 개선 폭
합계 +23점과 악화 폭 -19점이 상쇄돼 +4점만 남았다.

## 4. 검색 개선이 생성으로 전달된 정도

| 검색 지표 | Baseline | Tuned |
|---|---:|---:|
| Document Hit@5 | 29/45 | 40/45 |
| Any-Gold-Chunk@8 | 23/45 | 28/45 |
| All-Gold-Chunks@8 | 10/45 | 13/45 |
| GoldChunkRecall@8 | .359 | .456 |

문서 hit는 11개 늘었지만, answer-bearing gold chunk가 새로 들어온 문항은 5개뿐이다.

- 문서와 gold chunk가 함께 신규 유입: `svc_reg_03`, `svc_reg_04`,
  `svc_adm_02`, `svc_intl_04`
- 문서 hit는 기존부터 있었고 gold chunk만 신규 유입: `svc_reg_06`
- 문서만 맞고 gold chunk는 계속 없음: `svc_grad_04`, `svc_core_01`,
  `svc_intl_02`, `svc_intl_03`, `svc_core_01_role_staff`,
  `svc_intl_03_role_free`

| Evidence 상태 | 문항 | 문항 W/T/L | paired 응답 점수 기여 |
|---|---:|---:|---:|
| miss → hit | 5 | 4/1/0 | +11 |
| hit → hit | 23 | 3/15/5 | -11 |
| miss → miss | 17 | 3/13/1 | +4 |

실제 근거 복구는 분명히 효과가 있었다. 문제는 이미 근거가 있던 문항에서 동일한
크기의 손실이 발생한 점이다. miss→miss의 +4는 DEV gold 불완전성, 비-gold 관련
문맥과 생성 변동이 섞여 있어 retrieval 개선으로 귀속하지 않는다.

## 5. 후처리와 완전성 병목

Historical 최종 답변의 고정 fallback 문구와 날짜 절단 형태로 비인과적 산술
분해를 하면 다음과 같다.

| 최종 답변 signature | Baseline 응답 / 손실점 | Tuned 응답 / 손실점 | 효과 |
|---|---:|---:|---:|
| raw abstention 추정 | 33 / 66 | 32 / 64 | +2점 |
| 모든 claim 후처리 거부 추정 | 15 / 30 | 4 / 8 | +22점 |
| `2026.` 등 날짜 절단 | 6 / 8 | 12 / 18 | -10점 |
| 나머지 정상 형식 답변 | 81 / 69 | 87 / 79 | -10점 |
| 총 손실 | 173 | 169 | +4점 |

- 모든 claim이 거부된 fallback은 15개에서 4개로 줄어 좋아졌다.
- 생성 단계의 전체 abstention 추정치는 33개에서 32개로 거의 줄지 않았다.
- 날짜가 잘린 응답은 6개에서 12개로 늘었고 18개가 모두 0점 또는 1점이었다.
- Tuned 1점 65개 중 61개의 Judge 사유에 핵심 조건의 누락·부족·불완전이
  나타났다. 검색 다음 병목은 답변 완전성이다.
- 당시 인접 commit의 `MAX_CLAIMS=5`이고 historical 답변 26/270개가 정확히
  5개 bullet이지만 raw claim 수가 없어 실제 cap 절단으로 확정할 수 없다.

대표 사례:

- `svc_acad_03`: gold recall 1.0인데 답변이
  `2026학년도 2학기 수강신청은 2026.`으로 잘려 3회 모두 0점.
- `svc_reg_06`: All-Gold@8이 false→true가 됐지만 “분할납부 1회차 대출 불가”를
  계속 누락해 양 조건 모두 `[1,1,1]`.
- `svc_sch_02`, `svc_sch_05`, `svc_sch_06`: gold 일부는 유지됐지만 날짜·자격
  facet 누락, 구 문서·회의록·입학 distractor와 abstention이 겹쳐 합계 -13점.

> 2026-09-03 후속 원문 검수: `svc_reg_06`의 위 “1회차” gold는 세부
> 분할납부 규정과 불일치했다. 2026학년도 2학기 재학생 등록금 납부계획과 같은
> 게시물 FAQ가 모두 `분할 1·4차 학자금대출 불가`를 명시하므로 현재 DEV는 이를
> 정답으로 채택했다. 따라서 위 문단은 historical Judge 실패 설명이지 현재
> 정답 기준이 아니다.

## 6. 카테고리 편차

| 카테고리 | 문항 | W/T/L | 평균 변화 | 3회 총점 기여 |
|---|---:|---:|---:|---:|
| registration | 7 | 3/3/1 | .714→.952 | +5 |
| scholarship | 7 | 1/2/4 | 1.190→.667 | -11 |
| academic | 5 | 0/5/0 | 1.200→1.200 | 0 |
| graduation | 6 | 1/5/0 | .833→.889 | +1 |
| core | 5 | 0/5/0 | .000→.000 | 0 |
| admissions | 4 | 1/3/0 | .333→.583 | +3 |
| international | 5 | 1/3/1 | .467→.400 | -1 |
| employment | 3 | 1/2/0 | .667→1.000 | +3 |
| student_support | 3 | 2/1/0 | .778→1.222 | +4 |

특히 장학 7문항의 -11점이 다른 카테고리 이득을 크게 상쇄했다. 비배타적
일정·날짜형 14문항도 평균 .833→.619, 총 -9점이었다. 평균만 보고 채택하면
이런 국소 회귀를 놓친다.

## 7. 이번에 적용한 수정과 남은 검증

1. 날짜 점 표기 보호와 짧은 완전문 보존
   - known-good DEV45 draft에서 critical-value retention .7426→1.000,
     개선 20·동일 25·악화 0.
2. 공식 UI 조회 안내의 좁은 attribution bridge
   - `svc_reg_02`에서 올바른 이월·납부확인 문장을 복원하고, 근거 없는
     분할납부 확대 문장은 계속 거부.
3. 문서당 cap 2를 유지한 explicit multi-facet sibling 교체
   - 현재 DEV45 context 변경을 2/45로 제한하면서 All-Gold@8 17→18,
     recall .500→.511, 기존 Hit/MRR/Any-Gold는 유지.
4. 현재 generation prompt의 facet-aware partial abstention
   - 근거가 있는 항목은 반드시 답하고 없는 항목만 구분해 회피하도록 명시.

위 1–4는 historical 실패 원인에 대응하지만, 같은 historical 답변을 재집계해
현재 코드의 end-to-end 향상을 주장할 수는 없다. 다음 증거가 필요하다.

- 동일한 DEV trace에서 raw draft와 현재 postprocessor projection을 별도 Judge로
  비교한다.
- `svc_sch_02`, `svc_sch_05`, `svc_acad_03`, `svc_reg_06`을 포함한 표적 생성
  preflight를 먼저 수행한다.
- 이후 동결 HOLDOUT Core 27의 C0/C1 생성 3회와 사람 calibration으로 최종
  성능을 판단한다.

외부 Judge·생성 실행은 답변과 검색 context를 외부 모델에 전송하므로 사용자의
명시적 승인 뒤 수행한다.
