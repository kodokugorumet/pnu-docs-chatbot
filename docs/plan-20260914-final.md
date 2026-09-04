# 최종 계획 — 2026-09-14 보고서 제출까지 (작성 2026-08-31)

> 평가 방법론의 정본은 `docs/evaluation-protocol-20260914.md`다. 기존
> 45문항은 DEV, 신규 36문항은 FINAL HOLDOUT, 연구비 53문항은 별도 부록으로
> 고정한다.

마감: **9/14에 보고서까지 전부 제출**. 남은 기간 2주. 원칙은 세 가지다.

1. 보고서의 뼈대가 되는 숫자를 먼저 확보하고 **9/9에 동결**한다. 동결
   이후에는 코드·인덱스·평가를 건드리지 않는다 (수치 재측정의 연쇄 방지).
2. 검색 ablation은 retrieval-only로 수행하고, 생성 n=3은 사전 고정한
   C0(Cascade+BM25 tuning OFF)와 C1(ON)에만 쓴다.
3. 부산대 서비스 holdout·claim grounding·사람 검수를 headline으로 삼는다.
   연구비 53문항과 Dense/Hybrid는 각각 부록·exploratory 결과로 한정한다.

## 일정

| 기간 | 작업 | 산출물 |
|---|---|---|
| 8/31 (오늘) | 평가 프로토콜·C0/C1·주지표 GFC 확정 | evaluation protocol v1.0 |
| 9/1 | evaluator의 @5 오류, 명시적 설정, trace, atomic-claim schema 구현 | 재현 가능한 evaluator |
| 9/2–9/3 | Core 27 + Challenge 9, parser audit 18문서 작성·교차 검수·hash 동결 | FINAL HOLDOUT v2 |
| 9/4 | DEV 8조건 retrieval·parser audit·smoke, clean code-freeze | exploratory ablation 표·tag |
| 9/5 | HOLDOUT 4조건 retrieval one-shot, generation 시작 | 최종 검색 표·raw 일부 |
| 9/6 | Core C0/C1 162개 + Challenge C1 27개 + oracle 9개, structured judge | 최종 생성 raw·judge 산출물 |
| 9/7 | 사람 2인 blind 평가 | human labels |
| 9/8 | adjudication·통계·실패 분석·장애 복구 buffer | GFC·grounding·agreement 표 |
| 9/9 | **결과 동결일**: 최종 manifest와 모든 수치 확정 | result-freeze 태그 |
| 9/10–9/12 | **보고서 집필** + 자료 제작 (아래 구성안) | 보고서 초안 → 완성본 |
| 9/13 | 퇴고, 발표자료·데모 리허설(필요 시) | 제출본 확정 |
| 9/14 | 제출 (버퍼) | — |

지연 시 절단 순서는 cross-family/pairwise judge → Dense/Hybrid exploratory →
oracle 진단 순이다. **Core 27 holdout의 C0/C1 retrieval·generation,
structured judge, 사람 검수, clean manifest와 보고서**는 절단하지 않는다.

## 보고서 구성안 (9/10 착수 시 이 순서로)

1. 서론 — 문제 정의, 목표, 교수님 요구사항 목록과 대응 요약
2. 시스템 구조 — 아키텍처 다이어그램(`diagrams/` 기존 자산 활용)
3. Corpus 구축 — 수집(크롤 스코프·공식 출처 재수집), 파싱(HWP 파이프라인,
   게이트), 중복 제거
4. 검색·생성 — 3파서, BM25 서비스 튜닝, Dense/Hybrid exploratory,
   역할 라우팅, 생성과 claim attribution
5. 평가 방법론 — DEV 45와 FINAL HOLDOUT 36 분리, parser audit, atomic
   gold, GFC·근거·인용 지표, structured judge·사람 검수·paired 통계
6. 결과 — parser 보존, C0/C1 검색, 생성 n=3, oracle 진단, grounding·회피,
   latency·비용을 단계별로 분리해 제시
7. 한계와 향후 과제 — 작은 holdout, judge calibration, corpus 신선도,
   문서 내부 injection·멀티턴·다기관 확장
8. 부록 — 결정 로그 인덱스(워크로그 참조), 재현 명령 모음

자료 제작: parser 보존표·retrieval funnel·GFC/claim/citation 표·실패 taxonomy는
`processed/eval/final-20260909/` 산출물에서 스크립트로 추출한다. 연구비 53문항
스코어보드는 부록으로만 둔다. PDF 변환은 make-pdf 사용.

## 사용자 확인 필요 (계획 정밀도에 영향)

1. **교수님 요구사항 전체 목록** — 메모리 파일이 소실되어 요구 #2·#4의
   존재만 커밋에서 확인됨. 보고서 1장(요구사항 대응표)에 필수
2. **보고서 형식** — 분량·템플릿·국문/영문, 발표자료 필요 여부, 데모
   시연 여부
3. **브랜치 정리 방침** — feat/role-based-answers(미푸시 8커밋) 및
   feat/grant-rules-corpus의 main 병합·푸시 시점
