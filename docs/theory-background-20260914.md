# 최종 보고서 이론적 배경 조사

작성일: 2026-08-31
용도: `docs/final-report-20260914.md` 2장·5장의 근거와 과장 방지 체크리스트

원 논문, 학회 공식 proceedings, 저자 PDF, 공식 지표 문서만 사용했다. 논문의
영문·오픈도메인 결과를 부산대 한국어 행정 문서의 예상 성능으로 전용하지 않고,
우리 평가셋에서 검증한 결과와 문헌의 일반 원리를 구분한다.
[Papers with Code](https://paperswithcode.com/)는 관련 논문과 benchmark를 찾는 discovery 보조 수단으로만
사용하며, 결과 수치와 방법론의 근거는 반드시 원 논문 또는 공식 proceedings에서
확인한다.

## 1. 검색과 재정렬

| 주제 | 이론적 근거 | 본 과제 적용 | 보고서에서 피할 표현 | 1차 출처 |
|---|---|---|---|---|
| BM25 | 확률적 관련성 프레임워크에 기반하며 용어 빈도 포화와 문서 길이 정규화를 포함한다 | 규정 번호, 사업명, 학과명, 연도처럼 exact term이 중요한 문서의 기준선 | “의미를 전혀 이해하지 못한다”, “dense보다 항상 낮다” | [Robertson & Zaragoza, 2009](https://doi.org/10.1561/1500000019) |
| Dense/DPR | 질문과 passage를 dual encoder로 표현하고 내적으로 관련도를 계산한다 | 문서 표현과 다른 자연어 의역 질의를 보완하는 후보 lane | 원 논문의 영문 QA 9–19%p top-20 개선을 우리 예상치로 사용 | [Karpukhin et al., 2020](https://aclanthology.org/2020.emnlp-main.550/) |
| 도메인 일반화 | BEIR의 이질적 zero-shot 평가에서 BM25는 강한 기준선이고 reranking 계열은 계산 비용이 크다 | 부산대 45문항에서 BM25/dense/hybrid를 동일 조건으로 비교 | 한 데이터셋의 우위를 모든 행정 질의로 일반화 | [Thakur et al., 2021](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/65b9eea6e1cc6bb9f0cd2a47751a186f-Abstract-round2.html) |
| Hybrid | sparse와 dense는 lexical·semantic match라는 다른 신호를 제공한다 | 두 lane의 후보를 넉넉히 모아 상보성을 실험 | “결합하면 반드시 각 lane보다 좋다” | [Lee et al., 2023](https://aclanthology.org/2023.acl-long.746/) |
| RRF | 원 점수가 아닌 순위의 역수를 합산해 서로 다른 점수 스케일을 결합한다 | BM25와 cosine score의 직접 정규화 없이 초기 fusion | 원 논문의 `k=60`이 모든 corpus의 최적값 | [Cormack et al., 2009](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf) |
| Cross-encoder | query와 passage를 함께 인코딩해 후보별 관련도를 세밀하게 계산한다 | top-20/50 후보만 재정렬하고 latency도 측정 | MS MARCO 개선율을 우리 예상 성능으로 전용 | [Nogueira & Cho, 2019](https://arxiv.org/abs/1901.04085) |

대표 수식은 다음과 같다.

\[
BM25(q,d)=\sum_{t\in q} IDF(t)
\frac{f(t,d)(k_1+1)}
{f(t,d)+k_1(1-b+b|d|/\mathrm{avgdl})}
\]

\[
s_{\mathrm{DPR}}(q,p)=E_Q(q)^\top E_P(p)
\]

\[
RRF(d)=\sum_{r\in R}\frac{1}{k+r(d)}
\]

## 2. RAG, 청킹과 grounding

| 주제 | 이론적 근거 | 본 과제 적용 | 보고서에서 피할 표현 | 1차 출처 |
|---|---|---|---|---|
| RAG | parametric memory와 외부 non-parametric memory를 결합한다 | 부산대 공식 문서를 검색해 생성 컨텍스트와 출처로 사용 | 우리 구현이 RAG-Sequence/RAG-Token과 동일하거나 retriever를 공동학습했다고 표현 | [Lewis et al., 2020](https://proceedings.neurips.cc/paper/2020/hash/6b493230-Abstract.html) |
| 검색 단위 | 문서·passage·sentence·proposition 단위 선택이 retrieval과 QA에 영향을 준다 | 제목–절–항목, 표 행, 연도, 문서명을 보존하고 청크 크기를 실험 변수로 둔다 | “청크는 작을수록 항상 좋다” | [Chen et al., 2024](https://aclanthology.org/2024.emnlp-main.845/) |
| 긴 컨텍스트 | 입력 중간의 정보를 덜 이용하는 U자형 경향이 여러 모델·과제에서 관찰됐다 | 근거 수를 무작정 늘리지 않고 reranking과 컨텍스트 순서를 고정 | “모든 LLM은 중간 정보를 읽지 못한다” | [Liu et al., 2024](https://aclanthology.org/2024.tacl-1.9/) |
| Faithfulness/factuality | 제공된 근거와의 일치와 외부 사실과의 일치는 다른 축이다 | 정답성과 별도로 답변 핵심 주장의 근거 청크 지지를 평가 | 답이 맞으면 grounded이거나 출처가 있으면 factual하다고 단정 | [Maynez et al., 2020](https://aclanthology.org/2020.acl-main.173/) |
| 검색과 환각 | retrieval augmentation은 환각을 줄일 수 있지만 제거하지 않는다 | 근거 부족 시 회피, 날짜·숫자·대상 조건 검증, 문장별 출처 | “RAG가 환각을 해결했다”, “검색이 맞으면 답도 맞다” | [Shuster et al., 2021](https://aclanthology.org/2021.findings-emnlp.320/), [Stolfo, 2024](https://aclanthology.org/2024.findings-naacl.100/) |

최신성 실패는 dense retrieval만으로 해결되지 않는다. 질문 연도, 문서 개정일·
적용 기간 metadata, 구판·최신판 경쟁을 별도 변수로 평가해야 한다. 이번 서비스
실험에서도 `2026`이 많은 문서에 등장해 변별력이 낮았고, 연도 일치 후보의
stable partition이 MRR 개선에 기여했다.

## 3. 검색·생성 평가 지표

### 3.1 Hit@k

\[
Hit@k=\frac{1}{N}\sum_{i=1}^{N}
\mathbf{1}[TopK_i\cap G_i\neq\varnothing]
\]

상위 k개 안에 하나 이상의 gold 문서가 있는 질의 비율이다. 본 과제의 문서
Hit@5 0.889는 “88.9% 답변 정답률”이 아니다. 문서는 맞지만 필요한 조항·표
청크가 빠질 수 있으므로 DEV에서는 GoldChunk coverage@k를, atomic gold가 있는
holdout에서는 Evidence Recall/All-Evidence@k를 별도로 제시한다.

### 3.2 MRR

\[
MRR=\frac{1}{N}\sum_{i=1}^{N}\frac{1}{rank_i}
\]

첫 관련 결과가 없으면 0으로 처리한다. 첫 근거가 얼마나 앞에 나타나는지
반영하지만 그 뒤의 관련 문서 완전성은 측정하지 않는다. 정의와 TREC QA의
사용 맥락은 [NIST TREC QA](https://trec.nist.gov/data/qa.html)를 따른다.

### 3.3 RAGAS와 답변 점수

RAGAS는 retrieval의 관련·집중도, 생성의 faithfulness, 답변 품질처럼 RAG의
구성 요소를 분리한다
([Es et al., 2024](https://aclanthology.org/2024.eacl-demo.16/)). 우리
평가도 문서 Hit/MRR, 근거 청크 Hit, 0–2점 답변 judge를 합치지 않는다.
RAGAS Answer Correctness는 사실·의미 유사도를 결합한 연속형 score이므로
퍼센트 정답률로 표현하지 않는다
([공식 문서](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_correctness/)).

검색과 생성의 교차 해석은 다음과 같다.

| 검색 | 생성 | 우선 진단 |
|---|---|---|
| hit | 실패 | 답 청크 선택, 컨텍스트 활용, 프롬프트·후처리 |
| miss | 실패 | retrieval, 청킹, corpus 부재 |
| miss | 성공 | gold 불완전성, 다른 유효 근거, parametric memory |
| hit | 성공 | 근거 coverage와 세부 조건 완전성 확인 |

## 4. LLM-as-a-judge의 편향과 반복성

| 위험 | 문헌 근거 | 본 과제 통제 | 남는 한계 | 1차 출처 |
|---|---|---|---|---|
| 위치·장황성·self-enhancement 편향 | 강한 judge의 유용성과 함께 여러 편향이 관찰됨 | 모델·버전·prompt·rubric·temperature 고정 | 점수는 객관적 진실이 아님 | [Zheng et al., 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html) |
| 순서 편향 | 답변 순서만 바꿔도 pairwise 승패가 바뀔 수 있음 | 경쟁 비교 A/B와 B/A 양방향 blind 평가 | 순서 교환도 모든 편향을 제거하지 않음 | [Wang et al., 2024](https://aclanthology.org/2024.acl-long.511/) |
| 자기불일치 | 동일 설정의 judge 점수가 반복 실행에서 달라질 수 있음 | 고정 답변을 별도로 3회 재채점 | 생성 n=3은 judge 신뢰도 검증이 아님 | [Haldar & Hockenmaier, 2025](https://aclanthology.org/2025.findings-emnlp.1361/) |
| LLM 문체 선호 | LLM 생성문을 선호할 가능성이 보고됨 | 내용 중심 rubric과 사람 표본검증 | 인간 기준답안과 모델 답변 문체 차이 | [Liu et al., 2023](https://aclanthology.org/2023.emnlp-main.153/) |
| 사람과의 정렬·관대함 | 13개 judge를 비교했을 때 가장 강한 모델도 사람 점수와 차이가 남았고, 높은 percent agreement만으로 점수 정렬을 보장할 수 없었으며 prompt 복잡도 민감성과 leniency가 관찰됨 | 원시 일치율뿐 아니라 chance-corrected·class-wise 지표와 오판 유형을 보고 | 높은 반복 일치율을 사람 수준 정확성으로 해석 | [Thakur et al., 2025](https://aclanthology.org/2025.gem-1.33/) |
| 다국어 판정 불안정성 | 5개 과제·25개 언어 실험의 평균 Fleiss' kappa가 약 0.3이었고, 일관성은 언어별로 크게 달랐음 | 한국어 부산대 행정 질의에서 독립 2인 평가와 합의 라벨로 별도 calibration | 영어 benchmark 또는 모델 크기만으로 한국어 신뢰도를 가정 | [Fu & Liu, 2025](https://aclanthology.org/2025.findings-emnlp.587/) |
| 점수 범위 편향 | reference-free 요약 직접 채점에서 judge 출력이 사전 지정 score range에 민감했음 | 모든 조건에 동일한 0–2점 rubric과 의미를 고정하고 다른 점수 범위의 결과를 직접 비교하지 않음 | 고정 척도가 편향을 제거했다고 주장 | [Fujinuma, 2026](https://aclanthology.org/2026.findings-acl.657/) |
| 클래스 불균형 | accuracy·precision·F1은 클래스 비율과 positive-class 선택에 민감하며, Youden's J의 선형 변환인 balanced accuracy가 judge 선택에 더 적합하다고 제안됨 | GFC pass/fail의 혼동행렬·클래스 수와 balanced accuracy를 함께 보고 | 다수 클래스 위주의 높은 accuracy만으로 gate 통과 | [Collot et al., 2026](https://aclanthology.org/2026.eacl-industry.69/) |

최종 경쟁 비교는 서비스명을 지우고 우리/상대를 A/B와 B/A 순서로 각각
평가한다. 두 결과가 같은 실제 시스템을 선택할 때만 안정 승패로 보고, 순서에
따라 바뀌면 순서 민감 사례로 분리한다.

### 4.1 일치도와 분류 지표의 역할

단일 지표는 서로 다른 실패를 가린다. 본 과제는 사람–사람 GFC에 원시 일치율과
Cohen's kappa를, 순서가 있는 0–1–2점에는 quadratic-weighted kappa(QWK)를
사용한다. Judge–합의 사람 GFC에는 혼동행렬, Cohen's kappa, macro-F1,
balanced accuracy와 false-pass rate를 함께 제시한다. Balanced accuracy는

\[
\mathrm{BalancedAccuracy}=\frac{1}{2}
\left(\frac{TP}{TP+FN}+\frac{TN}{TN+FP}\right)
\]

로, GFC 통과·실패 양쪽 recall을 같은 비중으로 반영한다. 다만 한 클래스가 표본에
전혀 없으면 정의할 수 없으므로 gate를 통과시키지 않는다. 모든 지표에는 실제
클래스별 문항 수를 병기해 작은 표본과 불균형을 숨기지 않는다. 이 설계는 문헌의
영어·다국어 평균 결과를 한국어 성능 수치로 가져오는 것이 아니라, 한국어 대학
행정 도메인에서 사람 calibration을 직접 수행하기 위한 통제다.

## 5. 최종 평가에 적용할 최소 프로토콜

1. 반복 튜닝에 사용한 45문항은 개발셋이라고 명시한다.
2. 검색은 Hit@1·3·5, MRR, DEV GoldChunk coverage 또는 holdout atomic
   Evidence Recall, latency, 신규·상실 적중을 제시한다.
3. 생성은 설정별 n=3과 문항별 승·무·패, run 산포를 제시한다.
4. 역할 변형은 기본 문항과 같은 family로 묶어 bootstrap한다.
5. 고정 답변 judge 반복은 생성 반복과 분리한다.
6. 평균 차이는 paired cluster bootstrap 95% 신뢰구간과 함께 보고한다.
7. 한국어 도메인 답변을 두 독립 평가자가 blind 판정하고, 합의 라벨을 별도로
   만든다. 사람–사람 raw agreement·kappa·QWK와 Judge–사람 혼동행렬·
   kappa·macro-F1·balanced accuracy를 계산한다.
8. GFC 통과·실패의 클래스 수와 비율을 공개하고, 한 클래스가 없거나 kappa가
   정의되지 않으면 calibration gate를 통과시키지 않는다.
9. 최신성, 검색 miss, 검색 hit–생성 실패, 경계 점수를 층화해 사람이 검토한다.
10. 모든 주장에 “해당 평가셋·judge·수집일 기준”이라는 범위 한정을 붙인다.

현재 결과는 서로 다른 시점과 표본으로 분리해 해석한다.

- **현재 코드 DEV 검색 45문항:** C0→C1의 Hit@5는 0.644→0.889,
  MRR은 0.494→0.717이다. 반복 튜닝한 DEV 안의 검색 지표이며 최종
  일반화 성능이 아니다.
- **2026-08-31 historical 생성:** 당시 end-to-end 0–2점 inline judge 평균은
  0.7185→0.7481이지만 paired cluster bootstrap 95% CI가 0을 포함했다.
  현재 코드 재실행 결과로 섞어 쓰거나 생성 개선으로 확정하지 않는다.
- **현재 코드 `p0g` DEV 3문항:** GFC는 C0 1/3, C1 2/3이었으나 유효
  질문 표본은 3개이고 paired bootstrap 95% CI는 `[0.000, 1.000]`, exact
  sign-flip과 McNemar의 `p=1.0`이다. 방향성 진단일 뿐 headline이 아니다.
- **최종 holdout 36문항:** 실제 2인 사람 검수·calibration과 실행이 아직
  완료되지 않았다. 따라서 현 시점에는 최종 생성 성능이나 일반화 개선 수치가 없다.

검색과 생성 성과를 하나의 “정확도”로 합치면 안 된다.
