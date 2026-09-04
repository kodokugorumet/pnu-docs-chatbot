# 부산대학교 공문서 기반 검색 증강 생성 챗봇의 설계와 평가

> **작업 초안 — 2026-09-01.** 2026-09-09 결과 동결 뒤 확정한다.
> 수치의 상태와 원천은 `docs/report-evidence-ledger-20260914.md`를 따른다.

## 초록

대학의 학사·입학·등록·장학·국제교류·학생지원 정보는 여러 부서의 웹페이지와
PDF, HWP, 표 형식 문서에 분산되어 있다. 일반적인 생성 모델은 최신 교내 규정과
공고를 안정적으로 기억하지 못하며, 출처 없이 답할 경우 날짜·대상·절차를
혼동할 위험이 있다. 본 과제는 부산대학교 공식 문서를 수집·구조화하고, 검색된
근거 안에서 답변과 출처를 제공하는 검색 증강 생성(retrieval-augmented
generation, RAG) 챗봇을 구축했다.

총 10,977개의 크롤 완료 행에서 2,260개 문서를 선별했으며, cascade 파서로
2,247개 문서를 처리해 44,591개 청크와 7,813개 표를 추출했다. 연구비 규정
트랙에서는 lexical·dense 검색, 순위 융합, cross-encoder 재정렬과 구조 보존형
부모 확장을 단계적으로 비교했다. 현재 서비스 headline 조건은 Cascade 파싱
결과에 대한 BM25 검색과 근거 기반 생성이며, 평가는 검색과 생성을 분리했다.
서비스 개발셋 45문항의 기준선은 문서 Hit@5 0.644, MRR 0.494였고,
2026-09-02 현재 코드의 검색 후보 C1은 같은 DEV에서 Hit@5 0.889와
MRR 0.717을 기록했다. 2026-09-04 현재 코드로 C0/C1을 각각 3회 다시 생성한
0–2점 LLM judge 평균은 0.8815와 1.2148이었고, 문항 family 단위 paired cluster
bootstrap 평균 차이 95% 신뢰구간은 [0.0889, 0.5887]이었다. 반면 3회 중 2회
이상 완전정답인 문항은 13/45에서 20/45로 증가했지만 차이의 95% 신뢰구간
[-0.0217, 0.3201]이 0을 포함했다. 따라서 평균 답변 품질 개선은 동일 DEV와
해당 Judge 설정에 한정해 해석하며, 완전정답률이나 일반화 성능 개선은 확정적으로
주장하지 않는다. 연구비 규정 50문항
비교 결과도 최신 경쟁 서비스 재수집과 순서 교환 평가 후 최종 수치로 고정한다.

본 과제의 기여는 (1) 한국어 대학 행정 문서의 수집·파싱·품질 검사 파이프라인,
(2) 서비스 headline으로 채택한 BM25 단일 lane 근거 검색·출처 제시와
dense/hybrid 탐색 실험의 분리, (3) 검색·근거·생성 오류를 분리하는 평가 체계,
(4) 채택뿐 아니라 실패한 실험까지 보존한 재현 가능한 의사결정 기록에 있다.

**주제어:** 검색 증강 생성, RAG, BM25, dense retrieval, reciprocal rank
fusion, 공문서 질의응답, 근거성 평가

## 1. 서론

### 1.1 배경과 문제 정의

대학 행정 정보는 정확한 고유명사와 연도, 대상자, 신청 기간, 예외 조건이
답변의 의미를 결정한다. 그러나 공식 정보가 부서별 게시판, 정적 안내 페이지,
첨부파일에 흩어져 있고 동일 주제의 구판과 최신판이 함께 남아 있다. 단순 키워드
검색은 사용자가 문서의 표현을 모르면 적절한 근거를 찾기 어렵고, 생성 모델만
사용하면 최신성 및 출처 추적이 어렵다.

본 과제는 부산대학교 공식 문서를 외부 지식 저장소로 사용하고, 질문과 관련된
청크를 검색한 뒤 검색 근거에 제한된 답변을 생성하는 서비스를 구현한다. 여기서
성능은 하나의 숫자로 정의하지 않는다. 첫째, 올바른 문서와 근거 청크가 상위에
검색되는가. 둘째, 생성기가 검색 근거를 빠뜨리거나 왜곡하지 않는가. 셋째,
사용자가 답변의 출처와 적용 범위를 확인할 수 있는가를 별도로 평가한다.

### 1.2 연구 질문

본 보고서는 다음 질문에 답한다.

1. 이질적인 대학 웹 문서를 검색 가능한 구조로 어떻게 변환할 수 있는가?
2. exact term에 강한 sparse 검색과 의역에 강한 dense 검색을 어떻게 결합할
   때 한국어 행정 질의의 근거 검색이 개선되는가?
3. 검색 지표의 개선은 최종 답변 품질로 일관되게 전이되는가?
4. LLM judge의 변동과 편향을 고려하면서 서비스와 경쟁 시스템을 어떻게
   공정하게 비교할 수 있는가?

### 1.3 목표와 범위

범위는 부산대학교 및 과제에서 정의한 공공기관 공식 문서다. 본 구현은 원 RAG
논문의 retriever·generator 공동학습을 재현한 시스템이 아니라, RAG 원리를
적용한 retrieval-augmented QA 시스템이다. 평가 수치는 사용한 코퍼스, 질의셋,
모델, judge, 수집일에 한정한다.

### 1.4 요구사항 대응표

교수자 요구사항 원문을 확보하면 아래 표를 최종 확정한다.

| 요구사항 | 구현 | 검증 | 상태 |
|---|---|---|---|
| 공식 문서 기반 답변과 출처 | 검색 청크 기반 생성·출처 카드 | 검색/근거/생성 평가 | 구현 |
| 사용자 역할별 답변 | 프리셋 및 자유입력의 안전한 역할 매핑 | 역할 router 단위 테스트 | 구현 |
| 다양한 공문서 형식 처리 | HTML, PDF, HWP/HWPX, Office 문서 cascade 파싱 | 파서 프로필 비교 | 구현 |
| 성능 최적화와 비교 평가 | 단계별 A/B, DEV 45·연구비 53문항 평가, holdout 36문항 계획 | 본 보고서 5–6장 | 진행 중 |
| 나머지 요구사항 원문 | 사용자 제공 필요 | 대응 근거 연결 | 확인 필요 |

## 2. 이론적 배경과 관련 연구

### 2.1 BM25 기반 sparse retrieval

BM25는 확률적 관련성 프레임워크에 기반한 lexical ranking 함수로, 질의 용어의
희소성, 문서 내 용어 빈도의 포화, 문서 길이 정규화를 결합한다
([Robertson and Zaragoza, 2009](https://doi.org/10.1561/1500000019)). 질의
\(q\)와 문서 \(d\)에 대한 대표적 형태는 다음과 같다.

\[
BM25(q,d)=\sum_{t\in q} IDF(t)
\frac{f(t,d)(k_1+1)}
{f(t,d)+k_1(1-b+b|d|/\mathrm{avgdl})}
\]

부산대 행정 문서는 사업명, 학과명, 규정 번호, 연도처럼 정확한 문자열이
중요하므로 BM25가 강한 기준선이다. 반면 질문과 문서의 표현이 다르면 lexical
overlap이 부족하고, 한국어 대화형 어미와 저변별 연도 토큰이 후보 순위에 잡음을
줄 수 있다. 따라서 토큰화와 질의 정규화도 검색 모델의 일부로 평가해야 한다.

### 2.2 Dense retrieval과 도메인 일반화

DPR은 질문과 지문을 각각 dual encoder로 표현하고 두 벡터의 내적으로 관련도를
계산한다([Karpukhin et al., 2020](https://aclanthology.org/2020.emnlp-main.550/)).

\[
s_{\mathrm{DPR}}(q,p)=E_Q(q)^\top E_P(p)
\]

Dense retrieval은 문서와 표현이 다른 자연어 의역을 보완할 수 있다. 그러나
BEIR은 이질적 데이터셋의 zero-shot 평가에서 BM25가 여전히 강한 기준선이며,
고성능 reranking 계열은 더 큰 계산 비용을 요구함을 보였다
([Thakur et al., 2021](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/65b9eea6e1cc6bb9f0cd2a47751a186f-Abstract-round2.html)).
따라서 영문 open-domain QA에서 보고된 dense 우위를 한국어 대학 행정 문서에
그대로 일반화하지 않고, 동일 평가셋에서 BM25, dense, hybrid를 비교한다.

### 2.3 Hybrid retrieval과 Reciprocal Rank Fusion

Sparse와 dense 검색은 각각 lexical match와 semantic match라는 상이한 신호를
제공한다. 두 신호의 상보성은 hybrid retrieval의 근거가 되지만, 결합이 항상
개별 검색기보다 우수하다는 보장은 없으므로 ablation이 필요하다
([Lee et al., 2023](https://aclanthology.org/2023.acl-long.746/)).

BM25 점수와 cosine similarity는 스케일이 달라 직접 합산하기 어렵다.
Reciprocal Rank Fusion(RRF)은 원 점수 대신 각 검색기의 순위를 사용한다
([Cormack et al., 2009](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)).

\[
RRF(d)=\sum_{r\in R}\frac{1}{k+r(d)}
\]

원 논문은 \(k=60\)을 사용했지만 이를 보편 최적값으로 가정하지 않는다. 본
과제에서는 각 lane의 후보를 넉넉히 수집한 뒤 RRF로 결합하고, 실측 결과로
채택 여부를 결정한다.

### 2.4 Cross-encoder 재정렬

Cross-encoder는 질의와 후보 passage를 하나의 입력으로 함께 처리하여 세밀한
관련성 점수를 계산한다. 후보마다 추론해야 하므로 전체 코퍼스 검색보다는 초기
검색의 상위 후보를 재정렬하는 단계에 적합하다
([Nogueira and Cho, 2019](https://arxiv.org/abs/1901.04085)). 본 과제는 BM25와
dense 후보를 결합한 뒤 상위 후보만 재정렬하고, 검색 순위와 재정렬 점수를 다시
융합해 한 점수의 전횡을 방지한다.

### 2.5 Retrieval-Augmented Generation

RAG는 생성 모델의 parametric memory와 외부 검색 인덱스의 non-parametric
memory를 결합한다([Lewis et al., 2020](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html)).
외부 문서를 갱신하고 출처를 추적할 수 있다는 점이 대학 행정 질의에 적합하다.
다만 retrieval augmentation은 환각을 줄일 수 있어도 제거하지는 않는다
([Shuster et al., 2021](https://aclanthology.org/2021.findings-emnlp.320/)).
본 시스템은 근거가 부족하면 한계를 밝히고, 숫자·기한·대상 조건은 검색 문서로
확인하도록 프롬프트와 출력 구조를 설계한다.

### 2.6 청킹, 구조 보존과 긴 컨텍스트

문서, passage, sentence, proposition 중 어떤 단위를 색인하는지는 retrieval과
downstream QA 모두에 영향을 준다. Dense X Retrieval은 해당 실험 환경에서
proposition 단위가 passage보다 우수할 수 있음을 보였다
([Chen et al., 2024](https://aclanthology.org/2024.emnlp-main.845/)). 이 결과를
“작은 청크가 항상 좋다”로 일반화하지 않고, 본 과제에서는 제목–절–항목 계층,
표 행, 공고 연도, 문서명을 보존한다.

긴 컨텍스트를 입력할 수 있다는 사실은 그 안의 정보를 균일하게 이용한다는
뜻이 아니다. 관련 정보가 입력 중간에 있을 때 성능이 낮아지는 현상이 보고됐다
([Liu et al., 2024](https://aclanthology.org/2024.tacl-1.9/)). 따라서 문서를
무작정 많이 넣기보다 근거 밀도를 높이고, 핵심 근거의 순서와 컨텍스트 수를
평가 설정에 고정한다.

### 2.7 Faithfulness, factuality와 groundedness

Faithfulness는 생성문이 제공된 근거와 일치하는지, factuality는 외부의 실제
사실과 일치하는지를 뜻하며 동일한 개념이 아니다
([Maynez et al., 2020](https://aclanthology.org/2020.acl-main.173/)). RAG 장문
생성에서도 정답을 포함하면서 근거 없는 진술이 남을 수 있다
([Stolfo, 2024](https://aclanthology.org/2024.findings-naacl.100/)). 그러므로
정답성과 별도로 핵심 주장이 검색 청크에서 지지되는지를 평가한다.

### 2.8 RAG 평가와 LLM-as-a-judge

RAGAS는 retrieval의 집중도, 생성의 faithfulness, 답변 relevance 등 구성
요소를 분리해 평가하는 접근을 제안한다
([Es et al., 2024](https://aclanthology.org/2024.eacl-demo.16/)). 본 과제도
Hit@k와 MRR을 문서 검색 지표로, DEV GoldChunk coverage 또는 holdout atomic
Evidence Recall을 근거 지표로, 고정 rubric의 답변 점수를 생성 지표로
구분한다. Answer Correctness와 같은 연속형 evaluator
score는 이항 정답률로 해석하지 않는다
([RAGAS documentation](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_correctness/)).

LLM judge는 확장 가능한 평가 수단이지만 position, verbosity,
self-enhancement bias가 보고됐다
([Zheng et al., 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html)).
답변 순서만 바꿔도 판단이 달라질 수 있으므로 경쟁 비교에서는 A/B와 B/A를
모두 평가한다([Wang et al., 2024](https://aclanthology.org/2024.acl-long.511/)).
또한 동일 설정의 judge도 자기일관성이 낮을 수 있으므로 서로 다른 생성 답변의
3회 점수와 동일 답변의 judge 반복을 구분한다
([Haldar and Hockenmaier, 2025](https://aclanthology.org/2025.findings-emnlp.1361/)).

최근 검증 연구는 반복 일치율만으로 judge의 타당성을 주장할 수 없음을 더
구체화한다. Thakur et al.은 13개 judge를 비교해 가장 강한 모델에서도 사람
점수와 차이가 남고, percent agreement가 높아도 점수 정렬과 오류 크기를 가릴
수 있으며 prompt 복잡도 민감성과 관대한 채점이 나타날 수 있음을 보였다
([Thakur et al., 2025](https://aclanthology.org/2025.gem-1.33/)). Fu와 Liu는
5개 과제·25개 언어에서 다국어 판정의 평균 Fleiss' kappa가 약 0.3이고
언어별 편차가 큼을 보고했다. 이 결과를 한국어 수치로 전용하지는 않지만,
부산대 한국어 행정 질의에서 별도 사람 calibration이 필요한 근거로 삼는다
([Fu & Liu, 2025](https://aclanthology.org/2025.findings-emnlp.587/)).

직접 점수 채점은 척도 설계에도 민감하다. Fujinuma는 reference-free 요약
평가에서 사전 지정 score range에 따른 편향을 보였으므로, 본 과제는 모든
조건에서 0–1–2점의 의미와 rubric을 고정하고 다른 범위로 얻은 수치를 직접
비교하지 않는다. 다만 해당 실험 조건과 본 과제의 근거 기반 한국어 RAG는
다르므로 고정 척도가 편향을 제거한다고 주장하지 않는다
([Fujinuma, 2026](https://aclanthology.org/2026.findings-acl.657/)). GFC
pass/fail처럼 클래스 비율이 치우칠 수 있는 평가에서는 accuracy만으로 judge를
고르지 않는다. Collot et al.의 분석에 따라 클래스별 recall을 같은 비중으로
반영하는 balanced accuracy와 혼동행렬을 함께 사용한다
([Collot et al., 2026](https://aclanthology.org/2026.eacl-industry.69/)).

[Papers with Code](https://paperswithcode.com/)는 관련 연구와 benchmark를 찾는
discovery 보조 수단으로만 사용했다. 본 보고서의 결과 근거와 인용은 원 논문과
공식 proceedings를 따른다.

## 3. 시스템 설계

### 3.1 전체 구조

시스템은 다음 파이프라인으로 구성된다.

```text
공식 웹/첨부 문서
  → 수집 범위·중복·무결성 검사
  → 형식별 cascade 파싱
  → 구조 보존 청킹과 corpus gate
  → BM25 및 dense 인덱스
  → 후보 융합·재정렬·문서 다양성 제한
  → 근거 컨텍스트 구성
  → 역할 관점을 반영한 답변 생성
  → 문장별 출처와 한계 표시
```

기존 데모 아키텍처 자산은 `diagrams/pnu-rag-demo-architecture.*`에 있다. 이
그림은 BM25 중심 데모 시점의 구조이므로 최종 보고서에서는 최신 hybrid·역할
경로를 반영해 갱신하거나 “교수자 데모 기준 구성”으로 범위를 한정한다.

### 3.2 수집과 코퍼스 선별

공식 부산대학교 도메인을 robots 정책과 요청 간격을 지키며 수집했다. 크롤 상태
기록에는 완료 10,977행, 페이지 7,775개, 파일 3,202개가 남아 있다. 이후 허용
host·확장자·크기·무결성·중복 규칙으로 2,260개 문서, 473,678,602 bytes를
선별했다. SHA-256 중복 118행을 제거했고, 선별 manifest의 SHA-256은
`1fa7e0f2f5d03a2a1249584d91f7232c9b2cb4642b39115bf3c9e6a871845a2f`다.

### 3.3 다형식 파싱과 품질 gate

HTML, PDF, HWP/HWPX, DOCX, XLSX 등 형식별 파서를 cascade로 적용했다.
2,260개 입력 중 2,247개를 파싱했고 44,591개 청크, 7,813개 표를 추출했다.
검색 인덱스의 44,520청크는 파싱 산출물에서 corpus gate를 통과한 단위이며,
71개 차이는 최종 보고서에서 제거 규칙과 함께 설명한다.

파서 프로필 24문항 비교에서 Baseline은 23/24, Challenger와 Cascade는 각각
24/24 문서 Hit@5를 기록했다. 이 결과와 운영 복잡도를 함께 고려하여 Cascade를
채택했다. 이는 24문항 문서 retrieval 기준의 운영 결정이며 파서 추출 품질이나
전체 코퍼스에서의 우월성을 뜻하지 않는다.

### 3.4 검색 파이프라인

연구비 규정 트랙의 채택 파이프라인은 BM25와 한국어 dense 후보 수집, RRF,
cross-encoder, 검색순위와 CE순위의 재융합, 문서당 청크 제한, section_path
부모 확장 순서다. 서비스 트랙은 2026-08-31 기준 BM25 API 경로의 128개 후보에
대해 대화형 접미어로 생기는 저정보 토큰을 완화하고, 일반 안내 페이지 및 질문
연도와 불일치하는 문서를 안정적으로 뒤로 보내는 후보를 검증 중이다.

### 3.5 답변 생성과 역할 처리

생성기는 검색된 근거와 출처를 입력받아 답변한다. 사용자 역할은 학생·교직원·
연구자 등 프리셋 또는 자유입력으로 받을 수 있으나, 자유 텍스트를 그대로
프롬프트에 삽입하지 않는다. 규칙 기반 router가 가장 가까운 안전한 역할에
매핑하고, 역할은 hard filter가 아니라 답변 관점과 소프트 우선순위로만
사용한다. 부산대 단일기관 질의에서 역할 라우팅이 검색 점수를 직접 개선하지
않으므로 이 기능은 “성능 개선”보다 “요구사항 구현과 입력 안전성”으로 평가한다.

## 4. 성능 개선 방법

### 4.1 단계적 A/B 원칙

한 번에 여러 변수를 바꾸면 원인을 알 수 없으므로 후보 검색, 융합, 재정렬,
부모 확장, 컨텍스트 배치, 생성 프롬프트를 단계별로 비교했다. 비용이 없는 검색
스모크에서 Hit@k·MRR·근거 적중과 손실 문항을 먼저 확인하고, 채택 후보만 생성
n=1을 거쳐 n=3으로 재측정한다. 평균이 올라도 기존 적중 문항을 잃거나 특정
카테고리가 악화되면 문항 단위로 원인을 검토한다.

### 4.2 연구비 규정 트랙

초기 BM25의 RAGAS context recall 0.4522에서 cross-encoder, 부모 확장,
순위 융합, dense lane을 단계적으로 추가하여 0.5266을 기록했다. 동일 작업의
생성 평가는 0–2점 judge 평균 0.81에서 1.15로 상승했다. 다만 RAGAS 0점인데
생성은 1–2점인 지표 허상, 미수집 문서, 해석례 부재를 분리해 분석했다. 이
수치는 2026-08-06 설정의 결과이며 최종 표에는 모델·평가셋·judge 조건을 함께
기재한다.

### 4.3 서비스 검색 실패 원인과 튜닝

서비스 기준선의 16개 문서 miss를 추적한 결과, 대부분의 정답 문서는 128개
후보 pool 안에 있었으나 다음 잡음에 밀려 최종 top-5에서 탈락했다.

1. 한국어 대화형 어미에서 파생된 저정보 token과 bigram
2. 많은 문서에 등장해 변별력이 낮은 연도 token
3. FAQ·이용 안내처럼 여러 질의와 부분 일치하는 일반 문서
4. 질문 연도와 다른 구판 공고

서비스 전용 경로에서 질의의 핵심 명사 매칭은 유지하면서 대화형 접미어 토큰을
FTS OR query에서 제외하고, 일반 문서 및 연도 불일치 후보를 stable partition으로
뒤로 보냈다. 직접 benchmark 경로의 기본 동작은 유지해 기존 연구비 평가에
영향을 주지 않도록 했다.

검색된 문서 안에서도 질문이 요구한 날짜·금액·자격·방법 중 일부만 context에
남는 문제가 있었다. 이를 위해 질문이 둘 이상의 facet을 명시적으로 요구하고,
같은 문서의 현재 두 chunk가 한 facet만 중복 제공할 때에만 저기여 chunk를
누락 facet의 인접 sibling으로 교체했다. 문서당 cap은 2로 유지하며, 질문과
sibling의 학년도·학기가 충돌하면 교체하지 않는다.

### 4.4 일반 규칙과 표적 규칙

동결 시점의 production decision rule 68개를 특정 DEV 실패 문항 없이도 원리로
설명할 수 있는 일반 규칙 21개와, 관측된 DEV 실패를 고치기 위해 도입한 표적
규칙 47개로 분리했다. 표적 규칙이 겨냥한 DEV ID는 역할 변형을 포함해 중복 제거
37개다.[^rule-inventory] 일반 규칙에는 입력 안전 정규화, 제목·본문 매칭, 연도
충돌 처리, facet 분리의 공통 조건이 포함된다. 표적 규칙에는 학생증·증명서
위탁, 교환학생, 자부담금, D-2 단체접수, 하계방학 자격증, 제3자 인권신고,
모집인원 변경 등 반복 실패에서 추가된 질의 확장·점검표·후처리 bridge가
포함된다.

이 분류는 표적 규칙을 성능 기여로 정당화하기 위한 것이 아니라 DEV 과적합 위험을
드러내기 위한 것이다. 동결 후에는 규칙을 더 고치지 않고, 별도 분석기가 holdout의
실제 trace 발동과 질의상 발동 가능성을 분리 집계한다. stable rule ID가 trace에
없는 경우에는 실제 발동으로 간주하지 않고 `trace_absent`로 남긴다.

### 4.5 채택하지 않은 대안

| 대안 | 결과 | 판단 |
|---|---|---|
| CE 단독 교체 | context recall +0.012 | 새 근거를 찾지만 BM25 직격 결과를 밀어내 융합 필요 |
| 생성형 reranker | 판별형보다 낮은 0.4248 | 전용 입력 형식 민감도와 비용 때문에 기각 |
| 연구비 트랙의 당시 문서당 청크 cap 완화 | 0.3954로 악화 | 장황한 문서가 슬롯을 독식 |
| 서비스 DEV의 전역 cap 2→4 | GoldChunkRecall은 증가했지만 31/45 context 변경, 고유 문서 감소 | 영향 범위가 커서 미채택; cap 2에서 누락 facet만 교체 |
| 부모 확장 단독 | 0.4364 | 잘못 고른 청크를 확장하므로 reranker 뒤에서만 사용 |
| Dense 모델 KURE-v1 | 평균 동률, 0점 문항 2개 증가 | Snowflake-ko lane 채택 |
| 불확실 신규 실험 | 제출 일정 대비 근거 부족 | 9/14 이후로 이관 |

## 5. 평가 방법

### 5.1 평가셋

평가는 서로 다른 목적의 다섯 트랙으로 나눈다.

| 트랙 | 문항 수 | 목적 | 해석 범위 |
|---|---:|---|---|
| 파서·데모 검색 | 24 | 파서 프로필과 문서 검색 가능성 비교 | 문서 retrieval만 평가 |
| 파서 source-bound 감사 | 18문서·54 anchor | 세 프로필의 원자 근거 문자열 보존 진단 | 표적 기계 평가이며 full-corpus·RAG 성능이 아님 |
| 서비스 DEV | 45 | 학사·입학·졸업·국제·등록·장학·지원과 역할 변형 평가 | 반복 튜닝에 사용한 개발셋 |
| 서비스 final holdout | 36(Core 27 + Challenge 9) | 최종 일반화·공격·회피 평가 | AI 검토 12건 수정·기계 gate 완료, 2인 human signoff·실행 미완료 |
| 연구비 규정 | 53(확보 비교 50) | 복잡한 규정 검색·생성과 경쟁 서비스 비교 | 확보 가능 문항과 수집일에 한정 |

서비스 45문항은 9개 행정 영역의 기본 문항 39개와 역할 조건을 추가한 변형
문항 6개로 구성하며, 질문, 카테고리, 역할, 정답 문서 anchor, 기대 근거 문구를
포함한다. 총 127개의 evidence quote(85개 case–chunk 쌍, 71개 고유 chunk)는
인덱스에서의 존재 여부를 기계 검증했다. 그러나 이
평가셋을 실패 진단과 튜닝에 반복 사용했으므로 최종 일반화 성능이 아니라
개발셋 성능으로 보고한다.

Holdout v2는 독립 AI 검토에서 지적된 12건을 수정한 뒤 36문항·93 evidence
option으로 재검증했다. 현재 SHA-256은
`2fbedad63531491d3c069a5de6fe51fff623f7e571b9170c6a9aa7452e2d51f9`이며
schema·DEV·corpus pre-review gate는 PASS다. 읽기 전용 사람 검토 패킷과 별도
SHA-bound A/B 응답 template까지 준비했지만 실제 판정은 아직 모두 `PENDING`이므로
signoff gate는 실행 전 BLOCK 상태다. 최종 validator는 merge manifest뿐 아니라
packet과 두 원본 응답의 SHA·reviewer·36개 판정을 다시 대조하며, cases는 한 byte
snapshot에서 중복-key 없는 strict JSON으로 해석한다. 같은 파일 alias·빈 packet·
parent symlink 경로도 거부한다. 실제 2인 human
signoff와 최종 파일 동결 전에는 holdout을 실행하지 않는다.

### 5.2 검색 지표

문서 Hit@k는 상위 \(k\)개 결과에 하나 이상의 gold 문서가 포함된 질의 비율이다.

\[
Hit@k=\frac{1}{N}\sum_{i=1}^{N}
\mathbf{1}[TopK_i\cap G_i\neq\varnothing]
\]

MRR은 첫 관련 결과 순위의 역수 평균으로, 첫 정답 근거가 얼마나 앞에 나타나는지
반영한다([NIST TREC QA](https://trec.nist.gov/data/qa.html)).

\[
MRR=\frac{1}{N}\sum_{i=1}^{N}\frac{1}{rank_i}
\]

문서 URL이 맞더라도 실제 답에 필요한 청크가 없을 수 있으므로 DEV에서는
`Any-Gold-Chunk@k`, `All-Gold-Chunks@k`, `GoldChunkRecall@k`를 별도로
측정한다. 여기서 gold는 atomic claim이 아니라 평가셋에 저장된 Cascade chunk
ID 목록이다. 따라서 이 수치를 `Atomic Evidence Recall`이나 사실성으로 부르지
않는다. 최종 holdout은 profile-independent atomic evidence option을 사용해
Evidence Recall/All-Evidence를 별도로 계산한다. 최종 표에는 Hit@1·3·5, MRR,
청크 coverage@5/@8, latency, 카테고리별 결과와 신규·상실 적중을 함께 제시한다.

Final 검색은 `run_final_retrieval_schedule.py`가 Core 27문항을 PAR-B/PAR-CH/
C0/C1 순서로 총 108 slot에 고정하고 외부 생성 없이 `extractive` 경로만 쓴다.
각 slot 전후에는 holdout·DEV·index·sign-off와 sign-off가 가리키는 packet·A/B
원본까지 동일 byte binding인지 다시 확인한다.
`analyze_holdout_retrieval.py`는 schedule·WAL·요청 provenance를 다시 감사한 뒤
Source Hit@1/3/5, Evidence Recall·All-Evidence@5/8, MRR@50, Candidate Recall@50,
latency와 paired 검정을 생성한다. 합성 108-slot/216-WAL E2E는 통과했지만 실제
holdout 수치는 아직 없다.

### 5.3 생성 평가

2026-08-31 DEV historical 결과는 생성과 판정을 함께 실행한 고정 rubric의
0–2점 inline judge다. 기준선과 후보를 각각 3회 실행한 run 평균, 문항 평균의
평균, 표준편차, 문항별 개선·동률·악화 수를 제시하되, 0.7185와 같은 값을
“71.85% 정답률”로 쓰지 않는다.

최종 holdout은 answer 수집을 먼저 동결하고 별도 condition-blinded structured
Judge가 claim별 정확성·완전성·citation support를 판정한다. 사전 고정한 주지표는
모든 필수 claim이 정확하고 근거로 지지되는 문항 비율인 GFC이며, 0–2점 평균은
부분 개선을 설명하는 보조 지표다. 검색 hit×GFC 교차표로 검색 실패와 컨텍스트
활용 실패를 구분한다. Judge-human gate 통과 전의 GFC n=3 결과는 exploratory로만
취급한다.

직접 생성은 `run_final_generation_schedule.py`가 Core 162와 Challenge C1 27,
총 189 logical slot을 case-balanced AB/BA 순서로 고정한다. 별도 oracle 9개를
더해 생성 계획은 198개다. Clean local/server Git commit, index와 공통 source
manifest, holdout sign-off의 packet·A/B 원본, provider/model/request config를
매 호출에 재검증하고 배타 run-lock과 started/completed WAL로 동시 실행·불확실
재호출을 막는다. 이 경로는 합성 검증만
완료했으며 실제 holdout은 아직 실행하지 않았다.

### 5.4 Judge 신뢰도

Historical 생성 n=3은 생성과 judge를 함께 다시 실행한 end-to-end 변동이며
judge 자체 신뢰도는 보장하지 않는다. 최종 평가는 immutable answer JSONL과
judgment JSONL을 분리하고 `answer_id`·답변 hash·prompt hash로 연결한다. 고정
답변을 3회 재채점해 일치율을 계산한다. Core 조건에서 사람 판정과의 raw
agreement ≥ 0.80, balanced accuracy ≥ 0.80, Cohen's kappa ≥ 0.60,
macro-F1 ≥ 0.75를 모두 통과해야 headline GFC에 사용한다. GFC 클래스 한쪽이
없어 balanced accuracy 또는 kappa가 정의되지 않으면 gate를 통과시키지 않는다.
모델명·버전·실행일·temperature·prompt·rubric·parse failure를 보존하고
`uncertain` 및 경계 사례는 사람이 검토한다.

최종 답변 검토는 `build_answer_review_packet.py`가 condition·model·provider와
자동 support 판정을 숨긴 Markdown, 두 독립 평가자 라벨과 합의 라벨, 비공개
mapping을 분리해 생성한다. `analyze_judge_human_calibration.py`는 이 라벨을
직접 읽어 사람–사람 GFC raw agreement·Cohen's kappa, 순서형 0–1–2점 QWK,
Judge–합의 사람 GFC 혼동행렬·balanced accuracy·Cohen's kappa·macro-F1·
false-pass rate와 gate를 fail-closed로 계산한다. 클래스별 문항 수도 함께
제시해 불균형을 숨기지 않는다. 실제 사람 라벨과 최종 holdout 실행은 아직
완료되지 않았으므로 이 절의 수치는 확정 결과가 아니라 사전 고정한 protocol이다.

### 5.5 경쟁 서비스 비교

경쟁 서비스 결과는 동일 질문과 수집일을 기록한다. 제품명을 가린 뒤 A=우리,
B=경쟁과 A=경쟁, B=우리 순서를 모두 평가한다. 두 판정이 모순이면 tie 또는
사람 재검토로 처리한다. 확보되지 않은 문항은 0점으로 임의 처리하지 않고 비교
모수에서 제외한 뒤 이유를 각주로 제시한다.

### 5.6 통계와 재현성

개선 전후는 같은 문항의 paired 비교다. 평균 차이와 함께 개선·악화·동률 수,
paired bootstrap 95% 신뢰구간을 보고한다. 모든 실행은 코드 commit, 코퍼스
revision, model, prompt, 환경 설정과 JSONL 산출물을 연결하며 최종 산출물의
SHA-256 manifest를 부록에 포함한다.

반복 Judge 호출은 추가 표본으로 세지 않는다.
`analyze_final_generation_gfc.py`는 서로 다른 generation run 1/2/3을 질문
내부 반복으로 결합해 유효 표본을 Core 질문 `n=27`로 유지한다. 질문별 GFC 성공
비율 차이에 family bootstrap과 sign-flip을 적용하고 3회 중 다수결 GFC에는 exact
McNemar를 적용한다. Terminal generation error는 재생성으로 숨기지 않고 GFC=0으로
포함하며, Judge-human gate가 실패하면 adjudicated human run1을 headline으로
사용한다. 원본 answer/judgment/calibration artifact의 SHA·record hash·바인딩을
다시 검증한다.

## 6. 결과

### 6.1 코퍼스와 파서

| 항목 | 결과 |
|---|---:|
| 크롤 완료 행 | 10,977 |
| 선별 문서 | 2,260 |
| 선별 용량 | 473,678,602 bytes |
| 파싱 문서 | 2,247 |
| 파싱 청크 | 44,591 |
| 검색 인덱스 청크 | 44,520 |
| 추출 표 | 7,813 |

| 파서 프로필 | 문서 Hit@5 | MRR |
|---|---:|---:|
| Baseline | 23/24 | 0.8854 |
| Challenger | 24/24 | 0.8417 |
| Cascade | 24/24 | 0.8486 |

별도의 18문서·54 anchor source-bound 감사 결과는 다음과 같다.

| Profile | Anchor 보존 | 3/3 보존 문서 | HWP/HWPX | Digital PDF | OCR/Table PDF |
|---|---:|---:|---:|---:|---:|
| Baseline | 48/54 | 14/18 | 18/18 | 17/18 | 13/18 |
| Challenger | 54/54 | 18/18 | 18/18 | 18/18 | 18/18 |
| Cascade | 54/54 | 18/18 | 18/18 | 18/18 | 18/18 |

공통 source manifest SHA-256은
`1fa7e0f2f5d03a2a1249584d91f7232c9b2cb4642b39115bf3c9e6a871845a2f`다.
평가셋·JSON·CSV SHA-256은 각각
`5371353a52b36df8685d4892d8138ba1eaf691fb93a2fd335ced21e279208720`,
`9fabcd7525ba22dea1758a55418546e25627d2d7413faf7cccea1efc0dbf2755`,
`bd98a7469b33970af2c018a8e0af7a06a004fa3e366dd9957bdc638762d7de81`다.
이는 실패 모드를 섞은 표적 18문서의 결정적 문자열 보존 진단이며 Cascade
산출물을 canonical binding으로 사용했다. 따라서 full-corpus 파서 우월성,
Challenger 대비 Cascade 우월성, RAG 성능 향상 또는 54 anchor의 독립 2인 원문
육안검수 결과로 해석하지 않는다.

### 6.2 서비스 기준선

| 계층 | 지표 | 결과 | 해석 |
|---|---|---:|---|
| 검색 | 문서 Hit@5 | 29/45 = 0.644 | gold 문서 상위 5개 포함 |
| 검색 | MRR | 0.494 | 첫 gold 문서 순위 |
| 근거 | Any-Gold-Chunk@5 | 21/45 = 0.467 | 상위 5개에 하나 이상의 지정 gold chunk 포함 |
| 근거 | All-Gold-Chunks@5 | 8/45 = 0.178 | 상위 5개에 지정 gold chunk 전부 포함 |
| 근거 | GoldChunkRecall@5 | 0.315 | 지정 gold chunk 중 상위 5개에 포함된 비율의 평균 |
| 근거 | Any-Gold-Chunk@8 | 23/45 = 0.511 | 실제 생성 context 8개에서 하나 이상 포함 |
| historical 생성 | run 1 | 0.733 | 2026-08-31 0–2점 inline judge 평균 |
| historical 생성 | run 2 | 0.667 | 2026-08-31 0–2점 inline judge 평균 |
| historical 생성 | run 3 | 0.756 | 2026-08-31 0–2점 inline judge 평균 |
| historical 생성 | 문항 평균의 평균 | 0.7185 | 정답률이나 현재 코드 재실행 결과가 아님 |

### 6.3 서비스 검색 튜닝 결과

| 지표 | 기준선 | 튜닝 후보 | 차이 |
|---|---:|---:|---:|
| 문서 Hit@5 | 0.644 (29/45) | 0.889 (40/45) | +0.245, +11문항 |
| MRR | 0.494 | 0.717 | +0.223 |
| Any-Gold-Chunk@5 | 0.467 (21/45) | 0.600 (27/45) | +0.133, +6문항 |
| All-Gold-Chunks@5 | 0.178 (8/45) | 0.356 (16/45) | +0.178, +8문항 |
| GoldChunkRecall@5 | 0.315 | 0.478 | +0.163 |
| Any-Gold-Chunk@8 | 0.511 (23/45) | 0.622 (28/45) | +0.111, +5문항 |
| All-Gold-Chunks@8 | 0.222 (10/45) | 0.400 (18/45) | +0.178, +8문항 |
| GoldChunkRecall@8 | 0.359 | 0.511 | +0.152 |
| 기존 적중 상실 | — | 0 | — |

2026-09-02 현재 코드의 동일 45문항 DEV 재실행에서 위 검색 지표 개선을
확인했다. 반복 튜닝에 사용한 DEV이므로 이 결과는 최종 일반화 성능이 아니다.

별도의 2026-08-31 historical end-to-end 실행에서 생성 run별 평균은 기준선
0.733/0.667/0.756, 당시 튜닝 후보 0.733/0.756/0.756이었고, 문항 평균의
평균은 0.7185에서 0.7481로 0.0296 상승했다. 문항별로는 10개가 개선, 29개가
동일, 6개가 악화됐다. 그러나 역할 변형을 기본 문항과 같은 family로 묶은
paired cluster bootstrap 10,000회의 평균 차이 95% 신뢰구간은
[-0.1407, 0.1951]로 0을 포함했다. 이 historical 생성 결과를 현재 코드
DEV45 검색 재실행과 한 실험처럼 결합하지 않으며, 생성 개선은 확정하지 않는다.

Historical 135개 paired 응답을 다시 분해하면 0점은 61→52로 줄었지만 2점도
23→18로 줄어 총점이 97→101, 즉 4점만 증가했다. 실제 gold chunk가 새로
들어온 5문항은 +11점을 만들었으나, 이미 gold가 있던 23문항에서 -11점이
발생해 전부 상쇄됐다. 날짜 절단 응답도 6→12로 늘었으며 tuned 1점 65개 중
61개는 Judge 사유상 핵심 조건이 누락되거나 불완전했다. 세부 산술과 사례는
`docs/dev45-generation-root-cause-20260902.md`에 기록했다.

검색 hit와 evidence가 개선된 문항도 답변의 세부 조건을 모두 사용하지 못했고,
검색 결과가 같아도 생성·judge 변동으로 점수가 바뀐 사례가 있었다. 특히
일반 제목 강등이 문서 본문 품질을 보지 않아 유효 근거까지 뒤로 보낼 수 있는
반례가 확인됐다. 최종 서비스 구성에서는 본문 핵심어·숫자·표 coverage가 높은
일반 제목 문서를 예외 처리하는 후속 A/B가 필요하다.

2026-08-31에 보존된 문서당 후보 청크 cap 2→4 후속 DEV 실험에서는 문서 Hit@5와
MRR이 각각 0.889와 0.723으로 유지됐고, Any-Gold-Chunk@8은 28/45에서
30/45, All-Gold-Chunks@8은 13/45에서 16/45, GoldChunkRecall@8은
0.456에서 0.500으로 증가했다. 반면 단일 순차 실행의 p50 latency는
142.4ms에서 166.9ms로 늘고 평균 고유 문서 수는 6.13에서 5.04로 줄었다.
context 구성 또는 순서가 31/45에서, extractive 답변이 10/45에서 달라졌다.
따라서 전역 cap 4는 최종 서비스에 채택하지 않고 cap 2를 유지했다.

대신 명시적 multi-facet 질문의 누락 근거만 교체하는 보수적 수정으로 실제
`POST /chat` DEV45를 재수집했다. Source Hit@5 40/45, MRR .717407,
Any-Gold-Chunk@8 28/45와 평균 고유 문서 수 6.0222는 그대로였고,
All-Gold-Chunks@8은 17/45에서 18/45, GoldChunkRecall@8은 .500000에서
.511111로 변했다. 최종 context ID가 달라진 문항은 2/45뿐이었다.
`svc_sch_02`는 중복 금액 chunk 대신 신청기간을 얻어 recall .5→1.0과
All-Gold false→true를 기록했다. `svc_sch_05`는 질문 비필수 중복수령 규정
대신 신청자격을 얻었지만, legacy gold가 그 비필수 chunk도 포함하므로 exact
recall은 2/3으로 같았다. 이는 DEV에서의 표적 검색 개선이며 생성 품질이나
일반화 성능을 뜻하지 않는다.

이후 질문에 직접 요구되지 않은 참고 근거를 분모에서 제외하고, 합성된 quote를
원문의 원자 행으로 교정했다. 현재 DEV45는 140개 evidence 중 105개를 필수,
35개를 참고 근거로 구분한다. 이 기준으로 9월 2일 저장 C1과 현재 C1을 같은
분석기에서 다시 계산한 누적 결과는 다음과 같다.

| current required-evidence 기준, k=8 | 9월 2일 C1 | 현재 C1 | 변화 |
|---|---:|---:|---:|
| Source Hit@5 | 40/45 (.889) | 40/45 (.889) | 0 |
| Document MRR@5 | .7026 | .7137 | +.0111 |
| Any required gold chunk | 28/45 (.622) | 34/45 (.756) | +6문항 |
| All required gold chunks | 21/45 (.467) | 31/45 (.689) | +10문항 |
| Mean required gold chunk recall | .5444 | .7148 | +.1704 |
| All required evidence items | 21/45 (.467) | 33/45 (.733) | +12문항 |
| Mean required evidence recall | .5630 | .7481 | +.1852 |

마지막 두 행은 exact chunk ID 또는 동일 반환 chunk 안의 전체 정규화 quote만
인정하며 fuzzy·semantic match를 사용하지 않는다. 개선 12문항, 손실 0문항이지만
15/45의 source 배열이 달라진 누적 DEV 튜닝 결과다. 특히 수료후연구생의 멀리
떨어진 금액·신청방법, 이자지원 FAQ의 소득분위·입금계좌, AI학업장려대출의
대상학과·금리·한도, 성적표·학위논문의 제출기한, 교연비의 전체 예산·영역별
구성비를 질문 facet으로 분리해 문서당 cap 2를 유지하면서 필요한 chunk만
교체했다. 원천 비교는
`processed/eval/preflight-20260903/retrieval-budget-breakdown-v5/dev45-matrix-vs-current.json`에
보존했다. 이는 holdout이나 생성 성능이 아니며, 새 context의 end-to-end GFC는
별도로 측정한다.

9월 4일에는 남은 실패의 표현 불일치를 문서 유형·항목명 수준에서만 보정했다.
학생증·증명서 위탁, 입학 모집인원 변경, D-2 단체접수, 외국인 학부 신입학,
교환학생 선발, 하계방학 자격증 과정, 장애학생 학습지원, 제3자 인권신고가
대상이다. 답의 숫자·기관명·가능 여부는 검색어에 넣지 않았다. 교환학생의
선발 규모/일정과 제3자 신고의 허용/필수조건처럼 근거가 떨어져 있는 경우에만
강한 동일 문서 seed의 zero-marginal 청크를 교체했다.

| 추가 의도 튜닝, k=8 | 확장 전 C1 | 현재 C1 | 변화 |
|---|---:|---:|---:|
| Source Hit@5 | 40/45 (.889) | 45/45 (1.000) | +5문항 |
| Document MRR@5 | .7137 | .8137 | +.1000 |
| Any required evidence item | 34/45 (.756) | 45/45 (1.000) | +11문항 |
| All required evidence items | 33/45 (.733) | 40/45 (.889) | +7문항 |
| Mean required evidence recall | .7481 | .9407 | +.1926 |

추적한 각 지표에서 상실 문항은 없었고, 최종 증분 4문항도 모두 완전 필수근거를
새로 확보했다. 실제 서비스 경로의 extractive DEV45 45건이며 외부 LLM 호출은
없었다. 생성 prompt에는 대응하는 누락 점검표를 넣었지만 생성 개선 여부는 fresh
generation 전에는 판단하지 않는다. 또한 이 단계는 실패 사례를 반복 관찰한
DEV 튜닝이므로 100% Hit를 일반화 성능으로 표현하지 않는다. 원천은
`processed/eval/preflight-20260904/dev45-retrieval-query-expansion-v5/`이다.

새 context의 end-to-end 전달 효과를 확인하기 위해 현재 코드에서 C0/C1을 각각
45문항×3회 생성하고 각 답변을 Judge v11로 한 번씩 판정했다. Generator는
`gemini-3.5-flash-lite`, Judge는 `gemini-3.1-flash-lite`, temperature 0이며
생성 270개와 판정 270개에 terminal error나 최종 fallback은 없었다.

| 현재 코드 DEV45 독립 생성 n=3 | C0 | C1 | 차이 |
|---|---:|---:|---:|
| 0–2점 Judge 평균 | .8815 | 1.2148 | +.3333 |
| 0점 응답 | 52/135 | 29/135 | -23 |
| 2점 응답 | 36/135 | 58/135 | +22 |
| 2/3 majority GFC | 13/45 (.289) | 20/45 (.444) | +.1556 |

문항별 3-run 평균은 C1 개선/동률/악화가 19/15/11이고, 역할 변형을 기본 질문과
같은 family로 묶은 paired cluster bootstrap 10,000회의 평균 점수 차이 95% CI는
[+.0889, +.5887]로 0을 포함하지 않았다. 그러나 majority GFC의 family-cluster
bootstrap 95% CI는 [-.0217, +.3201]이고 exact McNemar 양측 p=.1185였다.
평균 점수 개선과 달리 엄격한 완전정답률 개선의 통계적 근거는 충분하지 않다.
사전 고정 주지표는 GFC이고 표의 0–2점 평균은 보조 지표다. Judge v11의 결정론적
guard는 명시적 회피, 자기모순, 인용 불일치를 점수를 낮추는
방향으로만 교정했으며 78/270에 적용됐다. 생성기와 Judge가 같은 Gemini 계열이고
사람 calibration이 없으며 반복 튜닝한 DEV라는 제한 때문에 이 수치는 조건부
개발셋 결과다. 원천 집계는
`processed/eval/preflight-20260903/dev45-generation-current-v1/judge/dev45-3run-service-ab-v11.json`이다.[^dev45-v11]

#### Judge v11 동일 답변 반복 안정성

생성 변동과 Judge 변동을 분리하기 위해 C0/C1의 고정된 `run1` 답변을 Judge
v11로 각각 세 번 판정했다. C0와 C1 모두 0–2점과 GFC가 문항별 **45/45
완전 일치**했다. C0는 세 반복 모두 평균 `.8667`, GFC `12/45(.267)`였고,
C1은 모두 평균 `1.2000`, GFC `18/45(.400)`였다. 반복 다수결 기준 차이는
`+6문항(+.1333)`이며 질문 단위 paired bootstrap 95% CI는
`[-.0444,+.3111]`, exact McNemar 양측 `p=.2379`였다.[^judge-v11-stability]

이 결과는 현재 API·rubric·temperature 설정에서 판정의 **재현성**이 높음을
보이지만, Judge가 사람과 같은 정답을 내린다는 **타당도**를 입증하지 않는다.
같은 답변을 세 번 채점한 값은 독립 표본 135개가 아니며 조건별 유효 표본은
각각 45문항이다. 또한 이 절의 C1 `18/45`는 고정 `run1`의 결과이므로, 서로
다른 세 생성의 2/3 majority인 앞 절의 `20/45`와 목적과 추정량이 다르다.

#### 동결 후 C2 quote-bound 생성 실험

검색 결과가 있어도 자유 서술 답변과 사후 휴리스틱 attribution 사이에서 claim이
누락되는 병목을 분리하기 위해, 서비스에 연결하지 않은 C2 실험 lane을 만들었다.
C2는 모델이 각 claim과 함께 `source_number` 및 원문에서 복사한 연속 quote를
구조화해 반환하도록 하고, application이 출처 번호·quote 포함·숫자·핵심 주체·
관계 방향을 결정적으로 검증한다. C1의 동결된 검색 trace와 context를 그대로
재사용했으므로 아래 차이는 검색 개선이 아니라 생성–근거 결속 방식의 차이다.

| DEV45 독립 생성 n=3 | C1 | C2 v3 | 차이 |
|---|---:|---:|---:|
| run별 GFC | 18/45, 20/45, 20/45 | 28/45, 25/45, 23/45 | — |
| 3-run 0–2점 평균 | 1.2148 | 1.3630 | +0.1481 |
| 2/3 majority GFC | 20/45 (.444) | 26/45 (.578) | +6문항, +.1333 |

평균 점수 차이의 family-cluster bootstrap 95% CI는 `[-.0000, +.3116]`이었다.
majority GFC는 paired gain/loss가 9/3, family-cluster bootstrap 95% CI가
`[.0000, +.2826]`, exact McNemar 양측 `p=.1460`이었다. 따라서 C2의 점 추정치는
일관되게 높지만 DEV45만으로 완전정답률 개선을 확정할 통계적 근거는 충분하지
않다. majority 비GFC 19문항 중 13문항은 필수 근거가 context@8에 모두 없었고,
6문항은 필수 근거가 모두 있었는데도 실패해, 남은 병목이 검색과 생성 양쪽에
있음을 보여준다.[^c2-v3]

이 실험의 run1은 최초 응답에서 발견한 일반 verifier false negative를 고친 뒤
같은 raw response를 API 호출 없이 v3로 재투영했고, run2·run3만 commit
`85709b5`의 동결 verifier로 새로 생성했다. 따라서 세 run은 서로 다른 모델
응답이지만 verifier 설계에 대해 run1까지 완전히 독립인 confirmatory 평가가
아니다. C2를 서비스에 채택하거나 일반화 성능으로 표현하지 않고, 독립 holdout과
사람 calibration 전까지 post-freeze exploratory 결과로만 유지한다.

이 평가 뒤 실패 답변을 추적해 등록금 관계문 두 종류의 후처리 false negative를
수정했다. 저장된 동일 초안·동일 full context 270개를 오프라인으로 재처리했을 때
답변 텍스트는 9개만 바뀌고 261개는 그대로였으며, 바뀐 9개를 같은 Judge v11로
재판정한 점수는 모두 2점이었다. 바뀌지 않은 261개의 Judge input SHA도 기존과
261/261 일치했다. 이를 결합한 score-only 진단 평균은 C0 .9037, C1 1.2963,
차이 +.3926이고 family-cluster bootstrap 95% CI는 `[+.1429,+.6519]`다.
그러나 이는 새 생성 없이 같은 저장 초안을 재처리한 projection이므로 공식 E2E,
GFC, citation 또는 일반화 성능으로 승격하지 않는다. 최종 수치는 코드 동결 뒤
fresh holdout 실행으로 확정한다. 진단 원천은
`processed/eval/preflight-20260903/dev45-generation-current-v1/postprocessor-direction-fix-v1/README.md`다.

추가 오류 분석에서는 같은 표 행의 날짜·시각 및 분야별 최대금액과, 인접한
`고지서출력: DATE`/UI 경로를 후처리가 분리해 버리는 문제를 수정했다. 이전
projection 대비 C1 답변 6/135만 바뀌고 129/135는 동일했으며, 음성 회귀를
포함한 전체 709 tests가 통과했다. 승인된 6개만 Judge v11로 재판정한 결과
오류 0건, `svc_sch_02` 3개와 `svc_adm_02` 1개가 `1→2`, 나머지 2개는
`1→1`이었다. 동일 입력 129개의 기존 판정을 재사용한 누적 score-only 진단은
C0 .9037, C1 1.3259, 차이 +.4222였고 질문 단위 20/16/9, family-cluster
bootstrap 95% CI `[+.1812,+.6742]`였다. 1점으로 남은 두 답변은 저장 초안에
각각 이수 기준과 전액장학 예외가 없어 새 프롬프트로 재생성해야 한다. 이 수치도
같은 초안 projection이므로 공식 E2E/GFC/citation/일반화 성능이 아니다. 진단 원천은
`processed/eval/preflight-20260903/dev45-generation-current-v1/postprocessor-grounding-fix-v4/README.md`다.

검색 0점군에서는 교연비 정답 문서가 1위임에도 문서당 cap 2가 일반 지급대상
청크를 남기고 교원 연간 한도표를 누락하는 병목을 확인했다. 단일 숫자 질문에
한해 강한 동일 문서 seed를 확장하고 질문 대상이 같은 금액 행을 거리보다 먼저
선택하자, 저장된 DEV45 후보 풀에서 바뀐 문항은 `svc_core_01`과 역할 변형 2개,
필수 evidence micro recall은 47/71(.662)→49/71(.690), 감소 문항은 0이었다.
이는 저장 후보 재선택 진단이지 fresh 검색·생성 성능이 아니다. 생성 프롬프트도
대체강좌 이수기준, 직무체험 방식, 등록금 고지서 시점·필수 예외를 질문 의도별로
요구하고, 질문하지 않은 서류·문의처를 일괄 요구하지 않도록 바꿨다. 이 프롬프트의
효과 역시 새 generation 전에는 성능 향상으로 주장하지 않는다.

이 두 개선을 알려진 실패 5문항에 한정해 실제 서비스에서 1회씩 생성한 결과,
Gemini 3.5 Flash Lite 생성과 Gemini 3.1 Flash Lite Judge가 각각 5/5 완료됐고
오류·fallback은 없었다. 이전 C1 세 run에서 항상 0점이던 교연비 기본·역할 문항은
각각 2점, 등록금 납부는 1점에서 2점이 됐다. 선택 5문항 평균은 .600에서 1.600,
GFC는 0/5에서 3/5였으나, 실패군을 의도적으로 골라 한 번만 생성한 표적 smoke라
전체 DEV 또는 일반화 성능으로 쓰지 않는다.

나머지 대체강좌·직무체험은 생성 초안에 정답이 있었음에도 semantic guard가
삭제한 것으로 trace에서 확인됐다. comparator 뒤의 가능 표현이 modality를
오염시키는 문제와 `모집대상` 제외 괄호 및 명사형 프로그램 활동을 좁게 보정하자,
동일 저장 초안 replay에서 이 두 답변만 완전한 형태로 복원되고 다른 3개는
그대로였다. 신규 양성 5개와 오탐 방지 음성 회귀를 포함해 전체 714 tests가
통과했다. 변경 두 답변을 별도 승인된 Judge v11 partial run으로 확인하자 둘 다
Gemini 3.1 Flash Lite 기준 2점·GFC true·citation support full이었고 오류와
fallback은 없었다. 동일 저장 초안의 5문항 진단값은 fresh pre-fix 1.600·GFC
3/5에서 post-fix 2.000·5/5가 됐다. 단, 이 replay는 알려진 실패군의 같은 생성
초안과 context를 다시 후처리한 것이므로 fresh E2E나 일반화 성능이 아니다. 원천은
`processed/eval/preflight-20260904/targeted-generation-prompt-retrieval-v1/README.md`다.

예산 질의는 검색과 생성의 단계 분리를 추가로 확인했다. 정답 표를 회수한 뒤에도
기존 추출형 응답은 주변 공개 공고만 나열했지만, 표의 대상 학년도 열을 해석하고
합계와 세 영역의 예산액·구성비를 문장화하도록 보강한 뒤 `svc_core_02`와 역할
변형 2/2가 정답 수치를 직접 출력했다. 두 문장 모두 같은 정답 표 chunk에
귀속되고 critical-value 검증을 통과했다. 이는 2문항 표적·결정론적 검증이며 전체
생성 품질이나 LLM Judge 성능의 대체 지표로 사용하지 않는다.

의도별 검색 보강의 실제 답변 전달 여부는 알려진 실패 9문항을 별도로 선정해
fresh 생성·Judge로 확인했다. Gemini 3.5 Flash Lite 생성과 Gemini 3.1 Flash
Lite Judge가 각각 9/9 한 번의 시도로 완료됐고 오류·fallback은 없었다. 같은
9문항의 이전 C1 세 run 평균은 .5556/.4444/.5556, GFC는 1/0/1건이었고,
fresh run은 평균 1.5556, GFC 6/9였다. 다만 개선 대상 실패군을 의도적으로 고른
n=1 표적 진단이므로 이 차이는 전체 DEV나 일반화 성능 추정치가 아니다.

남은 국제 문항은 원인을 다시 단계별로 분리했다. 교환학생 일정은 완전한 raw
draft를 요일·축약 종료일 처리 오류로 semantic guard가 삭제했으며, 좁은 수정 뒤
동일 저장초안 9개 중 해당 답변 하나만 복원됐다. D-2 단체접수는 예약·일정·D-2
서류/60,000원 수수료가 같은 PDF의 세 청크로 떨어져 전역 문서 cap 2로는 완전한
context를 만들 수 없었다. 전역 cap은 유지하고 명시적 D-2 다중 facet 질문에만
effective cap 3을 허용했다. fresh retrieval-only DEV45 v6에서 D-2 기본·역할
두 문항이 required evidence 1/3→3/3으로 바뀌고, All required evidence@8은
40/45(.889)→42/45(.933), 평균 recall은 .9407→.9704가 됐으며 loss는 0이었다.
그러나 이 후처리·검색 수정 뒤 fresh LLM 재생성과 Judge는 아직 하지 않았으므로
생성 성능 향상으로 승격하지 않는다.

추가로 자부담금 지원 질문은 지원 대상·전액지원, 선납부 후 증빙 확인 절차,
제출서류·구글폼 경로가 한 안내문의 세 청크로 분리돼 있었다. 이 질문은 이전
C1 생성 3회에서 1/1/0점이었고 제출 절차·서류 누락이 직접 감점 사유였으므로,
해당 정보통신보조기기 자부담금 의도에만 effective cap 3과 세 facet을 적용했다.
DEV45 v7은 v6 대비 `svc_sup_02` 하나만 바뀌고 loss는 0이었으며 All required
evidence@8이 42/45(.933)에서 43/45(.956), 평균 recall이 .9704에서 .9778로
증가했다. 남은 두 미완전 문항은 같은 교연비 질문의 기본·역할 변형이다. 실제
최대액 표는 이미 회수되고 두 답변 모두 이전 fresh 생성에서 GFC였으므로, 배경문
청크까지 넣어 45/45를 만드는 튜닝은 하지 않고 annotation 검토 대상으로 남겼다.

위 검색·절차 수정 뒤 알려진 실패 4문항을 fresh 생성·Judge로 다시 측정했다.
생성 4회와 Judge 4회는 모두 지정 모델의 첫 시도에 완료됐고 오류·fallback은
없었으며, 검색 hit·MRR·exact required gold all@5/@8은 모두 1.000이었다.
Judge는 교환 일정 2점, D-2 기본 1점, 자부담금 지원 2점, D-2 역할 변형 1점으로
평균 1.500과 GFC 2/4를 기록했다. 직전 동일 4문항도 평균 1.500·GFC 2/4였으므로
총량 개선은 없었다. 자부담금 문항의 1→2 개선이 D-2 역할 변형의 2→1 변동에
상쇄됐으며, 이 n=4×1 표적 DEV 결과를 대폭 향상이나 일반화 근거로 쓰지 않는다.

D-2 실패 trace에서는 검색이 아니라 후처리가 추가 병목으로 확인됐다. 기본 raw
draft의 정확한 1차 기간·시간·회차별 대상과 역할 raw draft의 서비스 정의 및
60,000원·현금·만원권·면제 문장이 관계/표기 검사에서 삭제됐다. 정확한 표 행과
조건문에만 작동하는 좁은 bridge와 음성 회귀를 추가한 뒤 같은 draft/context를
재투영하면 19/19 문장과 39/39 critical value가 유지되고 거부 claim은 0이 된다.
기본 raw draft 자체에 없었던 수수료는 별도 생성 문제이므로 D-2 점검표를 독립
항목으로 분리하고 값 대신 “확인하세요”라고 쓰지 못하게 했다. 이 후속 결과는
외부 모델을 다시 부르지 않은 결정론적 진단이며 새 end-to-end 점수가 아니다.
세부 원천은
`processed/eval/preflight-20260904/targeted-generation-postfix-v8/README.md`다.

후처리 병목이 D-2에만 국한되는지 확인하기 위해 기존 DEV45 C1 생성 3회의 저장
초안 135개를 전수 재투영했다. 원래 projection은 run별 47/48/48개 문장과
11/34/17개 핵심값을 제거했다. 이전 Judge 실패 사유와 trace를 함께 검토해,
단위가 한 번만 적힌 수량 범위, 명사형 지원자격·대출대상, 예산표의 예산 규모·
비중, 참여혜택 번호 행, 부서별 연락처, 자격증 특강 운영 표현의 여섯 가지
false negative를 동일 행·표제·수치 조건으로 좁게 수정했다. 반대 조건과 잘못된
수치·연도·소유자는 음성 회귀로 계속 거부한다. 서로 다른 근거 블록의 사실을
한 문장에 합치는 문제는 다중 source 허용으로 validator를 느슨하게 하지 않고,
생성 prompt에서 근거 블록별로 줄을 분리하도록 했다.

최종 동일초안 projection은 답변 15/135를 바꾸고 source-backed 문장 16개를
추가 보존했으며, 핵심값 손실을 합계 62개에서 44개로 18개 줄였다. 변경 record
15개 중 13개는 기존 Judge 2점 미만 문항이고 당시 실패 사유가 이번에 복원한
누락 사실을 지목했다. 이는 후처리 병목과 수정 대상의 타당성을 뒷받침하지만,
새 생성이나 Judge 판정이 아니므로 13개 점수 개선, GFC 향상 또는 일반화 성능으로
계산하지 않는다. 전체 747 tests가 통과했고 6개는 선택 의존성으로 skip됐다.
원천은
`processed/eval/preflight-20260904/dev45-postprocessor-capability-fix-v1/README.md`다.

### 6.4 연구비 규정 트랙

| 단계 | Context recall |
|---|---:|
| BM25 기준선 | 0.4522 |
| + cross-encoder | 0.4644 |
| + 부모 확장 | 0.4877 |
| + 순위 융합 | 0.5157 |
| + dense lane | 0.5266 |

동일 트랙의 53문항 생성 judge 평균은 기준 검색 0.81에서 채택 파이프라인
1.15로 증가했고, 완전 정답 13문항이 23문항으로 늘었다. 현재 확보50 비교는
우리 서비스 1.607, 2026-08-06 경쟁 서비스 snapshot 1.440이다. 이 값은 상대
재수집과 양방향 blind judge 이전의 조건부 결과이므로 최종 비교표는 9/9
result-freeze 전에 별도로 갱신한다.

### 6.5 2026-08-31 historical DEV 중간 결과

아래 수치는 현재 코드의 최종 결과가 아니라 보존된 historical DEV 결과다.

- 서비스 tuned 생성 n=3 평균: **0.7481** (run 평균 표준편차 0.0105)
- paired 개선/동률/악화 문항 수: **10/29/6**
- paired cluster bootstrap 평균 차이 95% 신뢰구간: **[-0.1407, 0.1951]**
- 최신 경쟁 서비스 양방향 blind 비교: **미착수**
- 사람 층화 표본 검토 결과: **미착수**

2026-09-01 재개 뒤 승인된 Gemini 호출로 현재 코드의 DEV 등록 3문항 C0/C1
`p0g` answer artifact를 수집했다. 아래 결과는 위 historical n=3과 분리한다.

### 6.6 현재 코드 DEV 3문항 `p0g` preflight

| 지표 | C0 | C1 |
|---|---:|---:|
| Retrieval hit | 2/3 | 3/3 |
| MRR | 0.250 | 0.778 |
| Any Gold @5 / @8 | 2/3 / 2/3 | 3/3 / 3/3 |
| All Gold @5 / @8 | 0/3 / 0/3 | 3/3 / 3/3 |
| Mean Gold Recall | 0.333 | 1.000 |
| Generation API error | 0 | 0 |
| Judge v8 문항별 점수 | `[2, 1, 0]` | `[2, 1, 2]` |
| Judge v8 평균 | 1.000 | 1.667 |
| Judge v8 GFC | 1/3 | 2/3 |
| Judge v8 반복 일치 | 전 문항 3/3 | 전 문항 3/3 |

두 answer artifact는
`processed/eval/preflight-20260901/generation-p0g/dev-smoke-c0-run1.answers.jsonl`과
`processed/eval/preflight-20260901/generation-p0g/dev-smoke-c1-run1.answers.jsonl`이다.
유효 Judge v8 결과는 `judge-p0g-v8/c0-summary.json`과 `c1-summary.json`에 있다.
차이는 `svc_reg_03`의 BIDV 검색 복구에서 발생했고, `svc_reg_02`는 두 조건 모두
복학 시 이월 세부를 누락해 1점이었다. v6/v7 실패·calibration artifact는 quota와
quote-format 문제로 무효다. v8은 18.8–24.3KB compact input과 exact match 및
line/list-marker canonical quote validation을 사용했다.

생성 오류 0건은 요청·응답 수집 성공만 뜻한다. Judge의 3회 일치는 판정 안정성
진단이지 질문 표본을 9개로 늘리지 않으며, 유효 표본은 여전히 n=3이다. 동일 모델
계열의 생성·판정에 따른 self-preference와 사람 calibration 미완 문제도 남는다.
따라서 CI, 생성 성능 headline, holdout 결과 또는 일반화 성능으로 사용하지 않는다.

질문별 GFC 성공 비율을 paired 분석하면 C0 0.333, C1 0.667, 차이는 +0.333이다.
그러나 paired bootstrap 95% CI는 `[0.000, 1.000]`이고 exact sign-flip과
majority-GFC McNemar의 `p`값은 모두 1.0이다. 이는 `svc_reg_03` 한 문항의 방향성
진단과 양립하지만, 표본 3개로 생성 성능 향상을 확정할 수 없음을 수치로 보여준다.

## 7. 오류 분석

### 7.1 검색 miss

서비스 기준선의 16개 miss 중 15개는 목표 문서가 후보 pool에 있었고, 1개는
FTS pool에도 없었다. 이는 코퍼스 부재만이 아니라 질의 정규화, 후보 경쟁,
문서 버전과 generic 문서의 문제를 보여준다. 튜닝 후 남은 5개 miss는 깊은
후보 순위, FTS 후보 부재, 같은 유형 문서의 구판 경쟁, 계절 표현 동의어 등으로
분류한다.

### 7.2 검색 hit–생성 실패

정답 문서와 근거가 검색돼도 생성 점수가 0인 문항은 retrieval보다 컨텍스트
선정, 핵심 조건 누락, 과잉 회피, 프롬프트 또는 judge 변동의 문제다. 특히
장학·등록 문항은 대상자와 예외 조건을 모두 포함해야 2점을 받을 수 있으므로,
문서 hit만으로 답변 가능성을 보장할 수 없다.

### 7.3 지표 허상과 수집 한계

연구비 트랙에서는 RAGAS context recall이 0인데 동일 컨텍스트의 생성 점수는
1–2인 사례가 있었다. 반대로 정답 문서가 존재해도 유권해석이나 세부기준이
수집되지 않아 완전한 답을 만들 수 없는 문항이 있었다. 따라서 자동 지표의
0점을 곧바로 실패로 간주하지 않고 원문·답변을 함께 검토한다.

### 7.4 날짜·짧은 사실 후처리 손실

생성 후처리기는 마침표 뒤 공백을 문장 경계로 처리해
`2026. 8. 24.(월)` 같은 한국식 점 표기 날짜를 `2026.`과 나머지 조각으로
분리했다. 이어지는 20자 미만 필터가 날짜 조각과
“학부 등록금은 동결되었습니다.” 같은 짧은 완전문을 제거했다. 이 문제는
등록·입학·취업·장학 문항의 날짜와 신청 조건을 실제로 훼손했다.

점 표기 날짜 안의 마침표를 문장 분리 전에 보호하고, 20자 필터 완화는 생성
초안에만 8자로 제한했다. 추출형 후보의 20자 노이즈 필터는 유지했다. 45개
공식 기준답안을 known-good draft로 사용한 외부 전송 없는 회귀에서 critical
value 보존 평균은 0.7426에서 1.0000으로 증가했고, 20문항 개선·25문항 동일·
악화 0문항이었다. 이는 splitter의 결정론적 안전성을 보여주지만 실제 모델
생성의 end-to-end 개선을 뜻하지 않으므로 표적 생성 n=3을 별도로 수행한다.

별도로 `svc_reg_02`에서는 정확한 생성 초안의 “학생지원시스템 → 등록 →
납부확인” 안내가 삭제됐다. claim의 “확인할 수 있다”는 permission 관계로
해석하면서 근거의 명사형 UI 안내는 같은 관계로 인식하지 못한 비대칭이
원인이었다. 공식 UI 경로, 동일 조회 동작·범위·critical value가 함께 있을
때만 허용하는 bridge를 추가하고, 명시적 불가 문구와 다른 메뉴는 계속
거부했다. 동일 초안·동일 context의 DEV 등록 3문항 결정론적 replay에서
정규화 문장 내용이 바뀐 것은 1/3이었고, 12문장 중 올바른 11문장은 유지하며
근거 없는 분할납부 확대 문장 1개만 제거했다. critical value 손실은 0이었다.
이 결과도 후처리의 유지·삭제 진단이며 품질 점수가 아니므로, 별도 LLM Judge
전후 평가는 외부 데이터 전송 승인 뒤 수행한다.

### 7.5 Claim별 인용과 관계 반전 방어

기존 claim attribution은 숫자가 청크 어딘가에 존재하고 단어가 충분히 겹치면
“분할 납부 가능”을 “불가능”, “80점 이상”을 “이하”로 뒤집거나 학부와 대학원의
인상률을 바꿔 붙인 문장도 지원된 것으로 판정할 수 있었다. 이를 막기 위해
허용·금지, 이상·이하·초과·미만·이내·미달·최소·최대,
인상·인하·동결, 날짜의 시작·종료 관계를 연도·학기·주체·값과 같은 문장 또는
표 행 안에서 결합하는 제한적 attribution veto를 추가했다. 복합 금액과 소수
수량을 정규화하고, 탭 행 경계를 보존하며, `검토 중`·`예정`을 완료된 사실과
구분한다. Citation도 claim별 supporting passage 중심 excerpt와 위치·hash를
반환한다.

77개 합성·공지형 adversarial microbenchmark에서 guard를 끈 ablation의 label
정확도는 0.4416, label+reason joint 정확도는 0.4156이고 false positive가
43건이었다. Guard를 켜면 두 정확도가 모두 1.0000, false positive와 false
negative가 0건이었다. 이 결과는 규칙 기반 attribution 구성요소의 표적 회귀
테스트이지 실제 검색·LLM 생성·서비스 일반화 성능이 아니다. 실제 효과는
동결 답변의 claim-level Judge 및 사람 검토에서 별도로 확인한다. 이 guard는
일반 NLI가 아니며, 의무·필요 관계(예: `제출해야 한다`와 `제출할 필요가 없다`)는
현재 표적 규칙의 범위 밖이다.

### 7.6 역할 변형 결과

6개 기본–역할 변형 pair의 검색 source 배열은 모두 동일했다. 생성 3회 점수는
5개 pair에서 동일했지만 `svc_reg_01`은 `[1,1,1]`, 학생 역할 변형은
`[1,0,0]`이었다. 검색이 같으므로 이 차이가 역할 prompt 때문인지 생성·judge
분산인지는 현재 결과만으로 단정할 수 없다. 따라서 역할 기능은 구현·입력
안전성은 확인됐지만 “생성 품질 6/6 무해”라는 주장은 사용하지 않는다.

## 8. 타당도 위협과 한계

1. **개발셋 과적합:** 서비스 45문항을 원인 진단과 튜닝에 반복 사용했고, 47개
   표적 규칙이 중복 제거 37개 DEV ID를 겨냥한다. 동결 holdout에서 규칙별 실제
   발동률·발동 가능률과 의도 밖 spillover를 측정해 일반화 여부를 사후 검증한다.
2. **Judge 변동과 편향:** 생성기와 Judge가 모두 Gemini 계열이라 자기 선호가
   생길 수 있고, 생성 n=3은 judge 자체의 일관성 검증과 다르다. 위치·장황성·
   모델 선호뿐 아니라 한국어 신뢰도, 점수 범위, 관대함과 클래스 불균형의 영향이
   남는다. 동일 답변 3회 반복에서는 점수·GFC가 45/45 일치했지만, 이는 재현성
   근거일 뿐 정확성 근거가 아니다. 독립 평가자 2인의 calibration 전에는
   Judge 타당도 문제가 해소되지 않는다.
3. **표본 크기와 검정력:** DEV는 45문항이고 최종 holdout의 headline Core는
   27문항이다. paired 설계와 family cluster bootstrap을 사용하더라도 작은 효과와
   category별 차이를 검출하는 힘이 제한된다.
4. **규칙 기반 코드 복잡도:** `search_api.py` 3,087→8,325줄,
   `bm25_search.py` 963→1,273줄, `generators.py` 1,053→1,294줄로 증가했다.
   증가분에는 표적 정규식·점검표·후처리 bridge가 많아 유지보수와 상호작용 회귀
   위험이 커졌다.[^freeze-snapshot]
5. **경쟁 비교 최신성:** 현재 경쟁 점수는 2026-08-06 snapshot이다.
6. **코퍼스 신선도:** 게시물 갱신과 삭제, 연도별 사본 누적과 구판 문서가 결과에
   영향을 준다.
7. **Gold 불완전성:** 하나의 anchor URL이나 evidence quote가 모든 유효한
   근거를 대표하지 못할 수 있다.
8. **코퍼스 구조 결함:** 약 11%의 suspect 문서를 허용한 인덱스이며, base64
   문자열 유출, 스크립트 렌더링 첨부 누락, 연도별 사본 누적을 완전히 해소하지
   못했다.
9. **단일 기관 일반화:** 역할 기반 검색 우선순위와 한국어 질의 정규화의 효과는
   부산대 코퍼스·질의 구성에 한정된다.
10. **비용·지연 평가:** 품질 개선과 함께 cross-encoder·dense·LLM 호출의
    latency와 자원 비용을 최종 표에 추가해야 한다.
11. **파서 감사 선택 편향:** 18문서는 표적 표본이고 Cascade 산출물을 canonical
    binding으로 사용했으며 독립 2인 원문 화면 검수를 하지 않았다.
12. **Holdout 사람 검수 미완료:** AI 검토 12건 수정과 기계 gate는 완료했지만
    실제 2인 human signoff와 최종 파일 동결 전이다.
13. **C2 확인 편향:** C2 v3 verifier는 첫 DEV run의 거부 사례를 보고 일반화된
    시간·날짜·복합어·관계 경계 규칙을 보정한 뒤 동결했다. run2·run3는 동결 뒤
    fresh 실행이지만, 전체 n=3 집계의 run1은 같은 raw response 재투영이므로
    독립 holdout 확인 전에는 탐색 결과다.

## 9. 결론

본 과제는 부산대학교의 이질적인 공식 문서를 수집·구조화하고, lexical 검색과
근거 기반 생성을 결합한 서비스 RAG 챗봇을 구현했다. Dense·융합·재정렬·부모
확장은 연구비 트랙에서 별도 평가했다. 단계별 평가를
통해 문서 검색, 근거 청크 검색, 생성 답변 품질이 서로 다른 병목임을 확인했다.
특히 2026-09-02 현재 코드의 동일 서비스 DEV 검색에서 Hit@5가 0.644에서
0.889, MRR이 0.494에서 0.717로 변했다. 이와 별개로 2026-08-31 historical
생성 n=3 평균은 0.7185에서 0.7481로 증가했으나 생성 차이의 bootstrap 신뢰구간은
0을 포함했다. 후속 현재 코드 DEV45 생성 n=3에서는 Judge 평균이 0.8815에서
1.2148로 증가하고 평균 점수 차이의 95% CI가 [0.0889, 0.5887]이었지만,
2/3 majority GFC는 0.289에서 0.444로 증가한 점 추정치와 달리 차이의 95% CI가
[-0.0217, 0.3201]로 0을 포함했다. 이는 retrieval 개선이 부분 답변의 평균 품질에는
전달됐지만 완전정답률을 확정적으로 개선했다고 말하려면 holdout과 사람 검증이
더 필요함을 보여준다.

후속 의도별 검색 보강 뒤 같은 DEV45의 C1은 Hit@5 45/45(1.000), MRR
.8137, Any required evidence@8 45/45(1.000), All required evidence@8
43/45(.956)를 기록했다. 선택된 실패 9문항의 fresh 생성·Judge는 평균 1.5556,
GFC 6/9였지만, 표적 선택 n=1이라는 경계 때문에 전체 성능 수치로 사용하지 않는다.
최종 국제 문항 수정도 fresh LLM 생성과 동결 holdout에서 재현되지 않았으므로
최종 일반화 성능이 아니다.

별도 C2 quote-bound 실험은 동일한 C1 검색 context에서 3-run 평균을 1.2148에서
1.3630으로, majority GFC를 20/45에서 26/45로 높였다. 그러나 GFC 신뢰구간의
하한이 0이고 exact McNemar `p=.1460`이며 verifier 개발에 DEV run1을 사용했으므로,
이는 생성–근거 결속 개선의 유망한 후보이지 최종 성능 향상 확정이 아니다.

파서 감사의 표적 기계 평가와 DEV45 생성 n=3·Judge v11, 고정 답변의 Judge
v11 3회 반복 안정성 평가는 완료했지만, 최종
결론은 holdout 36문항, Judge-사람 calibration, 실제 사람 검증, 필요 시 파서 원문 육안검수, 경쟁 서비스 최신
재수집·순서 교환 평가와 clean 전체 회귀를 완료한 뒤 갱신한다. 제출본은 성능이
오른 실험뿐 아니라 기각된 대안과
평가 한계를 함께 제시하여 결과의 재현 가능성과 해석 범위를 명확히 한다.

[^rule-inventory]: `docs/rule-inventory-20260904.md` (68개 규칙의 위치·트리거·분류·DEV 근거와 검증 명령). 정본 경로와 SHA는 `docs/progress-log-20260901.md`의 “2026-09-04 규칙 인벤토리 고정” 절에 기록했다.
[^dev45-v11]: 정본 결과는 `processed/eval/preflight-20260903/dev45-generation-current-v1/judge/dev45-3run-service-ab-v11.json`; 결합 재검증은 `processed/eval/post-freeze-analysis-20260904/generation-failures-v2/dev45-c0-c1-n3-generation-failures.json`이다. SHA는 progress log의 해당 두 절에 기록했다.
[^freeze-snapshot]: 동결 시점 코드 줄 수·SHA와 세 index SHA는 `evidence/20260914/tuning-freeze-snapshot-20260904.json`; 정본 기록은 progress log의 “2026-09-04 튜닝 동결” 절이다.
[^c2-v3]: 정본은 `processed/eval/preflight-20260904/dev45-grounded-claims-c2-v3/analysis/c1-vs-c2-v3-3run-majority-gfc.json`과 `c1-vs-c2-v3-3run-ab.json`이다. 입력 answer/Judge 및 분석 SHA, 호출 수, 실패 분해는 progress log의 “2026-09-04 C2 v3 동결·독립 n=3 평가” 절에 기록했다.
[^judge-v11-stability]: 정본은 `processed/eval/preflight-20260903/dev45-generation-current-v1/judge/stability-20260904/c0-run1-judge-v11-r1-r3.json`, `c1-run1-judge-v11-r1-r3.json`, `c0-vs-c1-run1-majority-gfc.json`이다. SHA와 호출 감사는 progress log의 “2026-09-04 C0/C1 Judge v11 안정성 반복 완료” 절에 기록했다.

## 참고문헌

1. Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework:
   BM25 and Beyond. *Foundations and Trends in Information Retrieval*, 3(4),
   333–389. https://doi.org/10.1561/1500000019
2. Karpukhin, V., et al. (2020). Dense Passage Retrieval for Open-Domain Question
   Answering. *EMNLP 2020*, 6769–6781. https://aclanthology.org/2020.emnlp-main.550/
3. Thakur, N., et al. (2021). BEIR: A Heterogeneous Benchmark for Zero-shot
   Evaluation of Information Retrieval Models. *NeurIPS Datasets and Benchmarks*.
4. Lee, H., et al. (2023). On Complementarity Objectives for Hybrid Retrieval.
   *ACL 2023*, 13357–13368. https://aclanthology.org/2023.acl-long.746/
5. Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009). Reciprocal Rank
   Fusion Outperforms Condorcet and Individual Rank Learning Methods. *SIGIR 2009*.
6. Nogueira, R., & Cho, K. (2019). Passage Re-ranking with BERT.
   https://arxiv.org/abs/1901.04085
7. Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive
   NLP Tasks. *NeurIPS 2020*.
8. Chen, T., et al. (2024). Dense X Retrieval: What Retrieval Granularity Should
   We Use? *EMNLP 2024*, 15159–15177.
9. Liu, N. F., et al. (2024). Lost in the Middle: How Language Models Use Long
   Contexts. *TACL*, 12, 157–173.
10. Maynez, J., et al. (2020). On Faithfulness and Factuality in Abstractive
    Summarization. *ACL 2020*, 1906–1919.
11. Shuster, K., et al. (2021). Retrieval Augmentation Reduces Hallucination in
    Conversation. *Findings of EMNLP 2021*, 3784–3803.
12. Stolfo, A. (2024). Groundedness in Retrieval-augmented Long-form Generation.
    *Findings of NAACL 2024*, 1537–1552.
13. Es, S., et al. (2024). RAGAs: Automated Evaluation of Retrieval Augmented
    Generation. *EACL System Demonstrations*, 150–158.
14. Zheng, L., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot
    Arena. *NeurIPS Datasets and Benchmarks*.
15. Wang, P., et al. (2024). Large Language Models are not Fair Evaluators.
    *ACL 2024*, 9440–9450.
16. Haldar, S., & Hockenmaier, J. (2025). Rating Roulette: Self-Inconsistency in
    LLM-As-A-Judge Frameworks. *Findings of EMNLP 2025*, 24986–25004.
17. Liu, Y., et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human
    Alignment. *EMNLP 2023*, 2511–2522.
18. Thakur, A. S., Choudhary, K., Ramayapally, V. S., Vaidyanathan, S., &
    Hupkes, D. (2025). Judging the Judges: Evaluating Alignment and
    Vulnerabilities in LLMs-as-Judges. *GEM² 2025*, 404–430.
    https://aclanthology.org/2025.gem-1.33/
19. Fu, X., & Liu, W. (2025). How Reliable is Multilingual LLM-as-a-Judge?
    *Findings of EMNLP 2025*, 11040–11053.
    https://aclanthology.org/2025.findings-emnlp.587/
20. Fujinuma, Y. (2026). Contrastive Decoding Mitigates Score Range Bias in
    LLM-as-a-Judge. *Findings of ACL 2026*, 13404–13418.
    https://aclanthology.org/2026.findings-acl.657/
21. Collot, S., Fraser, C., Zhao, J., Shen, W. F., Willi, T., & Leontiadis, I.
    (2026). Balanced Accuracy: The Right Metric for Evaluating LLM Judges —
    Explained through Youden's J Statistic. *EACL 2026 Industry Track*,
    927–936. https://aclanthology.org/2026.eacl-industry.69/

## 부록 A. 재현 산출물

최종 동결 뒤 다음을 삽입한다.

- Git commit/tag
- 실행 환경과 dependency lock hash
- 코퍼스·인덱스 revision
- 평가 JSONL/CSV 파일 목록과 SHA-256
- 검색 및 생성 재현 명령
- judge model·prompt·rubric·temperature
- 전체 회귀 로그
- `scripts/build_evidence_manifest.py`로 생성한 artifact hash·완전성 manifest

2026-09-01 작업 snapshot에서 이미 보존한 중간 근거는 다음과 같다.

후속 final schedule·WAL·배타 run-lock·retrieval 108-slot E2E·generation-run
GFC·partial Judge·사람 fallback 테스트까지 반영한 2026-09-02 전체 검사는
656 tests OK(6 skip), ESLint·TypeScript·Vite production build PASS다. 이는 아직
dirty 작업 snapshot이므로 최종 동결 커밋의 전체 check와 데모 smoke는 별도로
재실행한다. 실제 holdout·외부 LLM 호출은 이 검사에서 수행하지 않았다.

- `evidence/20260914/cap2-vs-cap4-retrieval-dev45.{json,csv,html}`
- `evidence/20260914/grounding-guard-ablation.{json,csv}`
- `evidence/20260914/postprocessor-offline-ab.json`
- `processed/eval/preflight-20260902/postprocessor-ab/dev-smoke-c1-postprocessor-pairs-v2.analysis.json`
- `processed/eval/dev45-c1-multifacet-20260902-v2/matrix-summary.json`
- `processed/eval/dev45-c1-multifacet-20260902-v2/old-c1-vs-multifacet.{json,csv,html}`
- `evidence/20260914/local-full-check-20260901.log`
- `processed/eval/20260901-parser-audit-v1.json`
- `processed/eval/20260901-parser-audit-v1.csv`
- `processed/eval/preflight-20260901/generation-p0g/dev-smoke-c0-run1.answers.jsonl`
- `processed/eval/preflight-20260901/generation-p0g/dev-smoke-c1-run1.answers.jsonl`
- `processed/eval/preflight-20260901/judge-p0g-v8/c0-summary.json`
- `processed/eval/preflight-20260901/judge-p0g-v8/c1-summary.json`
- `processed/eval/preflight-20260901/judge-p0g-v8/gfc-paired-analysis.json`
- `processed/eval/preflight-20260901/judge-p0g-v8/gfc-paired-cases.csv`
- `processed/eval/preflight-20260904/dev45-grounded-claims-c2-v3/analysis/c1-vs-c2-v3-3run-review-v2.html`
- `processed/eval/preflight-20260903/dev45-generation-current-v1/judge/stability-20260904/c0-run1-judge-v11-r1-r3.{json,csv}`
- `processed/eval/preflight-20260903/dev45-generation-current-v1/judge/stability-20260904/c1-run1-judge-v11-r1-r3.{json,csv}`
- `processed/eval/preflight-20260903/dev45-generation-current-v1/judge/stability-20260904/c0-vs-c1-run1-majority-gfc.{json,csv}`
- `docs/c2-grounded-claims-decision-20260904.md`
- `docs/holdout-v2-human-review.md`
- `evidence/holdout-v2-reviewer-a.json`
- `evidence/holdout-v2-reviewer-b.json`
- `scripts/build_holdout_signoff.py`

이 파일들은 최종 holdout·외부 생성·사람 평가를 대신하지 않으며, 코드 동결 뒤
생성할 `evidence/20260914/final/manifest.json`과 구분한다.
