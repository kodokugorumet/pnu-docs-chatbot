# 2026-09-04 서비스 규칙 인벤토리

기준: `evidence/20260914/tuning-freeze-snapshot-20260904.json`의 서비스 코드
SHA. 이 문서는 단순 URL·보안·입력 형식·범용 문자열 파싱 정규식이 아니라,
검색 결과·생성 지시·최종 claim 채택을 바꾸는 **production decision rule**을
1행 1규칙 그룹으로 기록한다. 따라서 아래 1:1은 `Rule ID ↔ 의사결정 함수 또는
함께 움직이는 정규식 그룹`의 대응이며, 모든 `re.compile` 호출 수를 의미하지
않는다.

분류 기준:

- 일반 규칙: 특정 DEV 문항을 몰라도 원리와 안전 조건을 설명할 수 있는 규칙
- 표적 규칙: 특정 DEV 실패를 관찰한 뒤 그 의도·문서 형태를 직접 겨냥해 도입한 규칙

## 1. 질의 정규화·후보 순위·중복 제거

| Rule ID | 규칙 이름 | 위치 | 트리거 조건 | 분류 | 도입 근거 DEV ID | 도입일·progress-log |
|---|---|---|---|---|---|---|
| `QRY-NORM-001` | 요청 상용구 제거 | `scripts/search_api.py:427-442` | 기관명과 `RETRIEVAL_REQUEST_TERMS`를 제거하되 내용어가 없으면 원문 토큰 유지 | 일반 | — | ≤2026-09-01 §2 |
| `QRY-NORM-002` | FTS UI 말투 토큰 제거 | `scripts/bm25_search.py:480-506` | service tuning에서 `SERVICE_QUERY_STOP_TOKENS`를 FTS OR 질의에서 제거 | 일반 | — | ≤2026-09-01 §2 |
| `RANK-001` | 상용 제목 강등·내용어 구제 | `scripts/bm25_search.py:509-631,717-782` | generic title은 강등하되 내용어 85%·최소 3개·숫자 전부 일치 시 구제 | 일반 | — | ≤2026-09-01 §2 |
| `RANK-002` | 제목 연도 불일치 강등 | `scripts/bm25_search.py:717-782` | 질의에 연도가 있고 비-generic 후보 제목/파일명에 다른 연도만 존재 | 일반 | — | ≤2026-09-01 §2 |
| `RANK-003` | 희소 영문 식별자 승격 | `scripts/bm25_search.py:633-648,717-782` | BIDV·TOPIK 등 비범용 영문 토큰이 질의와 제목/파일명에 정확히 공존 | 일반 | — | ≤2026-09-01 §2 |
| `RANK-004` | 절차 질의의 등록금 회의록 강등 | `scripts/bm25_search.py:650-682,717-782` | 신청·앱·절차 질의인데 후보가 등록금심의위원회 회의록이며 심의 결과 질의는 아님 | 표적 | `svc_reg_05` 주변 오염 | 2026-09-01 §2 |
| `RANK-005` | 일반 외국인 학부전형 우선 | `scripts/bm25_search.py:690-782` | 범위를 지정하지 않은 외국인 학부 신입 자격 질의에서 일반 모집요강을 우선하고 대학원·특화 track은 후순위 | 표적 | `svc_intl_01` | 2026-09-04 §의도 기반 검색 확장 |
| `RANK-006` | 문서 다양성 cap·BM25 anchor 보존 | `scripts/bm25_search.py:849-921` | 문서당 기본 2개 chunk, 보존 anchor가 cap에 밀리면 같은 문서의 최하위 선택을 교체 | 일반 | 연구비 `grant_031` | ≤2026-09-01 §2 |
| `RANK-007` | 최종 context exact/near 중복 제거 | `scripts/search_api.py:1433-1502` | 정규화 exact 또는 near duplicate context를 제거하고 overflow unique로 충원 | 일반 | — | 2026-09-01 §2 |

## 2. 의도 기반 검색어 확장

모든 확장은 `scripts/search_api.py:445-632`의
`_service_retrieval_query_expansions()`와 `expand_service_retrieval_query()`에
있으며, service-tuning lane에서만 적용된다.

| Rule ID | 규칙 이름 | 트리거 조건·추가 문서어 | 분류 | 도입 근거 DEV ID | 도입일·progress-log |
|---|---|---|---|---|---|
| `EXP-001` | 학생증·증명서 위탁 | `학생증|증명서` + `외부기관|위탁` → 개인정보처리·위탁현황·수탁기관 | 표적 | `svc_core_03` | 2026-09-04 §의도 기반 검색 확장 |
| `EXP-002` | 모집인원 변경 | 학년도+신입+모집인원+전년대비 → 대학입학전형·기본계획·주요변경사항 | 표적 | `svc_adm_03` | 2026-09-04 §의도 기반 검색 확장 |
| `EXP-003` | D-2 단체접수 | D-2+비자/체류+연장+단체 → 체류기간·사전예약·제출서류·수수료 | 표적 | `svc_intl_03`, `svc_intl_03_role_free` | 2026-09-04 §의도 기반 검색 확장 |
| `EXP-004` | 외국인 학부 신입 자격 | 외국인+학부+신입+자격 → 특별전형·모집요강·국적·언어능력·학력 | 표적 | `svc_intl_01` | 2026-09-04 §의도 기반 검색 확장 |
| `EXP-005` | 교환학생 선발·일정 | 교환학생/해외파견+선발규모+지원일정 → 교비프로그램·1차·온라인지원·합격자발표 | 표적 | `svc_intl_02` | 2026-09-04 §의도 기반 검색 확장 |
| `EXP-006` | 하계 자격증 과정 | 여름방학+자격증+강의/특강/프로그램/대비 → 하계방학·취업역량·비교과·대비반 | 표적 | `svc_emp_01` | 2026-09-04 §의도 기반 검색 확장 |
| `EXP-007` | 장애학생 수업 지원 | 장애+지원+수업/시험 → 장애학생지원센터·교수학습·강의지원·교재지원 | 표적 | `svc_sup_01` | 2026-09-04 §의도 기반 검색 확장 |
| `EXP-008` | 제3자 인권신고 | 신고+피해자 아님/대신/제3자/목격 → 센터이용·Q&A·제3자·신고 | 표적 | `svc_sup_03` | 2026-09-04 §의도 기반 검색 확장 |

## 3. facet 분리와 동일 문서 sibling 완결

| Rule ID | 규칙 이름 | 위치 | 트리거 조건 | 분류 | 도입 근거 DEV ID | 도입일·progress-log |
|---|---|---|---|---|---|---|
| `FACET-001` | 명시적 facet 추출 | `scripts/search_api.py:1787-2458` | 날짜·시간·금액·금리·자격·방법·입금처·예산액·구성비를 질의와 근거에서 관계 단위로 추출 | 일반 | — | 2026-09-02 §9.2 |
| `FACET-002` | 동일 문서 sibling completion | `scripts/search_api.py:2617-3352` | temporal/procedure/multipart 또는 2개 이상 명시 facet이 있고 동일 문서에 누락 facet 후보가 존재 | 일반 | — | 2026-09-02 §9.2 |
| `FACET-003` | 학년도·학기 충돌 veto | `scripts/search_api.py:2464-2520` | 질의와 candidate의 명시 연도·학기가 충돌하면 anchor/neighbor 사용 금지 | 일반 | — | 2026-09-02 §9.2 |
| `FACET-004` | marginal facet 교체 | `scripts/search_api.py:2617-3352` | 같은 문서의 중복 facet chunk를 누락 facet을 가진 sibling으로 교체 | 표적 | `svc_sch_02`, `svc_sch_05` | 2026-09-02 §9.2 |
| `FACET-005` | 문서 제출기한 객체 결속 | `scripts/search_api.py:1839-1844,2188-2230` | 질문의 성적표·서류·논문 등 객체, 제출 동작, 기한, 실제 날짜가 같은 행에 존재 | 표적 | `svc_grad_02`, `svc_grad_04` | 2026-09-03 §12 |
| `FACET-006` | 대상학과 원거리 sibling | `scripts/search_api.py:1815-1818,2825-2950` | `어떤/어느 학과` 자격 facet이 비었으면 관련 문서의 앞쪽 eligibility 행까지 탐색 | 표적 | `svc_sch_01` | 2026-09-03 §11 |
| `FACET-007` | 단일 숫자 원거리 sibling | `scripts/search_api.py:2790-2980` | 요구 facet이 amount/rate/budget 하나이고 강한 동일 문서 seed가 있으나 값 facet이 비어 있음 | 표적 | `svc_core_01`, `svc_core_01_role_staff` | 2026-09-04 §추가 후처리·단일 숫자 facet |
| `FACET-008` | 예산액·구성비 facet 분리 | `scripts/search_api.py:1831-1838,2314-2347` | 전체 예산 규모와 영역별 비중을 독립 facet으로 요구하고 완전한 표만 지원으로 인정 | 표적 | `svc_core_02`, `svc_core_02_role_researcher` | 2026-09-03 §12 |
| `FACET-009` | 외국인 자격 3-part | `scripts/search_api.py:1855-1870,2411-2458` | 일반 외국인 학부 신입 자격에서 국적·언어·학력을 독립 필수 facet으로 설정 | 표적 | `svc_intl_01` | 2026-09-04 §의도 기반 검색 확장 |
| `FACET-010` | 교환학생 2-part | `scripts/search_api.py:1872-1880,2424-2458` | 선발규모와 온라인 지원일정을 독립 필수 facet으로 설정 | 표적 | `svc_intl_02` | 2026-09-04 §의도 기반 검색 확장 |
| `FACET-011` | D-2 3-part·cap 3 | `scripts/search_api.py:1890-1902,2640-2737` | D-2 단체접수에서 예약·회차일정·수수료를 필수화하고 해당 의도만 문서 cap을 3으로 확대 | 표적 | `svc_intl_03`, `svc_intl_03_role_free` | 2026-09-04 §잔여 국제 문항·D-2 |
| `FACET-012` | 보조기기 자부담 3-part·cap 3 | `scripts/search_api.py:1903-1917,2640-2737` | 지원범위·선납부/증빙 절차·서류/구글폼을 필수화하고 해당 의도만 cap 3 | 표적 | `svc_sup_02` | 2026-09-04 §자부담금 지원 3-part |
| `FACET-013` | 제3자 신고 3-part | `scripts/search_api.py:1881-1889,2424-2458` | 제3자 신고 가능·피해자 의사·피해자 인적사항을 독립 facet으로 설정 | 표적 | `svc_sup_03` | 2026-09-04 §의도 기반 검색 확장 |
| `FACET-014` | 의도별 탐색 반경 | `scripts/search_api.py:2960-3005` | 기본 2, 외국인/제3자 4, 선택 의도 8, D-2·제출기한·단일숫자 최대 24 chunk | 표적 | `svc_sch_01`, `svc_grad_02`, `svc_grad_04`, `svc_core_01`, `svc_intl_01`, `svc_intl_02`, `svc_intl_03`, `svc_sup_02`, `svc_sup_03` 및 role 변형 | 2026-09-03~04 §§11–12·의도 확장 |

## 4. 생성 prompt와 점검표

| Rule ID | 규칙 이름 | 위치 | 트리거 조건 | 분류 | 도입 근거 DEV ID | 도입일·progress-log |
|---|---|---|---|---|---|---|
| `GEN-001` | 근거 한정·값 보존 기본 prompt | `scripts/rag/generators.py:349-426` | 모든 비-structured 생성 | 일반 | — | ≤2026-09-01 §3 |
| `GEN-002` | 부분 근거 답변·전체 회피 금지 | `scripts/rag/generators.py:416-423` | 여러 항목 중 일부만 근거가 있을 때 확인 가능한 항목은 답하고 나머지만 회피 | 일반 | — | 2026-09-03 §10 |
| `GEN-003` | 상반 선택지 분리 | `scripts/rag/generators.py:396-401` | 이월/반환 등 서로 다른 처리 결과를 한 문장에 합치지 않음 | 일반 | — | 2026-09-04 §후처리 관계 오판 |
| `GEN-004` | 한 문장 한 근거 블록 | `scripts/rag/generators.py:401-404` | 대상·자격, 신청 경로, 비용이 서로 다른 source면 각각 별도 줄 | 표적 | `svc_grad_03`, `svc_grad_03_role_free`, `svc_adm_04` | 2026-09-04 §저장초안 전수 감사 |
| `GEN-005` | 명시 facet 점검표 | `scripts/rag/generators.py:433-466` | 언제·금액·자격·방법·가능성 표현을 독립 점검 항목으로 삽입 | 일반 | — | 2026-09-02 §9.2 |
| `GEN-006` | multipart 별도 응답 | `scripts/rag/generators.py:538-543` | 물음표 2개 이상 또는 각각/모두/둘다/및/구분자 포함 | 일반 | — | 2026-09-02 §9.2 |
| `GEN-007` | 대체·면제 이수기준 | `scripts/rag/generators.py:467-471` | 대신/대체/면제/인정 + 시험/강좌/과목/이수 | 표적 | `svc_grad_05` | 2026-09-04 §추가 후처리·생성 점검표 |
| `GEN-008` | 직무체험 활동 방식 | `scripts/rag/generators.py:472-476` | 체험/경험 + 프로그램/직무/진로/업무/현장 | 표적 | `svc_emp_03` | 2026-09-04 §추가 후처리·생성 점검표 |
| `GEN-009` | 등록금 고지서·등록 예외 | `scripts/rag/generators.py:477-486` | 등록금+납부 시점/방법 → 고지서 출력 시점과 0원·전액장학 예외 | 표적 | `svc_adm_02` | 2026-09-04 §추가 후처리·생성 점검표 |
| `GEN-010` | 업무별 수탁기관 | `scripts/rag/generators.py:487-491` | 학생증/증명서 + 외부기관/위탁 | 표적 | `svc_core_03` | 2026-09-04 §의도 기반 검색 확장 |
| `GEN-011` | 모집인원·변경점 분리 | `scripts/rag/generators.py:492-499` | 학년도+신입+모집인원+전년대비 | 표적 | `svc_adm_03` | 2026-09-04 §의도 기반 검색 확장 |
| `GEN-012` | D-2 5항목 점검표 | `scripts/rag/generators.py:500-515` | D-2+비자/체류+연장+단체 → 1차·대상·예약·서류·수수료/현금/권종 | 표적 | `svc_intl_03`, `svc_intl_03_role_free` | 2026-09-04 §D-2 후처리 재분해 |
| `GEN-013` | 자부담금 절차 점검표 | `scripts/rag/generators.py:516-523` | 정보통신보조기기+자부담/개인부담+지원 | 표적 | `svc_sup_02` | 2026-09-04 §자부담금 지원 3-part |
| `GEN-014` | 외국인 자격 구분 | `scripts/rag/generators.py:524-529` | 외국인+학부+신입+자격 | 표적 | `svc_intl_01` | 2026-09-04 §의도 기반 검색 확장 |
| `GEN-015` | 교환학생 선발·일정 | `scripts/rag/generators.py:530-535` | 교환학생/해외파견+선발규모+지원일정 | 표적 | `svc_intl_02` | 2026-09-04 §의도 기반 검색 확장 |
| `GEN-016` | 하계 자격증 과정 | `scripts/rag/generators.py:544-549` | 여름방학+자격증+강의/특강/프로그램/대비 | 표적 | `svc_emp_01` | 2026-09-04 §의도 기반 검색 확장 |
| `GEN-017` | 장애학생 수업·시험 구분 | `scripts/rag/generators.py:550-555` | 장애+지원+수업/시험 | 표적 | `svc_sup_01` | 2026-09-04 §의도 기반 검색 확장 |
| `GEN-018` | 제3자 신고 조건 | `scripts/rag/generators.py:556-564` | 신고+피해자 아님/대신/제3자/목격 | 표적 | `svc_sup_03` | 2026-09-04 §의도 기반 검색 확장 |

## 5. 후처리 허용·거부와 claim attribution

| Rule ID | 규칙 이름 | 위치 | 트리거 조건 | 분류 | 도입 근거 DEV ID | 도입일·progress-log |
|---|---|---|---|---|---|---|
| `POST-001` | 날짜 보호 문장 분리 | `scripts/search_api.py:1655-1767` | 날짜 내부 마침표를 보호한 뒤 줄·문장·표식 단위로 분리, 저품질 조각 제거 | 일반 | — | 2026-09-01 §3 |
| `POST-002` | 모델 회피문 제거 | `scripts/search_api.py:3844-3848,6919-6953` | 표준 회피·근거 없음 문장을 `model_abstention`으로 분류 | 일반 | — | 2026-09-01 §3 |
| `POST-003` | critical-value 동일성 | `scripts/search_api.py:3780-4292,6768-6917` | 학년도·학기·회차·날짜·시간·금액·비율·수량·전화번호가 source unit에 모두 존재해야 함 | 일반 | — | 2026-09-01 §3 |
| `POST-004` | 공유 단위 수량 범위 | `scripts/search_api.py:3837-3842,4268-4287` | `3~4개월`, 대시, `에서/부터`처럼 뒤쪽에만 단위가 있는 범위의 두 끝값 추출 | 표적 | `svc_acad_02`, `svc_acad_04` | 2026-09-04 §저장초안 전수 감사 |
| `POST-005` | lexical support threshold | `scripts/search_api.py:4304-4320,6926-7075` | 비숫자 claim term 40%와 최소 anchor 수를 만족하거나 제한된 paraphrase bridge 필요 | 일반 | — | 2026-09-01 §3 |
| `POST-006` | permission·comparator·direction veto | `scripts/search_api.py:3849-3959,4691-6523` | 허용/금지, 이상/이하, 증가/감소, 예정/확정의 operator·scope 충돌 시 거부 | 일반 | — | 2026-09-03 §10·2026-09-04 rubric v11 준비 |
| `POST-007` | UI 조회·신청 path bridge | `scripts/search_api.py:5362-5408,5629-5830` | 같은 object/action/value와 UI marker가 한 행 또는 인접 행에 있고 불가 문구가 없음 | 표적 | `svc_reg_02`, `svc_adm_02`, `svc_sch_06` | 2026-09-02 §9.1·2026-09-04 후처리 관계 오판 |
| `POST-008` | 범주형 목록 가능 표현 | `scripts/search_api.py:5334-5510` | 수납은행 등 3개 이상 목록의 75%와 scope anchor가 일치할 때 `가능` paraphrase 허용 | 표적 | `svc_reg_01` | 2026-09-04 §후처리 관계 오판 |
| `POST-009` | 이름 있는 절차 가능 표현 | `scripts/search_api.py:5348-5627` | BIDV 등 고유 영문 토큰·방법 문구·동작이 일치하고 명시적 금지가 없음 | 표적 | `svc_reg_03` | 2026-09-04 §후처리 관계 오판 |
| `POST-010` | 명시 eligibility roster | `scripts/search_api.py:5409-5424,5832-5957` | 모집대상·지원자격·대출대상 표제 주변의 긍정 명사행과 claim scope가 강하게 일치 | 표적 | `svc_sch_01`, `svc_adm_01` | 2026-09-04 §저장초안 전수 감사 |
| `POST-011` | 참여혜택 행 | `scripts/search_api.py:5425-5428,5959-6014` | 참여혜택 표제 아래 같은 행의 숫자와 핵심어 60% 이상 일치, 지급 없음/제외는 거부 | 표적 | `svc_intl_04` | 2026-09-04 §저장초안 전수 감사 |
| `POST-012` | 연락처 소유자·전화번호 행 | `scripts/search_api.py:5429-5431,6016-6058` | 같은 행의 정확한 전화번호와 학과/학부/대학원/센터 소유자가 일치 | 표적 | `svc_acad_05` | 2026-09-04 §저장초안 전수 감사 |
| `POST-013` | 명사형 프로그램 capability | `scripts/search_api.py:5446-5461,6060-6156` | 이수/체험/경험/확인/준비 claim이 4개 이상 핵심어·기준값과 일치하고 부정문이 없음 | 표적 | `svc_grad_05`, `svc_emp_02`, `svc_emp_03` | 2026-09-04 §semantic guard·저장초안 전수 감사 |
| `POST-014` | 자부담 선납부·증빙 지원 bridge | `scripts/search_api.py:5463-5472,6158-6204` | 정보통신보조기기 범위에서 자부담 납부→증빙 확인→지원의 순서가 같은 문장에 존재 | 표적 | `svc_sup_02` | 2026-09-04 §자부담금 지원 3-part |
| `POST-015` | D-2 예약·수수료·회차 bridge | `scripts/search_api.py:5473-5484,6206-6333` | 단체접수 서비스/필수예약, 60,000원·현금·만원권·GKS 면제, 1·2차/3차 대상 표를 각각 엄격 검증 | 표적 | `svc_intl_03`, `svc_intl_03_role_free` | 2026-09-04 §D-2 후처리 재분해 |
| `POST-016` | same-row local relation bridge | `scripts/search_api.py:6335-6523` | 표 행의 날짜·시간·방향·비교값이 동일 scope와 정확히 결속될 때만 unscoped relation 허용 | 표적 | `svc_sch_02`, `svc_adm_02`, `svc_intl_02` | 2026-09-03~04 §후처리 표 행 수정 |
| `POST-017` | 예산표 다중 행 claim | `scripts/search_api.py:2314-2347,6768-6820` | claim이 요구한 예산액/예산규모·구성비/비중 facet이 완전한 표에 있고 모든 수치·연도가 일치 | 표적 | `svc_core_02`, `svc_core_02_role_researcher` | 2026-09-03 §12·2026-09-04 저장초안 감사 |
| `POST-018` | 다중 날짜·대상 행 집계 | `scripts/search_api.py:6822-6917` | 3개 이상 날짜 claim에서 동일 scope 행만 집계하고, 복수 대상은 각 대상이 모든 날짜를 독립 보유해야 함 | 표적 | `svc_acad_03`, `svc_adm_04` | 2026-09-03 §10 |
| `POST-019` | claim별 최대 2개 citation | `scripts/search_api.py:6926-7075` | lexical·critical value·semantic 검사를 모두 통과한 상위 2개 source만 claim에 귀속 | 일반 | — | 2026-09-01 §3 |

## 6. 집계

이 인벤토리는 총 **68개** decision rule을 기록한다.

- 일반 규칙: **21개**
- 표적 규칙: **47개**
- 표적 규칙이 겨냥한 DEV 문항: **37개 ID**(role 변형 포함, 중복 제거)

표적 DEV ID 37개:

`svc_reg_01`, `svc_reg_02`, `svc_reg_03`, `svc_reg_05`,
`svc_sch_01`, `svc_sch_02`, `svc_sch_05`, `svc_sch_06`,
`svc_acad_02`, `svc_acad_03`, `svc_acad_04`,
`svc_acad_05`, `svc_grad_02`, `svc_grad_03`, `svc_grad_03_role_free`,
`svc_grad_04`, `svc_grad_05`, `svc_core_01`, `svc_core_01_role_staff`,
`svc_core_02`, `svc_core_02_role_researcher`, `svc_core_03`, `svc_adm_01`,
`svc_adm_02`, `svc_adm_03`, `svc_adm_04`, `svc_intl_01`, `svc_intl_02`,
`svc_intl_03`, `svc_intl_03_role_free`, `svc_intl_04`, `svc_emp_01`,
`svc_emp_02`, `svc_emp_03`, `svc_sup_01`, `svc_sup_02`, `svc_sup_03`.

## 7. 누락 검증 명령

서비스 코드는 변경하지 않고 다음 read-only 명령으로 의사결정 표면과 문서의
대응을 검증한다.

```bash
rg -n '^def (normalize_retrieval_query|_service_retrieval_query_expansions|expand_service_retrieval_query|select_distinct_contexts|_required_query_facets|_intent_specific_supported_facets|_complete_with_adjacent_facets|attribute_claim)|^def (_categorical_availability_supported|_named_procedure_availability_supported|_verification_availability_supported|_official_ui_application_path_supported|_enumerated_eligibility_supported|_labeled_participation_benefit_supported|_contact_phone_supported|_documented_capability_supported|_assistive_copay_support_procedure_supported|_group_visa_)' scripts/search_api.py
rg -n '^def (demote_generic_candidates|select_document_diverse_results)|^SERVICE_QUERY_STOP_TOKENS|^GENERIC_SOURCE_TITLES' scripts/bm25_search.py
rg -n '^def (build_prompt|_build_answer_requirement_block)' scripts/rag/generators.py
rg -o '`(QRY-NORM|RANK|EXP|FACET|GEN|POST)-[0-9]{3}`' docs/rule-inventory-20260904.md | sort | uniq -c
```

범위에서 제외한 정규식은 parser profile 형식, API 인증·origin, provider 오류
정규화, 날짜·금액 자체의 lexical parsing처럼 독립적으로 검색 순위나 claim
채택 결정을 만들지 않는 보조 요소다. `POST-003`처럼 여러 lexical parser가 한
의사결정에서 함께 움직이는 경우에는 하나의 rule group으로 묶었다.
