"""연구비 규정 질의의 기관 스코프 라우터.

비교 대상 챗봇(yunju.work grant-rules)의 '경로고정/광역' 스코프 해석을
단순화한 규칙 기반 라우터. 질의에 특정 사업·기관이 명시되면 해당 기관으로
스코프를 고정하고, 명시가 없으면 국가 공통기준(국가법령 + 과기부 고시)과
부산대 자체규정을 기본 스코프로 사용한다.

색인된 institution 값 기준으로 동작하며, 반환된 스코프는 검색 결과
후처리 필터로 쓴다(스코프 밖 문서 제거 후 상위 k 유지).
"""

from __future__ import annotations

# 질의 키워드 → 명시 기관 스코프. 먼저 일치하는 규칙이 이긴다.
EXPLICIT_INSTITUTIONS: list[tuple[list[str], list[str]]] = [
    (["수산과학원"], ["국립수산과학원"]),
    (["인문사회"], ["한국연구재단"]),
    (["식약처", "식품의약품"], ["식품의약품안전처"]),
    (["중소기업", "중기부"], ["중소벤처기업부"]),
    (["산업기술혁신"], ["산업통상자원부"]),
    (["창의재단", "과학창의재단", "찾아가는 학교"], ["한국과학창의재단"]),
]

# 여비·출장 질의: 대학 내부규정(취업규칙 등) 잡음을 제거하고
# 공무원 여비 규정(국가법령)·연구개발비 사용 기준(과기부)에
# 유권해석(인사혁신처 100문100답·보수 업무지침)을 더한다 (D35).
TRAVEL_TERMS = ["여비", "출장", "일비", "숙박비", "운임"]
TRAVEL_SCOPE = ["국가법령", "과학기술정보통신부", "인사혁신처"]

# 기본 스코프: 국연법 계열 공통기준 + 부산대 자체규정 + NRF 공통 유권해석.
# 한국연구재단 추가 근거 (D42): 정부연구비 QA 사례집(NRF·연산협)은 혁신법
# 전반의 유권해석층인데, 기본 스코프가 8슬롯을 다 채우면 스코프 밖 문서는
# 보충 경로로도 진입 불가 — grant_017·021에서 정답 Q&A가 차단됨을 실증.
DEFAULT_SCOPE = ["국가법령", "과학기술정보통신부", "부산대학교 산학협력단", "한국연구재단"]

# 명시됐지만 corpus에 없는 기관(백필 대기) — 스코프 제한 없이 전체 검색.
# 창의재단은 D35에서 수집 완료되어 명시 기관으로 승격.
UNCOVERED_MARKERS = ["극지연구소", "가스공사", "전력연구원", "한국전력", "환경부", "선박해양플랜트"]


def route(query: str) -> list[str] | None:
    """질의에 적용할 institution 허용 목록. None이면 전체 검색."""
    for marker in UNCOVERED_MARKERS:
        if marker in query:
            return None
    for keywords, scope in EXPLICIT_INSTITUTIONS:
        if any(k in query for k in keywords):
            # 준용 관계 (D40): 기관 지침의 여비 조항은 대부분 공무원 여비
            # 규정을 준용한다. 기관 단독 고정은 준용 원문(금액·시간 기준)을
            # 차단한다 — grant_045에서 컨텍스트 8건에 금액 0건으로 실증.
            # 여비 질의면 기관 + 여비 스코프 합집합으로 넓힌다.
            if any(t in query for t in TRAVEL_TERMS):
                return scope + [s for s in TRAVEL_SCOPE if s not in scope]
            return scope
    if any(t in query for t in TRAVEL_TERMS):
        return TRAVEL_SCOPE
    return DEFAULT_SCOPE


# 기본 스코프에서 대학 언급이 없는 일반 질의는 국가 공통기준을 자체규정보다
# 우선한다(비교 대상 챗봇의 '상위 근거 우선' 관계와 같은 원칙).
COMMON_FIRST = ["국가법령", "과학기술정보통신부"]
LOCAL_MARKERS = ["부산대", "우리 대학", "우리대학", "본교"]


def filter_hits(
    hits: list[dict],
    scope: list[str] | None,
    top_k: int,
    query: str = "",
    fallback_files: set[str] | frozenset[str] | None = None,
) -> list[dict]:
    """스코프 필터 적용 후 상위 top_k. 스코프 결과가 부족하면 원본으로 보충.

    ``fallback_files``는 플러딩 완화의 폴백층(D50)이다. 여기 등록된 파일의
    청크는 1군(비폴백) 근거가 top_k를 못 채울 때만 잔여 슬롯에 진입한다 —
    "관련 있는 종합 문서"가 다양화 캡 경쟁에서 기존 정답 근거를 밀어내는
    실패(D46 매뉴얼 516p, D49 NRF 3종 121청크: 규모 불문 동일 패턴)를
    구조적으로 차단한다. 순서: 1군 스코프 → 폴백 스코프 → 스코프 밖 보충.
    """
    if scope is None:
        scoped = list(hits)
    else:
        scoped = [h for h in hits if (h.get("institution") or "") in scope]
        if scope == DEFAULT_SCOPE and not any(m in query for m in LOCAL_MARKERS):
            # 공통기준과 자체규정을 교차 배치: 공통기준을 먼저 세우되
            # 자체규정도 상위에 남겨 둘 다 근거로 쓸 수 있게 한다.
            common = [h for h in scoped if (h.get("institution") or "") in COMMON_FIRST]
            local = [h for h in scoped if (h.get("institution") or "") not in COMMON_FIRST]
            merged = []
            for i in range(max(len(common), len(local))):
                if i < len(common):
                    merged.append(common[i])
                if i < len(local):
                    merged.append(local[i])
            scoped = merged
    if fallback_files:
        primary = [h for h in scoped if (h.get("file_name") or "") not in fallback_files]
        demoted = [h for h in scoped if (h.get("file_name") or "") in fallback_files]
        scoped = primary + demoted
    if len(scoped) < top_k:
        seen = {id(h) for h in scoped}
        scoped += [h for h in hits if id(h) not in seen][: top_k - len(scoped)]
    return scoped[:top_k]
