"""사용자 역할 기반 응답 라우터.

교수님 요구 2번: 랜덤 사용자가 자신의 역할(부산대 학생, 행정원, 감독원
직원 등)을 입력하면 그 역할에 맞는 기관 문서 내용을 제공한다.

설계 (2026-08-20):
- 입력은 프리셋 선택 + 자유 텍스트 모두 허용. 자유 텍스트는 규칙 기반으로
  가장 가까운 프리셋에 매핑하고, 매핑 실패 시 일반 사용자 프로필로
  폴백한다. 프리셋 매핑이라 평가 재현이 가능하다.
- 검색 작용은 소프트 우선순위다: 역할의 우선 기관 문서를 후보 정렬에서
  앞으로 보내되 하드 필터는 걸지 않는다 — 역할 밖 질문(학생이 금감원
  규정을 물음)도 답할 수 있어야 한다. 강등이 아닌 승격이지만 원리는
  D50 폴백층 강등과 같다: 다양화 캡이 top-k를 뽑기 **이전**의 후보 풀
  순서를 조정한다.
- 프롬프트에는 매핑된 프리셋의 perspective 문장만 들어간다. 사용자
  자유 텍스트 원문은 절대 프롬프트에 넣지 않는다 — 역할 입력란이
  프롬프트 주입 통로가 되는 것을 차단한다 (교수님 요구 4번, 프롬프트
  오염 방지와 연결). SYSTEM_INSTRUCTION의 역할 변경 거부 지시와
  충돌하지 않도록, 역할은 "지시"가 아니라 "질문자 정보"로만 전달한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RoleProfile:
    """매핑 가능한 사용자 역할 하나.

    ``institutions``는 색인된 institution 값 기준의 우선 기관 목록이고,
    ``perspective``는 생성 프롬프트에 들어가는 신뢰된 서버 측 문장이다.
    자유 텍스트 매핑은 기관 표지(``institution_markers``)가 신분 표지
    (``keywords``)보다 우선한다 — "금감원에서 일하는 직원"이 신분 표지
    "직원"으로 다른 기관 프로필에 잡히는 오매핑을 막는다.
    """

    id: str
    label: str
    perspective: str
    institutions: tuple[str, ...] = ()
    institution_markers: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()


GENERAL_ROLE = RoleProfile(
    id="general",
    label="일반 사용자",
    perspective="질문자는 특정 소속을 밝히지 않은 일반 사용자입니다.",
)

# 프리셋 순서가 매핑 동률의 최종 우선순위다. 같은 기관 안에서는 더
# 흔한 역할(학생)이 먼저 온다 — 기관 표지만 있고 신분 표지가 없는
# 입력("부산대 소속입니다")은 그 기관의 첫 프로필로 간다.
PRESET_ROLES: tuple[RoleProfile, ...] = (
    RoleProfile(
        id="pnu-student",
        label="부산대학교 학생",
        perspective=(
            "질문자는 부산대학교 학생입니다. 학사·장학·등록·수강·국제·비자 "
            "등 질문 주제에서 학생에게 적용되는 기준을 우선해 설명하세요."
        ),
        institutions=("부산대학교",),
        institution_markers=("부산대", "부산대학교"),
        keywords=("학생", "대학원생", "학부생", "재학생", "휴학생", "신입생"),
    ),
    RoleProfile(
        id="pnu-staff",
        label="부산대학교 행정직원",
        perspective=(
            "질문자는 부산대학교 행정직원입니다. 업무 처리 절차·담당 "
            "부서·서류 요건 등 실무자 관점의 기준을 우선해 설명하세요."
        ),
        institutions=("부산대학교",),
        institution_markers=("부산대", "부산대학교"),
        keywords=("행정원", "행정직", "교직원", "조교", "행정실"),
    ),
    RoleProfile(
        id="pnu-researcher",
        label="부산대학교 연구자",
        perspective=(
            "질문자는 부산대학교 소속 연구자(교수·연구원)입니다. 연구 "
            "수행·연구비 사용에 적용되는 기준을 우선해 설명하세요."
        ),
        institutions=("부산대학교", "부산대학교 산학협력단"),
        institution_markers=("부산대", "부산대학교", "산학협력단", "산단"),
        keywords=("연구자", "교수", "연구원", "연구책임자", "박사후"),
    ),
    RoleProfile(
        id="fss-staff",
        label="금융감독원 직원",
        perspective=(
            "질문자는 금융감독원 직원입니다. 감독·검사 실무에 적용되는 "
            "기준을 우선해 설명하세요."
        ),
        institutions=("금융감독원",),
        institution_markers=("금융감독원", "금감원", "감독원"),
    ),
    RoleProfile(
        id="bok-staff",
        label="한국은행 직원",
        perspective=(
            "질문자는 한국은행 직원입니다. 통화·지급결제 등 한국은행 "
            "업무에 적용되는 기준을 우선해 설명하세요."
        ),
        institutions=("한국은행",),
        institution_markers=("한국은행", "한은"),
    ),
    RoleProfile(
        id="krx-staff",
        label="한국거래소 직원",
        perspective=(
            "질문자는 한국거래소 직원입니다. 상장·시장운영 실무에 "
            "적용되는 기준을 우선해 설명하세요."
        ),
        institutions=("한국거래소",),
        institution_markers=("한국거래소", "거래소"),
    ),
    RoleProfile(
        id="ksd-staff",
        label="한국예탁결제원 직원",
        perspective=(
            "질문자는 한국예탁결제원 직원입니다. 예탁·결제 실무에 "
            "적용되는 기준을 우선해 설명하세요."
        ),
        institutions=("한국예탁결제원",),
        institution_markers=("예탁결제원", "예탁원"),
    ),
    RoleProfile(
        id="kisa-staff",
        label="한국인터넷진흥원 직원",
        perspective=(
            "질문자는 한국인터넷진흥원(KISA) 직원입니다. 정보보호 "
            "실무에 적용되는 기준을 우선해 설명하세요."
        ),
        institutions=("한국인터넷진흥원(KISA)",),
        institution_markers=("인터넷진흥원", "KISA", "kisa"),
    ),
    RoleProfile(
        id="kiost-staff",
        label="한국해양과학기술원 직원",
        perspective=(
            "질문자는 한국해양과학기술원 직원입니다. 해양 연구·행정 "
            "실무에 적용되는 기준을 우선해 설명하세요."
        ),
        institutions=("한국해양과학기술원",),
        institution_markers=("해양과학기술원", "해양과기원", "KIOST", "kiost"),
    ),
    GENERAL_ROLE,
)

_BY_ID = {profile.id: profile for profile in PRESET_ROLES}
_BY_LABEL = {profile.label: profile for profile in PRESET_ROLES}


def resolve(text: str | None) -> RoleProfile:
    """역할 입력을 프리셋 프로필로 매핑한다.

    프리셋 id·라벨 정확 일치 → 키워드 규칙 순서로 판정하고, 아무것도
    맞지 않으면 일반 사용자 프로필을 반환한다. 반환값은 항상 프리셋
    중 하나이므로 프롬프트에는 신뢰된 문장만 들어간다.
    """

    value = str(text or "").strip()
    if not value:
        return GENERAL_ROLE
    if value in _BY_ID:
        return _BY_ID[value]
    if value in _BY_LABEL:
        return _BY_LABEL[value]
    # 1순위: 기관 표지 일치. 같은 기관이 여러 프로필을 가지면(부산대)
    # 신분 표지까지 일치하는 프로필이 이기고, 없으면 프리셋 순서상 첫
    # 프로필로 간다.
    institution_matched = [
        profile
        for profile in PRESET_ROLES
        if any(marker in value for marker in profile.institution_markers)
    ]
    for profile in institution_matched:
        if any(keyword in value for keyword in profile.keywords):
            return profile
    if institution_matched:
        return institution_matched[0]
    # 2순위: 신분 표지만 일치 ("저는 대학원생이에요").
    for profile in PRESET_ROLES:
        if any(keyword in value for keyword in profile.keywords):
            return profile
    return GENERAL_ROLE


def prioritize_hits(
    rows: list[dict],
    profile: RoleProfile,
) -> list[dict]:
    """우선 기관 문서를 후보 풀 앞으로 보내는 안정 분할.

    top-k 선정(다양화·중복 제거) **이전**의 넓은 후보 풀에 적용해야
    한다. 우선 기관 안에서도, 나머지 안에서도 기존 검색 순위는 그대로
    유지되므로 역할 밖 질문의 정답 후보가 사라지지 않는다.
    """

    if not profile.institutions:
        return rows
    preferred = set(profile.institutions)
    return (
        [r for r in rows if (r.get("institution") or "") in preferred]
        + [r for r in rows if (r.get("institution") or "") not in preferred]
    )


def public_role(profile: RoleProfile, requested: str | None) -> dict:
    """API 응답에 싣는 역할 진단 정보."""

    return {
        "requested": str(requested or "").strip() or None,
        "id": profile.id,
        "label": profile.label,
        "institutions": list(profile.institutions),
    }
