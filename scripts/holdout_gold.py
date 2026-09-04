#!/usr/bin/env python3
"""Schema validation and profile-independent evidence matching for holdout v2.

The final holdout is intentionally stricter than the legacy DEV cases.  Gold
evidence identifies source documents rather than parser-specific chunk IDs and
keeps atomic claims separate.  This module contains no experiment-side writes;
it only loads, validates, and matches immutable JSON/JSONL inputs.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence
from urllib.parse import urlsplit, urlunsplit


CORE_SPLIT = "holdout-core"
CHALLENGE_SPLIT = "holdout-challenge"
EXPECTED_CASE_COUNT = 36
EXPECTED_CORE_COUNT = 27
EXPECTED_CHALLENGE_COUNT = 9
MIN_ROLE_CASES = 6
MIN_REQUIRED_CLAIMS = 2
MAX_REQUIRED_CLAIMS = 5
FUZZY_MATCH_THRESHOLD = 0.85
MIN_SEMANTIC_ANCHORS = 2
MAX_INTENTIONAL_TYPO_CASES = 3

CORE_CATEGORIES = (
    "academic",
    "admissions",
    "core",
    "employment",
    "graduation",
    "international",
    "registration",
    "scholarship",
    "student_support",
)
CORE_DIFFICULTY_TYPES = (
    "single_fact",
    "multi_evidence",
    "structure_sensitive",
)
ALLOWED_ROLES = ("pnu-student", "pnu-staff", "pnu-researcher")

# Canonical challenge labels used in the frozen holdout.  A few transparent
# aliases are accepted so the validator can read an early authoring manifest,
# but reports and hashes should use the canonical values below.
CHALLENGE_TYPES = (
    "unanswerable",
    "scope_version_ambiguity",
    "prompt_injection",
)
CHALLENGE_TYPE_ALIASES = {
    "unanswerable": "unanswerable",
    "insufficient_evidence": "unanswerable",
    "corpus_unanswerable": "unanswerable",
    "scope_version_ambiguity": "scope_version_ambiguity",
    "scope_or_version_conflict": "scope_version_ambiguity",
    "scope_conflict": "scope_version_ambiguity",
    "ambiguous_scope": "scope_version_ambiguity",
    "prompt_injection": "prompt_injection",
    "instruction_injection": "prompt_injection",
    "role_injection": "prompt_injection",
}
EXPECTED_BEHAVIOR_BY_CHALLENGE = {
    "unanswerable": "abstain",
    "scope_version_ambiguity": "clarify_or_abstain",
    "prompt_injection": "resist_injection_and_answer",
}
CANONICAL_EXPECTED_BEHAVIORS = frozenset(
    {"answer", *EXPECTED_BEHAVIOR_BY_CHALLENGE.values()}
)

ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,127}$")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
YEAR_RE = re.compile(
    r"(?<!\d)(?:19|20)\d{2}\s*(?:학\s*년\s*도|학년도|년도|년)?(?!\d)"
)
FILE_EXTENSION_RE = re.compile(r"\.(?:pdf|hwp|hwpx|docx?)$", re.IGNORECASE)
FILE_SIZE_RE = re.compile(
    r"\(\s*\d+(?:[.,]\d+)?\s*(?:bytes?|kb|mb|gb)\s*\)\s*$",
    re.IGNORECASE,
)
LAYOUT_SEPARATORS = frozenset("|¦│┃┆┊\t\r\n~∼～–—")
KOREAN_GRAMMATICAL_SUFFIXES = frozenset(
    {
        "가",
        "같이",
        "고",
        "과",
        "까지",
        "는",
        "도",
        "라고",
        "라는",
        "라도",
        "를",
        "마다",
        "마저",
        "만",
        "며",
        "보다",
        "부터",
        "이",
        "이고",
        "이나",
        "이라고",
        "이라는",
        "이라도",
        "이며",
        "이었습니다",
        "인",
        "입니다",
        "은",
        "을",
        "의",
        "이다",
        "에",
        "에게",
        "에게서",
        "에서",
        "였습니다",
        "와",
        "으로",
        "조차",
        "처럼",
        "하다",
        "합니다",
        "해요",
        "한테",
    }
)
POLARITY_PATTERN_GROUPS = (
    (
        re.compile(
            r"불가능|가능\s*하지\s*않|할\s*수\s*없|"
            r"허용\s*되지\s*않|금지|불허"
        ),
        re.compile(r"(?<!불)가능|할\s*수\s*있|허용"),
    ),
    (
        re.compile(
            r"불필요|필요\s*하지\s*않|필요\s*없|"
            r"하지\s*않아도|안\s*해도"
        ),
        re.compile(r"필수|필요|해야|하여야"),
    ),
)


class HoldoutValidationError(ValueError):
    """Aggregate all preflight failures so authors can fix them in one pass."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(str(error) for error in errors)
        super().__init__("holdout validation failed:\n- " + "\n- ".join(self.errors))


def _nonempty_text(value: Any) -> str:
    return str(value or "").strip()


def _fold_identifier(value: Any) -> str:
    return unicodedata.normalize("NFKC", _nonempty_text(value)).casefold()


def normalize_evidence_text(value: Any) -> str:
    """Normalize text for the protocol's layout-insensitive exact match.

    NFKC and case folding are followed by punctuation/layout folding.  Punctuation
    is turned into a boundary rather than deleted so unrelated neighboring tokens
    cannot be concatenated into an accidental match.
    """

    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    output: list[str] = []
    for character in normalized:
        if (
            character.isspace()
            or character in LAYOUT_SEPARATORS
            or unicodedata.category(character).startswith("P")
        ):
            output.append(" ")
        else:
            output.append(character)
    return re.sub(r"\s+", " ", "".join(output)).strip()


def normalize_title_without_year(value: Any) -> str:
    """Return the reproducible title fingerprint used for leak detection."""

    title = unicodedata.normalize("NFKC", str(value or "")).strip()
    title = FILE_SIZE_RE.sub("", title).strip()
    title = FILE_EXTENSION_RE.sub("", title)
    title = YEAR_RE.sub(" ", title)
    return normalize_evidence_text(title)


def canonical_source_url(value: Any) -> str:
    raw = _nonempty_text(value)
    if not raw:
        return ""
    parts = urlsplit(raw)
    if parts.scheme.casefold() not in {"http", "https"} or not parts.netloc:
        return ""
    path = parts.path.rstrip("/") or "/"
    return urlunsplit(
        (parts.scheme.casefold(), parts.netloc.casefold(), path, parts.query, "")
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        records.append(record)
    return records


def load_json_or_jsonl(path: Path) -> Any:
    """Load a DEV manifest or input in either JSON or JSONL form."""

    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        return []
    if stripped.startswith("["):
        return json.loads(text)
    if stripped.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return load_jsonl(path)


def collect_source_identities(payload: Any) -> dict[str, set[str]]:
    """Collect leak-resistant source identities from a DEV manifest."""

    families: set[str] = set()
    source_sha256s: set[str] = set()
    normalized_titles: set[str] = set()
    source_urls: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return

        plural = value.get("source_document_family_ids")
        if isinstance(plural, list):
            families.update(
                _nonempty_text(item) for item in plural if _nonempty_text(item)
            )
        for key in ("source_document_family_id", "document_family_id"):
            item = _nonempty_text(value.get(key))
            if item:
                families.add(item)
        plural_sha = value.get("source_sha256s")
        if isinstance(plural_sha, list):
            source_sha256s.update(
                _nonempty_text(item).casefold()
                for item in plural_sha
                if SHA256_RE.fullmatch(_nonempty_text(item))
            )
        source_sha = _nonempty_text(value.get("source_sha256"))
        if SHA256_RE.fullmatch(source_sha):
            source_sha256s.add(source_sha.casefold())
        plural_titles = value.get("normalized_source_titles")
        if isinstance(plural_titles, list):
            normalized_titles.update(
                normalize_title_without_year(item)
                for item in plural_titles
                if normalize_title_without_year(item)
            )
        normalized_title = _nonempty_text(
            value.get("normalized_title_without_year")
        )
        if normalized_title:
            normalized_titles.add(normalize_title_without_year(normalized_title))
        source_title = _nonempty_text(value.get("source_title"))
        if source_title:
            normalized_titles.add(normalize_title_without_year(source_title))
        source_url = canonical_source_url(value.get("source_url"))
        if source_url:
            source_urls.add(source_url)
        document_like = any(
            value.get(key) not in (None, "")
            for key in ("document_id", "doc_id", "source_url", "source_sha256")
        )
        if document_like:
            generic_family = _nonempty_text(value.get("family_id"))
            if generic_family:
                families.add(generic_family)
        for nested in value.values():
            if isinstance(nested, (dict, list)):
                visit(nested)

    visit(payload)
    return {
        "family_ids": families,
        "source_sha256s": source_sha256s,
        "normalized_titles": normalized_titles,
        "source_urls": source_urls,
    }


def collect_source_family_ids(payload: Any) -> set[str]:
    """Backward-compatible family-only view of ``collect_source_identities``."""

    return collect_source_identities(payload)["family_ids"]


def _string_list(
    record: Mapping[str, Any],
    key: str,
    path: str,
    errors: list[str],
    *,
    required: bool,
) -> list[str]:
    value = record.get(key)
    if not isinstance(value, list):
        errors.append(f"{path}.{key}: must be a list")
        return []
    items: list[str] = []
    for index, item in enumerate(value):
        text = _nonempty_text(item)
        if not text:
            errors.append(f"{path}.{key}[{index}]: must be a non-empty string")
            continue
        items.append(text)
    if required and not items:
        errors.append(f"{path}.{key}: must not be empty")
    if len(items) != len(set(items)):
        errors.append(f"{path}.{key}: contains duplicates")
    return items


def _canonical_challenge_type(value: Any) -> Optional[str]:
    normalized = _fold_identifier(value)
    return normalized if normalized in CHALLENGE_TYPES else None


def _canonical_expected_behavior(value: Any) -> Optional[str]:
    normalized = _fold_identifier(value)
    return normalized if normalized in CANONICAL_EXPECTED_BEHAVIORS else None


def _validate_challenge_oracle(
    case: Mapping[str, Any],
    path: str,
    challenge_type: Optional[str],
    errors: list[str],
) -> None:
    oracle = case.get("challenge_oracle")
    if not isinstance(oracle, dict):
        errors.append(f"{path}.challenge_oracle: must be an object")
        return
    for key in ("must_do", "must_not_do"):
        values = oracle.get(key)
        if (
            not isinstance(values, list)
            or not values
            or not all(_nonempty_text(value) for value in values)
        ):
            errors.append(
                f"{path}.challenge_oracle.{key}: must contain non-empty strings"
            )
    if challenge_type == "prompt_injection":
        if _nonempty_text(oracle.get("injection_surface")) not in {
            "question",
            "role",
        }:
            errors.append(
                f"{path}.challenge_oracle.injection_surface: must be "
                "'question' or 'role'"
            )
        if not _nonempty_text(oracle.get("injection_payload")):
            errors.append(
                f"{path}.challenge_oracle.injection_payload: must be non-empty"
            )


def _table_spec_from(claim: Mapping[str, Any], option: Mapping[str, Any]) -> Any:
    for container in (option, claim):
        for key in ("table_evidence", "table_spec"):
            if key in container:
                return container.get(key)
    return None


def _validate_table_spec(spec: Any, path: str, errors: list[str]) -> None:
    if not isinstance(spec, dict):
        errors.append(f"{path}: must be an object")
        return
    headers = spec.get("headers")
    if (
        not isinstance(headers, list)
        or not headers
        or not all(_nonempty_text(item) for item in headers)
    ):
        errors.append(f"{path}.headers: must contain non-empty strings")
    relations = spec.get("row_relations", spec.get("header_value_pairs"))
    if not isinstance(relations, list) or not relations:
        errors.append(f"{path}.row_relations: must be a non-empty list")
        return
    for index, relation in enumerate(relations):
        relation_path = f"{path}.row_relations[{index}]"
        if not isinstance(relation, dict):
            errors.append(f"{relation_path}: must be an object")
            continue
        for key in ("header", "value", "row_anchor"):
            if not _nonempty_text(relation.get(key)):
                errors.append(f"{relation_path}.{key}: must be a non-empty string")


def _validate_claims(
    case: Mapping[str, Any],
    case_path: str,
    *,
    core_answerable: bool,
    source_families: set[str],
    source_sha256s: set[str],
    source_titles: set[str],
    errors: list[str],
) -> tuple[set[str], set[str], set[str], set[str]]:
    seen_claim_ids: set[str] = set()
    option_families: set[str] = set()
    option_sha256s: set[str] = set()
    option_titles: set[str] = set()
    option_urls: set[str] = set()

    required_claims = case.get("required_claims")
    if not isinstance(required_claims, list):
        errors.append(f"{case_path}.required_claims: must be a list")
        required_claims = []
    if core_answerable and not MIN_REQUIRED_CLAIMS <= len(required_claims) <= MAX_REQUIRED_CLAIMS:
        errors.append(
            f"{case_path}.required_claims: answerable Core must contain "
            f"{MIN_REQUIRED_CLAIMS}-{MAX_REQUIRED_CLAIMS} claims"
        )
    if case.get("answerable") is False and required_claims:
        errors.append(f"{case_path}.required_claims: unanswerable case must be empty")

    for claim_index, claim in enumerate(required_claims):
        claim_path = f"{case_path}.required_claims[{claim_index}]"
        if not isinstance(claim, dict):
            errors.append(f"{claim_path}: must be an object")
            continue
        claim_id = _nonempty_text(claim.get("claim_id"))
        if not claim_id:
            errors.append(f"{claim_path}.claim_id: must be a non-empty string")
        elif claim_id in seen_claim_ids:
            errors.append(f"{claim_path}.claim_id: duplicate {claim_id!r}")
        else:
            seen_claim_ids.add(claim_id)
        if not _nonempty_text(claim.get("description")):
            errors.append(f"{claim_path}.description: must be a non-empty string")
        critical_values = _string_list(
            claim, "critical_values", claim_path, errors, required=True
        )
        anchors = claim.get("semantic_anchors", claim.get("meaning_anchors"))
        if anchors is not None:
            normalized_anchors = (
                {
                    normalize_evidence_text(item)
                    for item in anchors
                    if normalize_evidence_text(item)
                }
                if isinstance(anchors, list)
                else set()
            )
            if len(normalized_anchors) < MIN_SEMANTIC_ANCHORS:
                errors.append(
                    f"{claim_path}.semantic_anchors: when present, must contain "
                    f"at least {MIN_SEMANTIC_ANCHORS} non-empty anchors"
                )

        options = claim.get("evidence_options")
        if not isinstance(options, list) or not options:
            errors.append(f"{claim_path}.evidence_options: must be a non-empty list")
            continue
        option_keys: set[tuple[str, str, str]] = set()
        for option_index, option in enumerate(options):
            option_path = f"{claim_path}.evidence_options[{option_index}]"
            if not isinstance(option, dict):
                errors.append(f"{option_path}: must be an object")
                continue
            required_fields = (
                "document_id",
                "source_document_family_id",
                "source_sha256",
                "source_path",
                "source_url",
                "source_title",
                "normalized_title_without_year",
                "quote",
            )
            values = {key: _nonempty_text(option.get(key)) for key in required_fields}
            for key, value in values.items():
                if not value:
                    errors.append(f"{option_path}.{key}: must be a non-empty string")

            family = values["source_document_family_id"]
            source_sha = values["source_sha256"]
            normalized_title = values["normalized_title_without_year"]
            source_url = canonical_source_url(values["source_url"])
            if source_sha and not SHA256_RE.fullmatch(source_sha):
                errors.append(f"{option_path}.source_sha256: must be 64 hexadecimal characters")
            if values["source_url"] and not source_url:
                errors.append(f"{option_path}.source_url: must be an absolute HTTP(S) URL")
            if normalized_title:
                expected_title = normalize_title_without_year(values["source_title"])
                if normalized_title != expected_title:
                    errors.append(
                        f"{option_path}.normalized_title_without_year: expected "
                        f"{expected_title!r}"
                    )
            if family and family not in source_families:
                errors.append(
                    f"{option_path}.source_document_family_id: not declared in "
                    "case.source_document_family_ids"
                )
            if source_sha and source_sha not in source_sha256s:
                errors.append(
                    f"{option_path}.source_sha256: not declared in case.source_sha256s"
                )
            if normalized_title and normalized_title not in source_titles:
                errors.append(
                    f"{option_path}.normalized_title_without_year: not declared in "
                    "case.normalized_source_titles"
                )

            if family:
                option_families.add(family)
            if source_sha:
                option_sha256s.add(source_sha)
            if normalized_title:
                option_titles.add(normalized_title)
            if source_url:
                option_urls.add(source_url)
            option_key = (
                _fold_identifier(values["document_id"]),
                source_sha.casefold(),
                normalize_evidence_text(values["quote"]),
            )
            if option_key in option_keys:
                errors.append(f"{option_path}: duplicate evidence option")
            option_keys.add(option_key)

            table_spec = _table_spec_from(claim, option)
            is_table = _fold_identifier(claim.get("evidence_type")) == "table"
            if is_table or table_spec is not None:
                _validate_table_spec(table_spec, f"{option_path}.table_evidence", errors)
                if not critical_values:
                    errors.append(f"{claim_path}: table claim requires critical_values")

    for list_name in ("optional_claims", "forbidden_claims"):
        claims = case.get(list_name)
        if not isinstance(claims, list):
            errors.append(f"{case_path}.{list_name}: must be a list")
            continue
        for claim_index, claim in enumerate(claims):
            claim_path = f"{case_path}.{list_name}[{claim_index}]"
            if not isinstance(claim, dict):
                errors.append(f"{claim_path}: must be an object")
                continue
            claim_id = _nonempty_text(claim.get("claim_id"))
            if not claim_id:
                errors.append(f"{claim_path}.claim_id: must be a non-empty string")
            elif claim_id in seen_claim_ids:
                errors.append(f"{claim_path}.claim_id: duplicate {claim_id!r}")
            else:
                seen_claim_ids.add(claim_id)
            if not _nonempty_text(claim.get("description")):
                errors.append(f"{claim_path}.description: must be a non-empty string")

    return option_families, option_sha256s, option_titles, option_urls


def collect_holdout_validation_errors(
    records: Sequence[Mapping[str, Any]],
    *,
    dev_family_ids: Iterable[str] = (),
    dev_source_sha256s: Iterable[str] = (),
    dev_normalized_titles: Iterable[str] = (),
    dev_source_urls: Iterable[str] = (),
) -> list[str]:
    """Return all schema, composition, and leakage errors for a 36-case holdout."""

    errors: list[str] = []
    if len(records) != EXPECTED_CASE_COUNT:
        errors.append(
            f"dataset: expected {EXPECTED_CASE_COUNT} cases, found {len(records)}"
        )

    seen_ids: set[str] = set()
    core_family_owners: dict[str, str] = {}
    core_query_owners: dict[str, str] = {}
    core_source_family_owners: dict[str, str] = {}
    core_sha_owners: dict[str, str] = {}
    core_title_owners: dict[str, str] = {}
    core_url_owners: dict[str, str] = {}
    category_counts: Counter[str] = Counter()
    challenge_counts: Counter[str] = Counter()
    category_difficulty_counts: dict[str, Counter[str]] = {
        category: Counter() for category in CORE_CATEGORIES
    }
    split_counts: Counter[str] = Counter()
    role_case_count = 0
    intentional_typo_count = 0

    for index, raw_case in enumerate(records):
        case_path = f"cases[{index}]"
        if not isinstance(raw_case, Mapping):
            errors.append(f"{case_path}: must be an object")
            continue
        case = raw_case
        case_id = _nonempty_text(case.get("id"))
        if not ID_RE.fullmatch(case_id):
            errors.append(f"{case_path}.id: invalid ID {case_id!r}")
        if case_id in seen_ids:
            errors.append(f"{case_path}.id: duplicate {case_id!r}")
        seen_ids.add(case_id)
        label = case_id or case_path

        split = _nonempty_text(case.get("split"))
        if split not in {CORE_SPLIT, CHALLENGE_SPLIT}:
            errors.append(f"{label}.split: must be {CORE_SPLIT!r} or {CHALLENGE_SPLIT!r}")
        else:
            split_counts[split] += 1
        query = _nonempty_text(case.get("query"))
        if not query:
            errors.append(f"{label}.query: must be a non-empty string")
        family_id = _nonempty_text(case.get("family_id"))
        if not family_id:
            errors.append(f"{label}.family_id: must be a non-empty string")

        answerable = case.get("answerable")
        if not isinstance(answerable, bool):
            errors.append(f"{label}.answerable: must be a JSON boolean")
        raw_behavior = _fold_identifier(case.get("expected_behavior"))
        behavior = _canonical_expected_behavior(raw_behavior)
        if behavior is None:
            errors.append(f"{label}.expected_behavior: unsupported value")
        elif isinstance(answerable, bool):
            answer_behaviors = {"answer", "resist_injection_and_answer"}
            if answerable and behavior not in answer_behaviors:
                errors.append(
                    f"{label}: answerable=true requires an answer-producing "
                    "expected_behavior"
                )
            if not answerable and behavior in answer_behaviors:
                errors.append(
                    f"{label}: answerable=false cannot use an answer-producing "
                    "expected_behavior"
                )

        role = case.get("role")
        if role is not None and _nonempty_text(role) not in ALLOWED_ROLES:
            errors.append(f"{label}.role: unsupported role {role!r}")

        source_families = set(
            _string_list(
                case,
                "source_document_family_ids",
                label,
                errors,
                required=split == CORE_SPLIT,
            )
        )
        source_sha256s = set(
            _string_list(
                case,
                "source_sha256s",
                label,
                errors,
                required=split == CORE_SPLIT,
            )
        )
        for source_sha in source_sha256s:
            if not SHA256_RE.fullmatch(source_sha):
                errors.append(f"{label}.source_sha256s: invalid SHA-256 {source_sha!r}")
        source_titles = set(
            _string_list(
                case,
                "normalized_source_titles",
                label,
                errors,
                required=split == CORE_SPLIT,
            )
        )
        for title in source_titles:
            if title != normalize_title_without_year(title):
                errors.append(
                    f"{label}.normalized_source_titles: title is not "
                    f"normalized/year-free: {title!r}"
                )

        is_core_answerable = split == CORE_SPLIT and answerable is True
        option_families, option_sha256s, option_titles, option_urls = _validate_claims(
            case,
            label,
            core_answerable=is_core_answerable,
            source_families=source_families,
            source_sha256s=source_sha256s,
            source_titles=source_titles,
            errors=errors,
        )

        if split == CORE_SPLIT:
            category = _nonempty_text(case.get("category"))
            if category not in CORE_CATEGORIES:
                errors.append(f"{label}.category: unsupported Core category {category!r}")
            else:
                category_counts[category] += 1
            difficulty_type = _fold_identifier(case.get("difficulty_type"))
            if difficulty_type not in CORE_DIFFICULTY_TYPES:
                errors.append(
                    f"{label}.difficulty_type: must be one of "
                    f"{CORE_DIFFICULTY_TYPES}"
                )
            elif category in category_difficulty_counts:
                category_difficulty_counts[category][difficulty_type] += 1
            intentional_typo = case.get("intentional_typo")
            if not isinstance(intentional_typo, bool):
                errors.append(f"{label}.intentional_typo: must be a JSON boolean")
            elif intentional_typo:
                intentional_typo_count += 1
            if answerable is not True:
                errors.append(f"{label}: every Core case must be answerable")
            if behavior not in (None, "answer"):
                errors.append(f"{label}: Core expected_behavior must be 'answer'")
            if role is not None and _nonempty_text(role) in ALLOWED_ROLES:
                role_case_count += 1

            normalized_query = normalize_evidence_text(query)
            if normalized_query:
                previous = core_query_owners.get(normalized_query)
                if previous:
                    errors.append(f"{label}.query: duplicates Core query from {previous}")
                core_query_owners[normalized_query] = label
            if family_id:
                previous = core_family_owners.get(family_id)
                if previous:
                    errors.append(f"{label}.family_id: duplicates Core family from {previous}")
                core_family_owners[family_id] = label

            if source_families != option_families:
                errors.append(
                    f"{label}: source_document_family_ids must exactly match evidence options"
                )
            if source_sha256s != option_sha256s:
                errors.append(f"{label}: source_sha256s must exactly match evidence options")
            if source_titles != option_titles:
                errors.append(
                    f"{label}: normalized_source_titles must exactly match evidence options"
                )

            uniqueness_groups = (
                (source_families, core_source_family_owners, "source document family"),
                (option_sha256s, core_sha_owners, "source SHA-256"),
                (option_titles, core_title_owners, "normalized source title"),
                (option_urls, core_url_owners, "source URL"),
            )
            for values, owners, field in uniqueness_groups:
                for value in values:
                    previous = owners.get(value)
                    if previous:
                        errors.append(
                            f"{label}: {field} {value!r} overlaps Core case {previous}"
                        )
                    owners[value] = label

        elif split == CHALLENGE_SPLIT:
            raw_challenge_type = _fold_identifier(case.get("challenge_type"))
            challenge_type = _canonical_challenge_type(raw_challenge_type)
            if challenge_type is None:
                alias = CHALLENGE_TYPE_ALIASES.get(raw_challenge_type)
                if alias:
                    errors.append(
                        f"{label}.challenge_type: use canonical value {alias!r}"
                    )
                else:
                    errors.append(f"{label}.challenge_type: unsupported value")
            else:
                challenge_counts[challenge_type] += 1
                expected = EXPECTED_BEHAVIOR_BY_CHALLENGE[challenge_type]
                if behavior not in (None, expected):
                    errors.append(
                        f"{label}: challenge_type={challenge_type!r} requires "
                        f"expected_behavior={expected!r}"
                    )
            if raw_behavior != _nonempty_text(case.get("expected_behavior")):
                errors.append(
                    f"{label}.expected_behavior: use canonical lowercase value"
                )
            if challenge_type == "prompt_injection":
                if answerable is not True:
                    errors.append(
                        f"{label}: prompt_injection Challenge must set "
                        "answerable=true"
                    )
                required = case.get("required_claims")
                if not isinstance(required, list) or not 1 <= len(required) <= 3:
                    errors.append(
                        f"{label}.required_claims: prompt_injection must contain "
                        "1-3 safe-answer claims"
                    )
            elif answerable is not False:
                errors.append(
                    f"{label}: non-injection Challenge cases must set "
                    "answerable=false"
                )
            forbidden = case.get("forbidden_claims")
            if not isinstance(forbidden, list) or not forbidden:
                errors.append(
                    f"{label}.forbidden_claims: Challenge oracle requires at "
                    "least one forbidden behavior/claim"
                )
            _validate_challenge_oracle(
                case, label, challenge_type, errors
            )

    if split_counts[CORE_SPLIT] != EXPECTED_CORE_COUNT:
        errors.append(
            f"dataset: expected {EXPECTED_CORE_COUNT} Core cases, "
            f"found {split_counts[CORE_SPLIT]}"
        )
    if split_counts[CHALLENGE_SPLIT] != EXPECTED_CHALLENGE_COUNT:
        errors.append(
            f"dataset: expected {EXPECTED_CHALLENGE_COUNT} Challenge cases, "
            f"found {split_counts[CHALLENGE_SPLIT]}"
        )
    for category in CORE_CATEGORIES:
        if category_counts[category] != 3:
            errors.append(
                f"dataset: Core category {category!r} must have 3 cases; "
                f"found {category_counts[category]}"
            )
        for difficulty_type in CORE_DIFFICULTY_TYPES:
            if category_difficulty_counts[category][difficulty_type] != 1:
                errors.append(
                    f"dataset: Core category {category!r} difficulty_type "
                    f"{difficulty_type!r} must occur once; found "
                    f"{category_difficulty_counts[category][difficulty_type]}"
                )
    for challenge_type in CHALLENGE_TYPES:
        if challenge_counts[challenge_type] != 3:
            errors.append(
                f"dataset: challenge type {challenge_type!r} must have 3 cases; "
                f"found {challenge_counts[challenge_type]}"
            )
    if role_case_count < MIN_ROLE_CASES:
        errors.append(
            f"dataset: at least {MIN_ROLE_CASES} Core cases must declare a role; "
            f"found {role_case_count}"
        )
    if intentional_typo_count > MAX_INTENTIONAL_TYPO_CASES:
        errors.append(
            f"dataset: intentional_typo may appear in at most "
            f"{MAX_INTENTIONAL_TYPO_CASES} Core cases; found "
            f"{intentional_typo_count}"
        )

    dev_families = {_nonempty_text(value) for value in dev_family_ids if _nonempty_text(value)}
    leaked = sorted(dev_families.intersection(core_source_family_owners))
    for family in leaked:
        errors.append(
            f"dataset: DEV source document family leak {family!r} in "
            f"{core_source_family_owners[family]}"
        )
    dev_shas = {
        _nonempty_text(value).casefold()
        for value in dev_source_sha256s
        if _nonempty_text(value)
    }
    for source_sha in sorted(dev_shas.intersection(core_sha_owners)):
        errors.append(
            f"dataset: DEV source SHA-256 leak {source_sha!r} in "
            f"{core_sha_owners[source_sha]}"
        )
    dev_titles = {
        normalize_title_without_year(value)
        for value in dev_normalized_titles
        if normalize_title_without_year(value)
    }
    for title in sorted(dev_titles.intersection(core_title_owners)):
        errors.append(
            f"dataset: DEV normalized source title leak {title!r} in "
            f"{core_title_owners[title]}"
        )
    dev_urls = {
        canonical_source_url(value)
        for value in dev_source_urls
        if canonical_source_url(value)
    }
    for source_url in sorted(dev_urls.intersection(core_url_owners)):
        errors.append(
            f"dataset: DEV source URL leak {source_url!r} in "
            f"{core_url_owners[source_url]}"
        )
    return errors


def validate_holdout_records(
    records: Sequence[Mapping[str, Any]],
    *,
    dev_family_ids: Iterable[str] = (),
    dev_source_sha256s: Iterable[str] = (),
    dev_normalized_titles: Iterable[str] = (),
    dev_source_urls: Iterable[str] = (),
) -> None:
    errors = collect_holdout_validation_errors(
        records,
        dev_family_ids=dev_family_ids,
        dev_source_sha256s=dev_source_sha256s,
        dev_normalized_titles=dev_normalized_titles,
        dev_source_urls=dev_source_urls,
    )
    if errors:
        raise HoldoutValidationError(errors)


def validate_holdout_file(
    path: Path,
    *,
    dev_family_ids: Iterable[str] = (),
    dev_source_sha256s: Iterable[str] = (),
    dev_normalized_titles: Iterable[str] = (),
    dev_source_urls: Iterable[str] = (),
) -> list[dict[str, Any]]:
    records = load_jsonl(path)
    validate_holdout_records(
        records,
        dev_family_ids=dev_family_ids,
        dev_source_sha256s=dev_source_sha256s,
        dev_normalized_titles=dev_normalized_titles,
        dev_source_urls=dev_source_urls,
    )
    return records


def _candidate_field(candidate: Mapping[str, Any], *keys: str) -> Any:
    metadata = candidate.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    for key in keys:
        value = candidate.get(key)
        if value not in (None, ""):
            return value
        value = metadata.get(key)
        if value not in (None, ""):
            return value
    return None


def _candidate_text(candidate: Any) -> str:
    if isinstance(candidate, str):
        return candidate
    if not isinstance(candidate, Mapping):
        return ""
    return str(_candidate_field(candidate, "text", "preview", "excerpt", "content") or "")


def _claim_values(claim: Optional[Mapping[str, Any]]) -> list[str]:
    if not isinstance(claim, Mapping):
        return []
    values = claim.get("critical_values")
    if not isinstance(values, list):
        return []
    return [_nonempty_text(value) for value in values if _nonempty_text(value)]


def _semantic_anchors(
    claim: Optional[Mapping[str, Any]], option: Mapping[str, Any]
) -> list[str]:
    for container in (option, claim or {}):
        for key in ("semantic_anchors", "meaning_anchors"):
            values = container.get(key)
            if isinstance(values, list):
                return [_nonempty_text(value) for value in values if _nonempty_text(value)]
    return []


def _contains_normalized(text: str, value: str) -> bool:
    normalized = normalize_evidence_text(value)
    if not normalized:
        return False
    cursor = 0
    while True:
        position = text.find(normalized, cursor)
        if position < 0:
            return False
        end = position + len(normalized)
        before = text[position - 1] if position else ""
        after = text[end] if end < len(text) else ""
        starts_with_digit = normalized[0].isdigit()
        ends_with_digit = normalized[-1].isdigit()
        starts_with_hangul = "가" <= normalized[0] <= "힣"
        ends_with_hangul = "가" <= normalized[-1] <= "힣"

        start_ok = not (
            (starts_with_digit and before.isdigit())
            or (starts_with_hangul and "가" <= before <= "힣")
        )
        end_ok = True
        if ends_with_digit and after.isdigit():
            end_ok = False
        elif ends_with_hangul and "가" <= after <= "힣":
            suffix_end = end
            while suffix_end < len(text) and "가" <= text[suffix_end] <= "힣":
                suffix_end += 1
            suffix = text[end:suffix_end]
            end_ok = _is_korean_grammatical_suffix_sequence(suffix)
        if start_ok and end_ok:
            return True
        cursor = position + 1


def _is_korean_grammatical_suffix_sequence(value: str) -> bool:
    """Accept stacked known particles/endings without accepting arbitrary tails.

    Values commonly occur as ``3일까지입니다`` or ``학교에서는``.  Exact
    segmentation over the frozen suffix vocabulary accepts those forms while
    continuing to reject lexical continuations such as ``학부`` in ``학부모``.
    """

    if not value:
        return True
    reachable = {0}
    for start in range(len(value)):
        if start not in reachable:
            continue
        for suffix in KOREAN_GRAMMATICAL_SUFFIXES:
            if value.startswith(suffix, start):
                reachable.add(start + len(suffix))
    return len(value) in reachable


def _polarity_conflicts(reference: str, candidate: str) -> bool:
    """Reject high-similarity passages that reverse a clear relation polarity."""

    for negative_pattern, positive_pattern in POLARITY_PATTERN_GROUPS:
        reference_negative = bool(negative_pattern.search(reference))
        candidate_negative = bool(negative_pattern.search(candidate))
        reference_without_negative = negative_pattern.sub(" ", reference)
        candidate_without_negative = negative_pattern.sub(" ", candidate)
        reference_positive = bool(positive_pattern.search(reference_without_negative))
        candidate_positive = bool(positive_pattern.search(candidate_without_negative))
        if (
            reference_positive
            and not reference_negative
            and candidate_negative
            and not candidate_positive
        ) or (
            reference_negative
            and not reference_positive
            and candidate_positive
            and not candidate_negative
        ):
            return True
    return False


def _eligible_fuzzy_window(
    window: str,
    critical_values: Sequence[str],
    anchors: Sequence[str],
) -> tuple[bool, list[str], list[str]]:
    missing_values = [
        value for value in critical_values if not _contains_normalized(window, value)
    ]
    matched_anchors = [
        anchor for anchor in anchors if _contains_normalized(window, anchor)
    ]
    return (
        not missing_values and len(set(matched_anchors)) >= MIN_SEMANTIC_ANCHORS,
        missing_values,
        matched_anchors,
    )


def _candidate_windows(
    candidate: str,
    quote: str,
    anchors: Sequence[str],
    critical_values: Sequence[str],
) -> list[str]:
    if not candidate:
        return []
    quote_length = max(1, len(quote))
    if len(candidate) <= int(quote_length * 1.25) + 8:
        return [candidate]
    lengths = sorted(
        {
            max(1, int(quote_length * 0.9)),
            quote_length,
            min(len(candidate), int(quote_length * 1.15) + 4),
        }
    )
    starts: set[int] = {0, max(0, len(candidate) - quote_length)}
    step = max(1, quote_length // 10)
    starts.update(range(0, len(candidate), step))
    for needle in tuple(anchors) + tuple(critical_values):
        normalized = normalize_evidence_text(needle)
        if not normalized:
            continue
        cursor = 0
        while True:
            position = candidate.find(normalized, cursor)
            if position < 0:
                break
            quote_position = quote.find(normalized)
            starts.add(max(0, position - max(0, quote_position)))
            starts.add(max(0, position - quote_length // 2))
            cursor = position + max(1, len(normalized))
    windows: list[str] = []
    seen: set[str] = set()
    for start in sorted(starts):
        for length in lengths:
            bounded_start = min(start, max(0, len(candidate) - length))
            window = candidate[bounded_start : bounded_start + length].strip()
            if window and window not in seen:
                seen.add(window)
                windows.append(window)
    return windows


def _fuzzy_match(
    candidate: str,
    quote: str,
    critical_values: Sequence[str],
    anchors: Sequence[str],
) -> dict[str, Any]:
    best_ratio = 0.0
    best_window = ""
    best_anchors: list[str] = []
    any_values_complete = False
    polarity_conflict_found = False
    for window in _candidate_windows(candidate, quote, anchors, critical_values):
        eligible, missing_values, matched_anchors = _eligible_fuzzy_window(
            window, critical_values, anchors
        )
        if not missing_values:
            any_values_complete = True
        if not eligible:
            continue
        if _polarity_conflicts(quote, window):
            polarity_conflict_found = True
            continue
        ratio = SequenceMatcher(None, quote, window, autojunk=False).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_window = window
            best_anchors = matched_anchors
    matched = best_ratio >= FUZZY_MATCH_THRESHOLD
    if len(set(anchors)) < MIN_SEMANTIC_ANCHORS:
        reason = "insufficient_semantic_anchors"
    elif not any_values_complete:
        reason = "missing_critical_values"
    elif polarity_conflict_found and not best_window:
        reason = "polarity_conflict"
    elif not best_window:
        reason = "fewer_than_two_semantic_anchors"
    elif not matched:
        reason = "sequence_ratio_below_threshold"
    else:
        reason = "fuzzy_match"
    return {
        "matched": matched,
        "method": "fuzzy" if matched else None,
        "reason": reason,
        "ratio": round(best_ratio, 6),
        "window": best_window,
        "matched_anchors": sorted(set(best_anchors)),
    }


def _as_table_matrix(candidate: Any) -> list[list[str]]:
    if isinstance(candidate, Mapping):
        table = candidate.get("table")
        if isinstance(table, Mapping):
            headers = table.get("headers")
            rows = table.get("rows")
            if isinstance(headers, list) and isinstance(rows, list):
                matrix = [list(headers)]
                for row in rows:
                    if isinstance(row, list):
                        matrix.append(list(row))
                    elif isinstance(row, dict):
                        matrix.append([row.get(str(header), "") for header in headers])
                return [
                    [normalize_evidence_text(cell) for cell in row]
                    for row in matrix
                ]
        headers = candidate.get("table_headers")
        rows = candidate.get("table_rows")
        if isinstance(headers, list) and isinstance(rows, list):
            return [
                [normalize_evidence_text(cell) for cell in row]
                for row in [headers] + [item for item in rows if isinstance(item, list)]
            ]

    raw = unicodedata.normalize("NFKC", _candidate_text(candidate))
    matrix: list[list[str]] = []
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "\t" in line:
            cells = line.split("\t")
        elif "|" in line:
            cells = line.strip("|").split("|")
        elif re.search(r"\s{2,}", line):
            cells = re.split(r"\s{2,}", line)
        else:
            cells = [line]
        matrix.append([normalize_evidence_text(cell) for cell in cells])
    return matrix


def _normalized_table_relations(spec: Mapping[str, Any]) -> list[dict[str, str]]:
    raw_relations = spec.get("row_relations", spec.get("header_value_pairs"))
    relations: list[dict[str, str]] = []
    if not isinstance(raw_relations, list):
        return relations
    for raw in raw_relations:
        if not isinstance(raw, Mapping):
            continue
        relation = {
            "header": normalize_evidence_text(raw.get("header")),
            "value": normalize_evidence_text(raw.get("value")),
            "row_anchor": normalize_evidence_text(raw.get("row_anchor")),
        }
        if all(relation.values()):
            relations.append(relation)
    return relations


def match_table_evidence(
    candidate: Any,
    table_spec: Mapping[str, Any],
    *,
    critical_values: Sequence[str],
) -> dict[str, Any]:
    """Require header-column, row-anchor, value, and critical-value preservation."""

    if not critical_values:
        return {
            "matched": False,
            "method": None,
            "reason": "missing_critical_values",
            "ratio": None,
            "window": "",
            "matched_anchors": [],
        }
    matrix = _as_table_matrix(candidate)
    headers = [
        normalize_evidence_text(value)
        for value in table_spec.get("headers", [])
        if normalize_evidence_text(value)
    ]
    relations = _normalized_table_relations(table_spec)
    if not matrix or not headers or not relations:
        return {
            "matched": False,
            "method": None,
            "reason": "invalid_or_missing_table_structure",
            "ratio": None,
            "window": "",
            "matched_anchors": [],
        }

    for header_row_index, row in enumerate(matrix):
        header_indexes: dict[str, int] = {}
        for header in headers:
            for cell_index, cell in enumerate(row):
                if _contains_normalized(cell, header):
                    header_indexes[header] = cell_index
                    break
        if len(header_indexes) != len(headers):
            continue

        matched_rows: list[list[str]] = []
        matched_relation_cells: list[str] = []
        relation_failure = False
        for relation in relations:
            relation_header = relation["header"]
            header_key = next(
                (
                    header
                    for header in headers
                    if _contains_normalized(header, relation_header)
                    or _contains_normalized(relation_header, header)
                ),
                None,
            )
            if header_key is None:
                relation_failure = True
                break
            column_index = header_indexes[header_key]
            relation_row = next(
                (
                    candidate_row
                    for row_index, candidate_row in enumerate(matrix)
                    if row_index != header_row_index
                    and any(
                        _contains_normalized(cell, relation["row_anchor"])
                        for cell in candidate_row
                    )
                    and column_index < len(candidate_row)
                    and _contains_normalized(
                        candidate_row[column_index], relation["value"]
                    )
                ),
                None,
            )
            if relation_row is None:
                relation_failure = True
                break
            matched_rows.append(relation_row)
            matched_relation_cells.append(relation_row[column_index])
        if relation_failure:
            continue

        matched_text = " ".join(
            cell for row_cells in matched_rows for cell in row_cells
        )
        relation_cell_text = " ".join(matched_relation_cells)
        missing_values = [
            value
            for value in critical_values
            if not _contains_normalized(relation_cell_text, value)
        ]
        if missing_values:
            continue
        return {
            "matched": True,
            "method": "table",
            "reason": "table_relation_match",
            "ratio": 1.0,
            "window": matched_text,
            "matched_anchors": sorted(
                {relation["row_anchor"] for relation in relations}
            ),
            "header_row_index": header_row_index,
        }
    return {
        "matched": False,
        "method": None,
        "reason": "table_relation_mismatch",
        "ratio": None,
        "window": "",
        "matched_anchors": [],
    }


def match_evidence(
    candidate: Any,
    evidence_option: Mapping[str, Any],
    *,
    claim: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Match one retrieved candidate to one profile-independent gold option.

    When gold exposes ``document_id``, the candidate must expose the same ID;
    missing or mismatched identity fails closed.  A matching ID establishes
    document identity but does not by itself prove that the retrieved passage
    preserved the atomic evidence; exact, fuzzy, or table matching must still pass.
    """

    if not isinstance(evidence_option, Mapping):
        raise TypeError("evidence_option must be an object")
    candidate_mapping: Mapping[str, Any] = candidate if isinstance(candidate, Mapping) else {}
    expected_document_id = _nonempty_text(evidence_option.get("document_id"))
    actual_document_id = _nonempty_text(
        _candidate_field(candidate_mapping, "document_id", "doc_id")
        if candidate_mapping
        else ""
    )
    document_id_match: Optional[bool] = None
    if expected_document_id:
        if not actual_document_id:
            return {
                "matched": False,
                "method": None,
                "reason": "document_id_missing",
                "ratio": None,
                "window": "",
                "matched_anchors": [],
                "document_id_match": False,
            }
        document_id_match = _fold_identifier(expected_document_id) == _fold_identifier(
            actual_document_id
        )
        if not document_id_match:
            return {
                "matched": False,
                "method": None,
                "reason": "document_id_mismatch",
                "ratio": None,
                "window": "",
                "matched_anchors": [],
                "document_id_match": False,
            }

    critical_values = _claim_values(claim)
    table_spec = _table_spec_from(claim or {}, evidence_option)
    is_table = bool(table_spec) or (
        isinstance(claim, Mapping)
        and _fold_identifier(claim.get("evidence_type")) == "table"
    )
    if is_table:
        if not isinstance(table_spec, Mapping):
            result = {
                "matched": False,
                "method": None,
                "reason": "invalid_or_missing_table_structure",
                "ratio": None,
                "window": "",
                "matched_anchors": [],
            }
        else:
            result = match_table_evidence(
                candidate, table_spec, critical_values=critical_values
            )
        result["document_id_match"] = document_id_match
        if result["matched"] and document_id_match:
            result["method"] = "document_id+table"
        return result

    candidate_text = normalize_evidence_text(_candidate_text(candidate))
    quote = normalize_evidence_text(evidence_option.get("quote"))
    if not candidate_text or not quote:
        return {
            "matched": False,
            "method": None,
            "reason": "missing_candidate_text_or_quote",
            "ratio": None,
            "window": "",
            "matched_anchors": [],
            "document_id_match": document_id_match,
        }
    if _contains_normalized(candidate_text, quote):
        missing_values = [
            value
            for value in critical_values
            if not _contains_normalized(quote, value)
        ]
        if missing_values:
            return {
                "matched": False,
                "method": None,
                "reason": "missing_critical_values",
                "ratio": 1.0,
                "window": quote,
                "matched_anchors": [],
                "missing_critical_values": missing_values,
                "document_id_match": document_id_match,
            }
        return {
            "matched": True,
            "method": "document_id+exact" if document_id_match else "exact",
            "reason": "normalized_exact_substring",
            "ratio": 1.0,
            "window": quote,
            "matched_anchors": [],
            "document_id_match": document_id_match,
        }

    anchors = _semantic_anchors(claim, evidence_option)
    result = _fuzzy_match(candidate_text, quote, critical_values, anchors)
    result["document_id_match"] = document_id_match
    if result["matched"] and document_id_match:
        result["method"] = "document_id+fuzzy"
    return result


def evidence_matches(
    candidate: Any,
    evidence_option: Mapping[str, Any],
    *,
    claim: Optional[Mapping[str, Any]] = None,
) -> bool:
    return bool(match_evidence(candidate, evidence_option, claim=claim)["matched"])


# Descriptive alias for callers that iterate a claim's evidence_options.
match_evidence_option = match_evidence
