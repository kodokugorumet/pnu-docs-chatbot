"""Artifact manifest loading and checksum helpers.

This module is deliberately stdlib-only and compatible with Python 3.9.  It
does not download or install anything; mutation is confined to ``prepare`` in
``manager.py`` and only happens when its ``execute`` argument is true.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union


PathLike = Union[str, os.PathLike]
DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "parser-artifacts.json"
)
MANIFEST_ENV_VAR = "PARSER_ARTIFACT_MANIFEST"
SUPPORTED_CHECKSUMS = frozenset({"sha256", "sha512"})


class ArtifactManifestError(ValueError):
    """Raised when the parser artifact manifest is missing or malformed."""


def file_digest(
    path: PathLike,
    algorithm: str = "sha256",
    chunk_size: int = 1024 * 1024,
) -> str:
    """Return a lowercase hexadecimal digest for *path*.

    Only SHA-256 and SHA-512 are accepted.  Keeping the algorithm allow-list
    small prevents a typo from silently selecting a weak or unexpected hash.
    """

    normalized = algorithm.lower().replace("-", "")
    if normalized not in SUPPORTED_CHECKSUMS:
        raise ValueError(
            "unsupported checksum algorithm {!r}; expected one of {}".format(
                algorithm, sorted(SUPPORTED_CHECKSUMS)
            )
        )
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    digest = hashlib.new(normalized)
    with Path(path).open("rb") as source:
        while True:
            chunk = source.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(
    path: PathLike,
    expected: str,
    algorithm: str = "sha256",
) -> bool:
    """Return whether *path* matches *expected* using constant-time comparison."""

    normalized_expected = expected.strip().lower()
    normalized_algorithm = algorithm.lower().replace("-", "")
    expected_lengths = {"sha256": 64, "sha512": 128}
    if normalized_algorithm not in expected_lengths:
        raise ValueError("unsupported checksum algorithm {!r}".format(algorithm))
    if len(normalized_expected) != expected_lengths[normalized_algorithm]:
        raise ValueError(
            "{} checksum must contain {} hexadecimal characters".format(
                normalized_algorithm, expected_lengths[normalized_algorithm]
            )
        )
    try:
        int(normalized_expected, 16)
    except ValueError as exc:
        raise ValueError("expected checksum is not hexadecimal") from exc

    actual = file_digest(path, normalized_algorithm)
    return hmac.compare_digest(actual, normalized_expected)


def verify_sha256(path: PathLike, expected: str) -> bool:
    """Convenience wrapper for the checksum used by most parser artifacts."""

    return verify_checksum(path, expected, "sha256")


def _require_mapping(value: Any, label: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactManifestError("{} must be a JSON object".format(label))
    return value


def _validate_manifest(manifest: Dict[str, Any], source: Path) -> None:
    if manifest.get("schema_version") != 1:
        raise ArtifactManifestError(
            "{}: unsupported schema_version {!r}".format(
                source, manifest.get("schema_version")
            )
        )

    profiles = _require_mapping(manifest.get("profiles"), "profiles")
    capabilities = _require_mapping(manifest.get("capabilities"), "capabilities")
    artifacts = _require_mapping(manifest.get("artifacts"), "artifacts")
    if not profiles:
        raise ArtifactManifestError("profiles must not be empty")

    for profile_name, profile in profiles.items():
        profile = _require_mapping(profile, "profiles.{}".format(profile_name))
        for bucket in ("required", "optional"):
            names = profile.get(bucket, [])
            if not isinstance(names, list) or not all(
                isinstance(name, str) for name in names
            ):
                raise ArtifactManifestError(
                    "profiles.{}.{} must be a string array".format(
                        profile_name, bucket
                    )
                )
            unknown = sorted(set(names) - set(capabilities))
            if unknown:
                raise ArtifactManifestError(
                    "profiles.{}.{} references unknown capabilities: {}".format(
                        profile_name, bucket, ", ".join(unknown)
                    )
                )

        artifact_names = profile.get("artifacts", [])
        if not isinstance(artifact_names, list) or not all(
            isinstance(name, str) for name in artifact_names
        ):
            raise ArtifactManifestError(
                "profiles.{}.artifacts must be a string array".format(profile_name)
            )
        unknown_artifacts = sorted(set(artifact_names) - set(artifacts))
        if unknown_artifacts:
            raise ArtifactManifestError(
                "profiles.{}.artifacts references unknown artifacts: {}".format(
                    profile_name, ", ".join(unknown_artifacts)
                )
            )

    allowed_kinds = {"command", "file", "python_import", "tessdata", "model"}
    for name, capability in capabilities.items():
        capability = _require_mapping(capability, "capabilities.{}".format(name))
        if capability.get("kind") not in allowed_kinds:
            raise ArtifactManifestError(
                "capabilities.{} has unsupported kind {!r}".format(
                    name, capability.get("kind")
                )
            )

    for name, artifact in artifacts.items():
        artifact = _require_mapping(artifact, "artifacts.{}".format(name))
        variants = _require_mapping(
            artifact.get("variants"), "artifacts.{}.variants".format(name)
        )
        if not variants:
            raise ArtifactManifestError(
                "artifacts.{}.variants must not be empty".format(name)
            )
        for variant_name, variant in variants.items():
            variant = _require_mapping(
                variant,
                "artifacts.{}.variants.{}".format(name, variant_name),
            )
            for field in ("url", "checksum", "destination"):
                if not isinstance(variant.get(field), str) or not variant[field]:
                    raise ArtifactManifestError(
                        "artifacts.{}.variants.{}.{} must be a non-empty string".format(
                            name, variant_name, field
                        )
                    )
            algorithm = variant.get("checksum_algorithm", "sha256")
            try:
                expected_length = {"sha256": 64, "sha512": 128}[algorithm]
            except KeyError as exc:
                raise ArtifactManifestError(
                    "artifacts.{}.variants.{} has unsupported checksum algorithm".format(
                        name, variant_name
                    )
                ) from exc
            checksum = variant["checksum"].lower()
            if len(checksum) != expected_length:
                raise ArtifactManifestError(
                    "artifacts.{}.variants.{}.checksum must be {} hex characters".format(
                        name, variant_name, expected_length
                    )
                )
            try:
                int(checksum, 16)
            except ValueError as exc:
                raise ArtifactManifestError(
                    "artifacts.{}.variants.{}.checksum is not hexadecimal".format(
                        name, variant_name
                    )
                ) from exc


def load_artifact_manifest(path: Optional[PathLike] = None) -> Dict[str, Any]:
    """Load and validate the checked-in parser artifact manifest.

    Tests and controlled deployments may override the path with the function
    argument or ``PARSER_ARTIFACT_MANIFEST``.  Relative paths are resolved from
    the caller's current directory, not from the repository.
    """

    selected = path or os.environ.get(MANIFEST_ENV_VAR) or DEFAULT_MANIFEST_PATH
    manifest_path = Path(selected).expanduser().resolve()
    try:
        with manifest_path.open("r", encoding="utf-8") as source:
            value = json.load(source)
    except FileNotFoundError as exc:
        raise ArtifactManifestError(
            "artifact manifest not found: {}".format(manifest_path)
        ) from exc
    except json.JSONDecodeError as exc:
        raise ArtifactManifestError(
            "invalid JSON in artifact manifest {}: {}".format(manifest_path, exc)
        ) from exc

    manifest = _require_mapping(value, "manifest")
    _validate_manifest(manifest, manifest_path)
    # The source is useful in doctor output and remains JSON serializable.
    manifest["_manifest_path"] = str(manifest_path)
    return manifest
