"""Local-only runtime provisioning and diagnostics for parser pipelines.

The public functions intentionally return plain dictionaries so callers can
serialize their result directly in the CLI without depending on a third-party
package.
"""

from .artifacts import (
    ArtifactManifestError,
    file_digest,
    load_artifact_manifest,
    verify_checksum,
    verify_sha256,
)
from .manager import doctor, prepare

__all__ = [
    "ArtifactManifestError",
    "doctor",
    "file_digest",
    "load_artifact_manifest",
    "prepare",
    "verify_checksum",
    "verify_sha256",
]
