#!/usr/bin/env python3
"""No-clobber, crash-resistant publication helpers for final artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping, Sequence


def paths_alias(left: Path, right: Path) -> bool:
    if left.resolve() == right.resolve():
        return True
    try:
        return left.exists() and right.exists() and left.samefile(right)
    except OSError:
        return False


def reject_symlink_inputs(paths: Sequence[Path]) -> None:
    symlinks = [str(path) for path in paths if path.is_symlink()]
    if symlinks:
        raise ValueError("input artifacts must not be symlinks: " + ", ".join(symlinks))


def require_new_outputs(paths: Sequence[Path]) -> None:
    if any(
        paths_alias(left, right)
        for index, left in enumerate(paths)
        for right in paths[index + 1 :]
    ):
        raise ValueError("output paths must be distinct")
    existing = [str(path) for path in paths if os.path.lexists(path)]
    if existing:
        raise ValueError(
            "immutable output path already exists; use a new path: "
            + ", ".join(existing)
        )


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def exclusive_run_lock(path: Path, owner: Mapping[str, object]) -> Iterator[None]:
    """Hold one no-clobber process lock; a crash deliberately leaves it stale.

    Final schedules must never issue the same model sample twice.  ``O_EXCL``
    closes the audit-to-request race between concurrent runner processes.  A
    stale lock is not guessed away automatically: the experiment operator must
    inspect it and start a new experiment if a request may have been sent.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise ValueError(
            f"final schedule run lock already exists: {path}; concurrent or "
            "stale execution requires manual audit"
        ) from exc
    locked_stat = os.fstat(descriptor)
    try:
        payload = dict(owner)
        payload["pid"] = os.getpid()
        data = (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise OSError("could not write final schedule run lock")
            offset += written
        os.fsync(descriptor)
        _fsync_directory(path.parent)
        yield
    finally:
        try:
            current = os.lstat(path)
            if (
                current.st_dev != locked_stat.st_dev
                or current.st_ino != locked_stat.st_ino
            ):
                raise RuntimeError(
                    f"final schedule run lock identity changed: {path}"
                )
            path.unlink()
            _fsync_directory(path.parent)
        finally:
            os.close(descriptor)


def _stage_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            temp_path.unlink()
        except OSError:
            pass
        raise
    return temp_path


def publish_immutable_texts(
    outputs: Mapping[Path, str],
    *,
    authoritative_path: Path,
) -> None:
    """Atomically publish files without replacement, authority last.

    Each file is fully written and fsynced in its destination directory before
    an atomic hard-link publication. The authoritative artifact is linked last;
    therefore its existence certifies that every companion artifact was already
    published and directory-fsynced.
    """

    if authoritative_path not in outputs:
        raise ValueError("authoritative output is not present in output mapping")
    paths = list(outputs)
    require_new_outputs(paths)
    publish_order = [path for path in paths if path != authoritative_path]
    publish_order.append(authoritative_path)
    staged: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for path in paths:
            staged[path] = _stage_text(path, outputs[path])
        # Re-check after staging; os.link below is the final no-clobber guard.
        require_new_outputs(paths)
        for path in publish_order:
            os.link(staged[path], path)
            published.append(path)
            _fsync_directory(path.parent)
    except BaseException:
        for path in reversed(published):
            temp_path = staged.get(path)
            try:
                if temp_path is not None and path.samefile(temp_path):
                    path.unlink()
                    _fsync_directory(path.parent)
            except OSError:
                pass
        raise
    finally:
        for temp_path in staged.values():
            try:
                temp_path.unlink()
            except OSError:
                pass
