"""Baseline, structured challenger, and production cascade orchestration."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from scripts.document_parsing.adapters import (
    AdapterContext,
    overlay_pdf_tables,
    parse_docling,
    parse_java_hwp,
    parse_native_html,
    parse_native_office,
    parse_paddle,
    parse_pdfplumber_tables,
    parse_pymupdf,
    parse_pypdf,
    parse_subprocess,
    parse_tesseract,
    parse_tika,
    parse_unhwp,
    sniff_source,
)
from scripts.document_parsing.core import (
    Attempt,
    Block,
    ParseResult,
    QualityAssessment,
    SourceDocument,
    assess_quality,
    make_block_id,
    make_document_id,
    make_table_id,
    normalize_relative_path,
)
from scripts.document_parsing.runtime.manager import (
    verified_artifact_path,
    verified_hwp_worker_path,
)


PROFILES = ("baseline", "challenger", "cascade")
SUPPORTED_FORMATS = frozenset(
    {
        "pdf",
        "hwp",
        "hwpx",
        "html",
        "image",
        "docx",
        "xlsx",
        "xls",
        "pptx",
    }
)
PROFILE_PINS = {
    "hwplib": "1.1.10",
    "hwpxlib": "1.0.8",
    "unhwp": "0.3.0",
    "docling": "2.114.0",
    "paddleocr": "3.7.0",
    "pp_layout": "PP-DocLayout_plus-L",
    "pp_ocr_detection": "PP-OCRv5_mobile_det",
    "pp_ocr_korean": "korean_PP-OCRv5_mobile_rec",
    "tesseract": "5.5.2",
    "tika": "3.3.2",
}


@dataclass(frozen=True)
class PipelineConfig:
    profile: str
    tools_dir: Path
    raw_output_dir: Path
    repo_root: Path
    runtime_report: Optional[Mapping[str, Any]] = None
    expect_korean: bool = False
    min_chars: int = 20
    enable_vl_review: bool = False
    fast_timeout_seconds: int = 120
    tesseract_timeout_seconds: int = 120
    heavy_timeout_seconds: int = 1800
    max_blocks_per_document: int = 250000

    def __post_init__(self) -> None:
        if self.profile not in PROFILES:
            raise ValueError(
                "unknown profile {!r}; expected {}".format(
                    self.profile, ", ".join(PROFILES)
                )
            )
        if self.min_chars < 0:
            raise ValueError("min_chars must be non-negative")
        if self.max_blocks_per_document <= 0:
            raise ValueError("max_blocks_per_document must be positive")
        object.__setattr__(self, "tools_dir", Path(self.tools_dir).resolve())
        object.__setattr__(
            self, "raw_output_dir", Path(self.raw_output_dir).resolve()
        )
        object.__setattr__(self, "repo_root", Path(self.repo_root).resolve())


@dataclass
class PipelineOutcome:
    source: SourceDocument
    sniffed_format: str
    mime_type: str
    result: ParseResult
    quality: QualityAssessment
    status: str
    reason: str = ""
    selected_parsers: Sequence[str] = field(default_factory=tuple)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while True:
            chunk = source.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_source_document(
    path: Path,
    input_root: Path,
    profile: str,
    repo_root: Optional[Path] = None,
) -> SourceDocument:
    input_path = Path(path).resolve()
    root = Path(input_root).resolve()
    relative = normalize_relative_path(input_path.relative_to(root).as_posix())
    digest = file_sha256(input_path)
    sniffed = sniff_source(input_path)
    source_path = relative
    if repo_root is not None:
        try:
            source_path = input_path.relative_to(Path(repo_root).resolve()).as_posix()
        except ValueError:
            pass
    relative_parts = Path(relative).parts
    institution = relative_parts[0] if len(relative_parts) > 1 else ""
    return SourceDocument(
        path=input_path,
        document_id=make_document_id(relative, digest),
        relative_path=relative,
        source_sha256=digest,
        source_path=source_path,
        file_name=input_path.name,
        extension=input_path.suffix.lower(),
        size_bytes=input_path.stat().st_size,
        institution=institution,
        mime_type=sniffed.mime_type,
        profile=profile,
    )


class RuntimeCommands:
    """Resolve repo-local workers without performing installation."""

    def __init__(
        self,
        repo_root: Path,
        tools_dir: Path,
        runtime_report: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.tools_dir = Path(tools_dir)
        self.runtime_report = runtime_report
        self._java = self._find_java()

    @staticmethod
    def _first_file(candidates: Iterable[Path], executable: bool = False) -> Optional[Path]:
        for candidate in candidates:
            if candidate.is_file() and (
                not executable or os.access(str(candidate), os.X_OK)
            ):
                return candidate.resolve()
        return None

    def _capability_path(
        self,
        name: str,
        *,
        directory: bool = False,
        executable: bool = False,
    ) -> Optional[Path]:
        if self.runtime_report is None:
            return None
        capability = (
            self.runtime_report.get("capabilities", {}).get(name, {})
        )
        if not capability.get("available") or not capability.get("location"):
            return None
        try:
            path = Path(
                os.path.abspath(
                    str(Path(str(capability["location"])).expanduser())
                )
            )
        except OSError:
            return None
        if directory:
            return path if path.is_dir() else None
        if not path.is_file():
            return None
        if executable and not os.access(str(path), os.X_OK):
            return None
        return path

    def _pin(self, name: str) -> str:
        if self.runtime_report is not None:
            value = self.runtime_report.get("pins", {}).get(name)
            if value:
                return str(value)
        return PROFILE_PINS[name]

    def _approved_value(self, capability: str, pin: str) -> str:
        if self.runtime_report is not None:
            value = (
                self.runtime_report.get("capabilities", {})
                .get(capability, {})
                .get("version")
            )
            if value:
                return str(value)
        return self._pin(pin)

    def _tessdata_directory(self) -> Optional[Path]:
        if self.runtime_report is None:
            for directory in (
                self.tools_dir / "tesseract" / "share" / "tessdata",
                self.tools_dir / "share" / "tessdata",
            ):
                try:
                    complete = all(
                        (directory / filename).is_file()
                        and not (directory / filename).is_symlink()
                        and (directory / filename).stat().st_size > 0
                        for filename in (
                            "kor.traineddata",
                            "eng.traineddata",
                        )
                    )
                except OSError:
                    complete = False
                if complete:
                    return directory.resolve()
            return None
        korean = self._capability_path("tesseract_kor")
        english = self._capability_path("tesseract_eng")
        if (
            korean is None
            or english is None
            or korean.parent != english.parent
        ):
            return None
        try:
            if any(
                path.is_symlink() or path.stat().st_size <= 0
                for path in (korean, english)
            ):
                return None
        except OSError:
            return None
        return korean.parent

    def _find_java(self) -> Optional[Path]:
        if self.runtime_report is not None:
            return self._capability_path("java", executable=True)
        candidates: List[Path] = []
        configured = os.environ.get("PARSER_JAVA")
        if configured:
            candidates.append(Path(configured).expanduser())
        candidates.append(self.tools_dir / "jdk" / "bin" / "java")
        java_home = os.environ.get("JAVA_HOME")
        if java_home:
            candidates.append(Path(java_home).expanduser() / "bin" / "java")
        local_share = Path.home() / ".local" / "share"
        if local_share.is_dir():
            candidates.extend(
                sorted(
                    local_share.glob("jdk*/**/bin/java"),
                    key=lambda item: str(item),
                )
            )
        discovered = shutil.which("java")
        if discovered:
            candidates.append(Path(discovered))
        for candidate in candidates:
            if not candidate.is_file() or not os.access(str(candidate), os.X_OK):
                continue
            try:
                completed = subprocess.run(
                    [str(candidate), "-version"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=5,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            version_output = "{}\n{}".format(
                completed.stdout.decode("utf-8", errors="replace"),
                completed.stderr.decode("utf-8", errors="replace"),
            )
            match = re.search(r"(?<!\d)(\d+)(?:\.\d+)*(?!\d)", version_output)
            if (
                completed.returncode == 0
                and match is not None
                and int(match.group(1)) >= 21
            ):
                return candidate.resolve()
        return None

    def java_worker(self, profile: str) -> Optional[Sequence[str]]:
        installed_worker = verified_hwp_worker_path(self.tools_dir)
        if self.runtime_report is not None:
            approved_worker = self._capability_path("hwp_worker_jar")
            required = (
                approved_worker,
                self._capability_path("hwplib_jar"),
                self._capability_path("hwpxlib_jar"),
            )
            if (
                any(path is None for path in required)
                or installed_worker is None
                or approved_worker != installed_worker.resolve()
            ):
                return None
        jar = self._first_file(
            [installed_worker] if installed_worker is not None else []
        )
        if self._java is None or jar is None:
            return None
        return (
            str(self._java),
            "-Xmx1024m",
            "-XX:+ExitOnOutOfMemoryError",
            "-jar",
            str(jar),
            "--input",
            "{input}",
            "--output",
            "{output}",
            "--profile",
            profile,
        )

    def unhwp(self) -> Optional[Sequence[str]]:
        if self.runtime_report is not None:
            executable = self._capability_path("unhwp", executable=True)
            if executable is not None:
                try:
                    executable.relative_to(self.tools_dir)
                except ValueError:
                    pass
                else:
                    verified = verified_artifact_path("unhwp", self.tools_dir)
                    if verified is None or verified.resolve() != executable:
                        executable = None
        else:
            executable = verified_artifact_path("unhwp", self.tools_dir)
        if executable is not None and not os.access(
            str(executable), os.X_OK
        ):
            executable = None
        if executable is None:
            return None
        return (
            str(executable),
            "convert",
            "{input}",
            "-o",
            "{output_dir}",
            "--all",
            "--cleanup",
            "none",
        )

    def tesseract(self) -> Optional[Sequence[str]]:
        if self.runtime_report is not None:
            executable = self._capability_path(
                "tesseract",
                executable=True,
            )
        else:
            executable = self._first_file(
                [
                    self.tools_dir / "tesseract" / "bin" / "tesseract",
                    self.tools_dir / "bin" / "tesseract",
                ],
                executable=True,
            )
        if executable is None or self._tessdata_directory() is None:
            return None
        return (str(executable), "{input}", "stdout", "-l", "kor+eng")

    def docling(self, profile: str) -> Optional[Sequence[str]]:
        if self.runtime_report is not None:
            model_dir = self._capability_path(
                "docling_models",
                directory=True,
            )
        else:
            model_dir = Path(
                os.environ.get(
                    "DOCLING_ARTIFACTS_PATH",
                    str(self.tools_dir / "models" / "docling"),
                )
            ).expanduser()
        if model_dir is None:
            return None
        if not self._complete_docling_models(model_dir):
            return None
        interpreter = self._python_for(
            "docling",
            "docling",
            capability="docling",
        )
        worker = self.repo_root / "parser-workers" / "docling_worker.py"
        if interpreter is None or not worker.is_file():
            return None
        return (
            str(interpreter),
            str(worker),
            "--input",
            "{input}",
            "--output",
            "{output}",
            "--profile",
            profile,
        )

    def paddle(self, profile: str) -> Optional[Sequence[str]]:
        if self.runtime_report is not None:
            layout_model = self._capability_path(
                "pp_structure_v3_model",
                directory=True,
            )
            detection_model = self._capability_path(
                "pp_ocrv5_detection_model",
                directory=True,
            )
            recognition_model = self._capability_path(
                "pp_ocrv5_korean_model",
                directory=True,
            )
        else:
            paddle_models = self.tools_dir / "models" / "paddle"
            layout_model = (
                paddle_models / "PP-StructureV3" / "layout_detection"
            )
            detection_model = (
                paddle_models / "PP-StructureV3" / "text_detection"
            )
            recognition_model = (
                paddle_models / self._pin("pp_ocr_korean")
            )
        if any(
            path is None
            for path in (layout_model, detection_model, recognition_model)
        ):
            return None
        if not all(
            self._complete_paddle_model(path)
            for path in (
                layout_model,
                detection_model,
                recognition_model,
            )
        ):
            return None
        if self.runtime_report is not None:
            paddleocr_python = self._capability_path(
                "paddleocr",
                executable=True,
            )
            paddle_python = self._capability_path(
                "paddlepaddle",
                executable=True,
            )
            if (
                paddleocr_python is None
                or paddle_python is None
                or paddleocr_python != paddle_python
            ):
                return None
            interpreter = paddleocr_python
        else:
            interpreter = self._python_for("paddle", "paddleocr")
        worker = self.repo_root / "parser-workers" / "paddle_worker.py"
        if interpreter is None or not worker.is_file():
            return None
        return (
            str(interpreter),
            str(worker),
            "--input",
            "{input}",
            "--output",
            "{output}",
            "--profile",
            profile,
            "--layout-model-dir",
            str(layout_model),
            "--text-detection-model-dir",
            str(detection_model),
            "--detection-model",
            self._approved_value(
                "pp_ocrv5_detection_model",
                "pp_ocr_detection",
            ),
            "--recognition-model-dir",
            str(recognition_model),
            "--recognition-model",
            self._approved_value(
                "pp_ocrv5_korean_model",
                "pp_ocr_korean",
            ),
            "--device",
            "cpu",
        )

    def tika(self) -> Optional[Sequence[str]]:
        jar = verified_artifact_path("tika", self.tools_dir)
        if self.runtime_report is not None:
            approved_jar = self._capability_path("tika_jar")
            if (
                approved_jar is None
                or jar is None
                or approved_jar != jar.resolve()
            ):
                return None
        if self._java is None or jar is None:
            return None
        return (
            str(self._java),
            "-Xmx1024m",
            "-XX:+ExitOnOutOfMemoryError",
            "-jar",
            str(jar),
            "--text",
            "{input}",
        )

    def worker_env(self) -> Mapping[str, str]:
        values: Dict[str, str] = {}
        tessdata = self._tessdata_directory()
        if tessdata is not None:
            values["TESSDATA_PREFIX"] = str(tessdata)
        if self.runtime_report is not None:
            recognition_model = self._capability_path(
                "pp_ocrv5_korean_model",
                directory=True,
            )
            model_home = (
                recognition_model.parent
                if recognition_model is not None
                else None
            )
        else:
            model_home = self.tools_dir / "models" / "paddle"
        if model_home is not None and model_home.is_dir():
            values["PADDLE_PDX_CACHE_HOME"] = str(model_home)
        if self.runtime_report is not None:
            vl_model = self._capability_path(
                "paddleocr_vl_model",
                directory=True,
            )
            if vl_model is not None:
                values["PARSER_PADDLE_VL_MODEL_DIR"] = str(vl_model)
        if self.runtime_report is not None:
            docling_models = self._capability_path(
                "docling_models",
                directory=True,
            )
        else:
            docling_models = self.tools_dir / "models" / "docling"
        if docling_models is not None and docling_models.is_dir():
            values["DOCLING_ARTIFACTS_PATH"] = str(docling_models)
            values["HF_HUB_OFFLINE"] = "1"
            values["TRANSFORMERS_OFFLINE"] = "1"
            values["HF_DATASETS_OFFLINE"] = "1"
        return values

    @staticmethod
    def _nonempty_directory(path: Path) -> bool:
        if not path.is_dir():
            return False
        try:
            next(path.iterdir())
        except (StopIteration, OSError):
            return False
        return True

    @staticmethod
    def _complete_paddle_model(path: Path) -> bool:
        if not path.is_dir():
            return False
        try:
            return all(
                (path / name).is_file()
                and not (path / name).is_symlink()
                and (path / name).stat().st_size > 0
                for name in (
                    "inference.json",
                    "inference.pdiparams",
                    "inference.yml",
                )
            )
        except OSError:
            return False

    @staticmethod
    def _complete_docling_models(path: Path) -> bool:
        if not path.is_dir():
            return False
        required_patterns = (
            "docling-project--docling-layout-heron/*.safetensors",
            "docling-project--docling-layout-heron/*config*.json",
            "**/model_artifacts/tableformer/accurate/*.safetensors",
            "**/model_artifacts/tableformer/accurate/*config*.json",
        )
        for pattern in required_patterns:
            try:
                match = next(
                    (
                        item
                        for item in path.glob(pattern)
                        if (
                            item.is_file()
                            and not item.is_symlink()
                            and item.stat().st_size > 0
                        )
                    ),
                    None,
                )
            except OSError:
                return False
            if match is None:
                return False
        return True

    def _python_for(
        self,
        environment: str,
        module: str,
        capability: Optional[str] = None,
    ) -> Optional[Path]:
        if self.runtime_report is not None:
            return self._capability_path(
                capability or module,
                executable=True,
            )
        candidates = [
            self.tools_dir / "venvs" / environment / "bin" / "python",
            Path(sys.executable),
        ]
        discovered = shutil.which("python3.12")
        if discovered:
            candidates.append(Path(discovered))
        seen = set()
        for interpreter in candidates:
            try:
                resolved = Path(
                    os.path.abspath(str(interpreter.expanduser()))
                )
            except OSError:
                continue
            if (
                str(resolved) in seen
                or not resolved.is_file()
                or not os.access(str(resolved), os.X_OK)
            ):
                continue
            seen.add(str(resolved))
            try:
                completed = subprocess.run(
                    [
                        str(resolved),
                        "-c",
                        (
                            "import importlib.util,sys;"
                            "sys.exit(0 if importlib.util.find_spec({!r}) else 1)"
                        ).format(module),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            if completed.returncode == 0:
                return resolved
        return None


class PipelineRunner:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.commands = RuntimeCommands(
            config.repo_root,
            config.tools_dir,
            config.runtime_report,
        )
        self._limits = {
            "java": threading.BoundedSemaphore(1),
            "docling": threading.BoundedSemaphore(1),
            "paddle": threading.BoundedSemaphore(1),
            "tesseract": threading.BoundedSemaphore(2),
        }

    def run(self, source: SourceDocument) -> PipelineOutcome:
        sniffed = sniff_source(source.path)
        if sniffed.format not in SUPPORTED_FORMATS:
            attempt = Attempt(
                parser="{}/router".format(self.config.profile),
                status="unsupported",
                reason="unsupported_format:{}:{}".format(
                    sniffed.format, sniffed.reason
                ),
                metadata={
                    "detected_format": sniffed.format,
                    "sniff_reason": sniffed.reason,
                },
            )
            result = ParseResult(source, attempts=[attempt])
            quality = self._assess(result)
            return PipelineOutcome(
                source=source,
                sniffed_format=sniffed.format,
                mime_type=sniffed.mime_type,
                result=result,
                quality=quality,
                status="unsupported",
                reason=attempt.reason,
            )

        if sniffed.format in {"docx", "xlsx", "xls", "pptx"}:
            capability = {
                "docx": "python_docx",
                "xlsx": "openpyxl",
                "xls": "xlrd",
                "pptx": None,
            }[sniffed.format]
            parser_suffix = {
                "docx": "python-docx",
                "xlsx": "openpyxl",
                "xls": "xlrd",
                "pptx": "pptx-xml",
            }[sniffed.format]
            result = self._call_capabilities(
                tuple(
                    name
                    for name in ("core_python", capability)
                    if name is not None
                ),
                "native",
                parse_native_office,
                source,
                self._context(
                    "{}/{}".format(self.config.profile, parser_suffix),
                    self.config.fast_timeout_seconds,
                    options={
                        "version": self._capability_version(capability)
                    }
                    if capability
                    else None,
                ),
            )
            return self._finish(source, sniffed.format, sniffed.mime_type, result)
        if sniffed.format == "html":
            result = self._call_capability(
                "core_python",
                "native",
                parse_native_html,
                source,
                self._context(
                    "{}/html-dom".format(self.config.profile),
                    self.config.fast_timeout_seconds,
                    options={
                        "trafilatura_fallback": self._capability_available(
                            "trafilatura"
                        ),
                        "trafilatura_version": self._capability_version(
                            "trafilatura"
                        ),
                    },
                ),
            )
            return self._finish(source, sniffed.format, sniffed.mime_type, result)

        if self.config.profile == "baseline":
            result = self._baseline(source, sniffed.format)
        elif self.config.profile == "challenger":
            result = self._challenger(source, sniffed.format)
        else:
            result = self._cascade(source, sniffed.format)
        return self._finish(source, sniffed.format, sniffed.mime_type, result)

    def _baseline(self, source: SourceDocument, format_name: str) -> ParseResult:
        if format_name in {"hwp", "hwpx"}:
            parser = "baseline/{}".format(
                "hwpxlib" if format_name == "hwpx" else "hwplib"
            )
            return self._call(
                "java",
                parse_java_hwp,
                source,
                self._context(
                    parser,
                    self.config.fast_timeout_seconds,
                    command=self.commands.java_worker("baseline"),
                    options={
                        "version": self._approved_version(
                            "hwpxlib_jar"
                            if format_name == "hwpx"
                            else "hwplib_jar",
                            "hwpxlib"
                            if format_name == "hwpx"
                            else "hwplib",
                        )
                    },
                ),
            )
        if format_name == "image":
            return self._call(
                "tesseract",
                parse_tesseract,
                source,
                self._context(
                    "baseline/tesseract",
                    self.config.tesseract_timeout_seconds,
                    command=self.commands.tesseract(),
                    options={
                        "version": self._approved_version(
                            "tesseract",
                            "tesseract",
                        )
                    },
                ),
            )
        if format_name == "pdf":
            return self._baseline_pdf(source)
        return ParseResult(source)

    def _baseline_pdf(self, source: SourceDocument) -> ParseResult:
        base = self._call_capabilities(
            ("core_python", "pymupdf"),
            "native",
            parse_pymupdf,
            source,
            self._context(
                "baseline/pymupdf",
                self.config.fast_timeout_seconds,
                options={"version": self._capability_version("pymupdf")},
            ),
        )
        preflight = self._preflight(base)
        page_count = self._page_count(base)
        scan_pages = {
            item["page"]
            for item in preflight
            if item.get("classification") == "scan"
        }
        attempts = [base]
        selected = base
        if scan_pages:
            ocr_results: List[ParseResult] = []
            for page in sorted(scan_pages):
                ocr_results.append(
                    self._call(
                        "tesseract",
                        parse_tesseract,
                        source,
                        self._context(
                            "baseline/tesseract",
                            self.config.tesseract_timeout_seconds,
                            command=self.commands.tesseract(),
                            options={
                                "version": self._approved_version(
                                    "tesseract",
                                    "tesseract",
                                ),
                                "page": page,
                            },
                        ),
                    )
                )
            attempts.extend(ocr_results)
            selected = self._compose_pages(
                source,
                page_count,
                default=base,
                overrides=[
                    (set([page]), result)
                    for page, result in zip(sorted(scan_pages), ocr_results)
                    if result.blocks
                ],
                results=attempts,
            )
        if any(item.get("has_tables") for item in preflight):
            tables = self._call_capabilities(
                ("core_python", "pdfplumber"),
                "native",
                parse_pdfplumber_tables,
                source,
                self._context(
                    "baseline/pdfplumber",
                    self.config.fast_timeout_seconds,
                    options={
                        "version": self._capability_version("pdfplumber")
                    },
                ),
            )
            selected = overlay_pdf_tables(
                source, selected, tables, "baseline/pdf-composite"
            )
        return selected

    def _challenger(self, source: SourceDocument, format_name: str) -> ParseResult:
        if format_name in {"hwp", "hwpx"}:
            return self._chain(
                source,
                [
                    (
                        "unhwp",
                        lambda: self._call(
                            "native",
                            parse_unhwp,
                            source,
                            self._context(
                                "challenger/unhwp",
                                self.config.fast_timeout_seconds,
                                command=self.commands.unhwp(),
                                options={
                                    "version": self._approved_version(
                                        "unhwp",
                                        "unhwp",
                                    )
                                },
                            ),
                        ),
                    )
                ],
            )
        if format_name == "image":
            return self._call(
                "paddle",
                parse_paddle,
                source,
                self._context(
                    "challenger/pp-structurev3",
                    self.config.heavy_timeout_seconds,
                    command=self.commands.paddle("challenger"),
                    options={
                        "version": self._approved_version(
                            "paddleocr",
                            "paddleocr",
                        ),
                        "model": self._approved_version(
                            "pp_ocrv5_korean_model",
                            "pp_ocr_korean",
                        ),
                        "device": "cpu",
                    },
                ),
            )
        if format_name == "pdf":
            return self._challenger_pdf(source)
        return ParseResult(source)

    def _challenger_pdf(self, source: SourceDocument) -> ParseResult:
        preflight_result = self._call_capabilities(
            ("core_python", "pymupdf"),
            "native",
            parse_pymupdf,
            source,
            self._context(
                "challenger/pymupdf-preflight",
                self.config.fast_timeout_seconds,
                options={"version": self._capability_version("pymupdf")},
            ),
        )
        docling = self._call(
            "docling",
            parse_docling,
            source,
            self._context(
                "challenger/docling",
                self.config.heavy_timeout_seconds,
                command=self.commands.docling("challenger"),
                options={
                    "version": self._approved_version(
                        "docling",
                        "docling",
                    )
                },
            ),
        )
        preflight = self._preflight(preflight_result)
        scan_pages = {
            item["page"]
            for item in preflight
            if item.get("classification") == "scan"
        }
        results = [preflight_result, docling]
        selected = docling
        if scan_pages:
            paddle = self._call(
                "paddle",
                parse_paddle,
                source,
                self._context(
                    "challenger/pp-structurev3",
                    self.config.heavy_timeout_seconds,
                    command=self.commands.paddle("challenger"),
                    options={
                        "version": self._approved_version(
                            "paddleocr",
                            "paddleocr",
                        ),
                        "model": self._approved_version(
                            "pp_ocrv5_korean_model",
                            "pp_ocr_korean",
                        ),
                        "device": "cpu",
                    },
                ),
            )
            results.append(paddle)
            if self._has_page_labels(docling) or self._has_page_labels(paddle):
                selected = self._compose_pages(
                    source,
                    self._page_count(preflight_result),
                    default=docling if docling.blocks else preflight_result,
                    overrides=[(scan_pages, paddle)] if paddle.blocks else [],
                    results=results,
                )
        assessment = self._assess(
            selected,
            expected_pages=self._page_count(preflight_result),
            table_signal=any(item.get("has_tables") for item in preflight),
        )
        if assessment.hard_fail:
            return self._chain_from_results(
                source, [selected, preflight_result], expected_pages=self._page_count(preflight_result)
            )
        return self._aggregate(source, selected, results)

    def _cascade(self, source: SourceDocument, format_name: str) -> ParseResult:
        if format_name in {"hwp", "hwpx"}:
            java_name = "hwpxlib" if format_name == "hwpx" else "hwplib"
            return self._chain(
                source,
                [
                    (
                        java_name,
                        lambda: self._call(
                            "java",
                            parse_java_hwp,
                            source,
                            self._context(
                                "cascade/{}".format(java_name),
                                self.config.fast_timeout_seconds,
                                command=self.commands.java_worker("cascade"),
                                options={
                                    "version": self._approved_version(
                                        "{}_jar".format(java_name),
                                        java_name,
                                    )
                                },
                            ),
                        ),
                    ),
                    (
                        "unhwp",
                        lambda: self._call(
                            "native",
                            parse_unhwp,
                            source,
                            self._context(
                                "cascade/unhwp",
                                self.config.fast_timeout_seconds,
                                command=self.commands.unhwp(),
                                options={
                                    "version": self._approved_version(
                                        "unhwp",
                                        "unhwp",
                                    )
                                },
                            ),
                        ),
                    ),
                    (
                        "tika",
                        lambda: self._call(
                            "java",
                            parse_tika,
                            source,
                            self._context(
                                "cascade/tika",
                                self.config.fast_timeout_seconds,
                                command=self.commands.tika(),
                                options={
                                    "version": self._approved_version(
                                        "tika_jar",
                                        "tika",
                                    )
                                },
                            ),
                        ),
                    ),
                ],
            )
        if format_name == "image":
            return self._chain(
                source,
                [
                    (
                        "paddle",
                        lambda: self._call(
                            "paddle",
                            parse_paddle,
                            source,
                            self._context(
                                "cascade/pp-structurev3",
                                self.config.heavy_timeout_seconds,
                                command=self.commands.paddle("cascade"),
                                options={
                                    "version": self._approved_version(
                                        "paddleocr",
                                        "paddleocr",
                                    ),
                                    "model": self._approved_version(
                                        "pp_ocrv5_korean_model",
                                        "pp_ocr_korean",
                                    ),
                                    "device": "cpu",
                                },
                            ),
                        ),
                    ),
                    (
                        "tesseract",
                        lambda: self._call(
                            "tesseract",
                            parse_tesseract,
                            source,
                            self._context(
                                "cascade/tesseract",
                                self.config.tesseract_timeout_seconds,
                                command=self.commands.tesseract(),
                                options={
                                    "version": self._approved_version(
                                        "tesseract",
                                        "tesseract",
                                    )
                                },
                            ),
                        ),
                    ),
                ],
            )
        if format_name == "pdf":
            return self._cascade_pdf(source)
        return ParseResult(source)

    def _cascade_pdf(self, source: SourceDocument) -> ParseResult:
        pymupdf = self._call_capabilities(
            ("core_python", "pymupdf"),
            "native",
            parse_pymupdf,
            source,
            self._context(
                "cascade/pymupdf",
                self.config.fast_timeout_seconds,
                options={"version": self._capability_version("pymupdf")},
            ),
        )
        preflight = self._preflight(pymupdf)
        page_count = self._page_count(pymupdf)
        scan_pages = {
            item["page"] for item in preflight if item.get("classification") == "scan"
        }
        structured_pages = {
            item["page"]
            for item in preflight
            if item.get("classification") in {"complex", "table"}
        }
        has_tables = any(item.get("has_tables") for item in preflight)
        results: List[ParseResult] = [pymupdf]
        overrides: List[Tuple[set, ParseResult]] = []

        if structured_pages or not pymupdf.blocks:
            docling = self._call(
                "docling",
                parse_docling,
                source,
                self._context(
                    "cascade/docling",
                    self.config.heavy_timeout_seconds,
                    command=self.commands.docling("cascade"),
                    options={
                        "version": self._approved_version(
                            "docling",
                            "docling",
                        )
                    },
                ),
            )
            results.append(docling)
            if structured_pages:
                overrides.append((structured_pages, docling))
            elif docling.blocks:
                all_pages = (
                    set(range(1, page_count + 1)) if page_count else set()
                )
                overrides.append((all_pages, docling))

        if scan_pages:
            paddle = self._call(
                "paddle",
                parse_paddle,
                source,
                self._context(
                    "cascade/pp-structurev3",
                    self.config.heavy_timeout_seconds,
                    command=self.commands.paddle("cascade"),
                    options={
                        "version": self._approved_version(
                            "paddleocr",
                            "paddleocr",
                        ),
                        "model": self._approved_version(
                            "pp_ocrv5_korean_model",
                            "pp_ocr_korean",
                        ),
                        "device": "cpu",
                    },
                ),
            )
            results.append(paddle)
            overrides.append((scan_pages, paddle))

        selected = self._compose_pages(
            source,
            page_count,
            default=pymupdf,
            overrides=overrides,
            results=results,
        )
        if has_tables:
            tables = self._call_capabilities(
                ("core_python", "pdfplumber"),
                "native",
                parse_pdfplumber_tables,
                source,
                self._context(
                    "cascade/pdfplumber",
                    self.config.fast_timeout_seconds,
                    options={
                        "version": self._capability_version("pdfplumber")
                    },
                ),
            )
            results.append(tables)
            selected = overlay_pdf_tables(
                source, selected, tables, "cascade/pdf-composite"
            )

        assessment = self._assess(
            selected,
            expected_pages=page_count,
            table_signal=has_tables,
        )
        if assessment.hard_fail or assessment.suspect:
            pypdf = self._call_capabilities(
                ("core_python", "pypdf"),
                "native",
                parse_pypdf,
                source,
                self._context(
                    "cascade/pypdf",
                    self.config.fast_timeout_seconds,
                    options={"version": self._capability_version("pypdf")},
                ),
            )
            results.append(pypdf)
            candidates = [selected, pypdf]
            selected = max(
                candidates,
                key=lambda result: (
                    self._assess(result, expected_pages=page_count).score,
                    self._non_whitespace(result),
                ),
            )

        aggregate = self._aggregate(source, selected, results)
        final_assessment = self._assess(
            aggregate, expected_pages=page_count, table_signal=has_tables
        )
        if final_assessment.hard_fail and self.config.enable_vl_review:
            review = self._call_capability(
                "paddleocr_vl_model",
                "paddle",
                parse_subprocess,
                source,
                self._context(
                    "cascade/paddleocr-vl-review",
                    self.config.heavy_timeout_seconds,
                    options={
                        "profile": "cascade",
                        "model": "PaddleOCR-VL",
                        "allow_custom_command": True,
                        "runtime_override": "custom_review_only",
                    },
                ),
                default_parser="cascade/paddleocr-vl-review",
                command_env="PARSER_PADDLE_VL_CMD",
            )
            # Review output is deliberately not selected or indexed.
            aggregate.attempts.extend(review.attempts)
            aggregate.raw_artifacts.extend(review.raw_artifacts)
        return aggregate

    def _chain(
        self,
        source: SourceDocument,
        attempts: Sequence[Tuple[str, Callable[[], ParseResult]]],
    ) -> ParseResult:
        results: List[ParseResult] = []
        selected: Optional[ParseResult] = None
        best_suspect: Optional[Tuple[float, ParseResult]] = None
        for _, run in attempts:
            result = run()
            assessment = self._assess(result)
            self._annotate_selected_attempt(result, assessment)
            results.append(result)
            if not assessment.hard_fail and not assessment.suspect:
                selected = result
                break
            if not assessment.hard_fail:
                if best_suspect is None or assessment.score > best_suspect[0]:
                    best_suspect = (assessment.score, result)
        if selected is None and best_suspect is not None:
            selected = best_suspect[1]
        if selected is None:
            selected = max(results, key=self._non_whitespace)
        return self._aggregate(source, selected, results)

    def _chain_from_results(
        self,
        source: SourceDocument,
        results: Sequence[ParseResult],
        expected_pages: Optional[int] = None,
    ) -> ParseResult:
        selected = max(
            results,
            key=lambda result: (
                self._assess(result, expected_pages=expected_pages).score,
                self._non_whitespace(result),
            ),
        )
        for result in results:
            self._annotate_selected_attempt(
                result, self._assess(result, expected_pages=expected_pages)
            )
        return self._aggregate(source, selected, results)

    def _finish(
        self,
        source: SourceDocument,
        format_name: str,
        mime_type: str,
        result: ParseResult,
    ) -> PipelineOutcome:
        if len(result.blocks) > self.config.max_blocks_per_document:
            result = ParseResult(
                source,
                attempts=list(result.attempts)
                + [
                    Attempt(
                        parser="{}/router".format(self.config.profile),
                        status="error",
                        reason="block_limit_exceeded:{}>{}".format(
                            len(result.blocks),
                            self.config.max_blocks_per_document,
                        ),
                        error_type="BlockLimitError",
                    )
                ],
                raw_artifacts=list(result.raw_artifacts),
            )
        normalized = ParseResult(
            source,
            blocks=self._normalize_blocks(result.blocks, source),
            attempts=list(result.attempts),
            raw_artifacts=list(dict.fromkeys(str(item) for item in result.raw_artifacts)),
        )
        schema_errors: List[str] = []
        try:
            normalized.validate()
        except (TypeError, ValueError) as exc:
            schema_errors.append(str(exc))
        expected_pages = self._page_count(normalized)
        table_signal = any(
            item.get("has_tables") for item in self._preflight(normalized)
        )
        quality = self._assess(
            normalized,
            expected_pages=expected_pages,
            table_signal=table_signal,
            schema_errors=schema_errors,
        )
        self._annotate_selected_attempt(normalized, quality)
        parsers = tuple(dict.fromkeys(block.parser for block in normalized.blocks))
        if quality.hard_fail:
            selected_attempt = self._selected_attempt(normalized)
            if normalized.blocks:
                status = "error"
            elif selected_attempt is not None and selected_attempt.status in {
                "error",
                "timeout",
            }:
                status = "error"
            else:
                status = "empty"
            reason = ";".join(quality.hard_fail_reasons)
        else:
            status = "parsed"
            reason = ";".join(quality.suspect_reasons)
        return PipelineOutcome(
            source=source,
            sniffed_format=format_name,
            mime_type=mime_type,
            result=normalized,
            quality=quality,
            status=status,
            reason=reason,
            selected_parsers=parsers,
        )

    def _context(
        self,
        parser: str,
        timeout: int,
        command: Optional[Sequence[str]] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> AdapterContext:
        values = {
            "profile": self.config.profile,
            "max_blocks": self.config.max_blocks_per_document,
            "managed_runtime": self.config.runtime_report is not None,
        }
        values.update(dict(options or {}))
        return AdapterContext(
            parser=parser,
            timeout_seconds=timeout,
            env=self.commands.worker_env(),
            raw_output_dir=self.config.raw_output_dir,
            command=command,
            options=values,
        )

    def _capability(
        self,
        name: Optional[str],
    ) -> Optional[Mapping[str, Any]]:
        if name is None or self.config.runtime_report is None:
            return None
        value = (
            self.config.runtime_report.get("capabilities", {}).get(name)
        )
        return value if isinstance(value, Mapping) else None

    def _capability_available(self, name: Optional[str]) -> bool:
        if name is None or self.config.runtime_report is None:
            return True
        capability = self._capability(name)
        return bool(capability and capability.get("available"))

    def _capability_version(self, name: Optional[str]) -> Optional[str]:
        capability = self._capability(name)
        if not capability or not capability.get("available"):
            return None
        version = capability.get("version")
        return str(version) if version else None

    def _pin(self, name: str) -> str:
        if self.config.runtime_report is not None:
            value = self.config.runtime_report.get("pins", {}).get(name)
            if value:
                return str(value)
        return PROFILE_PINS[name]

    def _approved_version(self, capability: str, pin: str) -> str:
        return self._capability_version(capability) or self._pin(pin)

    def _call_capability(
        self,
        capability_name: Optional[str],
        limit: str,
        function: Callable[..., ParseResult],
        source: SourceDocument,
        context: AdapterContext,
        **kwargs: Any
    ) -> ParseResult:
        return self._call_capabilities(
            (capability_name,) if capability_name is not None else (),
            limit,
            function,
            source,
            context,
            **kwargs
        )

    def _call_capabilities(
        self,
        capability_names: Sequence[str],
        limit: str,
        function: Callable[..., ParseResult],
        source: SourceDocument,
        context: AdapterContext,
        **kwargs: Any
    ) -> ParseResult:
        unavailable_name = next(
            (
                name
                for name in capability_names
                if not self._capability_available(name)
            ),
            None,
        )
        if unavailable_name is None:
            return self._call(
                limit,
                function,
                source,
                context,
                **kwargs
            )
        capability = self._capability(unavailable_name) or {}
        return ParseResult(
            source,
            attempts=[
                Attempt(
                    parser=context.parser or str(capability_name),
                    status="unavailable",
                    reason=(
                        "runtime capability {!r} was not approved by "
                        "doctor: {}"
                    ).format(
                        unavailable_name,
                        capability.get("detail", "not available"),
                    ),
                    metadata={
                        "capability": unavailable_name,
                        "doctor_location": capability.get("location"),
                        "doctor_version": capability.get("version"),
                    },
                )
            ],
        )

    def _call(
        self,
        limit: str,
        function: Callable[..., ParseResult],
        source: SourceDocument,
        context: AdapterContext,
        **kwargs: Any
    ) -> ParseResult:
        semaphore = self._limits.get(limit)
        if semaphore is None:
            return function(source, context, **kwargs)
        with semaphore:
            return function(source, context, **kwargs)

    def _assess(
        self,
        result: ParseResult,
        expected_pages: Optional[int] = None,
        table_signal: bool = False,
        schema_errors: Iterable[str] = (),
    ) -> QualityAssessment:
        attempt = self._selected_attempt(result)
        return assess_quality(
            blocks=result.blocks,
            expected_page_count=expected_pages,
            table_signal=table_signal,
            expect_korean=self.config.expect_korean,
            attempt=attempt,
            schema_errors=schema_errors,
            min_non_whitespace_chars=self.config.min_chars,
        )

    @staticmethod
    def _selected_attempt(result: ParseResult) -> Optional[Attempt]:
        """Return the attempt responsible for the selected canonical blocks.

        Aggregate results retain every fallback and enrichment attempt.  The
        final attempt can therefore be an unselected pdfplumber/Paddle/pypdf
        failure, which must not invalidate otherwise good selected content.
        """

        if not result.attempts:
            return None
        selected_keys = {
            (block.parser, block.page)
            for block in result.blocks
        }
        selected_parsers = {parser for parser, _ in selected_keys}
        if selected_keys:
            for attempt in reversed(result.attempts):
                attempt_page = attempt.metadata.get("page")
                if (
                    attempt_page is not None
                    and (attempt.parser, attempt_page) in selected_keys
                ):
                    return attempt
                if attempt_page is None and attempt.parser in selected_parsers:
                    return attempt
        return result.attempts[-1]

    def _annotate_selected_attempt(
        self, result: ParseResult, assessment: QualityAssessment
    ) -> None:
        attempt = self._selected_attempt(result)
        if attempt is None:
            return
        attempt_index = next(
            (
                index
                for index in range(len(result.attempts) - 1, -1, -1)
                if result.attempts[index] is attempt
            ),
            len(result.attempts) - 1,
        )
        metadata = dict(attempt.metadata)
        metadata["quality"] = assessment.to_dict()
        status = attempt.status
        reason = attempt.reason
        fallback_reason = attempt.fallback_reason
        if attempt.status == "success" and assessment.suspect:
            status = "suspect"
            fallback_reason = ";".join(assessment.suspect_reasons)
        elif attempt.status == "success" and assessment.hard_fail:
            status = "empty"
            fallback_reason = ";".join(assessment.hard_fail_reasons)
        if fallback_reason and not reason:
            reason = fallback_reason
        result.attempts[attempt_index] = replace(
            attempt,
            status=status,
            reason=reason,
            metadata=metadata,
            fallback_reason=fallback_reason,
        )

    def _normalize_blocks(
        self, blocks: Sequence[Block], source: SourceDocument
    ) -> List[Block]:
        table_ids: Dict[str, str] = {}
        normalized: List[Block] = []
        # Composition already establishes document order. Adapter-local
        # reading_order values commonly restart at zero for each PDF page, so
        # sorting here would interleave pages and undo the router's ordering.
        for order, block in enumerate(blocks):
            table_id = None
            if block.table_id is not None:
                key = str(block.table_id)
                if key not in table_ids:
                    table_ids[key] = make_table_id(
                        source.document_id, len(table_ids), self.config.profile
                    )
                table_id = table_ids[key]
            normalized.append(
                Block(
                    document_id=source.document_id,
                    block_id=make_block_id(
                        source.document_id, order, self.config.profile
                    ),
                    block_type=block.block_type,
                    text=block.text,
                    reading_order=order,
                    parser=block.parser,
                    page=block.page,
                    section_path=block.section_path,
                    table_id=table_id,
                    row=block.row,
                    column=block.column,
                )
            )
        return normalized

    @staticmethod
    def _aggregate(
        source: SourceDocument,
        selected: ParseResult,
        results: Sequence[ParseResult],
    ) -> ParseResult:
        attempts: List[Attempt] = []
        raw: List[Any] = []
        seen_attempts = set()
        for result in results:
            for attempt in result.attempts:
                identity = id(attempt)
                if identity not in seen_attempts:
                    seen_attempts.add(identity)
                    attempts.append(attempt)
            raw.extend(result.raw_artifacts)
        return ParseResult(
            source,
            blocks=list(selected.blocks),
            attempts=attempts,
            raw_artifacts=list(dict.fromkeys(str(item) for item in raw)),
        )

    @staticmethod
    def _page_count(result: ParseResult) -> Optional[int]:
        for attempt in result.attempts:
            value = attempt.metadata.get("page_count")
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass
            preflight = attempt.metadata.get("page_preflight")
            if isinstance(preflight, list) and preflight:
                return len(preflight)
        pages = [block.page for block in result.blocks if block.page is not None]
        return max(pages) if pages else None

    @staticmethod
    def _preflight(result: ParseResult) -> List[Mapping[str, Any]]:
        for attempt in result.attempts:
            value = attempt.metadata.get("page_preflight")
            if isinstance(value, list):
                return [
                    item for item in value if isinstance(item, Mapping)
                ]
        return []

    @staticmethod
    def _has_page_labels(result: ParseResult) -> bool:
        return bool(result.blocks) and any(
            block.page is not None for block in result.blocks
        )

    @staticmethod
    def _non_whitespace(result: ParseResult) -> int:
        return sum(
            sum(not character.isspace() for character in block.text)
            for block in result.blocks
            if block.block_type != "table_cell"
        )

    def _compose_pages(
        self,
        source: SourceDocument,
        page_count: Optional[int],
        default: ParseResult,
        overrides: Sequence[Tuple[set, ParseResult]],
        results: Sequence[ParseResult],
    ) -> ParseResult:
        override_results = [
            result for _, result in overrides if result.blocks
        ]
        unlabeled = [
            result
            for result in [default] + override_results
            if result.blocks and not self._has_page_labels(result)
        ]
        if unlabeled:
            # Page-level replacement is unsafe when a worker emits a complete
            # document without provenance (Docling's text fallback can do
            # this). Select one whole-document candidate instead of silently
            # dropping every page=None block.
            complete_pages = (
                set(range(1, page_count + 1)) if page_count else set()
            )
            if default.blocks and not self._has_page_labels(default):
                whole_document_candidates = [default]
                whole_document_candidates.extend(
                    result
                    for result in override_results
                    if complete_pages
                    and {
                        block.page
                        for block in result.blocks
                        if block.page is not None
                    }
                    >= complete_pages
                )
            else:
                whole_document_candidates = [default] + unlabeled
            candidate_ids = list(
                dict.fromkeys(id(result) for result in whole_document_candidates)
            )
            by_identity = {
                id(result): result for result in whole_document_candidates
            }
            selected = max(
                (by_identity[identity] for identity in candidate_ids),
                key=lambda result: (
                    self._assess(
                        result, expected_pages=page_count
                    ).score,
                    self._non_whitespace(result),
                ),
            )
            return self._aggregate(source, selected, results)
        if not page_count:
            selected = next(
                (result for _, result in reversed(overrides) if result.blocks),
                default,
            )
            return self._aggregate(source, selected, results)
        blocks: List[Block] = []
        for page in range(1, page_count + 1):
            selected = default
            for pages, candidate in overrides:
                if page in pages and candidate.blocks:
                    selected = candidate
            page_blocks = [
                block for block in selected.blocks if block.page == page
            ]
            if not page_blocks and selected is not default:
                page_blocks = [
                    block for block in default.blocks if block.page == page
                ]
            blocks.extend(page_blocks)
        if not blocks:
            blocks = list(default.blocks)
        temporary = ParseResult(source, blocks=blocks)
        return self._aggregate(source, temporary, results)
