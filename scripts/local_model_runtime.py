"""Own the lifecycle of a lazy local MLX-LM server process.

The stock ``mlx_lm.server`` keeps the most recently used model and its prompt
cache alive until the whole process exits.  This runtime deliberately owns a
child server process so an explicit unload can return memory to macOS by
terminating only that child.  The next local generation starts a fresh child
and loads the requested model on demand.
"""

from __future__ import annotations

import ipaddress
import json
import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Mapping, Sequence
from urllib.parse import urlparse, urlunparse


DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_MLX_SERVER_EXECUTABLE = "mlx_lm.server"
DEFAULT_STARTUP_TIMEOUT_SECONDS = 15.0
DEFAULT_TERMINATE_TIMEOUT_SECONDS = 10.0
WORKER_ENV_ALLOWLIST = frozenset(
    {
        "PATH",
        "HOME",
        "TMPDIR",
        "USER",
        "LOGNAME",
        "LANG",
        "SHELL",
        "VIRTUAL_ENV",
        "XDG_CACHE_HOME",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "NO_PROXY",
        "HF_HOME",
        "HF_ENDPOINT",
        "HF_HUB_CACHE",
        "HF_HUB_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "TRANSFORMERS_OFFLINE",
        "TOKENIZERS_PARALLELISM",
        "MLX_METAL_CACHE_DIR",
        "PYTHONNOUSERSITE",
        "PYTHONUNBUFFERED",
        "SYSTEMROOT",
        "COMSPEC",
        "PATHEXT",
    }
)


class LocalModelRuntimeError(RuntimeError):
    """Safe runtime failure with a stable public error code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class LocalModelBusyError(LocalModelRuntimeError):
    def __init__(self) -> None:
        super().__init__("local_model_busy")


class LocalModelUnmanagedError(LocalModelRuntimeError):
    def __init__(self) -> None:
        super().__init__("local_model_unload_not_supported")


def _env_float(
    name: str,
    default: float,
    minimum: float = 0.1,
    *,
    environment: Mapping[str, str] | None = None,
) -> float:
    values = os.environ if environment is None else environment
    try:
        value = float(values.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


def _env_int(
    name: str,
    default: int,
    minimum: int = 1,
    *,
    environment: Mapping[str, str] | None = None,
) -> int:
    values = os.environ if environment is None else environment
    try:
        value = int(values.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


def _is_loopback_host(hostname: str | None) -> bool:
    if not hostname:
        return False
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _default_managed_mode(base_url: str) -> bool:
    parsed = urlparse(base_url)
    try:
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme == "http"
        and _is_loopback_host(parsed.hostname)
        and port == 8080
    )


def _health_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            "/health",
            "",
            "",
            "",
        )
    )


def _http_ready(url: str, timeout: float) -> bool:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= int(response.status) < 300
    except (
        TimeoutError,
        socket.timeout,
        urllib.error.HTTPError,
        urllib.error.URLError,
        OSError,
    ):
        return False


def _worker_environment(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    values = os.environ if environment is None else environment
    return {
        key: value
        for key, value in values.items()
        if key in WORKER_ENV_ALLOWLIST or key.startswith("LC_")
    }


class ExternalLocalModelRuntime:
    """No-op lifecycle for Ollama or another externally managed endpoint."""

    managed = False

    def public_status(self) -> dict[str, Any]:
        return {
            "runtime_state": "external",
            "loaded_model": None,
            "unload_supported": False,
            "worker_running": False,
        }

    @contextmanager
    def generation(self, model: str | None = None) -> Iterator[None]:
        del model
        yield

    def note_loaded(self, model: str | None) -> None:
        del model

    def unload(self) -> dict[str, Any]:
        raise LocalModelUnmanagedError()

    def close(self) -> None:
        return


class ManagedLocalModelRuntime:
    """Start, serialize, and explicitly stop one owned MLX-LM child."""

    managed = True

    def __init__(
        self,
        *,
        base_url: str,
        executable: str = DEFAULT_MLX_SERVER_EXECUTABLE,
        worker_args: Sequence[str] = (),
        startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
        terminate_timeout_seconds: float = DEFAULT_TERMINATE_TIMEOUT_SECONDS,
        popen_factory: Callable[..., Any] = subprocess.Popen,
        readiness_probe: Callable[[str, float], bool] = _http_ready,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        worker_environment: Mapping[str, str] | None = None,
    ) -> None:
        parsed = urlparse(base_url)
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("RAG_LOCAL_BASE_URL has an invalid port") from exc
        if (
            parsed.scheme != "http"
            or not _is_loopback_host(parsed.hostname)
            or port is None
        ):
            raise ValueError(
                "managed MLX runtime requires an http loopback RAG_LOCAL_BASE_URL"
            )

        self.base_url = base_url.rstrip("/")
        self.health_url = _health_url(base_url)
        self.host = parsed.hostname or "127.0.0.1"
        self.port = port
        self.executable = executable
        self.worker_args = tuple(str(value) for value in worker_args)
        self.startup_timeout_seconds = max(0.1, startup_timeout_seconds)
        self.terminate_timeout_seconds = max(0.1, terminate_timeout_seconds)
        self._popen_factory = popen_factory
        self._readiness_probe = readiness_probe
        self._monotonic = monotonic
        self._sleep = sleep
        self._worker_environment = _worker_environment(worker_environment)

        self._lock = threading.RLock()
        self._generation_lock = threading.Lock()
        self._process: Any | None = None
        self._active_generations = 0
        self._loaded_model: str | None = None
        self._state = "unloaded"
        self._external_conflict = False
        self._last_error: str | None = None

    def _command(self) -> list[str]:
        executable = shutil.which(self.executable) or self.executable
        return [
            executable,
            "--host",
            self.host,
            "--port",
            str(self.port),
            *self.worker_args,
        ]

    def _refresh_process_locked(self) -> None:
        if self._process is None:
            return
        return_code = self._process.poll()
        if return_code is None:
            return
        self._process = None
        self._active_generations = 0
        self._loaded_model = None
        self._state = "error"
        self._last_error = "local_model_process_exited"

    def _terminate_process_locked(self) -> bool:
        process = self._process
        if process is None:
            return False
        process.terminate()
        try:
            process.wait(timeout=self.terminate_timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=self.terminate_timeout_seconds)
        self._process = None
        self._active_generations = 0
        self._loaded_model = None
        return True

    def _ensure_started_locked(self) -> None:
        self._refresh_process_locked()
        if self._process is not None:
            return
        if self._readiness_probe(self.health_url, 0.2):
            self._external_conflict = True
            self._state = "external"
            self._last_error = "local_model_runtime_unmanaged"
            raise LocalModelUnmanagedError()

        self._external_conflict = False
        self._state = "starting"
        self._last_error = None
        try:
            self._process = self._popen_factory(
                self._command(),
                env=dict(self._worker_environment),
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        except (OSError, ValueError) as exc:
            self._state = "error"
            self._last_error = "local_model_start_failed"
            raise LocalModelRuntimeError("local_model_start_failed") from exc

        deadline = self._monotonic() + self.startup_timeout_seconds
        while self._monotonic() < deadline:
            self._refresh_process_locked()
            if self._process is None:
                raise LocalModelRuntimeError(
                    self._last_error or "local_model_start_failed"
                )
            if self._readiness_probe(self.health_url, 0.25):
                self._state = "ready"
                return
            self._sleep(0.05)

        self._terminate_process_locked()
        self._state = "error"
        self._last_error = "local_model_start_timeout"
        raise LocalModelRuntimeError("local_model_start_timeout")

    def public_status(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_process_locked()
            if self._process is None:
                if self._readiness_probe(self.health_url, 0.2):
                    self._external_conflict = True
                    self._state = "external"
                    self._last_error = "local_model_runtime_unmanaged"
                elif self._external_conflict:
                    self._external_conflict = False
                    self._state = "unloaded"
                    self._last_error = None
            running = self._process is not None
            state = self._state
            if self._external_conflict:
                state = "external"
            elif self._active_generations > 0:
                state = "loading" if self._loaded_model is None else "busy"
            elif running and self._loaded_model:
                state = "loaded"
            elif running and state not in {"starting", "error"}:
                state = "ready"
            elif not running and state not in {"error", "external"}:
                state = "unloaded"
            return {
                "runtime_state": state,
                "loaded_model": self._loaded_model,
                "unload_supported": not self._external_conflict,
                "worker_running": running,
                **(
                    {"runtime_reason": self._last_error}
                    if self._last_error
                    else {}
                ),
            }

    @contextmanager
    def generation(self, model: str | None = None) -> Iterator[None]:
        self._generation_lock.acquire()
        entered = False
        try:
            with self._lock:
                self._ensure_started_locked()
                self._active_generations += 1
                entered = True
                if self._loaded_model != model:
                    self._state = "loading"
            try:
                yield
            finally:
                if entered:
                    with self._lock:
                        self._active_generations = max(
                            0,
                            self._active_generations - 1,
                        )
                        if self._process is not None:
                            self._state = (
                                "loaded" if self._loaded_model else "ready"
                            )
        finally:
            self._generation_lock.release()

    def note_loaded(self, model: str | None) -> None:
        normalized = str(model or "").strip()
        if not normalized:
            return
        with self._lock:
            self._refresh_process_locked()
            if self._process is not None:
                self._loaded_model = normalized
                self._state = "loaded"

    def unload(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_process_locked()
            if self._active_generations > 0:
                raise LocalModelBusyError()
            if self._process is None:
                if self._readiness_probe(self.health_url, 0.2):
                    self._external_conflict = True
                    self._state = "external"
                    self._last_error = "local_model_runtime_unmanaged"
                    raise LocalModelUnmanagedError()
                self._external_conflict = False
                self._state = "unloaded"
                self._loaded_model = None
                self._last_error = None
                return {
                    "ok": True,
                    "state": "unloaded",
                    "loaded_model": None,
                    "released": False,
                }

            self._state = "stopping"
            try:
                released = self._terminate_process_locked()
            except (OSError, subprocess.SubprocessError) as exc:
                self._state = "error"
                self._last_error = "local_model_stop_failed"
                raise LocalModelRuntimeError("local_model_stop_failed") from exc
            self._state = "unloaded"
            self._last_error = None
            return {
                "ok": True,
                "state": "unloaded",
                "loaded_model": None,
                "released": released,
            }

    def close(self) -> None:
        with self._lock:
            if self._process is None:
                return
            try:
                self._terminate_process_locked()
            except (OSError, subprocess.SubprocessError):
                pass
            self._state = "unloaded"
            self._last_error = None


def build_local_model_runtime(
    environment: Mapping[str, str] | None = None,
) -> ExternalLocalModelRuntime | ManagedLocalModelRuntime:
    values = os.environ if environment is None else environment
    base_url = values.get("RAG_LOCAL_BASE_URL", DEFAULT_LOCAL_BASE_URL).strip()
    explicit_mode = values.get("RAG_LOCAL_RUNTIME", "").strip().lower()
    if explicit_mode:
        if explicit_mode in {"managed", "managed_mlx", "mlx"}:
            managed = True
        elif explicit_mode in {"external", "unmanaged"}:
            managed = False
        else:
            raise ValueError(
                "RAG_LOCAL_RUNTIME must be managed_mlx or external"
            )
    else:
        managed = _default_managed_mode(base_url)
    if not managed:
        return ExternalLocalModelRuntime()

    max_tokens = _env_int(
        "RAG_LOCAL_MAX_OUTPUT_TOKENS",
        900,
        environment=values,
    )
    decode_concurrency = _env_int(
        "RAG_LOCAL_DECODE_CONCURRENCY",
        1,
        environment=values,
    )
    prompt_concurrency = _env_int(
        "RAG_LOCAL_PROMPT_CONCURRENCY",
        1,
        environment=values,
    )
    prompt_cache_size = _env_int(
        "RAG_LOCAL_PROMPT_CACHE_SIZE",
        1,
        environment=values,
    )
    raw_chat_template_args = values.get(
        "RAG_LOCAL_CHAT_TEMPLATE_ARGS",
        '{"enable_thinking":false}',
    ).strip()
    try:
        chat_template_args = json.loads(raw_chat_template_args)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "RAG_LOCAL_CHAT_TEMPLATE_ARGS must be a JSON object"
        ) from exc
    if not isinstance(chat_template_args, dict):
        raise ValueError(
            "RAG_LOCAL_CHAT_TEMPLATE_ARGS must be a JSON object"
        )
    executable = values.get(
        "RAG_LOCAL_SERVER_EXECUTABLE",
        DEFAULT_MLX_SERVER_EXECUTABLE,
    ).strip() or DEFAULT_MLX_SERVER_EXECUTABLE
    startup_timeout = _env_float(
        "RAG_LOCAL_STARTUP_TIMEOUT_SECONDS",
        DEFAULT_STARTUP_TIMEOUT_SECONDS,
        environment=values,
    )
    terminate_timeout = _env_float(
        "RAG_LOCAL_TERMINATE_TIMEOUT_SECONDS",
        DEFAULT_TERMINATE_TIMEOUT_SECONDS,
        environment=values,
    )
    return ManagedLocalModelRuntime(
        base_url=base_url,
        executable=executable,
        worker_args=(
            "--max-tokens",
            str(max_tokens),
            "--decode-concurrency",
            str(decode_concurrency),
            "--prompt-concurrency",
            str(prompt_concurrency),
            "--prompt-cache-size",
            str(prompt_cache_size),
            "--chat-template-args",
            json.dumps(
                chat_template_args,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        ),
        startup_timeout_seconds=startup_timeout,
        terminate_timeout_seconds=terminate_timeout,
    )
