from __future__ import annotations

import subprocess
import sys
import threading
import time
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from local_model_runtime import (
    ExternalLocalModelRuntime,
    LocalModelBusyError,
    LocalModelRuntimeError,
    LocalModelUnmanagedError,
    ManagedLocalModelRuntime,
    build_local_model_runtime,
)


class FakeProcess:
    def __init__(
        self,
        *,
        wait_requires_kill: bool = False,
        wait_fails_after_kill: bool = False,
    ) -> None:
        self.return_code: int | None = None
        self.terminated = False
        self.killed = False
        self.wait_requires_kill = wait_requires_kill
        self.wait_fails_after_kill = wait_fails_after_kill

    def poll(self) -> int | None:
        return self.return_code

    def terminate(self) -> None:
        self.terminated = True
        if not self.wait_requires_kill:
            self.return_code = 0

    def kill(self) -> None:
        self.killed = True
        if not self.wait_fails_after_kill:
            self.return_code = -9

    def wait(self, timeout: float | None = None) -> int:
        if self.return_code is None:
            raise subprocess.TimeoutExpired("fake-worker", timeout)
        return self.return_code


class ProcessFactory:
    def __init__(
        self,
        *,
        wait_requires_kill: bool = False,
        wait_fails_after_kill: bool = False,
    ) -> None:
        self.calls: list[tuple[list[str], dict[str, object]]] = []
        self.processes: list[FakeProcess] = []
        self.wait_requires_kill = wait_requires_kill
        self.wait_fails_after_kill = wait_fails_after_kill

    def __call__(self, command: list[str], **kwargs: object) -> FakeProcess:
        process = FakeProcess(
            wait_requires_kill=self.wait_requires_kill,
            wait_fails_after_kill=self.wait_fails_after_kill,
        )
        self.calls.append((list(command), dict(kwargs)))
        self.processes.append(process)
        return process


def sequence_probe(*values: bool):
    remaining = deque(values)

    def probe(_url: str, _timeout: float) -> bool:
        return remaining.popleft() if remaining else values[-1]

    return probe


class ManagedLocalModelRuntimeTests(unittest.TestCase):
    def make_runtime(
        self,
        factory: ProcessFactory,
        *,
        probe=None,
    ) -> ManagedLocalModelRuntime:
        return ManagedLocalModelRuntime(
            base_url="http://127.0.0.1:8080/v1",
            executable="mlx_lm.server",
            worker_args=("--max-tokens", "450"),
            startup_timeout_seconds=1,
            terminate_timeout_seconds=0.1,
            popen_factory=factory,
            readiness_probe=probe or sequence_probe(False, True),
            sleep=lambda _seconds: None,
        )

    def test_lazy_start_reuse_explicit_unload_and_reload(self) -> None:
        factory = ProcessFactory()
        runtime = self.make_runtime(
            factory,
            probe=sequence_probe(
                False,
                False,
                True,
                False,
                False,
                True,
            ),
        )

        self.assertEqual(runtime.public_status()["runtime_state"], "unloaded")
        self.assertEqual(factory.calls, [])

        with runtime.generation("local/qwen"):
            runtime.note_loaded("local/qwen")

        self.assertEqual(len(factory.calls), 1)
        command, kwargs = factory.calls[0]
        self.assertNotIn("--model", command)
        self.assertEqual(command[-2:], ["--max-tokens", "450"])
        self.assertTrue(kwargs["start_new_session"])
        worker_env = kwargs["env"]
        self.assertIsInstance(worker_env, dict)
        self.assertNotIn("GEMINI_API_KEY", worker_env)
        self.assertNotIn("RAG_API_TOKEN", worker_env)
        self.assertEqual(runtime.public_status()["runtime_state"], "loaded")
        self.assertEqual(runtime.public_status()["loaded_model"], "local/qwen")

        with runtime.generation("local/qwen"):
            runtime.note_loaded("local/qwen")
        self.assertEqual(len(factory.calls), 1)

        result = runtime.unload()
        self.assertTrue(result["released"])
        self.assertTrue(factory.processes[0].terminated)
        self.assertEqual(runtime.public_status()["runtime_state"], "unloaded")

        with runtime.generation("local/gemma"):
            runtime.note_loaded("local/gemma")
        self.assertEqual(len(factory.calls), 2)
        self.assertEqual(runtime.public_status()["loaded_model"], "local/gemma")
        runtime.close()
        self.assertTrue(factory.processes[1].terminated)

    def test_generation_context_serializes_auto_and_direct_local_callers(self) -> None:
        factory = ProcessFactory()
        runtime = self.make_runtime(factory)
        first_entered = threading.Event()
        release_first = threading.Event()
        second_entered = threading.Event()

        def first_generation() -> None:
            with runtime.generation("local/qwen"):
                first_entered.set()
                release_first.wait(timeout=1)

        def second_generation() -> None:
            with runtime.generation("local/gemma"):
                second_entered.set()

        first = threading.Thread(target=first_generation)
        second = threading.Thread(target=second_generation)
        first.start()
        self.assertTrue(first_entered.wait(timeout=1))
        second.start()
        time.sleep(0.05)
        self.assertFalse(second_entered.is_set())

        release_first.set()
        first.join(timeout=1)
        second.join(timeout=1)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertTrue(second_entered.is_set())

    def test_unload_is_idempotent_when_no_worker_exists(self) -> None:
        factory = ProcessFactory()
        runtime = self.make_runtime(factory, probe=lambda _url, _timeout: False)

        result = runtime.unload()

        self.assertTrue(result["ok"])
        self.assertFalse(result["released"])
        self.assertEqual(factory.calls, [])

    def test_unload_rejects_an_active_generation(self) -> None:
        factory = ProcessFactory()
        runtime = self.make_runtime(factory)

        with runtime.generation("local/qwen"):
            with self.assertRaises(LocalModelBusyError):
                runtime.unload()
            self.assertFalse(factory.processes[0].terminated)

        self.assertTrue(runtime.unload()["released"])

    def test_terminate_timeout_kills_only_the_owned_child(self) -> None:
        factory = ProcessFactory(wait_requires_kill=True)
        runtime = self.make_runtime(factory)

        with runtime.generation("local/qwen"):
            runtime.note_loaded("local/qwen")
        result = runtime.unload()

        self.assertTrue(result["released"])
        self.assertTrue(factory.processes[0].terminated)
        self.assertTrue(factory.processes[0].killed)

    def test_failed_kill_keeps_worker_ownership_for_a_later_retry(self) -> None:
        factory = ProcessFactory(
            wait_requires_kill=True,
            wait_fails_after_kill=True,
        )
        runtime = self.make_runtime(factory)

        with runtime.generation("local/qwen"):
            runtime.note_loaded("local/qwen")

        with self.assertRaises(LocalModelRuntimeError):
            runtime.unload()

        status = runtime.public_status()
        self.assertTrue(status["worker_running"])
        self.assertTrue(status["unload_supported"])

    def test_existing_external_server_is_never_adopted_or_stopped(self) -> None:
        factory = ProcessFactory()
        runtime = self.make_runtime(
            factory,
            probe=lambda _url, _timeout: True,
        )

        with self.assertRaises(LocalModelUnmanagedError):
            with runtime.generation("local/qwen"):
                pass
        with self.assertRaises(LocalModelUnmanagedError):
            runtime.unload()

        self.assertEqual(factory.calls, [])
        status = runtime.public_status()
        self.assertEqual(status["runtime_state"], "external")
        self.assertFalse(status["unload_supported"])

    def test_health_status_detects_an_external_server_before_generation(self) -> None:
        factory = ProcessFactory()
        runtime = self.make_runtime(
            factory,
            probe=lambda _url, _timeout: True,
        )

        status = runtime.public_status()

        self.assertEqual(status["runtime_state"], "external")
        self.assertFalse(status["unload_supported"])
        self.assertFalse(status["worker_running"])
        self.assertEqual(factory.calls, [])

    def test_unload_rechecks_when_an_external_server_has_gone_away(self) -> None:
        factory = ProcessFactory()
        runtime = self.make_runtime(
            factory,
            probe=sequence_probe(True, False),
        )

        with self.assertRaises(LocalModelUnmanagedError):
            runtime.unload()

        result = runtime.unload()
        self.assertTrue(result["ok"])
        self.assertFalse(result["released"])
        self.assertEqual(runtime.public_status()["runtime_state"], "unloaded")

    def test_environment_selects_managed_mlx_or_external_runtime(self) -> None:
        managed = build_local_model_runtime(
            {
                "RAG_LOCAL_RUNTIME": "managed_mlx",
                "RAG_LOCAL_BASE_URL": "http://127.0.0.1:8080/v1",
                "RAG_LOCAL_MAX_OUTPUT_TOKENS": "321",
            }
        )
        external = build_local_model_runtime(
            {
                "RAG_LOCAL_RUNTIME": "external",
                "RAG_LOCAL_BASE_URL": "http://127.0.0.1:11434/v1",
            }
        )

        self.assertIsInstance(managed, ManagedLocalModelRuntime)
        self.assertIn("321", managed.worker_args)
        chat_args_index = managed.worker_args.index("--chat-template-args")
        self.assertEqual(
            managed.worker_args[chat_args_index + 1],
            '{"enable_thinking":false}',
        )
        self.assertIsInstance(external, ExternalLocalModelRuntime)
        with self.assertRaises(ValueError):
            build_local_model_runtime(
                {
                    "RAG_LOCAL_RUNTIME": "managed_mxl",
                    "RAG_LOCAL_BASE_URL": "http://127.0.0.1:8080/v1",
                }
            )
        with self.assertRaises(ValueError):
            build_local_model_runtime(
                {
                    "RAG_LOCAL_RUNTIME": "managed_mlx",
                    "RAG_LOCAL_BASE_URL": "http://127.0.0.1:8080/v1",
                    "RAG_LOCAL_CHAT_TEMPLATE_ARGS": "not-json",
                }
            )

    def test_worker_environment_excludes_api_secrets(self) -> None:
        factory = ProcessFactory()
        with patch.dict(
            "os.environ",
            {
                "PATH": "/safe/bin",
                "HOME": "/safe/home",
                "GEMINI_API_KEY": "do-not-copy",
                "RAG_API_TOKEN": "do-not-copy",
            },
            clear=True,
        ):
            runtime = self.make_runtime(factory)
            with runtime.generation("local/qwen"):
                pass

        worker_env = factory.calls[0][1]["env"]
        self.assertEqual(worker_env["PATH"], "/safe/bin")
        self.assertEqual(worker_env["HOME"], "/safe/home")
        self.assertNotIn("GEMINI_API_KEY", worker_env)
        self.assertNotIn("RAG_API_TOKEN", worker_env)


if __name__ == "__main__":
    unittest.main()
