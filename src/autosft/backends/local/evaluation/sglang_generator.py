"""Managed, offline SGLang inference on this host's GPUs.

The Engine owns local worker processes; no HTTP endpoint is contacted or
published. Callers must close it (or use a context manager), including when
generation or scoring fails.
"""

from __future__ import annotations

from contextlib import contextmanager
import multiprocessing
import os
from pathlib import Path
import signal
import time
from typing import Any, Iterator, Mapping


@contextmanager
def runtime_environment(*, allow_longer_context: bool = False) -> Iterator[None]:
    """Limit runtime overrides to this local engine and its child processes."""
    settings = {
        name: "1"
        for name in (
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "HF_DATASETS_OFFLINE",
        )
    }
    if allow_longer_context:
        settings["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    previous = {name: os.environ.get(name) for name in settings}
    os.environ.update(settings)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _engine_worker(
    connection: Any,
    options: dict[str, Any],
    allow_longer_context: bool,
    engine_factory: Any = None,
) -> None:
    # Engine.shutdown() kills all descendants of its current process. Isolate
    # it from the caller's CPU scorer and any other task workers.
    # Keep the supervisor's process group so stopping a local task also stops
    # its inference workers. Engine itself is isolated by process ancestry.
    def stop_worker(signum: int, frame: Any) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop_worker)
    engine = None
    try:
        with runtime_environment(allow_longer_context=allow_longer_context):
            if engine_factory is None:
                from sglang import Engine

                engine_factory = Engine
            engine = engine_factory(**options)
        connection.send(("ready", None))
        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            if command != "generate":
                raise ValueError(f"unknown SGLang worker command: {command}")
            connection.send(("result", engine.generate(**payload)))
    except EOFError:
        pass
    except BaseException as exc:
        try:
            connection.send(("error", f"{type(exc).__name__}: {exc}"))
        except (OSError, EOFError):
            pass
    finally:
        try:
            if engine is not None:
                engine.shutdown()
        finally:
            # Initialization can fail after workers were launched but before
            # Engine returned an object. Clean those descendants as well.
            import psutil

            for child in psutil.Process().children(recursive=True):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            connection.close()


class LocalSGLang:
    def __init__(
        self,
        model_path: str,
        *,
        context_length: int,
        tensor_parallel_size: int = 1,
        memory_fraction: float = 0.8,
        seed: int | None = None,
        allow_longer_context: bool = False,
        startup_timeout: float = 900,
        generation_timeout: float | None = None,
        _engine_factory: Any = None,
    ):
        model = Path(model_path).expanduser().resolve()
        if not model.is_dir() or not (model / "config.json").is_file():
            raise ValueError(
                "SGLang evaluation requires a local model directory with config.json"
            )
        options = dict(
            model_path=str(model),
            tokenizer_path=str(model),
            context_length=context_length,
            tp_size=tensor_parallel_size,
            mem_fraction_static=memory_fraction,
            trust_remote_code=False,
        )
        if seed is not None:
            options["random_seed"] = seed
            options["enable_deterministic_inference"] = True
        self.generation_timeout = generation_timeout
        self.closed = False
        context = multiprocessing.get_context("spawn")
        self.connection, child_connection = context.Pipe()
        self.process = context.Process(
            target=_engine_worker,
            args=(child_connection, options, allow_longer_context, _engine_factory),
        )
        try:
            self.process.start()
            child_connection.close()
            status, _ = self._receive(startup_timeout)
            if status != "ready":
                raise RuntimeError("SGLang worker did not become ready")
        except BaseException:
            child_connection.close()
            self.close()
            raise

    def _receive(self, timeout: float | None) -> tuple[str, Any]:
        deadline = time.monotonic() + timeout if timeout is not None else None
        while deadline is None or time.monotonic() < deadline:
            interval = (
                min(1.0, max(0.0, deadline - time.monotonic()))
                if deadline is not None
                else 1.0
            )
            if self.connection.poll(interval):
                try:
                    status, payload = self.connection.recv()
                except EOFError as exc:
                    raise RuntimeError(
                        "local SGLang worker exited without a result"
                    ) from exc
                if status == "error":
                    raise RuntimeError(f"local SGLang worker failed: {payload}")
                return status, payload
            if not self.process.is_alive():
                raise RuntimeError(
                    f"local SGLang worker exited with code {self.process.exitcode}"
                )
        raise TimeoutError("local SGLang worker timed out")

    def generate(
        self, prompts: list[str], sampling_params: list[Mapping[str, Any]]
    ) -> list[str]:
        if self.closed:
            raise RuntimeError("SGLang engine is closed")
        if len(prompts) != len(sampling_params):
            raise ValueError("each prompt must have one set of sampling parameters")
        self.connection.send(
            ("generate", dict(prompt=prompts, sampling_params=sampling_params))
        )
        status, results = self._receive(self.generation_timeout)
        if status != "result":
            raise RuntimeError("unexpected local SGLang worker response")
        if not isinstance(results, list) or len(results) != len(prompts):
            raise ValueError("incomplete local SGLang inference results")
        if any(
            not isinstance(result, dict) or not isinstance(result.get("text"), str)
            for result in results
        ):
            raise ValueError("non-text local SGLang inference result")
        for result in results:
            reason = result.get("meta_info", {}).get("finish_reason", {})
            if isinstance(reason, dict) and reason.get("type") in {"abort", "error"}:
                raise RuntimeError(f"local SGLang inference failed: {reason}")
        return [result["text"] for result in results]

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if self.process.pid is not None:
                if self.process.is_alive():
                    try:
                        self.connection.send(("close", None))
                    except (OSError, EOFError):
                        pass
                    self.process.join(5)
                if self.process.is_alive():
                    # Snapshot only this worker's descendants before stopping
                    # it; never signal the caller's shared process group here.
                    import psutil

                    try:
                        descendants = psutil.Process(self.process.pid).children(
                            recursive=True
                        )
                    except psutil.NoSuchProcess:
                        descendants = []
                    self.process.terminate()
                    self.process.join(2)
                    for descendant in descendants:
                        try:
                            descendant.kill()
                        except psutil.NoSuchProcess:
                            pass
                    if self.process.is_alive():
                        self.process.kill()
                        self.process.join(1)
                self.process.close()
        finally:
            self.connection.close()

    def __enter__(self) -> "LocalSGLang":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
