"""Append-only storage for successful SIGAP AI analysis interactions."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
from typing import Any, Iterator
from uuid import uuid4


EVALUATION_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_PATH = EVALUATION_DIR / "interaction_logs.jsonl"
_THREAD_LOCK = threading.Lock()


class InteractionLogError(RuntimeError):
    """Raised when an interaction log cannot be read or written safely."""


@contextmanager
def _exclusive_lock(log_path: Path) -> Iterator[None]:
    """Use a lock file so concurrent local requests cannot interleave records."""
    lock_path = log_path.with_suffix(f"{log_path.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _THREAD_LOCK, lock_path.open("a+", encoding="utf-8") as handle:
        handle.seek(0)
        handle.write("0")
        handle.flush()
        handle.seek(0)
        locked = False
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            locked = True
            yield
        finally:
            if locked and os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            elif locked:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_interactions(log_path: Path = DEFAULT_LOG_PATH) -> list[dict[str, Any]]:
    """Read intact JSONL interaction records without altering the raw log."""
    if not log_path.exists():
        return []
    records = []
    with log_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise InteractionLogError(
                    f"JSONL tidak valid pada {log_path.name} baris {line_number}."
                ) from error
            if not isinstance(record, dict):
                raise InteractionLogError(
                    f"Record bukan objek JSON pada {log_path.name} baris {line_number}."
                )
            records.append(record)
    return records


def record_interaction(
    laporan: str,
    response: dict[str, Any],
    retrieved_contexts: list[dict[str, Any]],
    *,
    request_id: str | None = None,
    log_path: Path | None = None,
) -> dict[str, Any] | None:
    """Persist one successful response, returning ``None`` for a duplicate ID.

    The log stores only the report, actual API response, and retrieval context;
    request headers, credentials, and client identity are deliberately omitted.
    """
    if not laporan.strip():
        raise InteractionLogError("Laporan kosong tidak boleh dicatat.")
    if not isinstance(response, dict) or not response:
        raise InteractionLogError("Respons analisis tidak valid untuk dicatat.")
    if not isinstance(retrieved_contexts, list):
        raise InteractionLogError("Konteks retrieval harus berupa daftar.")

    destination = Path(log_path) if log_path is not None else DEFAULT_LOG_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(destination):
        if request_id and any(item.get("request_id") == request_id for item in read_interactions(destination)):
            return None

        record = {
            "interaction_id": str(uuid4()),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "laporan": laporan,
            "response": response,
            "retrieved_contexts": retrieved_contexts,
            "status": "success",
        }
        try:
            with destination.open("a", encoding="utf-8", newline="") as file:
                file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                file.flush()
                os.fsync(file.fileno())
        except OSError as error:
            raise InteractionLogError(f"Gagal menulis log interaksi: {destination}") from error
    return record
