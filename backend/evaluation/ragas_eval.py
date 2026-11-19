"""Preflight RAGAS for SIGAP AI; it never provisions or calls paid services."""

from __future__ import annotations

from pathlib import Path

try:  # Supports both `python file.py` and `python -m backend.evaluation...`.
    from .ragas_metrics import ragas_preflight
except ImportError:  # pragma: no cover - exercised by direct script execution
    from ragas_metrics import ragas_preflight

EVALUATION_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVALUATION_DIR / "test_dataset.csv"


if __name__ == "__main__":
    runnable, notes = ragas_preflight(DATASET_PATH)
    print("RAGAS siap dijalankan." if runnable else "RAGAS tidak dijalankan (preflight).")
    for note in notes:
        print(f"- {note}")
    print("Metrik retrieval berbasis ground truth tetap tersedia melalui ragas_metrics.py.")
