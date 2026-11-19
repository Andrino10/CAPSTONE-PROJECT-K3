"""Validate that SIGAP evaluation outputs are complete and mathematically valid."""

from __future__ import annotations

import csv
from pathlib import Path

try:  # Supports both `python file.py` and `python -m backend.evaluation...`.
    from .ragas_metrics import calculate_retrieval_metrics
except ImportError:  # pragma: no cover - exercised by direct script execution
    from ragas_metrics import calculate_retrieval_metrics


EVALUATION_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVALUATION_DIR / "test_dataset.csv"
RESULTS_PATH = EVALUATION_DIR / "hasil_similarity.csv"
REQUIRED_RESULT_COLUMNS = {
    "sample_id", "laporan", "ground_truth_kondisi", "ground_truth_risiko",
    "rank", "retrieved_kondisi", "retrieved_risiko", "faiss_squared_l2_distance",
    "cosine_similarity", "is_ground_truth",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def validate() -> dict[str, float]:
    if not DATASET_PATH.exists():
        raise ValueError(f"Dataset tidak ditemukan: {DATASET_PATH}")
    if not RESULTS_PATH.exists():
        raise ValueError("Hasil belum ada; jalankan similarity_eval_all.py terlebih dahulu.")

    dataset = read_csv(DATASET_PATH)
    results = read_csv(RESULTS_PATH)
    if not dataset or not results:
        raise ValueError("Dataset atau hasil evaluasi kosong.")
    if not REQUIRED_RESULT_COLUMNS.issubset(results[0]):
        raise ValueError("Kolom hasil evaluasi tidak lengkap.")

    dataset_ids = [row["sample_id"] for row in dataset]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("sample_id dataset duplikat.")

    by_sample: dict[str, list[dict[str, str]]] = {}
    for row in results:
        if any(value is None or not value.strip() for value in row.values()):
            raise ValueError(f"Nilai kosong pada hasil sampel {row.get('sample_id')}.")
        cosine = float(row["cosine_similarity"])
        l2_distance = float(row["faiss_squared_l2_distance"])
        if not -1.0 <= cosine <= 1.0:
            raise ValueError(f"Cosine similarity di luar [-1, 1]: {cosine}")
        if l2_distance < 0:
            raise ValueError(f"Jarak squared-L2 negatif: {l2_distance}")
        if row["is_ground_truth"] not in {"True", "False"}:
            raise ValueError("is_ground_truth harus True atau False.")
        by_sample.setdefault(row["sample_id"], []).append(row)

    if set(by_sample) != set(dataset_ids):
        raise ValueError("sample_id hasil tidak sesuai dengan dataset.")
    if len(results) != len(dataset) * 2:
        raise ValueError("Jumlah hasil harus tepat dua referensi per sampel (k=2).")

    relevance = []
    for sample_id, rows in by_sample.items():
        ranks = sorted(int(row["rank"]) for row in rows)
        if ranks != [1, 2]:
            raise ValueError(f"Rank retrieval tidak lengkap untuk {sample_id}: {ranks}")
        relevance.append([row["is_ground_truth"] == "True" for row in sorted(rows, key=lambda row: int(row["rank"]))])

    metrics = calculate_retrieval_metrics(relevance)
    if any(not 0.0 <= value <= 1.0 for value in metrics.values()):
        raise ValueError("Nilai metrik retrieval di luar [0, 1].")
    return metrics


if __name__ == "__main__":
    values = validate()
    print("VALID: hasil lengkap, k=2 konsisten, dan rentang metrik sah.")
    for name, value in values.items():
        print(f"{name}: {value:.4f}")
