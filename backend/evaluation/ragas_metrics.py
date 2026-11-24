"""Ground-truth retrieval metrics for SIGAP AI at k=2.

Despite its historical filename, this module does not call RAGAS.  It uses
explicit relevance labels from test_dataset.csv, so the metrics are locally
reproducible and do not require an LLM judge.
"""

from __future__ import annotations

import csv
import importlib.metadata
import importlib.util
import os
from pathlib import Path
from statistics import mean
from typing import Iterable


EVALUATION_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_PATH = EVALUATION_DIR / "hasil_similarity.csv"
RAGAS_REQUIRED_COLUMNS = {
    "laporan", "ground_truth_kondisi", "ground_truth_risiko",
}


def calculate_retrieval_metrics(
    relevance_by_query: Iterable[Iterable[bool]], *, relevant_per_query: int = 1
) -> dict[str, float]:
    """Calculate Precision@k, Recall@k, Hit@k, and MRR from relevance labels.

    ``relevant_per_query`` must come from the ground truth, never from model
    scores.  The supplied SIGAP synthetic dataset has exactly one reference
    acuan per report.
    """
    rows = [list(labels) for labels in relevance_by_query]
    if not rows:
        raise ValueError("Tidak ada query untuk dihitung.")
    if relevant_per_query < 1:
        raise ValueError("Jumlah referensi acuan per query harus minimal 1.")

    k = len(rows[0])
    if k == 0 or any(len(labels) != k for labels in rows):
        raise ValueError("Setiap query harus memiliki jumlah hasil retrieval yang sama.")

    relevant_retrieved = sum(sum(labels) for labels in rows)
    reciprocal_ranks = []
    for labels in rows:
        first_rank = next((rank for rank, relevant in enumerate(labels, start=1) if relevant), None)
        reciprocal_ranks.append(0.0 if first_rank is None else 1.0 / first_rank)

    return {
        "precision_at_2": relevant_retrieved / (len(rows) * k),
        "recall_at_2": relevant_retrieved / (len(rows) * relevant_per_query),
        "hit_at_2": mean(any(labels) for labels in rows),
        "mrr": mean(reciprocal_ranks),
    }


def metrics_from_results(results_path: Path = DEFAULT_RESULTS_PATH) -> dict[str, float]:
    """Read per-rank output from similarity_eval_all.py and aggregate it."""
    grouped: dict[str, list[tuple[int, bool]]] = {}
    with results_path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            sample_id = row["sample_id"]
            grouped.setdefault(sample_id, []).append(
                (int(row["rank"]), row["is_ground_truth"] == "True")
            )

    ordered_labels = [
        [is_relevant for _, is_relevant in sorted(ranks)]
        for _, ranks in sorted(grouped.items())
    ]
    return calculate_retrieval_metrics(ordered_labels)


def ragas_preflight(dataset_path: Path) -> tuple[bool, list[str]]:
    """Check prerequisites for LLM-judged RAGAS without making any API call.

    Faithfulness, Answer Relevancy, Context Precision, and Context Recall need
    an evaluator/judge model; the last two also need sufficient reference
    contexts or answers.  SIGAP's current synthetic retrieval dataset contains
    neither a generated-answer reference nor an approved evaluator model.
    """
    notes: list[str] = []
    try:
        notes.append(f"ragas terpasang: {importlib.metadata.version('ragas')}")
    except importlib.metadata.PackageNotFoundError:
        return False, ["ragas tidak terpasang."]
    if importlib.util.find_spec("datasets") is None:
        return False, notes + ["Paket datasets tidak tersedia."]
    if not dataset_path.exists():
        return False, notes + [f"Dataset tidak ditemukan: {dataset_path}"]

    with dataset_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        return False, notes + ["Dataset kosong."]
    if not RAGAS_REQUIRED_COLUMNS.issubset(fieldnames):
        return False, notes + ["Kolom dasar retrieval tidak lengkap."]

    notes.append("Dataset memiliki laporan dan satu referensi acuan untuk retrieval.")
    notes.append("Dataset tidak memiliki jawaban acuan atau konteks acuan lengkap untuk metrik RAGAS standar.")
    notes.append("Tidak ada evaluator generatif; metrik LLM RAGAS ditandai N/A.")
    if os.getenv("OPENAI_API_KEY"):
        notes.append("OPENAI_API_KEY terdeteksi, tetapi tidak digunakan tanpa persetujuan pengguna.")
    return False, notes


if __name__ == "__main__":
    metrics = metrics_from_results()
    print("Retrieval metrics (k=2)")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
