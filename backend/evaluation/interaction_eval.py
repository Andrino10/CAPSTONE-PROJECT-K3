"""Evaluate actual SIGAP AI interaction logs without changing raw records."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
import sys
from typing import Any

import numpy as np


EVALUATION_DIR = Path(__file__).resolve().parent
BACKEND_DIR = EVALUATION_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

import rag  # noqa: E402

try:
    from .interaction_logger import DEFAULT_LOG_PATH, InteractionLogError, read_interactions
    from .similarity_eval_all import cosine_similarity
except ImportError:  # Supports direct execution from backend/evaluation.
    from interaction_logger import DEFAULT_LOG_PATH, InteractionLogError, read_interactions
    from similarity_eval_all import cosine_similarity


DEFAULT_RESULTS_PATH = EVALUATION_DIR / "hasil_interaction_similarity.csv"
RESULT_COLUMNS = [
    "interaction_id", "timestamp_utc", "laporan", "level", "rank",
    "retrieved_kondisi", "retrieved_risiko", "faiss_squared_l2_distance",
    "cosine_similarity", "status",
]


def _context_text(document: dict[str, Any]) -> str:
    required = ("kondisi", "risiko", "penjelasan", "solusi")
    if not isinstance(document, dict) or any(not document.get(key) for key in required):
        raise InteractionLogError("Dokumen konteks interaksi tidak lengkap.")
    return " | ".join(str(document[key]) for key in required)


def evaluate_interactions(
    log_path: Path = DEFAULT_LOG_PATH,
    results_path: Path = DEFAULT_RESULTS_PATH,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Compute similarity for logged query/context pairs and write a new CSV."""
    records = read_interactions(log_path)
    successful = [record for record in records if record.get("status") == "success"]
    risk_distribution: Counter[str] = Counter()
    result_rows: list[dict[str, object]] = []

    for record in successful:
        interaction_id = record.get("interaction_id")
        laporan = record.get("laporan")
        response = record.get("response")
        contexts = record.get("retrieved_contexts")
        if not interaction_id or not isinstance(laporan, str) or not isinstance(response, dict):
            raise InteractionLogError("Record interaksi sukses tidak memiliki ID, laporan, atau respons.")
        if not isinstance(contexts, list):
            raise InteractionLogError("Record interaksi tidak memiliki konteks retrieval.")

        level = response.get("level")
        if level not in {"RENDAH", "SEDANG", "TINGGI"}:
            raise InteractionLogError(f"Level risiko tidak valid pada interaksi {interaction_id}.")
        risk_distribution[level] += 1
        query_embedding = np.asarray(rag.embed_model.encode([laporan]), dtype="float32")[0]

        for context in contexts:
            document = context.get("document") if isinstance(context, dict) else None
            text = _context_text(document)
            document_embedding = np.asarray(rag.embed_model.encode([text]), dtype="float32")[0]
            distance = context.get("faiss_squared_l2_distance")
            if not isinstance(distance, (int, float)) or distance < 0:
                raise InteractionLogError(f"Jarak L2 tidak valid pada interaksi {interaction_id}.")
            result_rows.append({
                "interaction_id": interaction_id,
                "timestamp_utc": record.get("timestamp_utc", ""),
                "laporan": laporan,
                "level": level,
                "rank": context.get("rank"),
                "retrieved_kondisi": document["kondisi"],
                "retrieved_risiko": document["risiko"],
                "faiss_squared_l2_distance": float(distance),
                "cosine_similarity": cosine_similarity(query_embedding, document_embedding),
                "status": record["status"],
            })

    if result_rows:
        with results_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS)
            writer.writeheader()
            writer.writerows(result_rows)

    summary: dict[str, object] = {
        "total_records": len(records),
        "successful_records": len(successful),
        "evaluated_contexts": len(result_rows),
        "risk_distribution": dict(sorted(risk_distribution.items())),
    }
    if result_rows:
        scores = [float(row["cosine_similarity"]) for row in result_rows]
        summary["cosine_mean"] = float(np.mean(scores))
        summary["cosine_min"] = float(np.min(scores))
        summary["cosine_max"] = float(np.max(scores))
    return result_rows, summary


def print_interaction_summary(summary: dict[str, object]) -> None:
    print(f"Record interaksi: {summary['total_records']}")
    print(f"Interaksi sukses: {summary['successful_records']}")
    print(f"Konteks dievaluasi: {summary['evaluated_contexts']}")
    print(f"Distribusi risiko sistem: {summary['risk_distribution']}")
    if summary["evaluated_contexts"]:
        print("Cosine similarity interaksi (laporan vs konteks): "
              f"rata-rata={summary['cosine_mean']:.4f}, "
              f"min={summary['cosine_min']:.4f}, maks={summary['cosine_max']:.4f}")
    print("Precision@2, Recall@2, Hit@2, dan MRR tidak dihitung tanpa label relevansi manual.")


if __name__ == "__main__":
    rows, summary = evaluate_interactions()
    print_interaction_summary(summary)
    if rows:
        print(f"Hasil disimpan: {DEFAULT_RESULTS_PATH}")
    else:
        print("Tidak ada interaksi sukses yang dapat dievaluasi.")
