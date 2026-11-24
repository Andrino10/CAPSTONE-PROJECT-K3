"""RAGAS evaluation layer for SIGAP AI."""

from __future__ import annotations

from pathlib import Path
import csv
import importlib.metadata
import importlib.util
from datetime import datetime, timezone
from typing import Any

try:  # Supports both `python file.py` and `python -m backend.evaluation...`.
    from .ragas_metrics import ragas_preflight
except ImportError:  # pragma: no cover - exercised by direct script execution
    from ragas_metrics import ragas_preflight

EVALUATION_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVALUATION_DIR / "test_dataset.csv"
RESULTS_PATH = EVALUATION_DIR / "hasil_ragas.csv"

RAGAS_COLUMNS = [
    "timestamp_utc", "query", "answer", "retrieved_documents",
    "average_similarity", "maximum_similarity", "minimum_similarity",
    "context_precision", "context_recall", "faithfulness",
    "answer_relevancy", "answer_correctness", "ground_truth",
    "status", "status_detail", "ragas_version",
]


def ground_truth_for_query(question: str) -> tuple[str | None, str | None]:
    """Resolve approved context; answer correctness remains N/A without a reference answer."""
    if not question.strip() or not DATASET_PATH.exists():
        return None, None
    with DATASET_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row.get("laporan", "").strip() == question.strip():
                reference = row.get("ground_truth_kondisi", "").strip()
                return (reference or None), None
    return None, None


def _score_in_range(value: Any) -> float | None:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= score <= 1.0:
        return score
    return None


def ragas_status() -> dict[str, Any]:
    """Return a safe, local-only RAGAS availability status."""
    try:
        import ragas  # noqa: F401
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample  # noqa: F401
        from ragas.metrics import (  # noqa: F401
            answer_correctness,
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        ragas_version = importlib.metadata.version("ragas")
    except Exception as error:
        return {
            "available": False,
            "status": "BELUM_TERSEDIA",
            "detail": f"Import RAGAS/evaluator gagal: {error}",
            "ragas_version": None,
        }

    return {
        "available": True,
        "status": "TERSEDIA",
        "detail": "RAGAS 0.4.3 tersedia untuk metrik non-generatif.",
        "ragas_version": ragas_version,
    }


def evaluate_ragas(
    *,
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth_context: str | None = None,
    ground_truth_answer: str | None = None,
    similarity_scores: list[float] | None = None,
) -> dict[str, Any]:
    """Evaluate RAGAS metrics that do not require an additional generative model."""
    scores = [_score_in_range(score) for score in (similarity_scores or [])]
    valid_scores = [score for score in scores if score is not None]
    result: dict[str, Any] = {
        "average_similarity": sum(valid_scores) / len(valid_scores) if valid_scores else None,
        "maximum_similarity": max(valid_scores) if valid_scores else None,
        "minimum_similarity": min(valid_scores) if valid_scores else None,
        "context_precision": None,
        "context_recall": None,
        "faithfulness": None,
        "answer_relevancy": None,
        "answer_correctness": None,
        "ground_truth": ground_truth_answer or None,
        "ground_truth_context": ground_truth_context or None,
    }

    if not question.strip() or not answer.strip() or not contexts:
        result.update({"status": "BELUM_TERSEDIA", "status_detail": "Query, jawaban, atau konteks kosong."})
        return result

    result.update({
        "faithfulness": None,
        "answer_relevancy": None,
        "answer_correctness": None,
        "status_detail_faithfulness":
            "N/A: Faithfulness RAGAS memerlukan evaluator generatif.",
        "status_detail_answer_relevancy":
            "N/A: Answer Relevancy RAGAS memerlukan evaluator generatif.",
        "status_detail_answer_correctness":
            "N/A: Jawaban referensi dan evaluator generatif tidak digunakan.",
        "status_detail_context_precision":
            "N/A: Ground truth konteks tidak tersedia.",
        "status_detail_context_recall":
            "N/A: Ground truth konteks tidak tersedia.",
    })

    try:
        from ragas import evaluate
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        from ragas.metrics import NonLLMContextPrecisionWithReference, NonLLMContextRecall

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference_contexts=[ground_truth_context] if ground_truth_context else None,
            reference=ground_truth_answer or ground_truth_context or None,
        )
        dataset = EvaluationDataset(samples=[sample])
        metrics = []
        if ground_truth_context:
            metrics = [
                NonLLMContextPrecisionWithReference(),
                NonLLMContextRecall(),
            ]
        if not metrics:
            result.update({
                "status": "TIDAK_MEMENUHI_INPUT",
                "status_detail":
                    "N/A: Ground truth konteks tidak tersedia; Context Precision dan Context Recall tidak dihitung.",
            })
            return result

        evaluation = evaluate(dataset=dataset, metrics=metrics, raise_exceptions=True)
        first = evaluation.scores[0]
        result["context_precision"] = _score_in_range(
            first.get("non_llm_context_precision_with_reference")
        )
        result["context_recall"] = _score_in_range(
            first.get("non_llm_context_recall")
        )
        result["status_detail_context_precision"] = (
            "Dihitung RAGAS non-LLM menggunakan reference_contexts."
        )
        result["status_detail_context_recall"] = (
            "Dihitung RAGAS non-LLM menggunakan reference_contexts."
        )
        result.update({
            "status": "SEBAGIAN_BERHASIL",
            "status_detail":
                "Context Precision dan Context Recall dihitung RAGAS tanpa model generatif; "
                "metrik berbasis LLM ditandai N/A.",
        })
    except Exception as error:
        result.update({
            "status": "BELUM_TERSEDIA",
            "status_detail": f"Evaluasi RAGAS gagal: {error}",
        })
    return result


def record_ragas_result(
    *,
    question: str,
    answer: str,
    contexts: list[str],
    similarity_scores: list[float],
    ground_truth: str | None,
    evaluation: dict[str, Any],
    results_path: Path = RESULTS_PATH,
) -> None:
    """Append a valid RAGAS attempt to hasil_ragas.csv."""
    if evaluation.get("status") not in {"BERHASIL", "SEBAGIAN_BERHASIL"}:
        return
    results_path.parent.mkdir(parents=True, exist_ok=True)
    exists = results_path.exists()
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "query": question,
        "answer": answer,
        "retrieved_documents": " || ".join(contexts),
        "average_similarity": evaluation["average_similarity"],
        "maximum_similarity": evaluation["maximum_similarity"],
        "minimum_similarity": evaluation["minimum_similarity"],
        "context_precision": evaluation["context_precision"],
        "context_recall": evaluation["context_recall"],
        "faithfulness": evaluation["faithfulness"],
        "answer_relevancy": evaluation["answer_relevancy"],
        "answer_correctness": evaluation["answer_correctness"],
        "ground_truth": ground_truth or evaluation.get("ground_truth"),
        "status": evaluation["status"],
        "status_detail": evaluation["status_detail"],
        "ragas_version": importlib.metadata.version("ragas") if importlib.util.find_spec("ragas") else "",
    }
    with results_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RAGAS_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    runnable, notes = ragas_preflight(DATASET_PATH)
    print("RAGAS siap dijalankan." if runnable else "RAGAS tidak dijalankan (preflight).")
    for note in notes:
        print(f"- {note}")
    print("Metrik retrieval berbasis ground truth tetap tersedia melalui ragas_metrics.py.")
