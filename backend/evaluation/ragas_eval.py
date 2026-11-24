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
        "detail": "RAGAS 0.4.3 tersedia; evaluator lokal dipilih melalui konfigurasi lingkungan.",
        "ragas_version": ragas_version,
    }


def evaluate_ragas(
    *,
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str | None = None,
    similarity_scores: list[float] | None = None,
) -> dict[str, Any]:
    """Evaluate one RAG result if an approved local evaluator is configured."""
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
        "ground_truth": ground_truth or None,
    }

    if not question.strip() or not answer.strip() or not contexts:
        result.update({"status": "BELUM_TERSEDIA", "status_detail": "Query, jawaban, atau konteks kosong."})
        return result

    try:
        import os
        if os.getenv("RAGAS_EVALUATOR", "").strip().lower() != "ollama":
            result.update({
                "status": "BELUM_TERSEDIA",
                "status_detail": "Evaluator RAGAS lokal belum dikonfigurasi. Set RAGAS_EVALUATOR=ollama.",
            })
            return result

        model = os.getenv("RAGAS_OLLAMA_MODEL", "").strip()
        if not model:
            result.update({
                "status": "BELUM_TERSEDIA",
                "status_detail": "Nama model Ollama belum dikonfigurasi. Set RAGAS_OLLAMA_MODEL.",
            })
            return result

        from langchain_ollama import ChatOllama
        from langchain_huggingface import HuggingFaceEmbeddings
        from ragas import evaluate
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        llm = ChatOllama(model=model, base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=ground_truth or None,
        )
        dataset = EvaluationDataset(samples=[sample])
        metrics = [answer_relevancy, faithfulness]
        if ground_truth:
            metrics.extend([context_precision, context_recall, answer_correctness])

        evaluation = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=True,
        )
        first = evaluation.scores[0]
        for key, result_key in (
            ("context_precision", "context_precision"),
            ("context_recall", "context_recall"),
            ("faithfulness", "faithfulness"),
            ("answer_relevancy", "answer_relevancy"),
            ("answer_correctness", "answer_correctness"),
        ):
            score = _score_in_range(first.get(key))
            if score is not None:
                result[result_key] = score

        result.update({
            "status": "BERHASIL",
            "status_detail": "Evaluasi RAGAS berhasil dijalankan.",
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
    if evaluation.get("status") != "BERHASIL":
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
        "ground_truth": evaluation["ground_truth"],
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
