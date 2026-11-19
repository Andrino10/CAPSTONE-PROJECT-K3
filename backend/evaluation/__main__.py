"""Single command runner for all locally supported SIGAP evaluations."""

from __future__ import annotations

from .check_metrics import validate
from .interaction_eval import evaluate_interactions, print_interaction_summary
from .ragas_metrics import metrics_from_results, ragas_preflight
from .similarity_eval_all import DATASET_PATH, print_summary, run_evaluation


def main() -> int:
    print("=== SIGAP AI: evaluasi dari backend/evaluation ===")
    print("Input pertanyaan/laporan: kolom 'laporan'.")
    print("Ground truth retrieval: kolom 'ground_truth_kondisi'.")
    print("Label risiko acuan: kolom 'ground_truth_risiko'.")

    rows = run_evaluation()
    print_summary(rows)

    print("\nMetrik retrieval (k=2):")
    for name, value in metrics_from_results().items():
        print(f"{name}: {value:.4f}")

    validate()
    print("Validasi hasil: LULUS")

    print("\nEvaluasi interaksi aktual:")
    _, interaction_summary = evaluate_interactions()
    print_interaction_summary(interaction_summary)

    runnable, notes = ragas_preflight(DATASET_PATH)
    print("\nRAGAS: siap dijalankan" if runnable else "\nRAGAS: tidak dijalankan")
    for note in notes:
        print(f"- {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
