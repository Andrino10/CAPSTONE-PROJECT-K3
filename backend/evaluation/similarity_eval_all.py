"""Run reproducible semantic-similarity and retrieval evaluation for SIGAP AI.

The CSV output has one row per retrieved reference (rank 1 and 2), preserving
the report, its declared ground truth, raw FAISS squared-L2 distance, and a
separately calculated cosine similarity.  L2 distance is never interpreted as
cosine similarity.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np


EVALUATION_DIR = Path(__file__).resolve().parent
BACKEND_DIR = EVALUATION_DIR.parent
PROJECT_DIR = BACKEND_DIR.parent
DATASET_PATH = EVALUATION_DIR / "test_dataset.csv"
RESULTS_PATH = EVALUATION_DIR / "hasil_similarity.csv"
sys.path.insert(0, str(BACKEND_DIR))

import rag  # noqa: E402  -- loaded once and shared with retrieval evaluation


REQUIRED_COLUMNS = {
    "sample_id", "laporan", "ground_truth_kondisi", "ground_truth_risiko", "data_asal"
}


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Return cosine similarity for two embedding vectors."""
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0:
        raise ValueError("Embedding bernorma nol tidak dapat dihitung cosinenya.")
    return float(np.dot(left, right) / denominator)


def load_dataset(dataset_path: Path = DATASET_PATH) -> list[dict[str, str]]:
    with dataset_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            raise ValueError(f"Kolom dataset wajib: {sorted(REQUIRED_COLUMNS)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Dataset evaluasi kosong.")

    seen_ids = set()
    for line_number, row in enumerate(rows, start=2):
        if any(not row[column] or not row[column].strip() for column in REQUIRED_COLUMNS):
            raise ValueError(f"Nilai kosong pada baris dataset {line_number}.")
        if row["sample_id"] in seen_ids:
            raise ValueError(f"sample_id duplikat: {row['sample_id']}")
        seen_ids.add(row["sample_id"])
    return rows


def run_evaluation(
    dataset_path: Path = DATASET_PATH, results_path: Path = RESULTS_PATH
) -> list[dict[str, object]]:
    """Evaluate the same SentenceTransformer and FAISS IndexFlatL2 as rag.py."""
    dataset = load_dataset(dataset_path)
    condition_to_index = {item["kondisi"]: index for index, item in enumerate(rag.structured_data)}
    if len(condition_to_index) != len(rag.structured_data):
        raise ValueError("Kondisi knowledge base duplikat; ground truth tidak dapat dipetakan unik.")

    output_rows: list[dict[str, object]] = []
    for sample in dataset:
        condition = sample["ground_truth_kondisi"]
        if condition not in condition_to_index:
            raise ValueError(f"Ground truth tidak ditemukan di knowledge base: {condition}")
        expected_risk = rag.structured_data[condition_to_index[condition]]["risiko"]
        if expected_risk != sample["ground_truth_risiko"]:
            raise ValueError(f"Risiko acuan tidak cocok untuk {sample['sample_id']}: {expected_risk}")

        query_embedding, distances, indices = rag.retrieve_references(sample["laporan"], k=2)
        query_embedding = np.asarray(query_embedding, dtype="float32")
        for rank, (distance, document_index) in enumerate(zip(distances[0], indices[0]), start=1):
            reference_embedding = np.asarray(rag.doc_embeddings[document_index], dtype="float32")
            reference = rag.structured_data[int(document_index)]
            output_rows.append({
                "sample_id": sample["sample_id"],
                "laporan": sample["laporan"],
                "ground_truth_kondisi": condition,
                "ground_truth_risiko": sample["ground_truth_risiko"],
                "data_asal": sample["data_asal"],
                "rank": rank,
                "retrieved_kondisi": reference["kondisi"],
                "retrieved_risiko": reference["risiko"],
                "faiss_squared_l2_distance": float(distance),
                "cosine_similarity": cosine_similarity(query_embedding[0], reference_embedding),
                "is_ground_truth": reference["kondisi"] == condition,
            })

    fieldnames = list(output_rows[0])
    with results_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    return output_rows


def print_summary(rows: list[dict[str, object]]) -> None:
    cosine_scores = [float(row["cosine_similarity"]) for row in rows]
    l2_distances = [float(row["faiss_squared_l2_distance"]) for row in rows]
    print(f"Sampel diuji: {len({row['sample_id'] for row in rows})}")
    print(f"Referensi dievaluasi: {len(rows)} (k=2)")
    print("Cosine similarity (laporan vs referensi): "
          f"rata-rata={np.mean(cosine_scores):.4f}, min={np.min(cosine_scores):.4f}, "
          f"maks={np.max(cosine_scores):.4f}")
    print("FAISS squared-L2 distance (bukan cosine similarity): "
          f"rata-rata={np.mean(l2_distances):.4f}, min={np.min(l2_distances):.4f}, "
          f"maks={np.max(l2_distances):.4f}")


if __name__ == "__main__":
    rows = run_evaluation()
    print_summary(rows)
    print(f"Hasil disimpan: {RESULTS_PATH}")
