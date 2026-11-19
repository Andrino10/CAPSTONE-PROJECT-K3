"""Regression checks for the SIGAP retrieval data contract."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))


class KnowledgeBaseAlignmentTests(unittest.TestCase):
    def test_faiss_documents_and_structured_references_are_aligned(self):
        import rag

        self.assertGreater(len(rag.docs), 0)
        self.assertEqual(len(rag.docs), len(rag.structured_data))
        self.assertTrue(all(set(item) == {"kondisi", "risiko", "penjelasan", "solusi"}
                            for item in rag.structured_data))

    def test_shared_retrieval_helper_returns_two_ranked_references(self):
        import rag

        _, distances, indexes = rag.retrieve_references("Kabel listrik terbuka", k=2)

        self.assertEqual(distances.shape, (1, 2))
        self.assertEqual(indexes.shape, (1, 2))


class RetrievalMetricTests(unittest.TestCase):
    def test_metrics_use_relevance_labels_and_rank(self):
        from backend.evaluation.ragas_metrics import calculate_retrieval_metrics

        metrics = calculate_retrieval_metrics([[True, False], [False, True], [False, False]])

        self.assertAlmostEqual(metrics["precision_at_2"], 1 / 3)
        self.assertAlmostEqual(metrics["recall_at_2"], 2 / 3)
        self.assertAlmostEqual(metrics["hit_at_2"], 2 / 3)
        self.assertAlmostEqual(metrics["mrr"], 0.5)

    def test_validation_module_is_importable_from_project_root(self):
        from backend.evaluation import check_metrics

        self.assertTrue(callable(check_metrics.validate))


class ApiIntegrationTests(unittest.TestCase):
    def test_analyze_endpoint_preserves_response_contract(self):
        from app import app

        with tempfile.TemporaryDirectory() as temporary_directory:
            app.config["INTERACTION_LOG_PATH"] = Path(temporary_directory) / "interaction_logs.jsonl"
            try:
                response = app.test_client().post(
                    "/api/analyze", json={"laporan": "Kabel listrik terbuka berada di dekat pekerja"}
                )
            finally:
                app.config.pop("INTERACTION_LOG_PATH", None)

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(
            set(payload), {"level", "temuan", "penyebab", "rekomendasi", "referensi"}
        )
        self.assertIn(payload["level"], {"RENDAH", "SEDANG", "TINGGI"})
        self.assertEqual(payload["referensi"], "Knowledge SIGAP AI")

    def test_successful_analysis_is_logged_with_actual_response_and_context(self):
        from app import app

        with tempfile.TemporaryDirectory() as temporary_directory:
            log_path = Path(temporary_directory) / "interaction_logs.jsonl"
            app.config["INTERACTION_LOG_PATH"] = log_path
            try:
                response = app.test_client().post(
                    "/api/analyze",
                    headers={"X-Interaction-Request-ID": "test-request-001"},
                    json={"laporan": "Kabel listrik terbuka berada di dekat pekerja"},
                )
                self.assertEqual(response.status_code, 200)
                self.assertTrue(log_path.exists())
                from evaluation.interaction_logger import read_interactions

                records = read_interactions(log_path)
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["laporan"], "Kabel listrik terbuka berada di dekat pekerja")
                self.assertEqual(records[0]["response"], response.get_json())
                self.assertEqual(records[0]["request_id"], "test-request-001")
                self.assertEqual(len(records[0]["retrieved_contexts"]), 2)

                duplicate_response = app.test_client().post(
                    "/api/analyze",
                    headers={"X-Interaction-Request-ID": "test-request-001"},
                    json={"laporan": "Kabel listrik terbuka berada di dekat pekerja"},
                )
                self.assertEqual(duplicate_response.status_code, 200)
                self.assertEqual(len(read_interactions(log_path)), 1)

                from evaluation.interaction_eval import evaluate_interactions

                result_path = Path(temporary_directory) / "hasil_interaction_similarity.csv"
                rows, summary = evaluate_interactions(log_path, result_path)
                self.assertEqual(len(rows), 2)
                self.assertTrue(result_path.exists())
                self.assertEqual(rows[0]["interaction_id"], records[0]["interaction_id"])
                self.assertEqual(summary["successful_records"], 1)
            finally:
                app.config.pop("INTERACTION_LOG_PATH", None)

    def test_logging_failure_does_not_change_analyze_response(self):
        from app import app

        with tempfile.TemporaryDirectory() as temporary_directory:
            app.config["INTERACTION_LOG_PATH"] = Path(temporary_directory) / "interaction_logs.jsonl"
            try:
                with patch("app.record_interaction", side_effect=OSError("disk penuh")):
                    response = app.test_client().post(
                        "/api/analyze", json={"laporan": "Kabel listrik terbuka berada di dekat pekerja"}
                    )
            finally:
                app.config.pop("INTERACTION_LOG_PATH", None)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            set(response.get_json()), {"level", "temuan", "penyebab", "rekomendasi", "referensi"}
        )


if __name__ == "__main__":
    unittest.main()
