"""Regression checks for the SIGAP retrieval data contract."""

import csv
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

    def test_retrieval_matches_report_to_the_hazard_condition(self):
        import rag

        _, _, indexes = rag.retrieve_references(
            "Pekerja tidak menggunakan masker pada area penuh debu", k=1
        )

        self.assertEqual(
            rag.structured_data[indexes[0][0]]["kondisi"],
            "Tidak menggunakan masker di area berdebu",
        )

    def test_single_hazard_report_does_not_add_a_weak_related_finding(self):
        import rag

        _, findings = rag.rag_system("Tidak ada rencana tanggap darurat.")

        self.assertEqual(
            [item["kondisi"] for item in findings],
            ["Tidak ada rencana tanggap darurat"],
        )

    def test_report_with_two_hazards_returns_both_matching_findings(self):
        import rag

        level, findings = rag.rag_system(
            "Pekerja bekerja di ketinggian tanpa safety harness dan kabel listrik terbuka di dekat area kerja."
        )

        self.assertEqual(level, "TINGGI")
        self.assertEqual(
            {item["kondisi"] for item in findings},
            {
                "Pekerja bekerja di ketinggian tanpa safety harness",
                "Kabel listrik terbuka",
            },
        )

    def test_unrelated_report_is_not_assigned_a_risk_level(self):
        import rag

        level, findings = rag.rag_system("Bagaimana cara memasak nasi goreng?")

        self.assertEqual(level, "TIDAK DAPAT DINILAI")
        self.assertEqual(findings, [])

    def test_cold_report_is_not_overridden_by_an_unrelated_high_risk_match(self):
        import rag

        level, findings = rag.rag_system(
            "Lokasi kerja dingin sehingga berisiko mengganggu kesehatan"
        )

        self.assertEqual(level, "RENDAH")
        self.assertEqual(findings[0]["kondisi"], "Suhu terlalu dingin")

    def test_briefing_report_does_not_inherit_a_weak_high_risk_match(self):
        import rag

        level, findings = rag.rag_system(
            "Tidak ada briefing keselamatan sebelum aktivitas dimulai"
        )

        self.assertEqual(level, "SEDANG")
        self.assertTrue(all(item["risiko"] == "SEDANG" for item in findings))

    def test_synthetic_reports_retrieve_their_reference_and_risk(self):
        import rag

        dataset_path = BACKEND_DIR / "evaluation" / "test_dataset.csv"
        with dataset_path.open(encoding="utf-8-sig", newline="") as file:
            samples = list(csv.DictReader(file))

        for sample in samples:
            with self.subTest(sample_id=sample["sample_id"]):
                level, findings = rag.rag_system(sample["laporan"])

                self.assertIn(
                    sample["ground_truth_kondisi"],
                    {item["kondisi"] for item in findings},
                )
                self.assertEqual(level, sample["ground_truth_risiko"])


class RagasIntegrationTests(unittest.TestCase):
    def test_unconfigured_evaluator_is_unavailable_not_fake_score(self):
        from evaluation.ragas_eval import evaluate_ragas

        with patch.dict("os.environ", {}, clear=True):
            result = evaluate_ragas(
                question="Kabel listrik terbuka",
                answer="Matikan sumber listrik.",
                contexts=["Kabel listrik terbuka"],
                similarity_scores=[0.8, 0.6],
            )

        self.assertEqual(result["status"], "BELUM_TERSEDIA")
        self.assertIsNone(result["context_precision"])
        self.assertIsNone(result["faithfulness"])
        self.assertAlmostEqual(result["average_similarity"], 0.7)

    def test_empty_context_does_not_invoke_evaluator(self):
        from evaluation.ragas_eval import evaluate_ragas

        result = evaluate_ragas(
            question="Nasi goreng",
            answer="Belum dapat dinilai.",
            contexts=[],
        )

        self.assertEqual(result["status"], "BELUM_TERSEDIA")
        self.assertIsNone(result["maximum_similarity"])

    def test_only_successful_ragas_scores_are_appended(self):
        import tempfile
        from evaluation.ragas_eval import RAGAS_COLUMNS, record_ragas_result

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hasil_ragas.csv"
            record_ragas_result(
                question="Pertanyaan",
                answer="Jawaban",
                contexts=["Konteks"],
                similarity_scores=[0.5],
                ground_truth=None,
                evaluation={
                    "status": "BELUM_TERSEDIA",
                    "status_detail": "Tidak ada evaluator",
                    "average_similarity": 0.5,
                },
                results_path=path,
            )
            self.assertFalse(path.exists())

            record_ragas_result(
                question="Pertanyaan",
                answer="Jawaban",
                contexts=["Konteks"],
                similarity_scores=[0.5],
                ground_truth=None,
                evaluation={
                    "status": "BERHASIL",
                    "status_detail": "Berhasil",
                    "average_similarity": 0.5,
                    "maximum_similarity": 0.5,
                    "minimum_similarity": 0.5,
                    "context_precision": None,
                    "context_recall": None,
                    "faithfulness": 0.8,
                    "answer_relevancy": 0.9,
                    "answer_correctness": None,
                    "ground_truth": None,
                },
                results_path=path,
            )
            with path.open(encoding="utf-8", newline="") as file:
                reader = csv.DictReader(file)
                rows = list(reader)
                self.assertEqual(set(reader.fieldnames or []), set(RAGAS_COLUMNS))
            self.assertEqual(len(rows), 1)


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
        self.assertTrue({
            "level", "temuan", "penyebab", "rekomendasi", "referensi",
            "retrieved_documents", "similarity_evaluation", "answer", "ragas_evaluation",
        }.issubset(payload))
        self.assertIn(
            payload["level"], {"RENDAH", "SEDANG", "TINGGI", "TIDAK DAPAT DINILAI"}
        )
        self.assertEqual(payload["referensi"], "Knowledge SIGAP AI")

    def test_analyze_rejects_empty_or_non_text_reports(self):
        from app import app

        client = app.test_client()
        for payload in (
            {"laporan": "   "},
            {"laporan": "teks pendek"},
            {"laporan": None},
            {"laporan": ["bahaya"]},
        ):
            with self.subTest(payload=payload):
                response = client.post("/api/analyze", json=payload)

                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.get_json())

    def test_random_sample_is_one_sentence_from_the_knowledge_base(self):
        import rag
        from app import app

        sample = rag.structured_data[0]
        with patch("app.random.choice", return_value=sample):
            response = app.test_client().get("/api/sample")

        self.assertEqual(response.status_code, 200)
        sentence = response.get_json()["laporan"]
        self.assertEqual(sentence, sample["kondisi"].rstrip(".!?") + ".")
        self.assertNotIn("\n", sentence)

        for condition in ("Tidak ada SOP", "Lingkungan kotor", "Area tergenang air", "Area sempit"):
            with self.subTest(condition=condition), patch(
                "app.random.choice", return_value={"kondisi": condition}
            ):
                response = app.test_client().get("/api/sample")

                self.assertEqual(response.status_code, 200)
                sentence = response.get_json()["laporan"]
                self.assertGreaterEqual(len(sentence), 20)
                self.assertTrue(sentence.endswith("."))
                self.assertNotIn("\n", sentence)
                self.assertTrue(sentence.endswith(condition + "."))

    def test_unassessed_report_can_be_included_in_interaction_evaluation(self):
        from app import app
        from evaluation.interaction_logger import read_interactions
        from evaluation.interaction_eval import evaluate_interactions

        with tempfile.TemporaryDirectory() as temporary_directory:
            log_path = Path(temporary_directory) / "interaction_logs.jsonl"
            results_path = Path(temporary_directory) / "results.csv"
            app.config["INTERACTION_LOG_PATH"] = log_path
            try:
                response = app.test_client().post(
                    "/api/analyze",
                    json={"laporan": "Bagaimana cara memasak nasi goreng?"},
                )
                self.assertEqual(response.status_code, 200)

                records = read_interactions(log_path)
                self.assertEqual(records[0]["retrieved_contexts"], [])
                rows, summary = evaluate_interactions(log_path, results_path)
            finally:
                app.config.pop("INTERACTION_LOG_PATH", None)

        self.assertEqual(rows, [])
        self.assertEqual(
            summary["risk_distribution"], {"TIDAK DAPAT DINILAI": 1}
        )

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
                self.assertGreaterEqual(len(records[0]["retrieved_contexts"]), 1)
                self.assertLessEqual(len(records[0]["retrieved_contexts"]), 2)

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
            {
                "level", "temuan", "penyebab", "rekomendasi", "referensi",
                "retrieved_documents", "similarity_evaluation", "answer", "ragas_evaluation",
            }.issubset(response.get_json()),
            True,
        )


if __name__ == "__main__":
    unittest.main()
