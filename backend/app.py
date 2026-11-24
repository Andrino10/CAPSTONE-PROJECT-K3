import random

from flask import Flask, request, jsonify
from flask_cors import CORS

from rag import rag_system
from rag import structured_data
from evaluation.interaction_logger import record_interaction
from evaluation.ragas_eval import evaluate_ragas, ground_truth_for_query, record_ragas_result

app = Flask(__name__)
CORS(app)

@app.route("/api/sample", methods=["GET"])
def sample():
    condition = random.choice(structured_data)["kondisi"].strip().rstrip(".!?")
    if len(condition) < 20:
        condition = f"Kondisi kerja: {condition}"
    return jsonify({"laporan": f"{condition}."})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Format laporan tidak valid"}), 400

    laporan = data.get("laporan")

    if not isinstance(laporan, str) or not laporan.strip():
        return jsonify({"error": "Laporan harus berupa teks dan tidak boleh kosong"}), 400

    laporan = laporan.strip()
    if len(laporan) < 20:
        return jsonify({"error": "Laporan harus berisi minimal 20 karakter"}), 400

    level, temuan, retrieval_contexts = rag_system(laporan, include_retrieval=True)

    if level == "TIDAK DAPAT DINILAI":
        hasil_temuan = ["Belum ada bahaya yang dapat dipastikan dari knowledge base."]
        penyebab = (
            "Laporan belum cukup cocok dengan referensi bahaya yang tersedia, "
            "jadi tingkat risiko belum dapat ditentukan."
        )
        rekomendasi = [
            "Tambahkan detail aktivitas, lokasi, kondisi bahaya, dan APD yang digunakan "
            "agar laporan dapat dicocokkan dengan lebih tepat."
        ]
        referensi = "Belum ditemukan referensi knowledge yang cukup sesuai"
    else:
        hasil_temuan = []
        rekomendasi = []
        penyebab = []

        for temuan_item in temuan:
            hasil_temuan.append(temuan_item["kondisi"])
            penyebab.append(temuan_item["penjelasan"])
            rekomendasi.append(temuan_item["solusi"])
        penyebab = " ".join(penyebab)
        referensi = "Knowledge SIGAP AI"

    jawaban_sistem = " ".join([
        penyebab if isinstance(penyebab, str) else " ",
        " ".join(rekomendasi) if isinstance(rekomendasi, list) else str(rekomendasi),
    ]).strip()
    contexts = [
        str(context["document"]["kondisi"])
        for context in retrieval_contexts
        if isinstance(context, dict) and isinstance(context.get("document"), dict)
    ]
    similarity_scores = [
        float(context["cosine_similarity"])
        for context in retrieval_contexts
        if isinstance(context, dict) and isinstance(context.get("cosine_similarity"), (int, float))
    ]
    ground_truth_context, ground_truth_answer = ground_truth_for_query(laporan)
    ragas_evaluation = evaluate_ragas(
        question=laporan,
        answer=jawaban_sistem,
        contexts=contexts,
        ground_truth_context=ground_truth_context,
        ground_truth_answer=ground_truth_answer,
        similarity_scores=similarity_scores,
    )
    try:
        record_ragas_result(
            question=laporan,
            answer=jawaban_sistem,
            contexts=contexts,
            similarity_scores=similarity_scores,
            ground_truth=ground_truth_answer or ground_truth_context,
            evaluation=ragas_evaluation,
        )
    except Exception:
        app.logger.exception("Gagal mencatat hasil evaluasi RAGAS")

    response_payload = {
        "level": level,
        "temuan": hasil_temuan,
        "penyebab": penyebab,
        "rekomendasi": rekomendasi,
        "referensi": referensi,
        "retrieved_documents": [
            {
                "rank": context["rank"],
                "document": context["document"],
                "cosine_similarity": context["cosine_similarity"],
                "faiss_squared_l2_distance": context["faiss_squared_l2_distance"],
            }
            for context in retrieval_contexts
        ],
        "similarity_evaluation": {
            "average": ragas_evaluation["average_similarity"],
            "maximum": ragas_evaluation["maximum_similarity"],
            "minimum": ragas_evaluation["minimum_similarity"],
        },
        "answer": jawaban_sistem,
        "ragas_evaluation": ragas_evaluation,
    }

    try:
        record_interaction(
            laporan,
            response_payload,
            retrieval_contexts,
            request_id=request.headers.get("X-Interaction-Request-ID"),
            log_path=app.config.get("INTERACTION_LOG_PATH"),
        )
    except Exception:
        # Logging failure must be visible to the server operator, but it must
        # not alter the established API response consumed by the frontend.
        app.logger.exception("Gagal mencatat interaksi analisis SIGAP AI")

    return jsonify(response_payload)

if __name__ == "__main__":
    app.run(debug=True)
