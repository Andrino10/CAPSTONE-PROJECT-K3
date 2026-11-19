from flask import Flask, request, jsonify
from flask_cors import CORS

from rag import rag_system
from evaluation.interaction_logger import record_interaction

app = Flask(__name__)
CORS(app)

@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.json
    laporan = data.get("laporan", "")

    if not laporan:
        return jsonify({"error": "Laporan kosong"}), 400

    level, temuan, retrieval_contexts = rag_system(laporan, include_retrieval=True)

    # Ambil isi detail
    hasil_temuan = []
    rekomendasi = []
    penyebab = []

    for t in temuan:
        hasil_temuan.append(t["kondisi"])
        penyebab.append(t["penjelasan"])
        rekomendasi.append(t["solusi"])

    response_payload = {
        "level": level,
        "temuan": hasil_temuan,
        "penyebab": " ".join(penyebab),
        "rekomendasi": rekomendasi,
        "referensi": "Knowledge SIGAP AI"
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
