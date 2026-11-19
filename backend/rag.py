from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pathlib import Path

print("Loading models...")

# =========================
# 1. Load Embedding Model
# =========================
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# =========================
# 2. Load Knowledge Base
# =========================
docs = []
structured_data = []

KNOWLEDGE_PATH = Path(__file__).with_name("knowledge.txt")

with KNOWLEDGE_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            # Only index records that have matching structured metadata.  The
            # first row of knowledge.txt is a header and must not shift FAISS
            # indices relative to structured_data.
            parts = line.split("|")
            if len(parts) == 4:
                kondisi = parts[0].strip()
                risiko = parts[1].strip()
                penjelasan = parts[2].strip()
                solusi = parts[3].strip()

                if (kondisi, risiko, penjelasan, solusi) == (
                    "[KONDISI]", "[RISIKO]", "[PENJELASAN]", "[SOLUSI]"
                ):
                    continue

                docs.append(line)
                structured_data.append({
                    "kondisi": kondisi,
                    "risiko": risiko,
                    "penjelasan": penjelasan,
                    "solusi": solusi
                })

# =========================
# 3. Embedding Knowledge
# =========================
doc_embeddings = embed_model.encode(docs)

dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

print("Sistem siap!\n")


# =========================
# 4. Analisis Risiko (Hybrid)
# =========================
def analisis_risiko(query, retrieved_indexes):
    teks = query.lower()

    hasil_temuan = []
    risiko_tertinggi = "RENDAH"

    for idx in retrieved_indexes:
        data = structured_data[idx]
        hasil_temuan.append(data)

        # Tentukan level tertinggi
        if data["risiko"] == "TINGGI":
            risiko_tertinggi = "TINGGI"
        elif data["risiko"] == "SEDANG" and risiko_tertinggi != "TINGGI":
            risiko_tertinggi = "SEDANG"

    return risiko_tertinggi, hasil_temuan


# =========================
# 5. RAG SYSTEM
# =========================
def retrieve_references(query, k=2):
    """Retrieve references with the same embedding model and FAISS index."""
    query_vec = embed_model.encode([query])
    distances, indexes = index.search(np.array(query_vec), k=k)
    return query_vec, distances, indexes


def rag_system(query, include_retrieval=False):
    _, distances, indexes = retrieve_references(query, k=2)

    # Ambil 2 referensi terbaik
    retrieved_indexes = indexes[0]

    level, temuan = analisis_risiko(query, retrieved_indexes)

    if include_retrieval:
        retrieval_contexts = [
            {
                "rank": rank,
                "faiss_squared_l2_distance": float(distance),
                "document": data,
            }
            for rank, (distance, data) in enumerate(zip(distances[0], temuan), start=1)
        ]
        return level, temuan, retrieval_contexts

    return level, temuan
