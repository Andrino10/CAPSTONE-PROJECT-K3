from collections import Counter
import math
from pathlib import Path
import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


embed_model = SentenceTransformer("all-MiniLM-L6-v2")

docs = []
structured_data = []
KNOWLEDGE_PATH = Path(__file__).with_name("knowledge.txt")

with KNOWLEDGE_PATH.open("r", encoding="utf-8") as file:
    for line in file:
        parts = [part.strip() for part in line.strip().split("|")]
        if len(parts) != 4 or parts[0] == "[KONDISI]":
            continue

        kondisi, risiko, penjelasan, solusi = parts
        docs.append(kondisi)
        structured_data.append({
            "kondisi": kondisi,
            "risiko": risiko,
            "penjelasan": penjelasan,
            "solusi": solusi,
        })

doc_embeddings = np.asarray(embed_model.encode(docs), dtype="float32")
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)


def cosine_similarity(left, right):
    """Return cosine similarity clamped to its mathematical range."""
    left_vector = np.asarray(left, dtype="float64")
    right_vector = np.asarray(right, dtype="float64")
    denominator = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
    if denominator == 0:
        raise ValueError("Embedding bernorma nol tidak dapat dihitung cosinenya.")
    score = float(np.dot(left_vector, right_vector) / denominator)
    return float(np.clip(score, -1.0, 1.0))


STOP_WORDS = {
    "agar", "akan", "area", "atau", "bagaimana", "bahaya", "bahwa",
    "bekerja", "berisiko", "dalam", "dan", "dari", "dengan", "di", "ini",
    "itu", "karena", "ke", "kepada", "kondisi", "kerja", "lebih", "lokasi",
    "melakukan", "memadai", "menggunakan", "oleh", "pada", "pekerja",
    "pekerjaan", "risiko", "sebagai", "sehingga", "saat", "secara", "tanpa",
    "tempat", "terhadap", "tidak", "untuk", "yang",
}


def _tokenize(text):
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in STOP_WORDS
    }


doc_tokens = [_tokenize(item["kondisi"]) for item in structured_data]
document_frequency = Counter(token for tokens in doc_tokens for token in tokens)
inverse_document_frequency = {
    token: math.log((len(docs) + 1) / (frequency + 1)) + 1
    for token, frequency in document_frequency.items()
}
MIN_RELEVANCE_SCORE = 0.45
MAX_RELATED_DOCUMENTS = 2


def _score_reference(query, query_embedding, document_index):
    query_vector = np.asarray(query_embedding[0], dtype="float32")
    document_vector = doc_embeddings[document_index]
    semantic_similarity = cosine_similarity(query_vector, document_vector)
    query_tokens = _tokenize(query)
    total_weight = sum(inverse_document_frequency.get(token, 1) for token in query_tokens)
    if total_weight:
        shared_tokens = query_tokens & doc_tokens[document_index]
        lexical_coverage = (
            sum(inverse_document_frequency.get(token, 1) for token in shared_tokens)
            / total_weight
        )
    else:
        lexical_coverage = 0.0

    relevance_score = 0.45 * semantic_similarity + 0.55 * lexical_coverage
    return relevance_score, semantic_similarity, lexical_coverage


def _rank_references(query, k):
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("Jumlah referensi harus berupa bilangan bulat positif.")

    query_embedding = np.asarray(embed_model.encode([query]), dtype="float32")
    all_distances, all_indexes = index.search(query_embedding, len(docs))
    distances_by_index = {
        int(document_index): float(distance)
        for distance, document_index in zip(all_distances[0], all_indexes[0])
    }

    ranked = []
    for document_index in all_indexes[0]:
        document_index = int(document_index)
        relevance_score, semantic_similarity, lexical_coverage = _score_reference(
            query, query_embedding, document_index
        )
        ranked.append((
            relevance_score,
            semantic_similarity,
            lexical_coverage,
            distances_by_index[document_index],
            document_index,
        ))

    ranked.sort(key=lambda result: (-result[0], -result[1], result[4]))
    selected = ranked[:min(k, len(ranked))]
    distances = np.asarray([[result[3] for result in selected]], dtype="float32")
    indexes = np.asarray([[result[4] for result in selected]], dtype="int64")
    scores = [(result[0], result[2]) for result in selected]
    return query_embedding, distances, indexes, scores


def retrieve_references(query, k=3):
    """Return references ranked by condition similarity and informative word overlap."""
    query_embedding, distances, indexes, _ = _rank_references(query, k)
    return query_embedding, distances, indexes


def analisis_risiko(_query, retrieved_indexes):
    hasil_temuan = [structured_data[int(idx)] for idx in retrieved_indexes]
    if not hasil_temuan:
        return "TIDAK DAPAT DINILAI", []

    risiko_tertinggi = max(
        (data["risiko"] for data in hasil_temuan),
        key={"RENDAH": 1, "SEDANG": 2, "TINGGI": 3}.get,
    )
    return risiko_tertinggi, hasil_temuan


def _split_query(query):
    parts = re.split(
        r"(?<=[.!?;])\s+|\s+(?:dan|serta|sedangkan|sementara)\s+",
        query,
        flags=re.IGNORECASE,
    )
    return [part.strip() for part in parts if part.strip()]


def rag_system(query, include_retrieval=False):
    query_embedding = np.asarray(embed_model.encode([query]), dtype="float32")[0]
    matching_results = []
    seen_indexes = set()
    risk_matching_results = []
    seen_risk_indexes = set()

    for query_part in _split_query(query):
        _, _, indexes, scores = _rank_references(
            query_part, k=MAX_RELATED_DOCUMENTS
        )
        top_score = scores[0][0]
        top_risk = structured_data[int(indexes[0][0])]["risiko"]
        for rank, (document_index, (relevance_score, lexical_coverage)) in enumerate(
            zip(indexes[0], scores)
        ):
            document_index = int(document_index)
            if relevance_score < MIN_RELEVANCE_SCORE:
                continue

            distance = float(np.sum((query_embedding - doc_embeddings[document_index]) ** 2))
            if document_index not in seen_indexes:
                seen_indexes.add(document_index)
                matching_results.append((
                    distance,
                    document_index,
                    relevance_score,
                    lexical_coverage,
                ))

            if rank > 0 and (
                top_score - relevance_score > 0.1
                or structured_data[document_index]["risiko"] != top_risk
            ):
                continue
            if document_index not in seen_risk_indexes:
                seen_risk_indexes.add(document_index)
                risk_matching_results.append((
                    distance,
                    document_index,
                    relevance_score,
                    lexical_coverage,
                ))

    matching_results.sort(key=lambda result: (-result[2], result[0], result[1]))
    matching_results = matching_results[:MAX_RELATED_DOCUMENTS]

    retrieved_indexes = [
        document_index for _, document_index, _, _ in risk_matching_results
    ]
    level, temuan = analisis_risiko(query, retrieved_indexes)

    if include_retrieval:
        retrieval_contexts = [
            {
                "rank": rank,
                "faiss_squared_l2_distance": float(distance),
                "cosine_similarity": cosine_similarity(
                    query_embedding, doc_embeddings[document_index]
                ),
                "relevance_score": float(relevance_score),
                "lexical_coverage": float(lexical_coverage),
                "document": structured_data[document_index],
            }
            for rank, (distance, document_index, relevance_score, lexical_coverage)
            in enumerate(matching_results, start=1)
        ]
        return level, temuan, retrieval_contexts

    return level, temuan
