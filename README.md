# SIGAP AI - Sistem Intelijen Garda Ancaman Pekerjaan

SIGAP AI adalah sistem berbasis Retrieval-Augmented Generation (RAG) untuk menganalisis laporan aktivitas kerja lapangan, mengidentifikasi potensi bahaya keselamatan dan kesehatan kerja (K3), menentukan tingkat risiko, serta menyusun rekomendasi tindakan mitigasi.

Proyek ini menggunakan FAISS dan Sentence Transformers untuk mencari kondisi bahaya yang relevan dari `backend/knowledge.txt`. RAGAS digunakan sebagai lapisan evaluasi tambahan untuk menilai kualitas retrieval dan jawaban. RAGAS tidak menggantikan proses RAG, FAISS, embedding, cosine similarity, atau pembentukan jawaban SIGAP yang sudah tersedia.

## Daftar Isi

- [Tujuan dan Teknologi](#tujuan-dan-teknologi)
- [Fitur Utama](#fitur-utama)
- [Arsitektur Singkat](#arsitektur-singkat)
- [Struktur Project](#struktur-project)
- [Persyaratan Sistem](#persyaratan-sistem)
- [Instalasi](#instalasi)
- [Konfigurasi](#konfigurasi)
- [Menjalankan Program](#menjalankan-program)
- [Evaluasi RAG dan RAGAS](#evaluasi-rag-dan-ragas)
- [Contoh Penggunaan](#contoh-penggunaan)
- [Penyimpanan Hasil Evaluasi](#penyimpanan-hasil-evaluasi)
- [Pengujian](#pengujian)
- [Troubleshooting](#troubleshooting)
- [Batasan Sistem](#batasan-sistem)

## Tujuan dan Teknologi

Tujuan pengembangan sistem ini adalah:

- Memproses laporan kerja menggunakan knowledge base K3.
- Menemukan kondisi bahaya yang relevan berdasarkan kemiripan teks.
- Menyusun temuan, penyebab, tingkat risiko, dan rekomendasi mitigasi.
- Menyajikan similarity retrieval yang dapat diperiksa secara kuantitatif.
- Menyediakan evaluasi RAGAS secara modular tanpa mengubah alur RAG utama.
- Menyimpan hasil evaluasi agar dapat ditinjau ulang.

Teknologi utama yang digunakan:

| Teknologi | Fungsi dalam project |
|---|---|
| Python | Backend, retrieval, dan evaluasi |
| Flask 3.0.3 | REST API `/api/analyze` dan `/api/sample` |
| Flask-CORS 4.0.1 | Mengizinkan pemanggilan API dari frontend |
| Sentence Transformers 3.0.1 | Membentuk embedding teks |
| Model `all-MiniLM-L6-v2` | Model embedding yang digunakan proyek |
| FAISS CPU 1.8.0 | Pencarian dokumen melalui `IndexFlatL2` |
| NumPy 1.26.4 | Operasi embedding dan cosine similarity |
| RAGAS 0.4.3 | Lapisan evaluasi retrieval dan jawaban |
| RAGAS 0.4.3 | Framework evaluasi non-generatif untuk Context Precision dan Context Recall |
| HTML, CSS, dan JavaScript | Antarmuka frontend statis |

Node.js tidak digunakan oleh frontend proyek ini. Frontend dijalankan sebagai file statis atau melalui server web statis sederhana.

## Fitur Utama

1. Validasi dan pemrosesan laporan kerja pengguna.
2. Pembuatan embedding laporan menggunakan `all-MiniLM-L6-v2`.
3. Pencarian referensi knowledge base menggunakan FAISS `IndexFlatL2`.
4. Peringkat hasil berdasarkan gabungan semantic similarity dan cakupan kata penting.
5. Perhitungan cosine similarity secara terpisah dari jarak FAISS squared-L2.
6. Pengambilan hingga dua dokumen relevan untuk setiap bagian query.
7. Penyusunan temuan, penyebab, tingkat risiko, dan rekomendasi.
8. Penampilan retrieved documents beserta similarity.
9. Evaluasi similarity rata-rata, maksimum, dan minimum.
10. Evaluasi RAGAS non-generatif untuk Context Precision dan Context Recall.
11. Penyimpanan hasil retrieval dan interaksi ke CSV/JSONL.
12. Dashboard evaluasi pada frontend dengan kartu nilai dan tabel retrieved documents.

## Arsitektur Singkat

Alur pemrosesan SIGAP AI:

```text
Laporan pengguna
      |
      v
Validasi teks dan pemisahan kondisi
      |
      v
Embedding dengan all-MiniLM-L6-v2
      |
      v
Pencarian kandidat dengan FAISS IndexFlatL2
      |
      v
Peringkat ulang berbasis relevance score
      |
      v
Cosine similarity dan penyusunan retrieved contexts
      |
      v
Pembentukan temuan, penyebab, dan rekomendasi
      |
      v
Evaluasi RAGAS opsional
      |
      v
Respons API, dashboard frontend, dan pencatatan hasil
```

FAISS digunakan untuk pencarian kandidat berdasarkan jarak kuadrat L2. Cosine similarity dihitung sebagai:

```text
cosine(q, d) = dot(q, d) / (||q|| * ||d||)
```

Nilai cosine similarity adalah ukuran kemiripan arah embedding dan tidak identik dengan skor RAGAS.

RAGAS dievaluasi setelah retrieval dan pembentukan jawaban. Metrik RAGAS dapat berupa Context Precision, Context Recall, Faithfulness, Answer Relevancy, dan Answer Correctness. Skor RAGAS hanya dianggap berhasil jika evaluator benar-benar mampu menghasilkan skor dalam rentang 0-1.

## Struktur Project

```text
SIGAP_AI-main/
├── backend/
│   ├── app.py
│   ├── rag.py
│   ├── knowledge.txt
│   └── evaluation/
│       ├── __main__.py
│       ├── check_metrics.py
│       ├── interaction_eval.py
│       ├── interaction_logger.py
│       ├── ragas_eval.py
│       ├── ragas_metrics.py
│       ├── similarity_eval_all.py
│       ├── test_single_eval.py
│       ├── test_dataset.csv
│       ├── hasil_similarity.csv
│       ├── hasil_interaction_similarity.csv
│       ├── interaction_logs.jsonl
│       ├── interaction_logs.jsonl.lock
│       └── EVALUATION.md
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── style.css
├── requirements.txt
└── README.md
```

### Penjelasan File Penting

| File atau folder | Fungsi |
|---|---|
| `backend/app.py` | Entry point Flask, endpoint API, penyusunan respons, dan pencatatan interaksi |
| `backend/rag.py` | Inisialisasi embedding, FAISS, knowledge base, retrieval, dan penyusunan temuan |
| `backend/knowledge.txt` | Knowledge base berformat pipe-separated dengan kondisi, risiko, penjelasan, dan solusi |
| `backend/evaluation/ragas_eval.py` | Lapisan evaluasi RAGAS, status evaluator, validasi skor, dan penulisan `hasil_ragas.csv` |
| `backend/evaluation/ragas_metrics.py` | Precision@2, Recall@2, Hit@2, dan MRR berbasis ground truth retrieval |
| `backend/evaluation/similarity_eval_all.py` | Evaluasi cosine dan jarak retrieval untuk seluruh `test_dataset.csv` |
| `backend/evaluation/check_metrics.py` | Validasi rentang skor, struktur CSV, rank, dan kelengkapan hasil |
| `backend/evaluation/interaction_logger.py` | Penyimpanan append-only setiap interaksi berhasil ke JSONL |
| `backend/evaluation/interaction_eval.py` | Evaluasi similarity untuk interaksi aktual yang telah dicatat |
| `backend/evaluation/test_single_eval.py` | Pengujian unit dan integrasi SIGAP |
| `backend/evaluation/test_dataset.csv` | Dataset sintetis 12 skenario beserta ground truth kondisi dan risiko |
| `frontend/index.html` | Struktur dashboard frontend |
| `frontend/app.js` | Pemanggilan API, pemetaan respons, dan rendering hasil |
| `frontend/style.css` | Tampilan dan responsive layout |
| `requirements.txt` | Dependency Python project |

## Persyaratan Sistem

### Sistem Operasi

Dokumentasi dan pengujian saat ini dilakukan pada Windows menggunakan PowerShell. Perintah juga dapat disesuaikan untuk Linux yang mendukung Python dan FAISS CPU.

### Versi Runtime

Lingkungan pengujian menggunakan:

- Python `3.13`
- Windows
- Flask `3.0.3`

Pastikan versi Python mendukung paket FAISS CPU dan PyTorch yang tercantum pada `requirements.txt`.

### Node.js

Node.js tidak digunakan. Tidak diperlukan `npm install`, `package.json`, atau proses build frontend.

### Package Utama

Versi dependency yang tercatat pada `requirements.txt`:

| Package | Versi |
|---|---|
| Flask | 3.0.3 |
| Flask-CORS | 4.0.1 |
| Sentence Transformers | 3.0.1 |
| FAISS CPU | 1.8.0 |
| NumPy | 1.26.4 |
| SciPy | 1.13.1 |
| PyTorch | 2.3.1 |
| Transformers | 4.41.2 |
| RAGAS | 0.4.3 |
| Datasets | 4.8.5 |
| LangChain | 1.3.11 |
| LangChain Community | 0.4.2 |
| LangChain Core | 1.4.8 |
| LangChain HuggingFace | 1.2.2 |

Beberapa package dapat memerlukan koneksi internet saat instalasi atau saat model Sentence Transformers pertama kali diunduh.

### Keterbatasan Evaluator RAGAS

Implementasi saat ini tidak menambahkan Ollama, layanan LLM, atau API berbayar.
Context Precision dan Context Recall dapat dihitung menggunakan metrik non-LLM
RAGAS apabila ground truth konteks tersedia.

Faithfulness, Answer Relevancy, dan Answer Correctness tidak dihitung karena:

- Faithfulness membutuhkan evaluator generatif.
- Answer Relevancy membutuhkan pemodelan generatif untuk menghasilkan pertanyaan.
- Answer Correctness membutuhkan jawaban referensi yang valid dan tidak tersedia di dataset saat ini.

## Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/Andrino10/CAPSTONE-PROJECT-K3.git
cd CAPSTONE-PROJECT-K3
```

### 2. Buat Virtual Environment

Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

Jika PowerShell menolak eksekusi script, gunakan Command Prompt:

```bat
venv\Scripts\activate.bat
```

Linux atau macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instal Dependency

File dependency berada pada direktori utama:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Perintah yang digunakan pada lingkungan pengujian adalah:

```powershell
python -m unittest backend.evaluation.test_single_eval -v
```

## Konfigurasi

Project saat ini tidak menyediakan file `.env` dan tidak mengharuskan API key.

Knowledge base dimuat langsung dari:

```text
backend/knowledge.txt
```

Jangan mengganti format kolom knowledge base menjadi format lain tanpa menyesuaikan parser di `backend/rag.py`. Baris yang tidak memiliki tepat empat kolom dipilih oleh parser sebagai baris yang dilewati.

FAISS dibangun di memori saat `backend/rag.py` dimuat. Tidak ada direktori penyimpanan index FAISS yang terpisah.

Tidak ada environment variable evaluator generatif yang digunakan.
Tidak ada API key atau layanan berbayar yang dikonfigurasi otomatis.

## Menjalankan Program

### 1. Aktifkan Virtual Environment

Windows:

```powershell
venv\Scripts\Activate.ps1
```

Linux atau macOS:

```bash
source venv/bin/activate
```

### 2. Jalankan Backend

Dari direktori utama:

```powershell
cd backend
python app.py
```

Backend kemudian berjalan pada:

```text
http://localhost:5000
```

Endpoint yang tersedia:

| Method | Endpoint | Fungsi |
|---|---|---|
| GET | `/api/sample` | Mengambil contoh kondisi dari knowledge base |
| POST | `/api/analyze` | Menganalisis laporan dan menjalankan retrieval/evaluasi |

Contoh request:

```json
{
  "laporan": "Pekerja tidak memakai helm di proyek konstruksi"
}
```

### 3. Buka Frontend

Frontend sekarang disajikan oleh Flask dari server yang sama dengan API.
Setelah backend berjalan, buka:

```text
http://localhost:5000
```

Tidak perlu menjalankan server frontend kedua. Frontend akan memanggil API
melalui origin yang sama, sehingga masalah koneksi lintas port dapat dihindari.

Jika ingin tetap memakai Live Server atau `python -m http.server`, frontend
tetap akan memakai backend lokal di `http://localhost:5000`.

### 4. Menggunakan Aplikasi

1. Buka `http://localhost:8080`.
2. Masukkan laporan kerja minimal 20 karakter atau pilih tombol **Contoh**.
3. Tekan **ANALISIS SEKARANG**.
4. Sistem menampilkan level risiko, temuan, penyebab, dan rekomendasi.
5. Bagian evaluasi menampilkan average similarity dan daftar retrieved documents.
6. Jika evaluator belum dikonfigurasi, metrik RAGAS ditampilkan sebagai tidak tersedia.

## Evaluasi RAG dan RAGAS

### Menjalankan Evaluasi Lokal

Perintah evaluasi lengkap dijalankan dari folder `backend`:

```powershell
cd backend
python -m evaluation
```

Perintah tersebut menjalankan:

- Evaluasi similarity untuk `test_dataset.csv`.
- Penulisan `hasil_similarity.csv`.
- Perhitungan Precision@2, Recall@2, Hit@2, dan MRR.
- Validasi hasil.
- Evaluasi interaksi dari `interaction_logs.jsonl`.
- Preflight RAGAS.

Hasil evaluasi yang sudah dijalankan pada project saat ini:

```text
Sampel diuji: 12
Referensi dievaluasi: 24
Cosine similarity rata-rata: 0.7390
Cosine similarity minimum: 0.4794
Cosine similarity maksimum: 0.9064
Precision@2: 0.5000
Recall@2: 1.0000
Hit@2: 1.0000
MRR: 0.8750
```

Nilai di atas merupakan hasil aktual evaluasi proyek, bukan ilustrasi.

### Menjalankan Preflight RAGAS

Dari folder `backend`:

```powershell
python -m evaluation.ragas_eval
```

Preflight ini hanya memeriksa package RAGAS dan kelengkapan dataset.
Penghitungan runtime menggunakan metrik non-generatif, tanpa Ollama atau model tambahan.

### Format Data RAGAS

Lapisan RAGAS menerima komponen:

| Komponen | Bentuk data | Keterangan |
|---|---|---|
| `question` atau `user_input` | `str` | Pertanyaan atau laporan pengguna |
| `answer` atau `response` | `str` | Jawaban yang benar-benar disusun sistem |
| `contexts` atau `retrieved_contexts` | `list[str]` | Dokumen yang ditemukan retrieval |
| `ground_truth` atau `reference` | `str` atau `None` | Jawaban acuan jika tersedia |
| `similarity_scores` | `list[float]` | Cosine similarity untuk statistik similarity |

Data tersebut dikonversi ke `ragas.dataset_schema.SingleTurnSample` dan `EvaluationDataset`.

Dataset `test_dataset.csv` saat ini hanya menyediakan:

- `laporan`
- `ground_truth_kondisi`
- `ground_truth_risiko`

Dataset tersebut tidak menyediakan jawaban acuan lengkap atau konteks acuan lengkap untuk semua metrik RAGAS. Oleh karena itu, metrik LLM RAGAS tidak boleh dianggap telah dijalankan hanya karena package RAGAS terpasang.

### Metrik RAGAS

| Metrik | Fungsi | Ground truth |
|---|---|---|
| Context Precision | Mengukur ketepatan urutan konteks terhadap pertanyaan | Dapat membutuhkan referensi |
| Context Recall | Mengukur cakupan informasi yang dibutuhkan | Membutuhkan referensi jawaban atau konteks |
| Faithfulness | Mengukur dukungan jawaban terhadap retrieved contexts | Tidak harus berupa jawaban acuan |
| Answer Relevancy | Mengukur relevansi jawaban terhadap pertanyaan | Biasanya tidak memerlukan ground truth |
| Answer Correctness | Membandingkan jawaban dengan jawaban acuan | Wajib |

Jika ground truth tidak tersedia, Answer Correctness tidak dihitung dan tidak disimpan sebagai skor valid.

### Evaluasi RAGAS Tanpa Model Tambahan

Implementasi saat ini tidak menyediakan Ollama, API LLM, atau model generatif.
Context Precision dan Context Recall dihitung oleh RAGAS non-LLM hanya jika
ground truth konteks tersedia. Faithfulness, Answer Relevancy, dan Answer
Correctness ditampilkan sebagai N/A.

## Contoh Penggunaan

Query:

```text
Pekerja menggunakan alat listrik tanpa grounding.
```

Contoh format retrieval aktual:

```text
Rank 1:
Pekerja tidak menggunakan helm di area konstruksi
Cosine Similarity: hasil aktual dari embedding laporan dan dokumen
```

Query grounding secara nyata akan bergantung pada isi knowledge base. Project saat ini memiliki kondisi dengan grounding pada entry terkait listrik, misalnya:

```text
Pekerja memperbaiki listrik tanpa mematikan sumber daya
Pekerja tidak mengetahui titik pemutus listrik darurat
Pekerja tidak menggunakan alat pelindung listrik
```

Format hasil evaluasi:

```text
Average Similarity: [hasil aktual]
Maximum Similarity: [hasil aktual]
Minimum Similarity: [hasil aktual]
Context Precision: N/A
Context Recall: N/A
Faithfulness: N/A
Answer Relevancy: N/A
Answer Correctness: N/A
Status RAGAS: BELUM_TERSEDIA
```

Contoh nilai aktual untuk query helm:

```text
Average Similarity: 0.8015
Maximum Similarity: 0.8489
Minimum Similarity: 0.7542
Status RAGAS: BELUM_TERSEDIA
```

Nilai similarity bukan nilai RAGAS. Nilai 0-1 pada kedua jenis pengukuran tidak boleh ditafsirkan sebagai skala yang identik.

## Penyimpanan Hasil Evaluasi

### `hasil_similarity.csv`

Lokasi:

```text
backend/evaluation/hasil_similarity.csv
```

File ini dipertahankan untuk evaluasi retrieval dataset. Kolom utamanya mencakup:

- `sample_id`
- `laporan`
- `ground_truth_kondisi`
- `ground_truth_risiko`
- `rank`
- `retrieved_kondisi`
- `retrieved_risiko`
- `faiss_squared_l2_distance`
- `cosine_similarity`
- `is_ground_truth`

Jalankan evaluasi ulang:

```powershell
cd backend
python -m evaluation.similarity_eval_all
```

### `hasil_interaction_similarity.csv`

Lokasi:

```text
backend/evaluation/hasil_interaction_similarity.csv
```

File ini menyimpan similarity untuk interaksi aktual yang berhasil dicatat oleh frontend/backend.

Jalankan:

```powershell
cd backend
python -m evaluation.interaction_eval
```

### `interaction_logs.jsonl`

Lokasi:

```text
backend/evaluation/interaction_logs.jsonl
```

Setiap record berisi:

- `interaction_id`
- `timestamp_utc`
- `laporan`
- `response`
- `retrieved_contexts`
- `status`

### `hasil_ragas.csv`

File ini dibuat oleh `record_ragas_result()` pada `backend/evaluation/ragas_eval.py` hanya ketika evaluasi RAGAS berhasil:

```text
backend/evaluation/hasil_ragas.csv
```

Kolomnya mencakup:

- `timestamp_utc`
- `query`
- `answer`
- `retrieved_documents`
- `average_similarity`
- `maximum_similarity`
- `minimum_similarity`
- `context_precision`
- `context_recall`
- `faithfulness`
- `answer_relevancy`
- `answer_correctness`
- `ground_truth`
- `status`
- `status_detail`
- `ragas_version`

Kegagalan RAGAS tidak ditulis sebagai skor valid.

## Pengujian

Pengujian dijalankan dari direktori utama:

```powershell
python -m unittest backend.evaluation.test_single_eval -v
```

Pengujian mencakup:

- Alignment knowledge base dan FAISS.
- Retrieval satu dan dua dokumen.
- Penanganan satu atau beberapa bahaya.
- Penolakan laporan yang tidak relevan.
- Validasi API.
- Pencatatan interaksi.
- Evaluasi similarity.
- Preflight RAGAS.
- Ketidaktersediaan evaluator yang aman.
- Pembatasan penyimpanan skor RAGAS yang valid.

Hasil pengujian terakhir pada lingkungan pengembangan:

```text
Ran 20 tests
OK
```

## Troubleshooting

### ModuleNotFoundError

Pastikan virtual environment aktif dan dependency telah dipasang:

```powershell
python -m pip install -r requirements.txt
```

Jalankan Python dari akar proyek atau folder yang sesuai agar import `rag`, `app`, dan package evaluasi berhasil.

### Instalasi FAISS Gagal

Gunakan `faiss-cpu==1.8.0` sesuai `requirements.txt`. Jangan mengganti ke FAISS GPU tanpa kebutuhan infrastruktur.

### Model Embedding Gagal Dimuat

Saat pertama dijalankan, proyek mencoba mengunduh `all-MiniLM-L6-v2` melalui Sentence Transformers. Pastikan koneksi internet dan cache Hugging Face dapat diakses.

### FAISS Index Tidak Ditemukan

Project saat ini membentuk index langsung dari `knowledge.txt`. Tidak ada file index FAISS tersimpan yang perlu dimuat. Jika terjadi kesalahan, periksa bahwa `backend/knowledge.txt` dapat dibaca dan memiliki baris yang valid.

### Knowledge Base Kosong

Pastikan `backend/knowledge.txt` tidak kosong dan setiap baris memiliki format:

```text
Kondisi | Risiko | Penjelasan | Solusi
```

Baris header `[KONDISI] | [RISIKO] | [PENJELASAN] | [SOLUSI]` dilewati oleh parser.

### Retrieved Documents Tidak Ditemukan

Sistem dapat mengembalikan `TIDAK DAPAT DINILAI` jika laporan terlalu tidak relevan atau berada di bawah ambang kecocokan. Tambahkan detail aktivitas, lokasi, kondisi bahaya, atau APD pada laporan.

### Evaluasi RAGAS Tidak Menghasilkan Skor

Jika Context Precision atau Context Recall kosong, periksa bahwa ground truth
konteks tersedia pada dataset dan bahwa retrieved contexts tidak kosong.
Faithfulness, Answer Relevancy, dan Answer Correctness harus tetap N/A karena
implementasi tidak menggunakan evaluator generatif.

Kegagalan RAGAS tidak mengubah hasil retrieval utama.

### Ground Truth Tidak Tersedia

Context Precision, Context Recall, dan Answer Correctness tidak boleh dianggap tersedia tanpa referensi yang sesuai. Sistem menandainya sebagai `N/A`.

### CSV Gagal Disimpan

Pastikan folder `backend/evaluation` dapat ditulis. Jangan membuka CSV dalam mode yang mengunci file selama proses penulisan.

### Port Sudah Digunakan

Backend dan frontend default menggunakan port `5000` yang sama. Jika menjalankan
frontend dari Live Server, pastikan backend tetap aktif di `http://localhost:5000`.

## Batasan Sistem

- Nilai retrieval bergantung pada kualitas dan cakupan `knowledge.txt`.
- Cosine similarity yang tinggi tidak menjamin jawaban benar.
- Hasil RAGAS memerlukan model evaluator yang terkonfigurasi.
- Context Recall dan Answer Correctness membutuhkan ground truth yang valid.
- Evaluasi LLM dapat memerlukan waktu komputasi tambahan.
- RAGAS dapat gagal tanpa mempengaruhi respons retrieval.
- Skor evaluasi bukan jaminan kebenaran mutlak pada kondisi lapangan nyata.
- Dataset evaluasi saat ini bersifat sintetis dan berjumlah 12 skenario.
- Sistem tidak menghasilkan penilaian ahli K3 sebagai pengganti penilaian profesional.
