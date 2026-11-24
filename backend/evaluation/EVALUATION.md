# Evaluasi SIGAP AI

## Ruang lingkup

Modul ini mengevaluasi retrieval SIGAP AI yang aktual: `all-MiniLM-L6-v2`,
FAISS `IndexFlatL2`, embedding hanya pada kolom kondisi knowledge, serta
peringkat ulang berdasarkan kemiripan semantik dan cakupan kata penting.
Endpoint analisis memecah laporan pada batas kalimat dan penghubung antarkondisi,
lalu mengambil satu referensi paling kuat untuk setiap bagian. Hasil di bawah
ambang kecocokan ditolak; laporan tanpa kecocokan yang cukup diberi status
`TIDAK DAPAT DINILAI`.

## Dataset

`test_dataset.csv` memiliki 12 skenario **sintetis** K3. Kolom
`ground_truth_kondisi` dan `ground_truth_risiko` menunjuk ke entri yang sudah
ada di `knowledge.txt`; kolom `data_asal` menyatakannya secara eksplisit.
Dataset ini menguji fungsi sistem dan bukan statistik kecelakaan kerja maupun
validasi penilaian ahli. Satu reference acuan per sampel memungkinkan Recall@3
yang terdefinisi.

| Kolom | Fungsi evaluasi |
| --- | --- |
| `sample_id` | Identitas sampel yang menautkan dataset dan hasil CSV. |
| `laporan` | Pertanyaan/laporan K3 yang dikirim ke retrieval SIGAP AI. |
| `ground_truth_kondisi` | Referensi knowledge base yang relevan untuk retrieval. |
| `ground_truth_risiko` | Risiko pada referensi acuan; diperiksa konsistensinya, bukan dipakai untuk mengklaim akurasi klasifikasi. |
| `data_asal` | Status sumber skenario; seluruh sampel saat ini sintetis. |

## Metode

`similarity_eval_all.py` memakai helper retrieval yang sama dengan `rag.py`
lalu menulis tiga kandidat teratas untuk seluruh laporan ke
`hasil_similarity.csv`. Endpoint analisis juga memeriksa setiap kalimat atau
bagian bahaya secara terpisah agar beberapa kondisi dalam satu laporan dapat
ditampilkan tanpa menambahkan referensi lemah dari kalimat lain. Nilai cosine
dihitung sebagai `dot(q, d) / (||q|| ||d||)`. Kolom
`faiss_squared_l2_distance` adalah jarak kuadrat L2 dari FAISS, sehingga bukan
cosine similarity. Evaluasi dataset mengukur peringkat kandidat; ia tidak
mengukur ambang kecocokan atau keputusan status tidak dapat dinilai.

`ragas_metrics.py` menghitung retrieval berbasis `is_ground_truth`:

- Precision@3 = referensi relevan yang diambil / (jumlah laporan x 3)
- Recall@3 = referensi acuan yang ditemukan / seluruh referensi acuan
- Hit@3 = proporsi laporan dengan sedikitnya satu reference acuan di top-3
- MRR = rata-rata kebalikan rank relevan pertama.

Similarity dan retrieval hanya mengukur kedekatan teks serta keberhasilan
menemukan reference acuan. Keduanya tidak membuktikan bahwa tingkat risiko,
penyebab, atau rekomendasi K3 benar pada kondisi kerja nyata.

## RAGAS

RAGAS 0.4.3 terintegrasi menggunakan metrik non-generatif yang didukung.
Context Precision dan Context Recall dapat dihitung dengan referensi konteks.
Faithfulness, Answer Relevancy, dan Answer Correctness ditampilkan N/A karena
memerlukan evaluator generatif atau jawaban referensi yang tidak digunakan.
Model utama tetap SentenceTransformer all-MiniLM-L6-v2 dan FAISS.

## Interaksi aktual dari frontend

Setiap respons `/api/analyze` yang berhasil dicatat backend ke
`interaction_logs.jsonl`. Satu baris JSONL berisi ID unik, waktu UTC, laporan
asli, respons API yang benar-benar dikirim, dan hingga dua konteks retrieval
beserta rank dan squared-L2. Konteks kosong menandakan laporan belum cukup
cocok. Header `X-Interaction-Request-ID` dari frontend digunakan
untuk menghindari duplikasi saat permintaan yang sama diproses ulang. Log tidak
menyimpan header lain, token, kata sandi, alamat IP, atau identitas klien.

Jalankan setelah menggunakan frontend:

```powershell
cd backend
python -m evaluation.interaction_eval
```

Output `hasil_interaction_similarity.csv` terpisah dari
`hasil_similarity.csv` agar interaksi aktual tidak bercampur dengan dataset
sintetis. Evaluasi interaksi menghitung jumlah record, distribusi level risiko,
dan cosine similarity laporan terhadap konteks yang benar-benar terambil.
Tanpa label relevansi manual atau jawaban acuan, metrik retrieval dan RAGAS
tidak dihitung serta tidak boleh ditafsirkan sebagai akurasi sistem.

## Menjalankan dari folder backend

Satu perintah mengevaluasi tiga referensi teratas tiap laporan, menampilkan
dokumen yang ditemukan beserta ringkasan cosine similarity, lalu memvalidasi
hasil CSV. Semua perhitungan dilakukan lokal dengan SentenceTransformer dan
FAISS; tidak ada evaluator/API generatif yang dipanggil.

```powershell
cd backend
python -m evaluation
```

Evaluasi interaksi aktual dan preflight RAGAS tetap dapat dijalankan terpisah
dengan perintah pada bagian masing-masing.

## Menjalankan per modul

Jalankan dari akar proyek:

```powershell
python backend/evaluation/similarity_eval_all.py
python backend/evaluation/ragas_metrics.py
python backend/evaluation/check_metrics.py
python backend/evaluation/ragas_eval.py
python -m unittest backend.evaluation.test_single_eval -v
```

`check_metrics.py` gagal dengan status non-nol untuk kolom yang hilang, data
kosong, nilai cosine/L2 yang tidak sah, ketidaksesuaian jumlah sampel, atau
rank selain 1, 2, dan 3.
