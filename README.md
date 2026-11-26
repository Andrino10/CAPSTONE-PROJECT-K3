# SIGAP AI

SIGAP AI adalah aplikasi untuk menganalisis laporan aktivitas kerja dan membantu mengidentifikasi risiko K3 (Keselamatan dan Kesehatan Kerja). Pengguna memasukkan laporan singkat, kemudian sistem menampilkan level risiko, temuan, penyebab, rekomendasi, dan maksimal dua dokumen knowledge base yang pembahasannya paling dekat.

## Fitur utama

- Menganalisis laporan kerja minimal 20 karakter.
- Menentukan level risiko: `RENDAH`, `SEDANG`, `TINGGI`, atau `TIDAK DAPAT DINILAI`.
- Menampilkan temuan risiko, penyebab, dan rekomendasi tindakan.
- Menampilkan maksimal dua dokumen terkait dari knowledge base K3.
- Menampilkan skor kemiripan dokumen untuk membantu pemeriksaan hasil retrieval.
- Mencatat interaksi dan hasil evaluasi lokal di folder `backend/evaluation`.

## Cara kerja singkat

```text
Laporan pengguna
      ↓
Pencarian knowledge base K3 dengan SentenceTransformer + FAISS
      ↓
Pemeringkatan berdasarkan kemiripan semantik dan kata penting
      ↓
Maksimal 2 dokumen terkait ditampilkan
      ↓
Level risiko, temuan, dan rekomendasi dikirim ke dashboard
```

Dua dokumen terkait digunakan untuk membantu pengguna memahami rujukan yang paling dekat dengan laporan. Penentuan risiko tetap memakai filter relevansi tersendiri agar dokumen pendamping tidak menaikkan risiko secara keliru.

## Teknologi

| Teknologi | Kegunaan |
|---|---|
| Python | Backend dan proses analisis |
| Flask | Server web dan API |
| HTML, CSS, JavaScript | Dashboard frontend |
| Sentence Transformers | Membuat embedding teks |
| `all-MiniLM-L6-v2` | Model embedding |
| FAISS | Pencarian dokumen serupa |
| RAGAS | Evaluasi retrieval opsional |

## Struktur proyek

```text
SIGAP_AI-main/
├── backend/
│   ├── app.py              # Server Flask dan API
│   ├── rag.py              # Retrieval, ranking, dan analisis risiko
│   ├── knowledge.txt       # Knowledge base K3
│   └── evaluation/         # Tes, log, dan evaluasi
├── frontend/
│   ├── index.html          # Halaman dashboard
│   ├── app.js              # Interaksi dashboard dan pemanggilan API
│   └── style.css           # Tampilan dashboard
├── requirements.txt        # Dependency Python
└── README.md
```

## Persiapan

Pastikan Python dan `pip` tersedia. Proyek ini telah diuji menggunakan Python 3.13 di Windows.

1. Buat virtual environment dari direktori utama proyek.

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   Jika PowerShell menolak aktivasi, gunakan Command Prompt:

   ```bat
   .venv\Scripts\activate.bat
   ```

2. Instal dependency.

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

Pada penggunaan pertama, model `all-MiniLM-L6-v2` mungkin perlu diunduh. Pastikan koneksi internet tersedia sampai proses tersebut selesai.

## Menjalankan aplikasi

Jalankan backend dari folder `backend`:

```powershell
cd backend
python app.py
```

Setelah muncul informasi server Flask, buka:

```text
http://localhost:5000
```

Frontend dan API berjalan dari alamat yang sama. Tidak perlu menjalankan Live Server, Node.js, atau server frontend kedua.

## Menggunakan dashboard

1. Buka `http://localhost:5000`.
2. Masukkan laporan aktivitas kerja minimal 20 karakter, atau klik **Contoh**.
3. Klik **ANALISIS SEKARANG**.
4. Periksa level risiko, temuan, penyebab, dan rekomendasi.
5. Lihat bagian **2 Dokumen Terkait Terdekat** untuk melihat dua pembahasan knowledge base yang paling relevan.

Contoh laporan:

```text
Pekerja bekerja di ketinggian tanpa safety harness dan kabel listrik terbuka di dekat area kerja.
```

## API

| Method | Endpoint | Kegunaan |
|---|---|---|
| `GET` | `/api/sample` | Mengambil contoh laporan dari knowledge base |
| `POST` | `/api/analyze` | Menganalisis laporan kerja |

Contoh request analisis:

```json
{
  "laporan": "Kabel listrik terbuka berada di dekat pekerja"
}
```

Contoh respons ringkas:

```json
{
  "level": "TINGGI",
  "temuan": ["Kabel listrik terbuka"],
  "rekomendasi": ["Isolasi kabel, gunakan pelindung kabel, dan hindari area basah."],
  "retrieved_documents": [
    {
      "rank": 1,
      "document": { "kondisi": "Kabel listrik terbuka" },
      "cosine_similarity": 0.73
    }
  ]
}
```

`retrieved_documents` berisi paling banyak dua dokumen terkait. Jika laporan tidak cukup relevan dengan knowledge base, `level` dapat bernilai `TIDAK DAPAT DINILAI` dan daftar dokumen dapat kosong.

## Perintah penting

Jalankan dari direktori utama proyek, kecuali disebutkan lain.

| Perintah | Kegunaan |
|---|---|
| `python -m unittest backend.evaluation.test_single_eval -v` | Menjalankan pengujian utama |
| `cd backend; python -m evaluation` | Menjalankan evaluasi retrieval dan pemeriksaan metrik |
| `cd backend; python -m evaluation.interaction_eval` | Mengevaluasi interaksi yang sudah tercatat |

## Lokasi data evaluasi

| File | Isi |
|---|---|
| `backend/evaluation/interaction_logs.jsonl` | Log interaksi analisis yang berhasil |
| `backend/evaluation/hasil_similarity.csv` | Hasil evaluasi retrieval dataset |
| `backend/evaluation/hasil_interaction_similarity.csv` | Evaluasi retrieval dari interaksi aktual |
| `backend/evaluation/hasil_ragas.csv` | Hasil RAGAS ketika evaluasi tersedia |

## Pemecahan masalah

### Halaman tidak dapat dibuka

Pastikan backend masih berjalan, lalu buka `http://localhost:5000`, bukan file `frontend/index.html` secara langsung.

### Port 5000 sedang digunakan

Tutup proses Flask lain yang memakai port tersebut, kemudian jalankan kembali `python app.py`.

### Model gagal dimuat

Pastikan dependency sudah terinstal dan koneksi internet tersedia saat model pertama kali diunduh.

```powershell
python -m pip install -r requirements.txt
```

### Hasil menjadi `TIDAK DAPAT DINILAI`

Tambahkan informasi aktivitas, lokasi, kondisi bahaya, dan APD pada laporan agar sistem dapat menemukan rujukan yang lebih sesuai.

### Dokumen terkait hanya satu atau tidak ada

Sistem hanya menampilkan dokumen yang melewati ambang relevansi. Ini dilakukan agar dokumen yang tidak berhubungan tidak ditampilkan sebagai rujukan.

## Batasan

- Hasil bergantung pada isi dan kualitas `backend/knowledge.txt`.
- Skor kemiripan menunjukkan kedekatan teks, bukan jaminan kebenaran penilaian.
- Sistem tidak menggantikan inspeksi atau keputusan ahli K3 di lapangan.
- RAGAS bersifat evaluasi tambahan; kegagalannya tidak menghentikan hasil analisis utama.
