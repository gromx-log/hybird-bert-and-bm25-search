# 🏠 Pencarian Properti Cerdas (Real Estate Hybrid Search Engine)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

Sebuah mesin pencari properti real estate hibrida yang menggabungkan kemampuan **IndoBERT** untuk klasifikasi syarat mutlak (hard filtering) serta kombinasi **BM25 (Pencarian Leksikal)** dan **Sentence-BERT (Pencarian Semantik)** untuk pemeringkatan relevansi (soft ranking).

Proyek ini dirancang untuk dapat dideploy langsung pada **Streamlit** dengan efisiensi tinggi, memanfaatkan data yang telah dipra-proses secara offline guna mempercepat waktu respons pencarian.

---

## 🚀 Fitur Utama

- **Two-Stage Hybrid Search**:
  - **Tahap 1 (Hard Filter)**: Menyaring hasil pencarian secara ketat berdasarkan niat/query pengguna atau filter manual pada aspek:
    - **Bebas Banjir**: Klasifikasi hibrida (IndoBERT + Regex) untuk mendeteksi properti yang aman dari banjir.
    - **Bisa KPR**: Klasifikasi IndoBERT untuk mendeteksi kesiapan fasilitas Kredit Pemilikan Rumah.
    - **Legalitas SHM**: Klasifikasi IndoBERT + Regex pendeteksi Sertifikat Hak Milik.
  - **Tahap 2 (Soft Ranking)**: Melakukan perangkingan terhadap properti yang lolos penyaringan menggunakan bobot optimal hibrida: **70% Lexical (BM25) + 30% Semantic (Sentence-BERT)**.
- **Advanced AI Settings**: Slider interaktif pada sidebar untuk menyesuaikan pengaruh pencarian leksikal (kata kunci eksak) vs semantik (makna kalimat).

---

## 📂 Arsitektur Data

Untuk memastikan aplikasi berjalan dengan cepat di lingkungan produksi (Streamlit Cloud), model klasifikasi IndoBERT yang berat dijalankan secara offline. Aplikasi memuat tiga file data utama dari folder `data/` yang harus saling selaras:

| Nama File | Deskripsi | Status |
|---|---|---|
| `data/properties_enriched.csv` | Dataset properti lengkap yang telah diperkaya dengan kolom prediksi AI (`AI_Bebas_Banjir`, `AI_Bisa_KPR`, `AI_Legalitas_SHM`, dll.) | **Wajib** |
| `data/bm25_index.pkl` | Indeks lexical serialized untuk pemrosesan BM25 instan | **Wajib** |
| `data/sbert_embeddings.npy` | Array representasi vektor (embeddings) dari Sentence-BERT untuk dokumen | **Wajib** |

> [!NOTE]
> File data mentah (`properties_raw.csv`) dan dataset pelatihan model (`labeled_training.csv`) dapat tetap disimpan di dalam folder `data/` untuk keperluan pengembangan/debug, namun tidak diakses secara langsung oleh aplikasi web Streamlit.

> [!WARNING]
> **Penyelarasan Indeks & Baris**: Ketiga file wajib di atas (`properties_enriched.csv`, `bm25_index.pkl`, dan `sbert_embeddings.npy`) harus memiliki jumlah baris/elemen yang sama dan diurutkan dalam urutan indeks yang sama. Jika Anda memperbarui dataset CSV, Anda harus membuat ulang indeks BM25 dan embeddings SBERT menggunakan skrip pembuatan indeks di dalam notebook `debug/[TEST]_Projek_UAS_NLP (1).ipynb`.

---

## 🛠️ Instalasi & Menjalankan Lokal

Ikuti langkah berikut untuk menjalankan aplikasi di komputer lokal Anda:

1. **Clone repositori ini**:
   ```bash
   git clone https://github.com/username/hybrid-bert-and-bm25-search.git
   cd hybrid-bert-and-bm25-search
   ```

2. **Buat & aktifkan virtual environment (opsional tapi disarankan)**:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependensi**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan aplikasi Streamlit**:
   ```bash
   streamlit run app.py
   ```

Aplikasi akan otomatis terbuka pada peramban Anda di alamat `http://localhost:8501`.

---

## ☁️ Panduan Deploy ke Streamlit Cloud

Aplikasi ini siap dideploy langsung ke **Streamlit Community Cloud**:

1. Pastikan seluruh perubahan kode telah di-push ke repositori GitHub Anda (termasuk folder `data/` yang berisi 3 file utama).
2. Kunjungi [Streamlit Share](https://share.streamlit.io/) dan masuk dengan akun GitHub Anda.
3. Klik tombol **New App**.
4. Pilih repositori, branch (misal: `main`), dan tentukan file utama yaitu `app.py`.
5. Klik **Deploy!** Streamlit Cloud akan membaca `requirements.txt` dan menginstal modul-modul yang dibutuhkan secara otomatis.

> [!NOTE]
> Karena model IndoBERT yang besar tidak dimuat pada saat startup (hanya menggunakan model SBERT yang ringan `paraphrase-multilingual-MiniLM-L12-v2`), penggunaan memori ram Streamlit Cloud akan tetap berada di bawah batas gratis 1 GB.

---

## 📊 Hasil Eksperimen & Visualisasi

Berikut adalah hasil pelatihan model dan analisis pembandingan metode pencarian hibrida yang diusulkan.

### 1. Kurva Pelatihan Model IndoBERT (Fine-Tuning)
<!-- PLACEHOLDER_START: training_progress_chart -->
*(Silakan tempatkan gambar kurva pelatihan model IndoBERT Anda di path repositori: `assets/training_progress.png`)*
```markdown
![Progres Pelatihan IndoBERT](assets/training_progress.png)
```
<!-- PLACEHOLDER_END: training_progress_chart -->

### 2. Optimasi Bobot Hybrid (BM25 Weight vs SBERT Weight)
Melalui evaluasi komprehensif pada **50 kueri uji** dengan relevansi manual (*ground truth*), bobot terbaik ditemukan pada kombinasi **BM25 = 0.7** dan **SBERT = 0.3** yang menghasilkan nilai **Precision@10 = 0.2294** dan **MRR = 0.5667**.

<!-- PLACEHOLDER_START: hybrid_optimization_chart -->
*(Silakan tempatkan gambar kurva optimasi bobot hibrida Anda di path repositori: `assets/hybrid_weight_optimization.png`)*
```markdown
![Grafik Optimasi Bobot Hybrid](assets/hybrid_weight_optimization.png)
```
<!-- PLACEHOLDER_END: hybrid_optimization_chart -->

### 3. Perbandingan Performa Evaluasi Akurasi (Benchmark Akhir)
Perbandingan kinerja model yang diusulkan (Hybrid Proposed) terhadap metode baseline lain (TF-IDF, BM25 murni, SBERT murni):

- **TF-IDF (Baseline)**: Presisi Rendah, MRR Rendah.
- **BM25 (Lexical Only)**: Cepat dan tepat untuk kueri eksak, namun kehilangan kedekatan makna kontekstual.
- **Sentence-BERT (Semantic Only)**: Baik dalam memahami makna kalimat tapi terkadang melewatkan detail kata kunci penting (nama stasiun, nomor, dll.).
- **Proposed (IndoBERT + Hybrid)**: Memberikan presisi tertinggi dengan menyaring properti secara cerdas dan merangking sisa hasil pencarian dengan kombinasi leksikal-semantik.

<!-- PLACEHOLDER_START: benchmark_comparison_chart -->
*(Silakan tempatkan gambar diagram perbandingan kinerja evaluasi akhir Anda di path repositori: `assets/results_chart_test.png`)*
```markdown
![Grafik Perbandingan Performa](assets/results_chart_test.png)
```
<!-- PLACEHOLDER_END: benchmark_comparison_chart -->
