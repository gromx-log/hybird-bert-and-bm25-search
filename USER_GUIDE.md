# 📖 Panduan Pengguna: Aplikasi Pencarian Properti Cerdas

---

## 👥 Kelompok 8 (Pengembang)
* **Filbert Ferdinand** (NIM: 535240135)
* **Arya Rava Pradana** (NIM: 535240023)
* **Rafael Theng** (NIM: 535240153)

Panduan ini disusun untuk memudahkan pengguna dalam mengoperasikan aplikasi **Pencarian Properti Cerdas**, baik dalam melakukan pencarian berbasis teks alami, menyaring spesifikasi fisik properti, mengatur konfigurasi algoritma pencarian, hingga menggunakan fitur pembandingan interaktif.

---

## 📂 Daftar Isi
1. [Subbab 1: Pengenalan Antarmuka Utama](#subbab-1-pengenalan-antarmuka-utama)
2. [Subbab 2: Melakukan Pencarian Bahasa Alami (Natural Language Search)](#subbab-2-melakukan-pencarian-bahasa-alami-natural-language-search)
3. [Subbab 3: Menggunakan Filter Konfigurasi di Sidebar](#subbab-3-menggunakan-filter-konfigurasi-di-sidebar)
4. [Subbab 4: Membaca Hasil Pencarian & Detail Properti](#subbab-4-membaca-hasil-pencarian--detail-properti)
5. [Subbab 5: Membandingkan Properti Secara Interaktif](#subbab-5-membandingkan-properti-secara-interaktif)

---

## 🖥️ Subbab 1: Pengenalan Antarmuka Utama

Aplikasi ini menggunakan tema **Google Antigravity Light**, yang mengedepankan kesederhanaan, kecepatan akses, dan visualisasi yang bersih. Layar utama terbagi menjadi beberapa area fungsional:
1. **Sidebar Konfigurasi (Kiri)**: Berisi panel kendali filter fisik (Harga, Luas Tanah, Luas Bangunan), filter klasifikasi AI mutlak, opsi pengurutan, pengaturan bobot algoritma, dan data statistik korpus.
2. **Header & Kolom Pencarian (Tengah atas)**: Area untuk memasukkan kueri teks alami pengguna, lengkap dengan tombol pencarian dan rekomendasi pencarian populer.
3. **Daftar Hasil Pencarian (Tengah bawah)**: Kartu properti interaktif yang menyajikan ringkasan spesifikasi, persentase tingkat kecocokan (*Match %*), ranking properti, dan badge kriteria properti.
4. **Floating Action Button (Sudut Kanan Bawah)**: Tombol pembanding properti mengambang (*FAB*) yang otomatis muncul ketika pengguna menambahkan properti ke dalam daftar perbandingan.

![Tampilan Antarmuka Utama Aplikasi]([Masukkan screenshot halaman beranda utama di sini])

---

## 🔍 Subbab 2: Melakukan Pencarian Bahasa Alami (Natural Language Search)

Mesin pencari ini dirancang untuk dapat memahami maksud pencarian kontekstual yang Anda tuliskan.

### Langkah-langkah Pencarian:
1. Klik kolom teks pencarian dengan placeholder *"cth: Rumah mewah di Jakarta Selatan bebas banjir bisa KPR"*.
2. Tuliskan kueri sesuai keinginan Anda, contohnya: **"Rumah murah di Depok bebas banjir dekat stasiun"**.
3. Tekan tombol **Cari Properti** di sebelah kanan kolom kueri, atau tekan tombol **Enter** pada keyboard Anda.
4. Jika Anda ingin mencari dengan cepat menggunakan contoh kueri populer, klik salah satu dari tombol rekomendasi chip di bawah kolom pencarian:
   - **Mansion Mewah Jakarta** (Chip Biru)
   - **Townhouse Dekat MRT** (Chip Merah)
   - **Rumah Murah Bekasi KPR** (Chip Hijau)
![Highlight Rainbow Kolom Pencarian saat Aktif]([Masukkan screenshot kolom pencarian dengan highlight pelangi di sini])
![Rekomendasi Chip Pencarian Populer]([Masukkan screenshot tombol chip populer di sini])

---

## ⚙️ Subbab 3: Menggunakan Filter Konfigurasi di Sidebar

Sidebar di sisi kiri layar menyediakan kendali manual tingkat lanjut. Panel ini terbagi ke dalam 3 tab navigasi utama:

### 1. Tab "Filter Fisik"
Tab ini digunakan untuk membatasi hasil pencarian berdasarkan karakteristik fisik properti:
* **Range Harga (Miliar Rp)**: Geser slider untuk menentukan batas harga minimum dan maksimum dalam skala Miliar Rupiah (misalnya: 1.5M - 5M).
* **Luas Tanah (LT) m²**: Geser slider untuk membatasi luas minimum dan maksimum tanah properti.
* **Luas Bangunan (LB) m²**: Geser slider untuk membatasi luas minimum dan maksimum bangunan properti.
* **Metode Pengurutan**: Selectbox untuk mengurutkan hasil pencarian berdasarkan:
  - *Kecocokan AI* (skor gabungan leksikal-semantik)
  - *Harga Terendah*
  - *Harga Tertinggi*
  - *Luas Tanah Terbesar*
* **Jumlah Hasil**: Slider untuk membatasi jumlah kartu properti yang ditampilkan pada halaman (rentang 3 s.d 20 properti).

![Sidebar Tab Filter Fisik]([Masukkan screenshot tab filter fisik pada sidebar di sini])

### 2. Tab "Filter AI"
Tab ini menampung penyaringan mutlak (*hard filters*) berbasis klasifikasi otomatis IndoBERT. Fitur ini memaksa hasil pencarian hanya menyajikan properti dengan kriteria:
* **Wajib Bebas Banjir** (Centang jika ingin menyaring properti yang berada di kawasan aman banjir).
* **Wajib Bisa KPR** (Centang jika ingin menyaring properti yang teridentifikasi bisa diajukan kredit bank).
* **Wajib Sertifikat SHM** (Centang jika hanya ingin properti dengan bukti kepemilikan Sertifikat Hak Milik).

![Sidebar Tab Filter AI]([Masukkan screenshot tab filter AI pada sidebar di sini])

### 3. Tab "Bobot & Info"
Tab ini diperuntukkan bagi konfigurasi tingkat lanjut (*Advanced AI Configuration*):
* **BM25 Weight (Lexical)**: Slider dinamis untuk menyesuaikan sensitivitas mesin pencari terhadap kecocokan kata kunci eksak (skala 0.0 - 1.0).
* **SBERT Weight (Semantic)**: Secara otomatis dihitung oleh sistem (`1.0 - BM25 weight`) untuk menyesuaikan kepekaan sistem terhadap makna kontekstual kueri.
* **Statistik Korpus**: Menyajikan data statistik jumlah properti terindeks serta jangkauan cakupan prediksi AI untuk status bebas banjir, KPR, dan sertifikasi SHM dalam basis data.

### 4. Tombol "Reset Semua Filter"
Tombol merah di bagian atas sidebar digunakan untuk membersihkan seluruh filter fisik dan AI serta mengembalikan posisi slider-slider properti secara instan ke batas nilai bawaan (*default*).

![Sidebar Tab Bobot dan Konfigurasi AI]([Masukkan screenshot tab bobot dan statistik korpus di sini])

---

## 📑 Subbab 4: Membaca Hasil Pencarian & Detail Properti

Setiap properti yang lolos dari filter fisik dan AI akan ditampilkan sebagai kartu informasi (*property card*) dengan detail yang terstruktur:

### Elemen Kartu Properti:
1. **Rank Badge**: Menunjukkan posisi urutan kecocokan properti (misal: *RANK 1*).
2. **Match Percentage**: Persentase tingkat kecocokan properti terhadap kueri Anda (misal: *92% MATCH*).
3. **Kriteria Badges**: Label status properti seperti *Bebas Banjir* (Hijau), *Bisa KPR* (Biru), dan *Legalitas SHM* (Kuning).
4. **Harga & Dimensi Fisik**: Menampilkan harga penawaran (misal: Rp 3.50 M) serta luas tanah dan bangunan (LT & LB).
5. **Cuplikan Deskripsi**: Potongan singkat deskripsi properti.
6. **Tombol Aksi**:
   - **Lihat Detail Lengkap**: Klik untuk membuka jendela dialog eksplorasi detail properti.
   - **Bandingkan / Hapus Banding**: Klik untuk memasukkan/mengeluarkan properti dari keranjang pembandingan.

![Kartu Properti dengan Badge dan Indikator Match]([Masukkan screenshot satu kartu properti hasil pencarian di sini])

### Membaca Detail Lengkap Properti:
Saat tombol **Lihat Detail Lengkap** diklik, jendela dialog modal akan muncul menyajikan deskripsi lengkap tanpa membebani memori browser. Di bagian bawah deskripsi, terdapat tautan langsung ke halaman sumber properti asli dari rumah123.com dengan format:
`Cek Sekarang: [Tautan URL Rumah123]`

![Modal Eksplorasi Detail Properti]([Masukkan screenshot dialog detail properti di sini])

---

## 📊 Subbab 5: Membandingkan Properti Secara Interaktif

Fitur pembandingan (*comparison grid*) mempermudah Anda menganalisis hingga **4 properti pilihan** sekaligus secara berdampingan.

### Cara Membandingkan:
1. Klik tombol **Bandingkan** pada kartu properti yang ingin dianalisis. Kartu akan diselimuti garis tepi berwarna biru dan memunculkan label *"TERPILIH UNTUK BANDING"*.
2. Tombol melayang **Buka Perbandingan (x/4)** akan muncul di sudut kanan bawah layar. Angka `x` menunjukkan jumlah properti yang telah ditambahkan.
3. Klik tombol mengambang tersebut untuk membuka jendela modal **Perbandingan Properti**.
4. Anda akan disajikan sebuah tabel pembandingan HTML premium yang teratur, dengan garis pemisah (*linings*) horizontal abu-abu yang jelas, baris tajuk abu-abu, teks status Ya (Hijau) / Tidak (Merah) yang menonjol, dan harga properti yang diwarnai biru tebal.
5. **Melakukan Aksi di Jendela Modal**:
   - Di baris terbawah tabel (*Aksi*), terdapat tombol **Lihat Detail** dan **Hapus** untuk masing-masing properti yang tersusun rapi secara vertikal.
   - Klik **Lihat Detail**: Deskripsi lengkap properti tersebut beserta link `Cek Sekarang` akan termuat langsung di dalam modal perbandingan bagian bawah (tanpa menutup modal perbandingan).
   - Klik **Hapus**: Properti bersangkutan akan langsung dihapus dari tabel perbandingan, dan kolom tabel akan merapat secara otomatis (modal tetap terbuka).
   - Klik tombol **X** di sudut kanan atas modal untuk kembali ke halaman pencarian utama.

![Floating Compare Button]([Masukkan screenshot tombol floating perbandingan di sini])
![Tabel Perbandingan Properti dalam Modal]([Masukkan screenshot modal perbandingan berisi tabel perbandingan properti di sini])
![Tampilan Detail Inline di dalam Modal Perbandingan]([Masukkan screenshot detail inline di bagian bawah modal perbandingan di sini])
