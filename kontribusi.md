# Kontribusi Anggota Kelompok

Dokumen ini menjelaskan pembagian tugas, kontribusi masing-masing anggota, proses kolaborasi, serta penggunaan alat bantu selama pengerjaan UTS Deep Learning. Pengerjaan Task 1 dan Task 2 dilakukan secara individu sesuai pembagian tanggung jawab, sedangkan Task 3 dikerjakan secara kolaboratif oleh kedua anggota kelompok.

## 1. Daftar Anggota

| No. | Nama | NIM |
|---|---|---|
| 1 | Muhammad Alief Reefandi | 101032330132 |
| 2 | Rofly Hi M Naser | 101032300085 |

## 2. Pembagian Tugas

Pembagian tanggung jawab dalam pengerjaan UTS Deep Learning dilakukan sebagai berikut.

| Anggota | Tanggung Jawab |
|---|---|
| Muhammad Alief Reefandi | Bertanggung jawab atas pengerjaan Task 1 (Multimodal Classification) dan berkolaborasi dalam pengerjaan Task 3 (Multimodal Clustering). |
| Rofly Hi M Naser | Bertanggung jawab atas pengerjaan Task 2 (Time-Series Regression RUL) dan berkolaborasi dalam pengerjaan Task 3 (Multimodal Clustering). |

## 3. Rincian Kontribusi Pengerjaan

### 3.1 Task 1 – Multimodal Classification

**Penanggung jawab: Muhammad Alief Reefandi**

Pengerjaan Task 1 meliputi analisis data, pengembangan model klasifikasi multimodal, pelaksanaan eksperimen, serta evaluasi performa model. Kontribusi yang dilakukan antara lain:

- Melakukan Exploratory Data Analysis (EDA) untuk memahami karakteristik data citra, sensor, distribusi kelas, dan kondisi dataset.
- Melakukan preprocessing data citra dan sensor, termasuk normalisasi, penanganan missing value, serta pembagian data training dan validation.
- Mengembangkan model CNN sebagai encoder untuk mengekstraksi fitur dari data citra.
- Mengembangkan encoder untuk memproses data sensor menggunakan pendekatan deep learning.
- Mengimplementasikan dan membandingkan dua strategi penggabungan modalitas, yaitu Feature Fusion dan Decision Fusion.
- Melakukan eksperimen baseline menggunakan pendekatan image-only dan sensor-only.
- Menguji ketahanan model terhadap label noise, domain shift, dan missing modality.
- Melakukan sanity check untuk memastikan model mempelajari pola data, bukan sekadar menghasilkan prediksi secara acak.
- Mengevaluasi kestabilan performa model menggunakan tiga random seed.
- Melakukan evaluasi menggunakan metrik Macro-F1 serta menyusun visualisasi hasil eksperimen.
- Melakukan inferensi terhadap data test dan menghasilkan file prediksi klasifikasi.

### 3.2 Task 2 – Time-Series Regression RUL

**Penanggung jawab: Rofly Hi M Naser**

Pengerjaan Task 2 berfokus pada pengembangan sistem prediksi Remaining Useful Life (RUL) berdasarkan data time-series sensor mesin. Kontribusi yang dilakukan antara lain:

- Melakukan EDA dan preprocessing data, termasuk normalisasi berdasarkan kondisi operasi, seleksi sensor, serta penanganan missing value dan spike.
- Membentuk label RUL dengan batas maksimum (*capping*) sebesar 150 siklus dan menggunakan sliding window berukuran 50.
- Menyusun validasi 5-fold berdasarkan mesin untuk menyesuaikan karakteristik pembagian data.
- Melakukan pengujian anti-kebocoran data (*data leakage*) dan uji mutasi pada pipeline preprocessing.
- Mengembangkan dan membandingkan model baseline Gradient Boosting dengan model deep learning LSTM, TCN, dan Transformer.
- Mengevaluasi performa model menggunakan RMSE dan asymmetric score, termasuk analisis berdasarkan rentang RUL dan mode kerusakan.
- Mengembangkan prediction interval menggunakan MC Dropout, ensemble, dan conformal calibration.
- Menyusun hasil prediksi serta melakukan analisis kebijakan servis berdasarkan batas bawah prediction interval.

### 3.3 Task 3 – Multimodal Clustering

**Penanggung jawab: Muhammad Alief Reefandi dan Rofly Hi M Naser**

Task 3 dikerjakan secara bersama-sama oleh kedua anggota kelompok. Kolaborasi mencakup analisis data, pengembangan representasi multimodal, eksperimen clustering, evaluasi, dan interpretasi hasil.

Kontribusi pengerjaan meliputi:

- Melakukan EDA terhadap data citra dan spektrum, termasuk analisis distorsi serta pengujian estimator menggunakan data sintetis.
- Menentukan strategi augmentasi data dan melakukan training SimCLR pada masing-masing modalitas.
- Mengembangkan model penggabungan modalitas menggunakan FusionAE dengan pendekatan modality dropout.
- Melakukan eksperimen ablasi untuk menganalisis pengaruh masing-masing modalitas terhadap hasil clustering.
- Membandingkan beberapa pendekatan clustering, yaitu KMeans, GMM, dan IDEC.
- Melakukan analisis representasi embedding menggunakan UMAP dan t-SNE.
- Mengevaluasi hasil clustering, termasuk pasangan data yang sulit dibedakan, heterogenitas cluster 0, serta alternatif jumlah cluster dengan k = 7.
- Melakukan pengujian menggunakan data sintetis untuk memeriksa kemampuan model dalam mengenali pola dan cluster yang telah diketahui.

Kedua anggota terlibat dalam diskusi dan evaluasi hasil eksperimen untuk menentukan pendekatan yang digunakan. Dengan demikian, keputusan akhir terkait metode dan hasil Task 3 merupakan hasil kolaborasi bersama.

## 4. Infrastruktur dan Integrasi Proyek

Dalam proses pengembangan dan integrasi proyek, kedua anggota menggunakan serta mengembangkan beberapa komponen pendukung agar seluruh task dapat dijalankan secara terstruktur.

Komponen yang digunakan meliputi:

- `run_all.py` sebagai skrip untuk menjalankan keseluruhan task.
- `kode/common/` sebagai direktori yang berisi fungsi dan komponen bersama.
- Flag `--debug`, `--data_dir`, dan `--out_dir` untuk mendukung konfigurasi eksekusi program.
- `kode/validasi_csv.py` untuk memeriksa kesesuaian format file prediksi.
- Google Colab dan Kaggle Notebook sebagai lingkungan untuk menjalankan eksperimen dan training model.

Integrasi dilakukan agar implementasi setiap task memiliki struktur yang konsisten dan dapat dijalankan kembali sesuai kebutuhan evaluasi.

## 5. Pull Request dan Kontribusi Repository

Seluruh implementasi dan dokumentasi dikelola melalui GitHub untuk mendukung kolaborasi, pengelolaan versi, serta integrasi hasil pengerjaan setiap anggota.

**Repository:** [AliefReefandi/UTS-Deep-Learning](https://github.com/AliefReefandi/UTS-Deep-Learning)

Berikut merupakan daftar Pull Request yang berkaitan dengan pengerjaan proyek:

| PR | Branch | Kontribusi |
|---|---|---|
| #1 | `task2/rul` | Implementasi Task 2, integrasi `run_all.py`, validasi CSV, dan notebook. |
| #2 | `task3/clustering` | EDA Task 3 dan implementasi SimCLR. |
| #3 | `task2/hasil` | Hasil training Task 2, dokumentasi, dan prediksi RUL. |
| #4 | `task3/fix-notebook` | Perbaikan deteksi lingkungan Google Colab dan Kaggle. |
| #5 | `task3/cluster` | Implementasi Task 3c–f, hasil clustering, dan `cluster.csv`. |
| #6 | `final/laporan` | Integrasi README, laporan, dokumen kontribusi, dan persiapan ujian lisan. |

## 6. Penggunaan Artificial Intelligence (AI)

AI digunakan sebagai alat bantu selama proses pengerjaan proyek, khususnya untuk mendukung pemahaman konsep, pengembangan kode, debugging, analisis, dan penyusunan dokumentasi. Penggunaan AI disesuaikan dengan tanggung jawab masing-masing anggota.

### 6.1 Muhammad Alief Reefandi – Task 1 dan Task 3

**Tools:** ChatGPT.

ChatGPT digunakan untuk membantu memahami tahapan pengerjaan Task 1, memberikan referensi implementasi kode, membantu proses debugging, serta mendukung analisis hasil EDA dan evaluasi model. AI juga digunakan sebagai alat bantu dalam memahami implementasi model, pembuatan visualisasi, dan penyusunan dokumentasi.

Pelaksanaan eksperimen, training model, pengujian, perbandingan hasil, serta penentuan model akhir dilakukan dan diverifikasi oleh anggota yang bersangkutan. Dalam pengerjaan Task 3, ChatGPT juga digunakan sebagai bantuan teknis selama proses kolaborasi.

### 6.2 Rofly Hi M Naser – Task 2 dan Task 3

**Tools:** Claude melalui Claude Code.

Claude digunakan sebagai alat bantu dalam penulisan dan perbaikan kode Python, pemahaman metode yang digunakan, serta penyusunan dokumentasi dan analisis hasil eksperimen.

Pada Task 2, AI dimanfaatkan untuk mendukung proses preprocessing, pengembangan model, pengujian anti-kebocoran data, dan evaluasi hasil. Pada Task 3, AI digunakan sebagai referensi dalam implementasi dan debugging beberapa bagian model clustering.

Eksperimen, training, pemeriksaan hasil, pengujian ulang, serta keputusan akhir terkait metode dan model tetap dilakukan dan diverifikasi oleh anggota yang bersangkutan.

### 6.3 Verifikasi dan Validasi Hasil

Untuk memastikan implementasi dan hasil eksperimen dapat diperiksa kembali, kedua anggota melakukan beberapa langkah verifikasi, antara lain:

- Menjalankan pengujian otomatis pada masing-masing task.
- Memeriksa hasil preprocessing dan prediksi.
- Melakukan sanity check serta pengujian menggunakan data sintetis.
- Melakukan debugging sebelum menjalankan training secara penuh.
- Memeriksa kembali angka dan hasil evaluasi yang dicantumkan dalam README dan laporan.
- Menjalankan ulang eksperimen tertentu untuk memastikan hasil dapat direproduksi.

Dengan demikian, AI berperan sebagai alat bantu selama proses pengerjaan, sedangkan pemahaman, pelaksanaan eksperimen, verifikasi hasil, dan pengambilan keputusan akhir tetap menjadi tanggung jawab masing-masing anggota kelompok.
