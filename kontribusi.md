# Kontribusi Anggota Kelompok

Dokumen ini menjelaskan pembagian tugas, kontribusi masing-masing anggota, proses kolaborasi, serta penggunaan alat bantu selama pengerjaan UTS Deep Learning. Pengerjaan Task 1 dan Task 2 dilakukan secara individu sesuai pembagian tanggung jawab, sedangkan Task 3 dikerjakan secara kolaboratif oleh kedua anggota kelompok.

## 1. Daftar Anggota

| No. | Nama | NIM |
|---|---|---|
| 1 | Muhammad Alief Reefandi | 101032330132 |
| 2 | Rofly Hi M Naser | 101032300085 |

## 2. Pembagian Tugas

| Anggota | Tanggung Jawab |
|---|---|
| Muhammad Alief Reefandi | Pengerjaan Task 1 – Multimodal Classification dan berkolaborasi dalam Task 3 – Multimodal Clustering. |
| Rofly Hi M Naser | Pengerjaan Task 2 – Time-Series Regression RUL dan berkolaborasi dalam Task 3 – Multimodal Clustering. |

## 3. Kontribusi Pengerjaan

### 3.1 Task 1 – Multimodal Classification: Product Defects

**Penanggung jawab: Muhammad Alief Reefandi**

Task 1 berfokus pada pengembangan model klasifikasi cacat produk menggunakan dua jenis masukan, yaitu citra dan data sensor. Kontribusi pengerjaan meliputi:

- Melakukan Exploratory Data Analysis (EDA) untuk memahami karakteristik data citra, sensor, distribusi kelas, dan kondisi dataset.
- Melakukan preprocessing data citra dan sensor, termasuk normalisasi, penanganan missing value, serta pembagian data training dan validation.
- Mengembangkan CNN sebagai encoder untuk mengekstraksi fitur dari data citra.
- Mengembangkan encoder deep learning untuk memproses data sensor.
- Mengimplementasikan dan membandingkan dua strategi penggabungan modalitas, yaitu Feature Fusion dan Decision Fusion.
- Melakukan eksperimen baseline menggunakan pendekatan image-only dan sensor-only.
- Menguji ketahanan model terhadap label noise, domain shift, dan missing modality.
- Melakukan sanity check untuk memastikan model mempelajari pola data dengan baik.
- Mengevaluasi kestabilan performa model menggunakan tiga random seed.
- Melakukan evaluasi menggunakan metrik Macro-F1 serta membuat visualisasi hasil eksperimen.
- Melakukan inferensi terhadap data test dan menghasilkan file `prediksi_klasifikasi.csv`.

Berdasarkan hasil eksperimen, Feature Fusion memperoleh rata-rata Macro-F1 sebesar **0,6591** dari tiga random seed dan menjadi pendekatan dengan performa terbaik dibandingkan model lainnya.

### 3.2 Task 2 – Time-Series Regression: Remaining Useful Life (RUL)

**Penanggung jawab: Rofly Hi M Naser**

Task 2 berfokus pada prediksi Remaining Useful Life (RUL) berdasarkan data time-series sensor mesin. Kontribusi pengerjaan meliputi:

- Melakukan EDA dan preprocessing data, termasuk normalisasi berdasarkan kondisi operasi, seleksi sensor, serta penanganan missing value dan spike.
- Membentuk label RUL dengan batas maksimum (*capping*) sebesar 150 siklus dan menggunakan sliding window berukuran 50.
- Menyusun validasi 5-fold berdasarkan mesin untuk menyesuaikan karakteristik pembagian data test.
- Melakukan pengujian anti-kebocoran data (*data leakage*) dan uji mutasi pada pipeline preprocessing.
- Mengembangkan dan membandingkan model baseline Gradient Boosting dengan model deep learning LSTM, TCN, dan Transformer.
- Mengevaluasi performa model menggunakan RMSE dan asymmetric score, termasuk analisis berdasarkan rentang RUL dan mode kerusakan.
- Mengembangkan prediction interval menggunakan MC Dropout, ensemble, dan conformal calibration.
- Menyusun hasil prediksi serta melakukan analisis kebijakan servis berdasarkan batas bawah prediction interval.
- Menghasilkan file prediksi RUL untuk 300 mesin test.

Berdasarkan hasil validasi, model ensemble LSTM dengan lima fold dipilih sebagai model akhir dengan RMSE sebesar **19,14** dan asymmetric score sebesar **4.673**.

### 3.3 Task 3 – Multimodal Clustering

**Penanggung jawab: Muhammad Alief Reefandi dan Rofly Hi M Naser**

Task 3 dikerjakan secara bersama-sama oleh kedua anggota kelompok. Pengerjaan mencakup analisis data, pengembangan representasi multimodal, eksperimen clustering, evaluasi, dan interpretasi hasil.

Kontribusi pengerjaan meliputi:

- Melakukan EDA terhadap data citra mikrostruktur dan spektrum, termasuk analisis distorsi serta pengujian estimator menggunakan data sintetis.
- Menentukan strategi augmentasi data dan melakukan training SimCLR pada masing-masing modalitas.
- Mengembangkan representasi fitur citra dan spektrum menggunakan pendekatan self-supervised learning.
- Mengembangkan model penggabungan modalitas menggunakan FusionAE dengan pendekatan modality dropout.
- Melakukan eksperimen ablasi untuk menganalisis pengaruh masing-masing modalitas terhadap hasil clustering.
- Membandingkan pendekatan clustering KMeans, Gaussian Mixture Model (GMM), dan IDEC.
- Melakukan pemilihan jumlah cluster menggunakan Silhouette Score, Davies-Bouldin Index, dan Calinski-Harabasz Index.
- Menganalisis stabilitas hasil clustering menggunakan bootstrap.
- Melakukan visualisasi embedding menggunakan UMAP dan t-SNE.
- Menganalisis pasangan cluster yang sulit dibedakan, heterogenitas cluster 0, serta alternatif jumlah cluster k = 7.
- Melakukan pengujian menggunakan data sintetis untuk memeriksa kemampuan model dalam mengenali pola dan cluster yang telah diketahui.

Berdasarkan hasil eksperimen, jumlah cluster yang dipilih adalah **k = 6**. Model final menggunakan IDEC dengan representasi latent dari FusionAE. Kedua anggota terlibat dalam diskusi, evaluasi hasil, serta penentuan pendekatan yang digunakan hingga menghasilkan file `cluster.csv`.

## 4. Infrastruktur dan Integrasi Proyek

Selain pengerjaan masing-masing task, pengembangan repository dilakukan dengan memperhatikan konsistensi struktur kode, kemudahan eksekusi, dan reproduksibilitas eksperimen.

Komponen pendukung yang digunakan meliputi:

- `run_all.py` sebagai skrip utama untuk menjalankan seluruh task maupun task tertentu.
- `kode/common/` sebagai direktori yang berisi fungsi bersama untuk pengelolaan argumen, random seed, path, utilitas, dan visualisasi.
- Flag `--debug`, `--data_dir`, dan `--out_dir` untuk mendukung konfigurasi eksekusi program.
- `kode/validasi_csv.py` untuk memeriksa kesesuaian format ketiga file prediksi.
- Google Colab dan Kaggle Notebook sebagai lingkungan untuk menjalankan eksperimen dan training model.
- Direktori `models/` untuk menyimpan checkpoint model yang digunakan dalam proses inferensi.
- Direktori `results/` untuk menyimpan hasil EDA, evaluasi, visualisasi, dan dokumentasi setiap task.

Kedua anggota turut berkontribusi dalam proses integrasi kode, pemeriksaan hasil, serta persiapan repository agar dapat dijalankan kembali sesuai kebutuhan evaluasi.

## 5. Penggunaan Artificial Intelligence (AI)

AI digunakan sebagai alat bantu selama proses pengerjaan proyek, khususnya untuk mendukung pemahaman konsep, pengembangan kode, debugging, analisis, dan penyusunan dokumentasi. Penggunaan AI disesuaikan dengan tanggung jawab masing-masing anggota.

### 5.1 Muhammad Alief Reefandi – Task 1 dan Task 3

**Tools:** ChatGPT.

ChatGPT digunakan untuk membantu memahami tahapan pengerjaan Task 1, memberikan referensi implementasi kode, membantu proses debugging, serta mendukung analisis hasil EDA dan evaluasi model. AI juga digunakan sebagai alat bantu dalam memahami implementasi model, pembuatan visualisasi, dan penyusunan dokumentasi.

Pelaksanaan eksperimen, training model, pengujian, perbandingan hasil, serta penentuan model akhir dilakukan dan diverifikasi oleh anggota yang bersangkutan. Dalam pengerjaan Task 3, ChatGPT juga digunakan sebagai bantuan teknis selama proses kolaborasi.

### 5.2 Rofly Hi M Naser – Task 2 dan Task 3

**Tools:** Claude melalui Claude Code.

Claude digunakan sebagai alat bantu dalam penulisan dan perbaikan kode Python, pemahaman metode yang digunakan, serta penyusunan dokumentasi dan analisis hasil eksperimen.

Pada Task 2, AI dimanfaatkan untuk mendukung proses preprocessing, pengembangan model, pengujian anti-kebocoran data, dan evaluasi hasil. Pada Task 3, AI digunakan sebagai referensi dalam implementasi dan debugging beberapa bagian model clustering.

Eksperimen, training, pemeriksaan hasil, pengujian ulang, serta keputusan akhir terkait metode dan model tetap dilakukan dan diverifikasi oleh anggota yang bersangkutan.

### 5.3 Verifikasi dan Validasi Hasil

Untuk memastikan implementasi dan hasil eksperimen dapat diperiksa kembali, kedua anggota melakukan beberapa langkah verifikasi, antara lain:

- Menjalankan pengujian otomatis pada masing-masing task.
- Memeriksa hasil preprocessing dan prediksi.
- Melakukan sanity check serta pengujian menggunakan data sintetis.
- Melakukan debugging sebelum menjalankan training secara penuh.
- Memeriksa kembali angka dan hasil evaluasi yang dicantumkan dalam README dan laporan.
- Menjalankan ulang eksperimen tertentu untuk memastikan hasil dapat direproduksi.

Dengan demikian, AI berperan sebagai alat bantu selama proses pengerjaan, sedangkan pemahaman, pelaksanaan eksperimen, verifikasi hasil, dan pengambilan keputusan akhir tetap menjadi tanggung jawab masing-masing anggota kelompok.
