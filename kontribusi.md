# Kontribusi Anggota Kelompok

Dokumen ini menjelaskan pembagian tugas, kontribusi, serta penggunaan alat bantu selama pengerjaan UTS Deep Learning. Pengerjaan dilakukan secara individu sesuai pembagian task, sedangkan Task 3 dikerjakan secara bersama-sama.

## 1. Daftar Anggota

| No. | Nama | NIM |
|---|---|---|
| 1 | Muhammad Alief Reefandi | 101032330132 |
| 2 | Rofly Hi M Naser | 101032300085 |

## 2. Pembagian Tugas

| Anggota | Tanggung Jawab |
|---|---|
| Muhammad Alief Reefandi | Pengerjaan Task 1 dan berkolaborasi dalam Task 3 |
| Rofly Hi M Naser | Pengerjaan Task 2 dan berkolaborasi dalam Task 3 |

## 3. Kontribusi Pengerjaan

### Task 1 – Multimodal Classification (Muhammad Alief Reefandi)

Task 1 dikerjakan oleh Muhammad Alief Reefandi, meliputi:

- Melakukan EDA dan preprocessing dataset.
- Membuat dan mengembangkan model CNN untuk data citra.
- Mengembangkan encoder untuk data sensor.
- Mengimplementasikan Feature Fusion dan Decision Fusion.
- Melakukan eksperimen image-only dan sensor-only.
- Menguji label noise, domain shift, missing modality, dan sanity check.
- Melakukan evaluasi menggunakan tiga random seed.
- Membuat prediksi data test serta visualisasi hasil eksperimen.

### Task 2 – Regresi Time-Series RUL (Rofly Hi M Naser)

Task 2 dikerjakan oleh Rofly Hi M Naser, meliputi:

- Melakukan EDA dan preprocessing data, termasuk normalisasi berdasarkan kondisi operasi, seleksi sensor, serta penanganan missing value dan spike.
- Membuat label RUL dengan cap 150 dan sliding window berukuran 50.
- Menyusun validasi 5-fold berdasarkan mesin untuk menyesuaikan kondisi data test.
- Melakukan pengujian anti-kebocoran data dan uji mutasi pada pipeline preprocessing.
- Mengembangkan dan membandingkan model baseline Gradient Boosting dengan LSTM, TCN, dan Transformer.
- Mengevaluasi model menggunakan RMSE dan asymmetric score, termasuk berdasarkan rentang RUL dan mode kerusakan.
- Mengembangkan prediction interval menggunakan MC Dropout, ensemble, dan conformal calibration.
- Menyusun hasil prediksi serta analisis kebijakan servis berdasarkan batas bawah prediction interval.

### Task 3 – Multimodal Clustering (Muhammad Alief Reefandi dan Rofly Hi M Naser)

Task 3 dikerjakan secara bersama-sama oleh kedua anggota, mulai dari analisis data, pengembangan model, eksperimen clustering, hingga evaluasi hasil.

- Melakukan EDA terhadap distorsi citra dan spektrum serta menguji estimator menggunakan data sintetis.
- Menentukan augmentasi dan melakukan training SimCLR pada masing-masing modalitas.
- Mengembangkan fusion model menggunakan FusionAE dengan modality dropout.
- Melakukan eksperimen ablasi untuk melihat pengaruh masing-masing modalitas.
- Membandingkan KMeans, GMM, dan IDEC dalam menentukan hasil clustering.
- Melakukan analisis embedding menggunakan UMAP dan t-SNE.
- Mengevaluasi cluster, pasangan data yang sulit dibedakan, heterogenitas cluster 0, serta alternatif jumlah cluster k = 7.
- Melakukan pengujian menggunakan data sintetis untuk memeriksa kemampuan model dalam mengenali pola dan cluster yang telah diketahui.

Pembagian pengerjaan Task 3 dilakukan secara kolaboratif, dengan diskusi mengenai hasil eksperimen, evaluasi model, dan penentuan pendekatan yang digunakan hingga memperoleh hasil akhir.

## 4. Infrastruktur dan Integrasi

Dalam pengerjaan proyek, kedua anggota turut menggunakan dan mengintegrasikan beberapa komponen pendukung, antara lain:

- `run_all.py` untuk menjalankan keseluruhan task.
- `kode/common/` untuk fungsi bersama.
- Flag `--debug`, `--data_dir`, dan `--out_dir`.
- `kode/validasi_csv.py` untuk memeriksa format file prediksi.
- Notebook Google Colab/Kaggle untuk menjalankan eksperimen dan training model.

## 5. Pull Request dan Kontribusi Repository

Repository: [AliefReefandi/UTS-Deep-Learning](https://github.com/AliefReefandi/UTS-Deep-Learning)

| PR | Branch | Kontribusi |
|---|---|---|
| #1 | `task2/rul` | Implementasi Task 2, integrasi `run_all.py`, validasi CSV, dan notebook |
| #2 | `task3/clustering` | EDA Task 3 dan implementasi SimCLR |
| #3 | `task2/hasil` | Hasil training Task 2, dokumentasi, dan prediksi RUL |
| #4 | `task3/fix-notebook` | Perbaikan deteksi lingkungan Colab/Kaggle |
| #5 | `task3/cluster` | Implementasi Task 3c–f, hasil clustering, dan `cluster.csv` |
| #6 | `final/laporan` | Integrasi README, laporan, kontribusi, dan persiapan ujian lisan |

## 6. Penggunaan AI

Dalam pengerjaan tugas ini, AI digunakan sebagai alat bantu untuk mendukung proses pemrograman dan pemahaman metode Deep Learning. Pengembangan kode, pelaksanaan eksperimen, pengujian, analisis hasil, serta penentuan model dilakukan oleh masing-masing anggota sesuai pembagian tugas.

### Rofly Hi M Naser (Task 2 dan Task 3)

**Tools:** Claude melalui Claude Code.

AI digunakan sebagai bantuan dalam penulisan dan perbaikan kode Python, pemahaman metode yang digunakan, serta penyusunan dokumentasi dan analisis hasil eksperimen. Pada Task 2, proses pengerjaan meliputi preprocessing, pengembangan model, pengujian anti-kebocoran, dan evaluasi hasil. Pada Task 3, AI digunakan sebagai referensi dalam implementasi dan debugging beberapa bagian model clustering.

Eksperimen, training, pemeriksaan hasil, pengujian ulang, serta keputusan akhir terkait metode dan model dilakukan sendiri.

### Muhammad Alief Reefandi (Task 1 dan Task 3)

**Tools:** ChatGPT.

AI digunakan sebagai bantuan untuk memahami tahapan pengerjaan Task 1, memperbaiki kode, dan membantu menganalisis hasil EDA maupun evaluasi model. Bantuan juga digunakan dalam memahami implementasi model, visualisasi, dan penyusunan dokumentasi.

Seluruh proses menjalankan notebook, training, melakukan eksperimen, membandingkan hasil, serta menentukan model yang digunakan dilakukan sendiri. Pada Task 3, penggunaan AI juga dilakukan sebagai bantuan teknis selama pengerjaan bersama.

### Verifikasi Hasil

Untuk memastikan hasil pengerjaan sesuai dengan implementasi dan dapat diperiksa kembali, dilakukan beberapa langkah berikut:

- Menjalankan pengujian otomatis pada masing-masing task.
- Memeriksa hasil preprocessing dan prediksi.
- Melakukan sanity check dan pengujian pada data sintetis.
- Menjalankan debugging sebelum training penuh.
- Memeriksa kembali angka dan hasil evaluasi pada README serta laporan.
- Menjalankan ulang eksperimen tertentu untuk memastikan hasilnya dapat direproduksi.

Dengan demikian, AI berperan sebagai alat bantu selama proses pengerjaan, sedangkan pemahaman, pelaksanaan eksperimen, verifikasi, dan keputusan akhir tetap menjadi tanggung jawab anggota kelompok.
