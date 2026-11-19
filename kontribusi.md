# Kontribusi Anggota Kelompok

Dokumen ini menjelaskan pembagian tugas dan kontribusi setiap anggota dalam pengerjaan UTS Deep Learning.

## Daftar Anggota

| No. | Nama | NIM |
|---:|---|---|
| 1 | Muhammad Alief Reefandi | 101032330132 |
| 2 | Rofly Hi M Naser | 101032300085 |


## Pembagian Tugas

| Anggota | Tanggung Jawab |
|---|---|
| Muhammad Alief Reefandi | Mengerjakan Task 1 |
| Rofly Hi M Naser | Mengerjakan Task 2 (regresi RUL) dan Task 3 (clustering multimodal); `run_all.py`, `kode/validasi_csv.py`, notebook Colab/Kaggle Task 2–3; penyusunan laporan |


## Kontribusi Task 1 – Multimodal Classification

- Exploratory Data Analysis (EDA) dan preprocessing dataset.
- Pengembangan model CNN untuk citra.
- Pengembangan encoder sensor menggunakan Deep Learning.
- Implementasi dan perbandingan Feature Fusion serta Decision Fusion.
- Eksperimen image-only dan sensor-only.
- Evaluasi label noise, domain shift, missing modality, dan sanity check.
- Evaluasi menggunakan tiga random seed.
- Pembuatan prediksi data test dan visualisasi hasil.

## Kontribusi Task 2 dan Task 3 (Rofly Hi M Naser)

### Task 2 – Regresi time-series RUL

- EDA dan preprocessing:
  - 6 rezim operasi dengan normalisasi z-score per kondisi;
  - seleksi sensor tanpa label (7 sensor white noise dibuang);
  - missing value dan spike ditangani secara kausal.
- Label RUL dengan cap 150, sliding window W = 50 dengan mask, dan validasi 5-fold per mesin yang meniru kondisi test.
- 6 uji anti-kebocoran plus uji mutasi (`kode/task2/test_leakage.py`).
- Baseline gradient boosting (non-DL) dibandingkan dengan LSTM, TCN, dan Transformer. Evaluasi RMSE dan skor asimetris, juga per rentang RUL dan per mode kerusakan.
- Prediction interval: MC dropout × ensemble 5 fold, dikalibrasi dengan conformal. Kebijakan servis berbasis batas bawah interval.

### Task 3 – Clustering multimodal

- EDA distorsi gambar (rotasi, skala, brightness, beda mikroskop) dan spektrum (shift, baseline, spike), dengan validasi estimator pada data sintetis.
- SimCLR per modalitas dengan augmentasi yang diturunkan dari EDA. Training di Kaggle GPU, bisa resume dan bisa dihentikan dengan batas waktu.
- Fusion:
  - blok per modalitas dengan koreksi mikroskop yang diuji dulu syaratnya;
  - FusionAE dengan modality dropout;
  - ablasi satu modalitas.
- Pemilihan k dengan aturan yang ditetapkan sebelum melihat hasil. Perbandingan KMeans dan GMM dengan IDEC.
- UMAP/t-SNE, pasangan cluster sulit, analisis heterogenitas cluster 0, dan analisis alternatif k = 7.
- Tes pada data sintetis yang kebenarannya diketahui (`kode/task3/test_ssl.py`, `kode/task3/test_cluster.py`).

### Infrastruktur bersama

- `run_all.py`: satu perintah untuk ketiga task, dengan seed dan versi library dicatat.
- `kode/common/`, flag `--debug`, `--data_dir` dan `--out_dir`.
- `kode/validasi_csv.py`: validasi format ketiga CSV prediksi, termasuk `prediksi_klasifikasi.csv`.
- Notebook `kode/colab_task2_task3.ipynb` dan `kode/colab_task3.ipynb`.

### Pull request (AliefReefandi/UTS-Deep-Learning)

| PR | Branch | Isi |
|---|---|---|
| [#1](https://github.com/AliefReefandi/UTS-Deep-Learning/pull/1) | `task2/rul` | Kode Task 2 (a–f), `run_all.py`, `validasi_csv.py`, notebook Colab |
| [#2](https://github.com/AliefReefandi/UTS-Deep-Learning/pull/2) | `task3/clustering` | Task 3a (EDA) dan 3b (SimCLR) |
| [#3](https://github.com/AliefReefandi/UTS-Deep-Learning/pull/3) | `task2/hasil` | Hasil training penuh Task 2, `results/task2/README.md`, prediksi RUL |
| [#4](https://github.com/AliefReefandi/UTS-Deep-Learning/pull/4) | `task3/fix-notebook` | Perbaikan deteksi lingkungan Colab/Kaggle |
| [#5](https://github.com/AliefReefandi/UTS-Deep-Learning/pull/5) | `task3/cluster` | Task 3c–f, hasil lengkap, `cluster.csv` |
| #6 | `final/laporan` | README, kontribusi, laporan, persiapan ujian lisan |

## Penggunaan AI

### Rofly Hi M Naser (Task 2 dan Task 3)

**Tool:** Claude (Anthropic) melalui **Claude Code** (asisten pemrograman di terminal).

**Bagian yang dibantu:**
- Menulis dan merapikan kode Python/PyTorch untuk Task 2 dan 3, `run_all.py`, `validasi_csv.py`, dan notebook Colab/Kaggle.
- Analisis EDA, pembuatan grafik, dan draf teks README hasil, laporan, serta persiapan ujian lisan.
- Saran desain (mis. normalisasi per kondisi, kalibrasi conformal, SimCLR, modality dropout, IDEC).
  Keputusan akhir saya ambil sendiri, terutama ketika ada beberapa pilihan, misalnya jumlah cluster di Task 3.

**Yang tidak dilakukan AI:** push ke GitHub, pembuatan PR, dan training penuh (Colab/Kaggle). Semuanya saya lakukan sendiri.

**Cara verifikasi:**
1. **Tes otomatis yang bisa gagal:**
   - `test_leakage`: 6 uji anti-kebocoran Task 2, ditambah uji mutasi (bfill dan median berpusat sengaja disuntikkan dan harus terdeteksi).
   - `test_models`.
   - `test_ssl` dan `test_cluster`: data sintetis dengan jawaban yang diketahui, mis. pipeline harus menemukan k = 6 dan IDEC harus memulihkan material dengan ARI > 0,9.
2. **`validasi_csv`** untuk ketiga CSV prediksi sebelum setiap PR.
3. **Cek ulang hasil:**
   - Angka di README/laporan dibaca ulang dari CSV/JSON hasil run.
   - Run Task 3c–f dijalankan ulang untuk memastikan hasilnya deterministik.
   - Embedding dihitung ulang dari bobot model di repo untuk memastikan `cluster.csv` dapat direproduksi.
4. **Debug lokal dulu:** setiap tahap diuji dengan `--debug` di CPU sebelum training penuh.
5. **Koreksi yang ditemukan selama verifikasi:**
   - *Task 2:* `lr_scheduler.step()` terpanggil sebelum `optimizer.step()` saat AMP melewati langkah; diperbaiki. `env_info` notebook tidak tersimpan; diperbaiki.
   - *Task 3b:* padding refleksi saat augmentasi rotasi menciptakan tekstur palsu, diganti crop di dalam gambar. Baseline PCA piksel terlalu lemah, diganti PSD radial.
   - *Task 3c–f:*
     - Pasangan sulit awalnya dihitung di ruang yang sudah dipertajam IDEC, sehingga semua overlap 0%.
     - Dua "pulau" di UMAP awalnya ditafsirkan sebagai tekstur. Setelah dicek ulang per modalitas, ternyata itu faktor spektrum (jarak puncak). Interpretasi dikoreksi di README dan laporan.
   - *Notebook:* deteksi Kaggle salah di Colab (folder `/kaggle/input` juga ada di Colab).

### Muhammad Alief Reefandi (Task 1)

*[Diisi oleh Alief: tool AI yang dipakai (jika ada), bagian yang dibantu, dan cara verifikasinya.]*
