# Hasil Eksperimen Task 2 – Prediksi Remaining Useful Life (RUL)

## 1. Deskripsi

Memprediksi sisa umur (RUL, dalam siklus) setiap mesin pada siklus terakhir yang tercatat, dari 21 sensor
dan 3 setelan operasi. Train: 900 mesin (dijalankan sampai rusak). Test: 300 mesin (histori terpotong,
50–455 siklus). Label dibatasi **cap 150**. Output: `prediksi_rul.csv` (`id_unit,rul`; 300 baris, 0–150)
di root repo dan salinannya di `prediksi/`.

Reproduksi: `python run_all.py --tasks 2` (atau `python -m kode.task2.main`). Training penuh di Colab T4
(`kode/colab_task2_task3.ipynb`) memakan ~10 menit untuk 4 model × 5 fold + MC dropout.
Lingkungan run ini (dari log Colab): Python 3.13.16, torch 2.14.1+cu130, GPU Tesla T4, seed 42.

| Bagian | Isi | Folder |
|---|---|---|
| a | EDA & preprocessing | `eda/` (lihat `eda/RINGKASAN.md`) |
| b–c | Label RUL & sliding window tanpa kebocoran | `windows/` (lihat `windows/RINGKASAN.md`) |
| d–e | Baseline vs 3 arsitektur sequence, evaluasi per mesin | `models/` |
| f | Prediction interval & kebijakan servis | `uncertainty/` |

## 2. Preprocessing singkat (detail: `eda/RINGKASAN.md`)

- 6 kondisi operasi (dari `setel_1`) → **z-score per kondisi**, sehingga efek kondisi hilang dan tren degradasi terlihat.
- 7 sensor dibuang karena hanya white noise (rasio std(Δz)/std(z) ≈ √2, tidak berkorelasi dengan umur). Sensor yang dipakai: **14**.
- Missing ~2% sel → forward-fill per mesin (hanya masa lalu). Spike 0,2% → dihapus secara kausal (median 10 siklus sebelumnya, ambang 5σ).
- Dua mode kerusakan (A: s11 turun, 558 mesin; B: s11 naik, 342 mesin) → model tidak diberi tahu mode; hasil dilaporkan per mode.
- Window W = 50 + channel mask + umur mesin → tensor (50, 16). Split **5-fold per mesin**; validasi meniru test (histori val dipotong di titik acak, hanya siklus terakhir yang dinilai; 900 mesin × 5 potongan = 4.500 sampel val).

## 3. Perbandingan model (validasi out-of-fold, 5 fold)

Skor asimetris: d = prediksi − aktual; s = Σ exp(−d/13) − 1 (d < 0) atau exp(d/10) − 1 (d ≥ 0).
Terlambat (prediksi terlalu besar) dihukum lebih berat. Skor dinormalisasi ke 300 mesin (= ukuran test).

| Model | Jenis | Parameter | RMSE (rata-rata fold ± std) | RMSE gabungan | MAE | Bias | Skor / 300 | Detik / fold |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **LSTM** | DL (sequence) | 69.761 | **19,13 ± 0,58** | **19,14** | 14,52 | **+0,51** | **4.673** | 17 |
| TCN | DL (konvolusi kausal) | 58.097 | 19,51 ± 0,52 | 19,51 | 14,97 | +2,46 | 5.226 | 23 |
| Transformer | DL (self-attention) | 103.041 | 19,75 ± 0,51 | 19,75 | 14,45 | +1,55 | 6.170 | 28 |
| Baseline | Gradient boosting, 58 fitur manual | – | 19,82 ± 0,80 | 19,83 | 14,44 | +2,24 | 6.654 | 12 |

Sumber: `models/tabel_perbandingan_model.csv`, per fold di `models/<model>/fold<k>.json`.

**Pilihan: LSTM** (`models/pilihan_model.json`). Pemilihan hanya di antara model DL, sesuai aturan bahwa
model utama harus deep learning. Alasan:
- RMSE dan skor asimetris terbaik. Skor 30% lebih rendah dari baseline (4.673 vs 6.654), karena bias LSTM
  paling kecil (+0,5). Bias positif = prediksi terlambat, dan itulah yang paling mahal di skor.
- Beda RMSE rata-rata (0,7) memang seukuran std antar-fold, tetapi perbandingannya **berpasangan**:
  di **kelima** fold LSTM lebih baik dari baseline (20,08 vs 20,89; 19,29 vs 20,46; 19,02 vs 19,71;
  18,29 vs 18,62; 18,98 vs 19,39), jadi keunggulannya konsisten, bukan kebetulan satu fold.
- Prediksi test = rata-rata 5 model fold (ensemble), bobot di `models/task2_lstm_fold{0-4}_best.pt`.

Kurva training (`models/12_kurva_training.png`): early stopping (patience 8) berhenti di epoch 19–21
(LSTM), 18–24 (TCN), 15–19 (Transformer). Val RMSE Transformer mulai naik setelah ~epoch 7 sementara train
terus turun, jadi model terbesar ini paling cepat overfit. Data hanya 900 mesin, sehingga kapasitas
tambahan tidak membantu.

### 3.1 Per rentang RUL aktual (`models/tabel_per_rentang_rul.csv`, grafik 10)

| RUL aktual | n | Baseline | LSTM | TCN | Transformer | Bias LSTM |
|---|---:|---:|---:|---:|---:|---:|
| 0–25 | 521 | **9,91** | 11,06 | 13,84 | 11,83 | +7,1 |
| 25–50 | 570 | 15,36 | 13,95 | 14,34 | **13,55** | +3,9 |
| 50–75 | 567 | 25,38 | **22,82** | 24,72 | 23,46 | +8,4 |
| 75–100 | 501 | 33,36 | **30,56** | 31,77 | 32,43 | +14,5 |
| 100–125 | 474 | 24,57 | 23,62 | **23,48** | 26,35 | +9,4 |
| 125–150 | 1.867 | **14,17** | 15,47 | 14,32 | 14,82 | −10,7 |

- **Galat terbesar di RUL 50–100** dan biasnya positif (terlambat). Pada tahap ini degradasi baru mulai,
  sinyal sensor masih lemah, dan kapan mesin mulai menurun berbeda-beda. Skor asimetris paling banyak
  berasal dari sini.
- RUL 125–150: bias negatif (−10,7). Mesin sehat diprediksi sedikit lebih pendek umurnya (arah yang aman).
  Ini efek cap 150: label tidak bisa lebih dari 150, sehingga model menarik ke tengah.
- RUL 0–25: baseline sedikit lebih baik (9,9 vs 11,1). Fitur manual "level 5 siklus terakhir" sangat
  informatif saat mesin hampir rusak, tetapi rentang ini hanya 12% sampel. Secara keseluruhan LSTM tetap
  lebih baik di 25–125, rentang yang paling sulit.

### 3.2 Per mode kerusakan (`models/tabel_per_mode.csv`)

| Mode | n | Baseline | LSTM | TCN | Transformer |
|---|---:|---:|---:|---:|---:|
| A (s11 turun) | 2.790 | 19,77 | **18,99** | 19,37 | 19,79 |
| B (s11 naik) | 1.710 | 19,93 | **19,39** | 19,75 | 19,70 |

Galat kedua mode hampir sama, jadi model tidak hanya bekerja untuk mode yang lebih banyak datanya.

## 4. Prediction interval & kapan mesin diservis (Task 2f)

**Metode.** MC dropout (dropout tetap aktif saat inferensi, 50 pass) × 5 model fold → rata-rata = prediksi,
std = ketidakpastian. Interval MC mentah terlalu sempit, sehingga dikalibrasi dengan **split conformal
ternormalisasi**: interval = [pred + q_lo·std, pred + q_hi·std], q dari kuantil residual val (aktual −
pred)/std. Cakupan dihitung **cross-fit** (q untuk fold k diestimasi dari 4 fold lain).

### 4.1 Kalibrasi (`uncertainty/tabel_kalibrasi_interval.csv`, grafik 13)

| Level | Cakupan MC mentah | Cakupan terkalibrasi | Lebar mentah | Lebar terkalibrasi |
|---|---:|---:|---:|---:|
| 50% | 0,232 | 0,499 | 11,0 | 22,4 |
| 80% | 0,464 | 0,797 | 19,2 | 43,2 |
| **90%** | 0,540 | **0,899** | 23,4 | 56,8 |
| 95% | 0,584 | 0,948 | 26,6 | 67,1 |

MC dropout saja hanya mencakup 54% (seharusnya 90%). MC dropout mengukur ketidakpastian **model**, bukan
noise data dan variasi antar-mesin. Setelah kalibrasi, cakupan tepat sesuai target di semua level.

Per rentang RUL (`uncertainty/tabel_interval_per_rentang.csv`), cakupan 90% tidak merata: 0,81–0,84 di
RUL 25–100 (kurang), 0,92–0,97 di RUL > 100 (lebih). Kalibrasinya global (satu pasang q untuk semua
sampel). Perbaikan yang mungkin: q berbeda per rentang prediksi. Lebar interval sudah ikut menyesuaikan:
36 siklus saat RUL < 50 vs 64–68 siklus saat RUL > 75, karena std MC lebih besar untuk mesin yang masih
jauh dari rusak.

### 4.2 Kebijakan servis (`uncertainty/tabel_kebijakan_servis.csv`, grafik 15)

Asumsi: servis perlu disiapkan **H = 20 siklus** sebelumnya. Mesin "perlu diservis" jika RUL aktual ≤ H.
Aturan yang dibandingkan pada data validasi: servis jika *X ≤ H*, dengan X = prediksi titik atau batas bawah
interval.

| Aturan (H = 20) | Kerusakan tertangkap | Terlewat | Servis terlalu dini | Rata-rata RUL aktual saat diservis |
|---|---:|---:|---:|---:|
| Prediksi titik ≤ 20 | 66,4% | 150 | 51 | 12,3 |
| Batas bawah 50% ≤ 20 | 83,9% | 72 | 142 | 15,4 |
| Batas bawah 80% ≤ 20 | 93,3% | 30 | 260 | 18,5 |
| **Batas bawah 90% ≤ 20** | **96,6%** | **15** | 331 | 20,3 |
| Batas bawah 95% ≤ 20 | 98,7% | 6 | 410 | 22,4 |

- Memakai **prediksi titik saja berbahaya**: sepertiga mesin yang sebenarnya hampir rusak terlewat.
- **Rekomendasi: servis jika batas bawah interval 90% ≤ 20 siklus.** Kerusakan terlewat turun dari 150 ke
  15 (−90%). Biayanya: lebih banyak servis dini, tetapi rata-rata mesin yang diservis masih punya ~20 siklus
  (servis hanya sedikit lebih awal dari perlu).
- Level dipilih sesuai biaya: jika kerusakan jauh lebih mahal dari servis dini → 95%. Jika servis mahal →
  80%. Tabel untuk H = 10 dan 30 juga tersedia di CSV.

### 4.3 Status 300 mesin test (`uncertainty/prediksi_rul_interval_test.csv`)

| Status | Aturan | Jumlah |
|---|---|---:|
| SERVIS SEGERA | batas bawah 90% ≤ 20 | 64 |
| JADWALKAN | 20 < batas bawah 90% ≤ 50 | 75 |
| AMAN | batas bawah 90% > 50 | 161 |

Hanya 20 mesin yang prediksi titiknya ≤ 20, tetapi 64 mesin perlu segera diservis setelah
ketidakpastian ikut dihitung. Itulah gunanya interval.
Statistik prediksi test: median 106, min 8,4, maks 146 siklus.

## 5. Keterbatasan & catatan

- RMSE ~19 didominasi rentang RUL 50–100, saat sinyal degradasi masih lemah. Batas ini datang dari data
  (informasi sensor), dan ketiga arsitektur DL mentok di angka yang mirip.
- Kalibrasi interval global → under-coverage di RUL 25–100 (lihat 4.1).
- Eksperimen window W = 100 tersedia (`--W 100`) tetapi tidak dijalankan pada run final ini.
- Hasil di folder ini berasal dari kode sebelum commit perbaikan scheduler AMP (`kode/task2/train.py`:
  scheduler kini tidak maju pada langkah yang dilewati GradScaler). Di CPU hasilnya identik. Di GPU hanya
  beberapa langkah awal OneCycle yang bergeser, sehingga rerun dapat berbeda sedikit (orde desimal RMSE).
- Grafik 15 digambar ulang dari `tabel_kebijakan_servis.csv` agar label tidak bertumpuk (angka sama).

## 6. File

| File | Isi |
|---|---|
| `../../prediksi_rul.csv`, `../../prediksi/prediksi_rul.csv` | Prediksi final (ensemble 5 fold LSTM) |
| `models/tabel_perbandingan_model.csv` | Tabel utama perbandingan model |
| `models/tabel_per_rentang_rul.csv`, `models/tabel_per_mode.csv` | Evaluasi per rentang RUL & per mode |
| `models/<model>/fold<k>.json` | Metrik + kurva training per fold |
| `models/test_prediksi_per_fold_lstm.csv` | Prediksi test per model fold |
| `models/log.txt` | Log training Colab |
| `uncertainty/*` | Kalibrasi, interval test, kebijakan servis |
| `../../models/task2_lstm_fold*_best.pt` (+ `_preprocessor.json`) | Bobot final (5 × 283 KB) + statistik normalisasi per fold |
