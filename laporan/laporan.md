# Laporan UTS Deep Learning

**Klasifikasi Multimodal, Regresi Remaining Useful Life, dan Clustering Multimodal**

| Nama | NIM | Bagian |
|---|---|---|
| Muhammad Alief Reefandi | 101032330132 | Task 1 |
| Rofly Hi M Naser | 101032300085 | Task 2, Task 3 |

Repositori: `github.com/AliefReefandi/UTS-Deep-Learning`. Semua angka di laporan ini dibaca dari
berkas hasil di `results/`. Reproduksi: `python run_all.py` (seed 42; versi library dicatat di
`env_info.json`). Hanya data yang diberikan yang dipakai; tidak ada data eksternal atau bobot pretrained.

## Ringkasan

| Task | Model utama (deep learning) | Pembanding | Hasil validasi | Berkas prediksi |
|---|---|---|---|---|
| 1. Klasifikasi cacat produk | Feature Fusion (CNN 2D + CNN 1D) | Decision Fusion, image-only, sensor-only | Macro-F1 0,659 ± 0,013 (3 seed) | `prediksi_klasifikasi.csv` |
| 2. Prediksi RUL | LSTM, ensemble 5 fold | Gradient boosting (non-DL), TCN, Transformer | RMSE 19,14; skor 4.673 / 300 mesin | `prediksi_rul.csv` |
| 3. Clustering material | SimCLR + FusionAE + IDEC | KMeans, GMM | k = 6; stabilitas 0,96 | `cluster.csv` |

# Task 1 – Klasifikasi Multimodal Cacat Produk

Tujuan: mengklasifikasikan unit produk ke 7 kelas (normal, gores, noda, retak, gelembung, las_dingin,
kontaminasi) dari foto permukaan 64×64 dan sinyal sensor 3 kanal × 128 titik (suhu, getaran, arus).
Data latih 48.000 unit, data uji 16.000 unit. Metrik: **Macro-F1**. Notebook: `kode/01_Task1_Classification.ipynb`.

## 1a. Eksplorasi data

- **Distribusi kelas tidak seimbang:** dari normal 23.888 (49,8%) sampai kontaminasi 2.357 (4,9%).
  Karena itu dipakai Macro-F1, split stratified, dan cross-entropy berbobot kelas.
- **Contoh tiap kelas:**
  - *Citra:* tekstur granular/bintik dan garis memanjang/diagonal; perbedaan antar-kelas halus.
  - *Sensor:* suhu terkonsentrasi di 181–182 °C, getaran bimodal di sekitar nol, arus 9–10 A dengan ekor ke bawah.
- **Rekaman sensor hilang:**
  - 5,9–6,0% nilai per kanal; 3.863 unit (8,1%) punya minimal satu nilai hilang.
  - Per kelas merata (5,2% kontaminasi sampai 6,5% retak), sehingga pola missing tidak membocorkan kelas.
- **Perbedaan latih–uji:** latih lini A 74,9% / B 25,1%, sedangkan uji lini A 40,0% / B 60,0%. Ini
  potensi domain shift yang diuji di 1e. Validasi (split acak) mengikuti distribusi latih: A 7.153 / B 2.447.

<!-- ALIEF: soal minta contoh tiap kelas di kedua modalitas. Grid contoh citra dan sinyal per kelas ada di notebook (sel 10–11) tetapi belum disimpan ke results/task1. Simpan satu gambar gabungan dan tambahkan di sini. -->

**Preprocessing:**
- Citra dibagi 255. Split stratified 80/20 (seed 42): 38.400 latih / 9.600 validasi.
- Nilai sensor yang hilang diisi rata-rata per titik waktu dari data latih, lalu z-score per kanal dengan
  statistik latih. Semua statistik hanya dari data latih, jadi tidak bocor ke validasi atau uji.

## 1b. Arsitektur dan strategi fusion

- **Encoder citra:** CNN 3 blok (Conv–BN–ReLU–Pool; 32/64/128 filter) dengan global average pooling → 128 dimensi.
- **Encoder sinyal:** CNN 1D 3 blok (kernel 5/5/3) dengan global average pooling → 128 dimensi.
- **Dua strategi fusion yang dibandingkan:**
  - *Feature-level (early/intermediate) fusion:* embedding kedua encoder dikonkatenasi (256) → MLP 128
    → 7 kelas. Interaksi antar-modalitas dapat dipelajari.
  - *Decision-level (late) fusion:* tiap modalitas punya kepala klasifikasi sendiri, lalu logit dirata-rata.
- **Training:** Adam (lr 3·10⁻⁴), cross-entropy berbobot kelas (*balanced*), batch 64, maks. 30 epoch,
  early stopping (patience 8) pada Macro-F1 validasi.

| Model (3 seed: 42, 123, 2026) | Mean Macro-F1 | Std |
|---|---:|---:|
| **Feature Fusion** | **0,6591** | 0,0126 |
| Decision Fusion | 0,6409 | 0,0242 |
| Sensor-only | 0,4339 | 0,0058 |
| Image-only | 0,3123 | 0,0170 |

![Perbandingan model, rata-rata 3 seed](../results/task1/comparison_3_seeds.png){width=62%}

**Feature Fusion dipilih sebagai model final** (`models/feature_fusion_final.pth`): skornya lebih tinggi
dan variasinya antar-seed separuh dari Decision Fusion. Rata-rata logit menggabungkan dua keputusan yang
dibuat terpisah, sehingga modalitas yang lemah (citra) ikut menarik prediksi.

## 1c. Nilai tiap modalitas

- **Sensor-only (0,434) jauh lebih informatif daripada image-only (0,312)**, tetapi gabungan keduanya
  +0,23 di atas sensor-only. Jadi kedua modalitas membawa informasi yang saling melengkapi.
- **Per kelas** (Decision Fusion, seed 42; Macro-F1 0,682, akurasi 71,3%):
  - *las_dingin* paling mudah (F1 0,94).
  - *gores* paling sulit (F1 0,29): 641 dari 1.124 sampel diprediksi *normal*. Goresan tipis sulit
    terlihat di citra 64×64, dan pola sensornya mirip normal.

![Confusion matrix (Decision Fusion, seed 42)](../results/task1/Confusion Matrix.png){width=55%}

<!-- ALIEF: soal minta "kelas mana butuh modalitas mana". Yang ada baru per kelas untuk Decision Fusion. Tambahkan tabel F1 per kelas untuk image-only, sensor-only, dan Feature Fusion (classification_report pada model yang sudah dilatih), lalu tulis kelas mana yang bergantung pada citra dan mana pada sensor. -->

## 1d. Label tidak sempurna

- **Eksperimen yang dilakukan:** sebagian label latih diacak (simulasi label noise), lalu Decision
  Fusion dilatih ulang.
- **Hasil:** Macro-F1 validasi 0,682 (0%) → 0,609 (10%) → 0,548 (20%). Penurunannya bertahap, tidak runtuh.

<!-- ALIEF: soal minta (1) cek kemungkinan mislabel pada data asli, mis. sampel dengan loss tinggi / confidence rendah pada label sendiri dari prediksi out-of-fold, lalu periksa contohnya; dan (2) uji satu teknik robust, mis. label smoothing (CrossEntropyLoss(label_smoothing=0.1)) atau co-teaching, dibandingkan dengan CE biasa pada noise 10%/20%. Keduanya belum ada di notebook. -->

## 1e. Domain shift antar-lini

| Model (Decision Fusion) | Data latih | Macro-F1 validasi lini B |
|---|---|---:|
| Normal | lini A + B | 0,587 |
| Tanpa lini B | lini A saja | **0,197** |

- Bahkan dengan kedua lini di data latih, Macro-F1 di lini B (0,587) lebih rendah daripada validasi
  keseluruhan (0,682), jadi lini B memang lebih sulit.
- Tanpa contoh lini B, performa runtuh ke 0,197.
- Data uji 60% lini B, sehingga skor uji kemungkinan lebih rendah daripada validasi.

<!-- ALIEF: soal minta (1) performa per lini di validasi: tambahkan Macro-F1 lini A untuk model normal; dan (2) satu teknik untuk mengatasi shift, mis. normalisasi sensor per lini (statistik z-score dihitung per lini) atau augmentasi, dievaluasi pada skenario latih A → uji B. Teknik ini belum ada di notebook. -->

## 1f. Modalitas hilang

- **Uji saat inferensi** pada validasi:
  - *Citra hilang:* diganti citra rata-rata data latih.
  - *Sensor hilang:* diisi nol, yaitu rata-rata latih setelah z-score.
- **Hasil (Decision Fusion):** lengkap 0,682; tanpa citra 0,351; **tanpa sensor 0,301**.
- Saat sensor tidak terekam, prediksi masih di atas tebakan acak dan setara model image-only (0,312).
  Model tidak runtuh, tetapi juga belum dilatih untuk kondisi ini.

<!-- ALIEF: soal minta model "tetap memberi prediksi masuk akal saat sensor tidak terekam". Pertimbangkan modality dropout saat training (sensor dinolkan acak, mis. p = 0,3) pada Feature Fusion, lalu bandingkan skor tanpa sensor dengan image-only. Uji juga pada Feature Fusion (model final), bukan hanya Decision Fusion. -->

## 1g. Training: sanity check, learning curve, dan seed

- **Sanity check:** melatih dengan seluruh label latih diacak menurunkan Macro-F1 validasi dari 0,682
  ke **0,137**, setara tebakan. Model memang mempelajari hubungan input–label, bukan artefak pipeline.
- **Learning curve** (Decision Fusion, seed 42): train loss turun terus, val loss berfluktuasi tajam
  (puncak di epoch 9 dan 20).
  - Epoch terbaik 16, dipilih dengan early stopping pada Macro-F1 validasi.
  - Validasi yang sama juga dipakai untuk pelaporan, sehingga angkanya sedikit optimistis.
- **3 seed:** semua model dilatih dengan seed 42, 123, dan 2026 (tabel 1b). Model final Feature Fusion: 0,659 ± 0,013.
- **Prediksi uji:** 16.000 unit → `prediksi_klasifikasi.csv` (`id_unit,kelas`), lolos `python -m kode.validasi_csv`.

![Kurva training Decision Fusion (seed 42)](../results/task1/learning_curves.png){width=85%}

<!-- ALIEF: (1) soal minta sanity check "overfit 1 batch" (latih pada satu batch sampai loss ≈ 0 / akurasi 100%); yang ada adalah uji label acak, jadi tambahkan overfit 1 batch atau jelaskan. (2) Learning curve, confusion matrix, dan eksperimen 1c–1f memakai Decision Fusion seed 42, sedangkan model final Feature Fusion: simpan learning curve Feature Fusion atau jelaskan alasannya. (3) Catat distribusi kelas prediksi test untuk mengecek pengaruh lini B. -->

# Task 2 – Prediksi Remaining Useful Life (RUL)

Data: 900 mesin latih (run-to-failure, 254.174 siklus) dan 300 mesin uji (berhenti acak; histori
50–455 siklus). Prediksi RUL di siklus terakhir tiap mesin uji, dengan cap 150. Kode: `kode/task2/`;
hasil: `results/task2/`.

## 2a. EDA dan preprocessing

| Temuan | Keputusan |
|---|---|
| 6 rezim operasi diskret dari `setel_1`. Rezim menjelaskan 37–100% varians sensor (η²); loncatan antar-rezim jauh lebih besar daripada tren degradasi | **z-score per rezim**, statistik hanya dari mesin latih tiap fold |
| 7 sensor (s01, s05, s06, s10, s16, s18, s19) murni white noise: std(Δz)/std(z) ≈ √2 | Dibuang (ambang 1,2; tanpa label → tidak bocor). Sisa 14 sensor |
| Dua mode kerusakan (A 558 mesin, B 342) dengan arah degradasi berlawanan (mis. s11). Korelasi global s11–RUL hanya 0,16, padahal per mesin 0,75 | Seleksi sensor **tidak** memakai korelasi dengan RUL. Mode tidak dijadikan fitur, hanya dipakai untuk analisis galat |
| Missing ~2% per sensor, acak; ada juga di baris terakhir data uji | **Forward-fill** per mesin (kausal, tanpa interpolasi) |
| Spike 0,2% titik, \|z\| sampai 10 | Median **10 siklus sebelumnya**; ganti jika > 5σ robust. Lalu clip ±5 |

![Efek rezim operasi sebelum/sesudah normalisasi per kondisi](../results/task2/eda/03_efek_kondisi_mentah_vs_norm.png){width=88%}

## 2b–c. Label RUL dan sliding window tanpa kebocoran

- **Label:** RUL = siklus_maks − siklus, dipotong pada 150. EDA menunjukkan sensor datar selama RUL >
  ~150, sehingga umur sisa 300 dan 200 memang tidak dapat dibedakan.
- **Window:** satu sampel per (mesin, siklus t) dengan input baris t−49..t mesin itu saja. W = 50 =
  histori uji terpendek. Histori < 50 di-pad kiri dengan 0 plus kanal **mask**. Fitur per langkah:
  14 sensor + umur + mask = 16.
- **Split per mesin (5-fold).** Validasi **meniru data uji**: histori tiap mesin validasi dipotong di
  5 titik acak, dan hanya siklus terakhir yang dinilai (4.500 sampel validasi).
- **Bukti tidak bocor:** 6 uji otomatis (`test_leakage`).
  - Uji utama: mengacak semua data sesudah t tidak mengubah window di t setelah preprocessing.
  - Uji lain: isolasi antar-mesin, preprocessor hanya di-fit di data latih, dan tidak ada mesin di dua sisi split.
  - **Uji mutasi:** interpolasi/bfill dan median berpusat sengaja disuntikkan, dan tes berhasil menangkapnya.

![Label RUL dengan cap 150 dan contoh window](../results/task2/windows/09_label_dan_window.png){width=80%}

## 2d. Model: baseline vs tiga arsitektur sequence

- **Baseline non-DL:** HistGradientBoosting pada 58 fitur manual: per sensor level 5 siklus terakhir, rata-rata, kemiringan, dan std window, ditambah umur dan panjang histori.
- **LSTM:** 2 lapis, 64 unit, kepala MLP. **TCN:** 4 blok konvolusi kausal (48 kanal, dilasi 1/2/4/8).
  **Transformer:** 2 blok self-attention (d = 64, 4 head).
- **Training:** MSE pada RUL/150, AdamW + OneCycle, AMP, early stopping (patience 8) pada RMSE validasi,
  checkpoint per epoch (bisa resume di Colab).

| Model | Parameter | RMSE (mean ± std fold) | MAE | Bias | Skor / 300 |
|---|---:|---:|---:|---:|---:|
| **LSTM** | 69.761 | **19,13 ± 0,58** | 14,52 | **+0,51** | **4.673** |
| TCN | 58.097 | 19,51 ± 0,52 | 14,97 | +2,46 | 5.226 |
| Transformer | 103.041 | 19,75 ± 0,51 | 14,45 | +1,55 | 6.170 |
| Baseline (GBM) | – | 19,82 ± 0,80 | 14,44 | +2,24 | 6.654 |

**Pilihan: LSTM** (hanya model DL yang dipertimbangkan).
- RMSE dan skor asimetris terbaik. Skornya 30% lebih rendah dari baseline, karena biasnya paling kecil:
  prediksi terlambat (d > 0) dihukum eksponensial lebih berat.
- Perbandingannya berpasangan: LSTM mengalahkan baseline di **kelima** fold.
- Transformer paling cepat overfit (val naik setelah ~epoch 7). Dengan hanya 900 mesin, kapasitas tambahan tidak membantu.

## 2e. Evaluasi per rentang RUL

![RMSE per rentang RUL dan prediksi vs aktual](../results/task2/models/10_rmse_per_rentang.png){width=49%} ![](../results/task2/models/11_prediksi_vs_aktual.png){width=49%}

| RUL aktual | n | Baseline | LSTM | TCN | Transformer | Bias LSTM |
|---|---:|---:|---:|---:|---:|---:|
| 0–25 | 521 | **9,9** | 11,1 | 13,8 | 11,8 | +7,1 |
| 25–50 | 570 | 15,4 | 14,0 | 14,3 | **13,6** | +3,9 |
| 50–75 | 567 | 25,4 | **22,8** | 24,7 | 23,5 | +8,4 |
| 75–100 | 501 | 33,4 | **30,6** | 31,8 | 32,4 | +14,5 |
| 100–125 | 474 | 24,6 | 23,6 | **23,5** | 26,4 | +9,4 |
| 125–150 | 1.867 | **14,2** | 15,5 | 14,3 | 14,8 | −10,7 |

- **Galat terbesar di RUL 50–100**, dan biasnya positif (terlambat). Di fase ini degradasi baru mulai
  dan sinyal sensor masih lemah. Ketiga arsitektur DL mentok di angka yang mirip, jadi batas ini datang
  dari informasi di data.
- **RUL > 125:** bias negatif, efek cap. Kesalahannya ke arah aman (servis sedikit lebih awal).
- **Per mode kerusakan:** galat hampir sama (A 18,99; B 19,39), jadi model tidak hanya bekerja untuk mode mayoritas.

## 2f. Prediction interval dan kebijakan servis

- **Interval:** MC dropout (50 pass) × 5 model fold. Interval mentah hanya mencakup **54%** pada target
  90%, karena MC dropout hanya mengukur ketidakpastian model, bukan noise data dan variasi antar-mesin.
- **Kalibrasi:** split conformal ternormalisasi, interval = pred + q·std, dengan q dari kuantil residual
  validasi. Cakupan dihitung cross-fit (q tiap fold diestimasi dari 4 fold lain), sehingga jujur.
- **Cakupan terkalibrasi:** 0,499 / 0,797 / **0,899** / 0,948 untuk target 50 / 80 / 90 / 95%.
- **Keterbatasan:** kalibrasinya global, sehingga cakupan 90% kurang di RUL 25–100 (0,81–0,84) dan lebih di RUL > 100.

![Kalibrasi interval dan kebijakan servis (H = 20)](../results/task2/uncertainty/13_kalibrasi_interval.png){width=45%} ![](../results/task2/uncertainty/15_kebijakan_servis.png){width=53%}

**Kapan mesin perlu diservis.** Asumsinya servis harus disiapkan H = 20 siklus sebelum rusak.

| Aturan "servis jika X ≤ 20" | Kerusakan tertangkap | Terlewat | Servis terlalu dini |
|---|---:|---:|---:|
| X = prediksi titik | 66,4% | 150 | 51 |
| X = batas bawah 80% | 93,3% | 30 | 260 |
| **X = batas bawah 90%** | **96,6%** | **15** | 331 |
| X = batas bawah 95% | 98,7% | 6 | 410 |

- **Rekomendasi: servis jika batas bawah interval 90% ≤ 20 siklus.** Kerusakan terlewat turun 90%
  dibanding prediksi titik. Biayanya lebih banyak servis dini, tetapi rata-rata mesin yang diservis
  masih punya ~20 siklus.
- **Pilihan level mengikuti biaya:** 95% jika kerusakan jauh lebih mahal daripada servis, 80% jika servis mahal.
- **Hasil pada 300 mesin uji:** **64 servis segera**, 75 dijadwalkan (batas bawah 20–50), 161 aman.
  Tanpa interval hanya 20 mesin yang akan ditandai.

# Task 3 – Clustering Multimodal Jenis Material

36.000 sampel tanpa label: citra mikrostruktur 64×64 dan spektrum komposisi 128 kanal, dari dua
mikroskop (M1 21.636, M2 14.364). Jumlah jenis material tidak diketahui. Kode: `kode/task3/`; hasil:
`results/task3/`.

## 3a. EDA distorsi

| Distorsi | Bukti | Keputusan |
|---|---|---|
| **Rotasi acak** | Orientasi dominan (structure tensor) menyebar di 0–180°. Ketidakrataan kecil juga muncul pada garis sintetis bersudut seragam, jadi itu bias estimator | Augmentasi rotasi bebas + flip |
| **Skala** | Tekstur garis membentuk 2 kelompok skala terpisah (λ ≈ 5 dan 11–12 px) | Augmentasi skala **ringan** (±15%) agar garis halus dan kasar tidak disamakan |
| **Brightness / mikroskop** | M2 lebih terang (146 vs 128). Setelah standardisasi, mikroskop **masih** tertebak dari PSD (AUC 0,88): M2 lebih berderau | Standardisasi per gambar + augmentasi noise/blur |
| **Spike spektrum** | 0,2% titik, selalu positif, 1 kanal. Residual bimodal (≤ 0,37 vs ≥ 0,80) | Despike dengan ambang 0,5 terhadap median 5 kanal |
| **Baseline** | Offset/kemiringan/lengkung lambat per spektrum | Kurangi baseline polinom-2 (modpoly), lalu normalisasi L2 |
| **Shift** | Ketiga puncak bergeser bersama (r = 0,90); 95% shift dalam ±7 kanal | Augmentasi shift ±7 dan stretch ±3%; CNN 1D dengan pooling |
| **Mikroskop pada spektrum** | AUC menebak mikroskop dari spektrum = 0,50 | Spektrum bebas efek alat, sehingga menjadi jangkar untuk menguji koreksi mikroskop (3c) |

![Beda mikroskop: PSD radial, kehalusan/noise, dan seberapa mudah mikroskop ditebak](../results/task3/eda/05_beda_mikroskop.png){width=92%}

![Struktur puncak spektrum: posisi puncak, pergeseran bersama, intensitas](../results/task3/eda/09_puncak_shift_intensitas.png){width=92%}

## 3b. Representation learning tanpa label: SimCLR

- **Metode:** SimCLR per modalitas dengan loss NT-Xent (τ = 0,2). Dua view augmentasi dari sampel yang
  sama ditarik mendekat; sampel lain dalam batch (512) didorong menjauh.
- **Alasan dipilih:** autoencoder piksel harus merekonstruksi semua detail, termasuk orientasi dan noise
  mikroskop. SimCLR diajari langsung untuk **mengabaikan** distorsi yang ditemukan di EDA.
- **Augmentasi gambar:** crop 36×36 yang dirotasi bebas dan selalu berada di dalam gambar, flip, skala
  ±15%, gamma, blur, noise, standardisasi.
  - Versi awal memakai padding refleksi. Padding itu ternyata menciptakan tekstur "chevron" palsu, jadi diganti crop di dalam gambar.
- **Augmentasi spektrum:** shift ±7, stretch ±3%, sisa baseline, noise, spike sintetis.
- **Encoder:** ResNet kecil untuk gambar (1,23 jt parameter, embedding 256, TTA 8 orientasi saat
  inferensi) dan CNN 1D untuk spektrum (0,24 jt, embedding 128).
- **Training:** 150/100 epoch di Kaggle GPU (14,2 + 1,7 menit), AMP, checkpoint dan resume.

| Diagnostik (tanpa label) | Gambar: PSD radial | Gambar: SimCLR | Spektrum: PCA | Spektrum: SimCLR |
|---|---:|---:|---:|---:|
| Retrieval view-ke-view top-1 (↑) | 0,016 | **0,367** | 0,018 | **0,796** |
| Rank efektif | 14,4 | 70,3 | 17,6 | 37,6 |
| AUC tebak mikroskop (ideal 0,5) | 0,88 | 0,99 | 0,50 | 0,50 |

- SimCLR jauh lebih tahan distorsi daripada fitur klasik: retrieval 20–40× lebih tinggi.
- Kesepakatan cluster gambar vs spektrum (AMI) tertinggi di k = 6 (0,80 vs 0,53 untuk fitur klasik).
- **Masalah yang tersisa:** embedding gambar tetap membawa jejak mikroskop (AUC 0,99), dan ini ditangani di 3c.

## 3c. Fusion dan apa yang hilang dengan satu modalitas

1. **Blok per modalitas:** L2 → PCA 95% varians → total varians 1 (gambar 16 dimensi, spektrum 14),
   sehingga kedua modalitas berbobot sama.
2. **Koreksi mikroskop bersyarat:** blok gambar dikurangi rata-ratanya per mikroskop. Ini hanya sah jika
   komposisi material tidak bergantung pada alat, dan syarat itu diuji pada spektrum yang bebas alat:
   Cramér's V (cluster spektrum, mikroskop) maks. **0,001** < 0,10. Hasilnya, semua cluster akhir punya
   V terhadap mikroskop 0,00 dan proporsi M2 39–41% di setiap cluster.
3. **FusionAE (deep fusion):** autoencoder MLP 30 → 256 → 128 → **16** dengan **modality dropout**
   (satu blok dinolkan dengan p = 0,25 dan harus direkonstruksi dari modalitas lain).
   - Latent dipaksa memuat informasi bersama **dan** unik per modalitas.
   - Fusion kontrastif ala CLIP tidak dipakai, karena hanya mempertahankan informasi bersama.
   - R² rekonstruksi silang: 0,61 (gambar dari spektrum) dan 0,67 (spektrum dari gambar).

**Ablasi:** KMeans (k = 6) pada tiap representasi.

| Representasi | ARI vs final | Akurasi kNN label final | Pasangan cluster tak terpisah (akurasi a-vs-b < 0,75) |
|---|---:|---:|---|
| Gambar saja | 0,999 | 1,000 | – |
| **Spektrum saja** | **0,687** | 0,772 | **c2–c5, c3–c4** (akurasi 0,48–0,52 = menebak) |
| Concat / FusionAE | 1,000 | 1,000 | – |

- **Yang hilang tanpa citra:** dua pasangan material dengan spektrum **identik** yang hanya berbeda
  tekstur (bintik terang vs pulau gelap; garis lurus vs garis bergelombang).
- **Peran spektrum:** pada k = 6 citra saja sudah cukup. Spektrum tetap berguna sebagai konfirmasi
  independen yang bebas efek mikroskop, dan sebagai satu-satunya sumber faktor jarak puncak di c0 (3f).

## 3d–e. Metode clustering dan jumlah cluster

**Pemilihan k (aturan ditetapkan sebelum melihat hasil):**
1. Hitung silhouette, Davies-Bouldin, Calinski-Harabasz, stabilitas bootstrap (ARI antar-subsampel 80%) dan BIC GMM untuk k = 2–15 pada latent FusionAE.
2. Kandidat = k dengan stabilitas ≥ 0,80.
3. Pilih jumlah peringkat (silhouette + DB + CH) terkecil.

Ketiga kriteria internal sepakat di **k = 6** (silhouette 0,472, DB 0,871, CH 20.859, stabilitas 0,963).
BIC terus turun tanpa siku, jadi hanya dipakai sebagai informasi. KMeans pada citra saja juga memilih k = 6.

![Pemilihan k](../results/task3/clustering/14_pemilihan_k.png){width=100%}

| Metode (k = 6, 3 seed) | Jenis | Silhouette (ruang sendiri) | Stabilitas antar-seed | Cluster terkecil |
|---|---|---:|---:|---:|
| KMeans (concat) | klasik | 0,511 | 1,000 | 10,2% |
| KMeans (FusionAE) | klasik | 0,472 | 1,000 | 10,2% |
| GMM kovarians penuh (FusionAE) | klasik | 0,378 | 0,838 | 10,2% |
| **IDEC (FusionAE)** | **deep** | **0,702** | 1,000 | 10,2% |

- **IDEC** (DEC + rekonstruksi): encoder FusionAE dan pusat cluster dilatih bersama dengan loss
  KL(P‖Q) + 0,1·rekonstruksi, dengan soft assignment Student-t. Rekonstruksi menjaga latent agar tidak collapse.
- **Hasil apa adanya:** IDEC konvergen setelah 1 epoch tanpa label yang berubah, dan partisinya
  **identik** dengan KMeans (ARI 1,00). Keenam cluster sudah terpisah jelas di latent FusionAE. IDEC
  hanya merapatkannya (silhouette 0,47 → 0,70).
- **GMM** paling tidak stabil (ARI 0,78 terhadap metode lain), karena kovarians penuh membuatnya lebih bergantung pada inisialisasi.
- **Model final:** IDEC (aturan "model utama deep learning"). Pengaman collapse terpenuhi (cluster terkecil 10,2% ≥ 0,5%).

## 3f. Visualisasi dan cluster yang sulit dipisahkan

![UMAP (semua sampel) dan t-SNE (8.000 sampel) pada latent IDEC](../results/task3/clustering/17_umap_tsne.png){width=100%}

UMAP dan t-SNE hanya dipakai untuk visualisasi; clustering dilakukan di latent 16 dimensi. Warna
mikroskop tercampur di setiap cluster. Cluster "spektrum saja" memotong cluster final secara acak di
c2/c5 dan c3/c4.

![Profil cluster: contoh citra dan spektrum rata-rata](../results/task3/clustering/18_profil_cluster.png){width=72%}

| Cluster | n | Tekstur | Puncak spektrum | Kelompok spektrum |
|---|---:|---|---|---|
| c0 | 10.043 | butiran poligonal + berawan | 20 / 40–44 / 85 | B |
| c1 | 8.685 | garis halus (λ ≈ 5 px) | 15 / 55 / 75 | A |
| c2 | 4.979 | bintik/sel terang | 30 / 62 / 100 | C |
| c3 | 4.340 | garis lurus kasar | 35 / 70 / 110 | D |
| c4 | 4.297 | garis kasar bergelombang | 35 / 70 / 110 | D |
| c5 | 3.656 | pulau gelap di latar terang | 30 / 62 / 100 | C |

**Pasangan sulit.**
- Di ruang gabungan, tumpang tindih kNN antar-cluster 0% untuk semua pasangan.
- Pasangan tersulit adalah yang **hanya dapat dipisahkan oleh satu modalitas**: **c3–c4** dan **c2–c5**.
  Spektrumnya identik (akurasi 0,48/0,52), dan hanya tekstur yang membedakannya (akurasi 1,00).
- Sistem dengan satu sensor spektrum akan menggabungkan keduanya.

**Cluster 0 heterogen.** Pemeriksaan di dalam cluster (`kode/task3/heterogen.py`) menemukan dua faktor
yang saling **independen** (Cramér's V antar-faktor 0,00):
1. **Tekstur:** butiran poligonal (5.203) vs berawan (4.840). Akurasi tebak dari blok gambar 0,998, dari spektrum 0,51.
2. **Jarak puncak 1–2:** 20 vs 24 kanal (puncak 2 di ~40 vs ~44). Akurasi tebak dari spektrum 0,98, dari gambar 0,59.
   - Distribusinya bimodal **hanya di c0**: ΔBIC 4.488, jarak antar-mode 4,3 std.
   - Lima cluster lain unimodal (ΔBIC < 0, jarak 1,1–1,5 std).

![Heterogenitas cluster 0](../results/task3/clustering/20_heterogenitas_cluster.png){width=100%}

**Mengapa c0 tidak dipecah:**
- Pemecahan tekstur (k = 7) memenuhi aturan refinement kami: pemecahan murni, dengan V mikroskop 0,098 < 0,10.
- Tetapi itu bertentangan dengan aturan k yang ditetapkan sebelumnya, dan V-nya tepat di batas.
- Faktor spektrum yang independen juga membuka kemungkinan k = 9 (2 × 2).
- Tanpa label tidak dapat dipastikan mana yang menandai jenis material.

Kami melaporkan k = 6 dan menyimpan k = 7 sebagai pembanding (`results/task3/cluster_alternatif_k7.csv`).

**Koreksi interpretasi.** Dua "pulau" c0 di UMAP awalnya kami kira dua tekstur. Pemeriksaan per
modalitas menunjukkan pulau ditentukan oleh **jarak puncak spektrum** (akurasi tebak dari spektrum 1,00,
dari gambar 0,58). Struktur di proyeksi 2D perlu diverifikasi di ruang aslinya sebelum dijadikan bukti.

## 3g. Keterbatasan Task 3

- Jejak mikroskop dihapus di tahap fusion (centering), bukan di encoder. Encoder yang invarian alat
  (mis. adversarial) akan lebih bersih.
- Deep clustering tidak mengubah partisi KMeans pada data yang sudah terpisah jelas.
- Validasi tanpa label bergantung pada ukuran internal, konsistensi antar-modalitas, dan inspeksi visual.
  Jumlah material di c0 (1, 2, atau 4) tidak dapat dipastikan.

# Reproduksibilitas dan verifikasi

- `python run_all.py` menjalankan Task 1 (notebook via nbconvert), Task 2, dan Task 3, lalu memvalidasi ketiga CSV.
  - `--debug` menjalankan subset kecil di CPU.
  - Seed 42; versi library dicatat ke `env_info.json`.
  - Task 3c–f dapat dijalankan ulang di CPU dari bobot SimCLR di `models/` (~19 menit). Uji pada folder
    kosong menghasilkan `cluster.csv` yang identik dengan yang dikumpulkan (ARI 1,00).
- **Tes:** `kode/task2/test_leakage.py` (anti-kebocoran + uji mutasi), `test_models.py`,
  `kode/task3/test_ssl.py`, dan `test_cluster.py`. Tes Task 3 memakai data sintetis yang jawabannya diketahui.
- **Lingkungan:** Task 2 di Colab (Tesla T4; Python 3.13, torch 2.14), Task 3b di Kaggle GPU, Task 3c–f
  lokal (Python 3.14.7, torch 2.15 CPU, scikit-learn 1.9.1).
- **Penggunaan AI:** lihat `kontribusi.md`.
