# Persiapan Ujian Lisan

Format tiap butir: **pertanyaan** → poin jawaban singkat. Angka diambil dari `results/` dan laporan.
Bagian Task 1 disusun dari notebook Alief; **Alief perlu memeriksa dan melengkapinya**.

---

## 0. Reproduksi angka secara live

Semua perintah dijalankan dari root repo dengan virtualenv aktif. (A)–(D) cepat; (E) butuh ~19 menit di CPU,
jadi untuk ujian cukup tunjukkan (A)–(D) dan sebutkan hasil (E).

**(A) Format ketiga CSV prediksi** (± 5 detik)

```bash
python -m kode.validasi_csv
```

**(B) Task 2: RMSE LSTM 19,13 ± 0,58 dan ensemble = prediksi_rul.csv** (< 1 detik)

```bash
python -c "import json,glob,numpy as np; r=[json.load(open(f))['rmse'] for f in sorted(glob.glob('results/task2/models/lstm/fold*.json'))]; print(np.round(r,2), np.mean(r).round(3), np.std(r).round(3))"
# -> [20.08 19.29 19.02 18.29 18.98] 19.133 0.576

python -c "import pandas as pd; f=pd.read_csv('results/task2/models/test_prediksi_per_fold_lstm.csv'); p=pd.read_csv('prediksi_rul.csv'); print((f.filter(like='pred').mean(axis=1).clip(0,150).round(2)-p.rul).abs().max())"
# -> 0.0   (prediksi akhir = rata-rata 5 model fold)

python -c "import pandas as pd; print(pd.read_csv('results/task2/uncertainty/prediksi_rul_interval_test.csv').status.value_counts())"
# -> AMAN 161, JADWALKAN 75, SERVIS SEGERA 64
```

- RMSE 19,13 adalah rata-rata RMSE 5 fold.
- RMSE 19,14 di README adalah RMSE gabungan semua sampel validasi (`tabel_perbandingan_model.csv`).

**(C) Task 2: uji anti-kebocoran** (± 10 detik)

```bash
python -m kode.task2.test_leakage    # semua [OK], termasuk uji ulang pada 60 mesin data asli
```

**(D) Task 3: ukuran cluster dan independensi terhadap mikroskop** (± 2 detik)

```bash
python -c "import pandas as pd; from kode.task3.fusion import cramers_v; c=pd.read_csv('cluster.csv').merge(pd.read_csv('data/clustering_meta.csv'),on='id_sampel'); t=pd.crosstab(c.cluster,c.mikroskop); print(t.assign(pct_M2=(100*t.M2/t.sum(axis=1)).round(1))); print('V =', cramers_v(c.cluster,c.mikroskop))"
# -> ukuran 10043/8685/4979/4340/4297/3656, M2 39,1–40,7% di setiap cluster, V = 0.0
```

- Versi tanpa koreksi bias (`scipy.stats.contingency.association`) memberi 0,0096. Kode memakai koreksi bias Bergsma.

**(E) Task 3c–f penuh dari bobot SimCLR** (~19 menit, CPU)

```bash
python -m kode.task3.main --no-eda --no-ssl --out_dir /tmp/rep --pred_dir /tmp/rep --results_dir /tmp/rep/results
```

- Jika `emb_*.npy` belum ada, embedding dihitung ulang dari `models/task3_simclr_*.pt`. Lalu fusion → k → IDEC → `cluster.csv`.
- Log mencetak `k final = 6`.
- Sudah diuji (20 Nov 2026, folder kosong, hanya bobot `models/`): embedding identik dengan hasil Kaggle
  (cosine 1,000), `cluster.csv` hasil ulang **identik** dengan yang di-commit (ARI 1,00), ukuran cluster sama.

**Tes lain:** `python -m kode.task3.test_cluster` (data sintetis: pipeline harus menemukan k = 6 dan IDEC
memulihkan material dengan ARI > 0,9) dan `python -m kode.task3.test_ssl`.

---

## 1. Pertanyaan umum

**Mengapa deep learning sebagai model utama, dan apa peran baseline?**
- Aturan soal. Baseline klasik dipakai untuk membuktikan model DL memang memberi nilai tambah:
  - Task 2: GBM pada 58 fitur manual.
  - Task 3: KMeans/GMM serta fitur PSD radial/PCA.
- Hasilnya di Task 2: LSTM menang di kelima fold, dengan skor 30% lebih rendah.

**Bagaimana menjamin reproduksibilitas?**
- `run_all.py` sebagai satu perintah, seed 42 (`set_seed`: random, numpy, torch, cudnn deterministic).
- `env_info.json` mencatat versi library dan GPU.
- Bobot model dan semua tabel hasil di-commit. Task 3c–f deterministik di CPU.

**Mengapa ada yang dilatih di Colab/Kaggle?**
- Training penuh LSTM/TCN/Transformer × 5 fold dan SimCLR 150 epoch butuh GPU.
- Kodenya sama; notebook hanya meng-clone repo dan memanggil `run_all.py`/`kode.task*.main`.
- Checkpoint per epoch, jadi bisa di-resume jika sesi terputus.

**Bagaimana AI dipakai dan diverifikasi?** (lihat `kontribusi.md`)
- Claude Code membantu menulis kode, analisis, dan draf teks.
- Verifikasi:
  - tes yang bisa gagal (uji mutasi);
  - `validasi_csv`;
  - cek ulang angka dari CSV/JSON;
  - run ulang Task 3 untuk memastikan hasilnya deterministik.
- Koreksi yang ditemukan dari verifikasi:
  - padding refleksi menciptakan tekstur palsu;
  - pasangan sulit dihitung di ruang IDEC;
  - salah tafsir pulau UMAP.

---

## 2. Task 1 – Klasifikasi multimodal (Alief)

**Mengapa Macro-F1, bukan akurasi?**
- Kelas tidak seimbang: normal 49,8%, kontaminasi 4,9%.
- Model yang menebak "normal" terus sudah mendapat akurasi ~50%. Macro-F1 memberi bobot sama ke tiap kelas.

**Bagaimana menangani ketidakseimbangan kelas?**
- Cross-entropy berbobot kelas (*balanced*) dan split stratified.

**Bagaimana preprocessing sensor menghindari kebocoran?**
- Rata-rata imputasi per titik waktu dan statistik z-score dihitung dari data latih saja.
- Statistik yang sama diterapkan ke validasi dan uji.

**Feature fusion vs decision fusion: apa bedanya dan mana yang dipilih?**
- Feature: konkatenasi embedding (256) → MLP, sehingga model bisa belajar interaksi antar-modalitas.
- Decision: logit tiap modalitas dirata-rata, sehingga tiap modalitas memutuskan sendiri.
- Feature Fusion terbaik dan paling stabil: 0,659 ± 0,013 vs 0,641 ± 0,024 (3 seed).

**Mengapa sensor-only jauh lebih baik dari image-only?**
- 0,434 vs 0,312. Cacat seperti *las_dingin* jelas di sinyal arus/suhu, sedangkan cacat permukaan halus
  (gores) sulit dilihat di citra 64×64.
- Gabungan keduanya +0,23 dari sensor-only, jadi informasinya saling melengkapi.

**Kelas mana yang paling sulit dan mengapa?**
- *gores*: F1 0,29; 641/1.124 diprediksi normal.
- *las_dingin* paling mudah: F1 0,94.

**Apa arti hasil domain shift?**
- Latih hanya lini A, uji lini B: Macro-F1 0,197 (vs 0,587).
- Data uji 60% lini B, jadi data latih wajib memuat kedua lini.
- Opsi lanjutan: normalisasi per lini, augmentasi, domain adaptation.

**Mengapa performa jatuh saat satu modalitas hilang?**
- 0,351 tanpa citra; 0,301 tanpa sensor.
- Model tidak pernah melihat input yang hilang saat training.
- Solusi: modality dropout saat training (dipakai di FusionAE Task 3).

**Apa gunanya sanity check label acak?**
- Label diacak → Macro-F1 0,137 ≈ tebakan.
- Membuktikan tidak ada kebocoran label lewat pipeline.

**Kelemahan yang mungkin ditanyakan (siapkan jawaban; tutup sebelum ujian jika sempat):**
- Confusion matrix, learning curve, dan eksperimen 1c–1f memakai Decision Fusion seed 42, bukan model final.
- Epoch terbaik dipilih pada validasi yang sama dengan pelaporan.
- Butir soal yang belum lengkap di notebook:
  - *1a:* gambar contoh tiap kelas belum disimpan ke `results/task1`.
  - *1c:* F1 per kelas untuk image-only dan sensor-only.
  - *1d:* cek mislabel dan teknik robust (label smoothing / co-teaching).
  - *1e:* Macro-F1 lini A dan teknik mengatasi shift (normalisasi per lini / augmentasi).
  - *1f:* modality dropout supaya prediksi tanpa sensor tetap masuk akal.
  - *1g:* sanity check overfit 1 batch.

**Jika ditanya label smoothing:** target one-hot diganti (1 − ε) untuk kelas benar dan ε/(K − 1) untuk
kelas lain. Model tidak dipaksa 100% yakin, sehingga label yang salah tidak menarik bobot terlalu jauh.

**Jika ditanya overfit 1 batch:** model harus bisa menghafal satu batch kecil (loss → 0). Jika gagal,
ada bug di arsitektur, loss, atau pemasangan label. Uji label acak (yang sudah dilakukan) menguji hal
sebaliknya: tanpa hubungan input–label, skor harus jatuh ke tingkat tebakan.

---

## 3. Task 2 – RUL

### Preprocessing dan label

**Mengapa normalisasi per kondisi operasi?**
- 6 rezim `setel_1` menjelaskan 37–100% varians sensor.
- Tanpa normalisasi per rezim, model mempelajari loncatan rezim, bukan degradasi.
- Statistik dihitung hanya dari mesin latih tiap fold.

**Bagaimana memilih sensor tanpa membocorkan label?**
- Kriteria white noise: std(Δz)/std(z) ≈ √2 untuk noise murni, ambang 1,2. Tidak memakai RUL.
- Korelasi global dengan RUL justru menyesatkan: s11 hanya 0,16 secara global karena dua mode kerusakan
  berlawanan arah, padahal 0,75 per mesin.

**Mengapa forward-fill dan median kausal?**
- Saat prediksi, masa depan tidak diketahui.
- Interpolasi/bfill dan median berpusat memakai titik sesudah t, sehingga bocor.
- Uji mutasi menyuntikkan keduanya dan tes berhasil menangkapnya.

**Mengapa cap 150?**
- Sensor datar selama RUL > ~150, sehingga umur sisa 300 dan 200 tidak dapat dibedakan.
- Tanpa cap, model dihukum untuk sesuatu yang tidak teramati. Prediksi juga dinilai dengan cap 150.

**Mengapa W = 50 dan bagaimana histori pendek ditangani?**
- 50 = histori uji terpendek, sehingga semua mesin uji punya window penuh.
- Pad kiri nol + kanal mask; fitur umur ditambahkan.

### Validasi

**Bagaimana split mencegah kebocoran?**
- 5-fold **per mesin**: tidak ada mesin di latih dan validasi sekaligus. Window mesin yang sama sangat
  berkorelasi, jadi split per window akan bocor.
- Validasi meniru uji: histori dipotong di 5 titik acak, dan hanya siklus terakhir yang dinilai.

**Apa skor asimetris?**
- d = pred − aktual. Telat (d ≥ 0): e^(d/10) − 1. Dini (d < 0): e^(−d/13) − 1.
- Telat lebih mahal karena mesin rusak sebelum diservis.

### Model

**Mengapa LSTM yang dipilih?**
- RMSE 19,13 dan skor 4.673, keduanya terbaik. Biasnya terkecil (+0,51), yang penting untuk skor asimetris.
- Transformer overfit paling cepat: 103 ribu parameter untuk 900 mesin.

**Arsitektur LSTM?**
- Linear 16 → 64, LSTM 2 lapis 64 unit (dropout 0,2), kepala MLP; 69.761 parameter.
- MSE pada RUL/150, AdamW + OneCycle, early stopping pada RMSE validasi.

**Di mana model paling salah?**
- RUL 50–100 (RMSE 22,8–30,6), dengan prediksi terlambat. Degradasi baru mulai.
- Semua model mentok serupa, jadi batasnya informasi di data.

### Ketidakpastian dan servis

**Mengapa MC dropout tidak cukup?**
- Cakupan nominal 90% hanya 54%.
- MC dropout mengukur ketidakpastian model (epistemik), bukan noise data dan variasi antar-mesin (aleatorik).

**Bagaimana kalibrasinya?**
- Split conformal ternormalisasi: z = (rul − pred)/std pada validasi out-of-fold; interval = pred ± q·std.
- q dihitung cross-fit dari 4 fold lain → cakupan 0,899.

**Kapan mesin diservis?**
- Aturan: batas bawah 90% ≤ 20 siklus. Menangkap 96,6% kerusakan (vs 66,4% dengan prediksi titik),
  dengan biaya lebih banyak servis dini.
- Di data uji: 64 servis segera, 75 dijadwalkan, 161 aman.

---

## 4. Task 3 – Clustering multimodal

### Representasi

**Distorsi apa yang ditemukan dan bagaimana ditangani?**
- **Gambar:** rotasi acak → augmentasi rotasi + flip. Skala → augmentasi ringan ±15% (λ 5 vs 11 px
  membedakan material). Brightness/noise mikroskop → standardisasi dan noise.
- **Spektrum:** spike → despike. Baseline → modpoly orde 2. Shift ±7 → augmentasi shift + CNN 1D.

**Mengapa SimCLR, bukan autoencoder?**
- Autoencoder harus merekonstruksi orientasi dan noise.
- SimCLR mengajari invariansi yang kita pilih lewat augmentasi.
- Retrieval view-ke-view: 0,37/0,80 vs 0,02 untuk fitur klasik.

**Apa NT-Xent dan τ?**
- Softmax similaritas cosinus: pasangan positif vs 2N − 2 negatif dalam batch.
- τ = 0,2 mempertajam distribusi, sehingga negatif sulit lebih berpengaruh.

**Mengapa embedding gambar masih "tahu" mikroskop (AUC 0,99)?**
- Beda noise/PSD antar-mikroskop tidak sepenuhnya dihapus augmentasi.
- Ditangani di fusion dengan centering per mikroskop.

### Fusion

**Bukankah centering per mikroskop curang?**
- Sah hanya jika komposisi material sama di kedua alat.
- Syarat itu diuji pada spektrum, yang bebas efek alat (AUC 0,50): V cluster spektrum vs mikroskop maks. 0,001.
- Hasil akhir: M2 39–41% di setiap cluster.

**Mengapa FusionAE dengan modality dropout?**
- Latent harus menyimpan informasi bersama **dan** unik.
- Fusion kontrastif (CLIP) hanya menyimpan yang bersama, sehingga akan membuang pembeda c2/c5 dan c3/c4
  yang hanya ada di citra.

**Apa yang hilang dengan satu modalitas?**
- Spektrum saja: ARI 0,687; c2–c5 dan c3–c4 tidak terpisah (akurasi ~0,5).
- Citra saja: ARI 0,999 pada k = 6, tetapi tidak bisa melihat faktor jarak puncak di c0.

### Clustering dan k

**Bagaimana k dipilih?**
- Aturan ditetapkan **sebelum** melihat hasil:
  1. stabilitas bootstrap ≥ 0,80;
  2. peringkat silhouette + DB + CH terkecil.
- Hasilnya k = 6, dan ketiga kriteria sepakat. BIC turun terus, jadi hanya informatif.

**IDEC identik dengan KMeans — apa gunanya deep clustering?**
- Hasil apa adanya: cluster sudah terpisah jelas di latent FusionAE. IDEC hanya merapatkannya
  (silhouette 0,47 → 0,70), tidak mengubah partisi.
- Itu konfirmasi, bukan kegagalan. IDEC tetap model final sesuai aturan DL.

**Mengapa GMM paling tidak stabil?**
- Kovarians penuh di 16 dimensi membuatnya sensitif terhadap inisialisasi: ARI antar-seed 0,838.

**Cluster mana yang sulit?**
- c3–c4 dan c2–c5: spektrum identik, hanya tekstur yang beda.
- c0 heterogen: tekstur (butiran/berawan) × jarak puncak (20/24 kanal), dua faktor independen (V 0,00).

**Mengapa tidak k = 7 atau 9?**
- k = 7 lolos aturan refinement, tetapi V mikroskop 0,098, tepat di batas.
- Mengubah k setelah melihat hasil melanggar aturan yang kami tetapkan.
- Tanpa label tidak bisa dipastikan apakah c0 itu 1, 2, atau 4 material.
- k = 7 disimpan sebagai pembanding (`cluster_alternatif_k7.csv`).

**Apa yang dikoreksi?**
- Pulau c0 di UMAP awalnya ditafsirkan sebagai tekstur. Uji per modalitas menunjukkan itu jarak puncak
  spektrum (akurasi tebak dari spektrum 1,00, dari gambar 0,58).
- Pelajaran: struktur di proyeksi 2D harus diverifikasi di ruang asli.

**Perkiraan ARI pada label tersembunyi?**
- Tidak bisa diukur. Jika c0 sebenarnya 2 atau 4 material, ARI turun sebanding ukuran c0 (28% data).
- Lima cluster lainnya didukung dua modalitas yang independen.

**UMAP vs t-SNE?**
- Hanya visualisasi. UMAP lebih menjaga struktur global; t-SNE (8.000 sampel) lebih menjaga struktur lokal.
- Clustering dilakukan di latent 16 dimensi, bukan di proyeksi 2D.
