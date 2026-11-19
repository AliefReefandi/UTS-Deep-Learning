# Task 3a – Ringkasan EDA distorsi gambar & spektrum

Data: 36.000 sampel tanpa label; citra mikrostruktur 64×64 (uint8) + spektrum komposisi 128 kanal.
Mikroskop M1 = 21.636, M2 = 14.364. Tidak ada NaN. Semua angka dari data penuh;
reproduksi: `python -m kode.task3.eda`. Kode preprocessing spektrum: `kode/task3/data.py`.
Semua ukuran di bawah dihitung **tanpa label**.

## Gambar

| # | Temuan | Bukti | Keputusan untuk representation learning (3b) | Grafik |
|---|---|---|---|---|
| 1 | **Beberapa jenis tekstur**: garis/lamela, sel/butir, pulau gelap (presipitat), tekstur halus berderau | Contoh acak | Encoder harus menangkap tekstur, bukan posisi objek → CNN dengan pooling global | 01 |
| 2 | **Brightness berbeda antar-mikroskop**: rata-rata M1 127,8 vs M2 146,1 (beda ±1,5 std); kontras hampir sama (19,7 vs 19,9) | Histogram; AUC menebak mikroskop dari brightness+kontras = **0,87** | **Standardisasi per gambar** (kurangi mean, bagi std) + augmentasi brightness/kontras acak | 02 |
| 3 | **Rotasi acak**: 43% gambar bertekstur berarah (koherensi > 0,5), dan arahnya menyebar di seluruh 0–180°. Histogram sedikit lebih tinggi di sekitar 45°/135° (rasio bin maks/min 1,40), tetapi pola yang **sama** (rasio 1,45) muncul pada garis sintetis dengan sudut yang benar-benar seragam → itu bias estimator gradien pada grid piksel, bukan sifat data. Koherensi bimodal (berarah vs isotropik) | Structure tensor per gambar; validasi sintetis: galat orientasi median 0,8°, λ terukur 4,9 / 10,5 untuk λ asli 5 / 11 (`ringkasan_eda.json` → `validasi_estimator_sintetis`) | Augmentasi **rotasi bebas 0–360° + flip** → representasi tidak bergantung arah | 03 |
| 4 | **Skala**: tekstur garis membentuk **2 kelompok skala terpisah** (λ ≈ 5 px dan λ ≈ 11–12 px, ada celah di 7–9 px) dengan sebaran kontinu di dalam tiap kelompok | Sentroid PSD radial | Augmentasi skala **ringan** (±15%, crop-resize): cukup untuk variasi di dalam kelompok, tapi tidak menyatukan garis halus dan kasar yang kemungkinan **jenis material berbeda** | 04 |
| 5 | **Beda mikroskop bukan hanya brightness**: setelah standardisasi, mikroskop masih bisa ditebak dari PSD radial (AUC **0,88**). M2 punya energi frekuensi tinggi jauh lebih besar (lebih **berderau**); M1 lebih halus | PSD radial rata-rata, rasio energi r ≥ 16 (AUC 0,72 dari satu angka ini saja) | Augmentasi **noise Gaussian + blur** acak, sehingga M1 dan M2 menjadi "view" yang mirip. Cek di 3b: embedding tidak boleh mengelompok menurut mikroskop | 05 |
| 6 | Skala M2 tampak sedikit lebih kecil, tetapi rasionya **tidak konsisten** antar-tekstur (M2/M1: 0,97 garis halus, 0,90 garis kasar, 0,83 isotropik) | Median λ per kelompok | **Belum bisa disimpulkan** sebagai beda perbesaran. Pola ini cocok dengan bias noise M2 pada sentroid frekuensi. Augmentasi skala ±15% sudah menutup kemungkinan itu | 04, `ringkasan_eda.json` |

## Spektrum

| # | Temuan | Bukti | Keputusan preprocessing / augmentasi | Grafik |
|---|---|---|---|---|
| 7 | **Spike**: 0,20% titik; 22,5% spektrum punya ≥ 1 spike (maks 4); hampir semua selebar **1 kanal**; **selalu positif** (residual minimum −0,53 hanya noise) | Residual maks thd median 5 kanal **bimodal**: ≤ 0,37 (noise) vs ≥ 0,80 (spike), celah kosong di antaranya | **Despike**: residual positif > **0,5** (tengah celah) → ganti median 5 kanal. Ambang absolut, karena σ robust per spektrum (MAD) hampir nol dan menandai noise biasa / puncak sempit. Padding `mirror` agar spike di kanal tepi juga tertangkap (144 kasus) | 06, 07 |
| 8 | **Baseline lambat** per spektrum: offset 5–95%: −0,14 … 0,64; kemiringan −0,35 … 0,34; lengkung −0,17 … 0,11 (bandingkan tinggi puncak median 1,06) | Fit polinom-2 modpoly | **Kurangi baseline polinom-2** (modpoly 30 iterasi: titik di atas kurva dipotong, sehingga puncak tidak menarik baseline). Sisa: lantai noise sedikit di atas 0 | 06, 08 |
| 9 | **Noise** cukup seragam: std residual 0,029–0,042 (5–95%) | Histogram | Tidak perlu penanganan khusus; augmentasi noise kecil di 3b | 08 |
| 10 | **Struktur puncak**: 96,7% spektrum punya **tepat 3 puncak**. Pola 3 puncak paling jelas terbagi menjadi **3 kelompok** (silhouette 0,695 di k = 3; 0,68 di k = 2; 0,66 di k = 4, lalu turun) | find_peaks + KMeans pada posisi 3 puncak | Petunjuk untuk 3c: spektrum saja hanya membedakan sedikit kelompok komposisi. Jika jenis material lebih banyak, sisanya harus dibedakan oleh **gambar** → alasan fusion | 09 |
| 11 | **Shift = translasi global per spektrum**: di dalam kelompok, deviasi posisi puncak 1 dan 2 berkorelasi **r = 0,90** (p1–p3: 0,91); jarak antar-puncak (std 1,8–2,0 kanal) lebih stabil daripada posisi absolut (std 3,1–4,3). Besar shift global 95%: **±7 kanal** | Scatter deviasi puncak 1 vs 2 | **Augmentasi shift acak ±7 kanal** + encoder 1D-CNN dengan pooling (tahan pergeseran). Std posisi absolut naik dari puncak 1 → 3, jadi mungkin ada sedikit peregangan (stretch) juga → augmentasi stretch kecil | 09 |
| 12 | **Intensitas** bervariasi: tinggi puncak tertinggi 0,66–1,72 (5–95%) | Histogram | **Normalisasi L2 per spektrum** setelah koreksi baseline | 09 |
| 13 | **Spektrum tidak dipengaruhi mikroskop**: AUC menebak mikroskop dari spektrum = 0,50 (mentah) / 0,51 (diproses) | Logistic regression CV 5-fold | Mikroskop hanya distorsi pada gambar. Spektrum bisa jadi "jangkar" yang bebas efek mikroskop saat fusion | 05, 10 |

## Urutan preprocessing spektrum (per spektrum, tanpa statistik lintas sampel)
1. Despike: residual positif terhadap median 5 kanal > 0,5 → nilai median
2. Baseline polinom derajat 2 (modpoly) → dikurangkan
3. Normalisasi L2

Karena semua langkah per spektrum, tidak ada informasi yang bocor antar-sampel.

## Poin jawab ujian lisan
- *Bagaimana tahu rotasinya acak?* Arah dominan dari structure tensor menyebar di seluruh 0–180° untuk gambar berarah. Ketidakrataan kecil di histogram juga muncul pada data sintetis bersudut seragam, jadi itu bias estimator. Orientasi bukan ciri material dan aman dijadikan augmentasi.
- *Kenapa augmentasi skala tidak boleh besar?* Garis membentuk 2 kelompok skala terpisah yang kemungkinan material berbeda. Scale ±50% bisa membuat garis halus mirip garis kasar, sehingga model diajari bahwa keduanya sama.
- *Kenapa standardisasi per gambar tidak cukup untuk beda mikroskop?* Setelah standardisasi, PSD masih menebak mikroskop dengan AUC 0,88 karena M2 lebih berderau. Maka perlu augmentasi noise/blur.
- *Kenapa ambang spike absolut, bukan k·σ?* Residual maksimum bimodal dengan celah jelas (0,37 vs 0,80). σ robust per spektrum hampir nol, sehingga k·σ menandai noise biasa.
- *Bagaimana tahu shift itu global?* Di dalam satu kelompok, ketiga puncak bergeser bersama (r = 0,90), dan jarak antar-puncak lebih stabil daripada posisinya.
