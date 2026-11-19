# Task 2a – Ringkasan EDA & preprocessing

Data: 900 mesin train (254.174 baris, run-to-failure), 300 mesin test (51.692 baris, berhenti
acak sebelum rusak). Semua angka dari data penuh; reproduksi: `python -m kode.task2.eda`.
Kode preprocessing: `kode/task2/data.py` (`Preprocessor`).

| # | Temuan | Bukti | Keputusan preprocessing | Grafik / tabel |
|---|---|---|---|---|
| 1 | **6 rezim operasi diskret**: `setel_1` ∈ {0, 10, 20, 25, 35, 42}; `setel_2` dan `setel_3` sepenuhnya ditentukan oleh `setel_1` (noise ±0,01). Jumlah baris per rezim seimbang (~42 rb) | Scatter setel_1 vs setel_2 membentuk 6 titik; hanya rezim 25 yang punya `setel_3` = 60 | Variabel `kond` = `setel_1` dibulatkan | 02, `tabel_kondisi_operasi.csv` |
| 2 | **Rezim operasi mendominasi bacaan sensor**: menjelaskan 37–100% varians (η²), mis. s05 99,98%, s11 37% | Pada sinyal mentah, nilai sensor melompat antar-rezim jauh lebih besar daripada tren degradasi | **Normalisasi z-score per kondisi**: mean & std tiap sensor dihitung per rezim, **hanya dari unit train**, lalu dipakai untuk val/test. Global z-score/min-max tidak cukup karena tidak menghapus loncatan antar-rezim | 03, `tabel_sensor.csv` (eta2_kondisi) |
| 3 | **7 sensor tidak informatif** (white noise murni): s01, s05, s06, s10, s16, s18, s19 | std(Δz)/std(z) ≈ 1,36 ≈ √2 (nilai teoretis white noise); sensor bertren < 1,0. Korelasinya dengan RUL ≈ 0,05 (setara noise) | Dibuang, sisa **14 sensor**. Kriteria **tanpa label** (ambang 1,2), jadi tidak bocor dari unit validasi | 04, `tabel_sensor.csv` (noise_ratio_diff) |
| 4 | **Dua mode kerusakan**: A (558 mesin), B (342 mesin), dengan sensor terdegradasi berbeda. A: s04, s12, s15 naik; s02, s03, s07, s11 turun. B: s04, s09, s11 naik; s13, s14, s17, s20, s21 turun | s11: korelasi global dengan RUL hanya 0,16, tetapi \|korelasi\| per mesin 0,75, karena arahnya berlawanan antar-mode | Seleksi sensor **tidak** memakai korelasi linear dengan RUL (akan salah membuang s11). Model sequence harus mengenali mode dari pola sensor; label mode **bukan** fitur (dibuat dari 10 siklus terakhir, tidak ada di test) dan hanya dipakai untuk analisis error per mode | 05, `mode_kerusakan_per_unit.csv` |
| 5 | **Sensor datar saat RUL > ~150**, lalu menyimpang makin cepat menjelang rusak | Profil rata-rata z per bin RUL; 46,6% baris train punya RUL asli > 150 | Mendukung **cap 150** (RUL piecewise-linear): di fase sehat, RUL 300 dan 200 tidak bisa dibedakan dari sensor | 05, 08 |
| 6 | **Missing ~2% per sensor** (35% baris punya ≥1 NaN, maks 5 per baris). Tidak bergantung RUL, tidak berkorelasi antar-sensor (\|r\| < 0,01), sedikit berkelompok di waktu (3.284 pasangan NaN berurutan vs 2.188 jika acak). Juga ada di **baris terakhir test** (2,2%) | Missing per sensor rata; % baris ber-NaN sama di semua rentang RUL | **Forward-fill per mesin** (hanya pakai nilai masa lalu → kausal), awal histori diisi 0 (= rata-rata kondisi). Tidak memakai interpolasi karena interpolasi melihat nilai sesudahnya (bocor masa depan) | 06 |
| 7 | **Spike ~0,20% titik**, merata di semua sensor, \|z\| sampai 10 | Lonjakan satu titik yang menyimpang jauh dari tetangganya | **Deteksi spike kausal**: bandingkan dengan median 10 siklus *sebelumnya*; jika selisih > 5σ robust (σ = 1,4826·MAD residual, dari train), ganti dengan median itu. Lalu **clip ±5** | 07 |
| 8 | **Panjang histori**: umur train 133–778 siklus (median 265); histori test 50–455 (median 162) | Histogram | Window sliding harus menangani histori pendek (minimal 50 di test); validasi meniru test dengan memotong histori val secara acak | 01 |

## Urutan preprocessing (per fold, fit hanya di unit train)
1. `kond` = round(`setel_1`)
2. z = (x − mean[kond]) / std[kond] untuk 14 sensor terpilih
3. Hapus spike (median kausal 10 siklus, ambang 5σ)
4. Forward-fill NaN per mesin, sisa → 0
5. Clip ke [−5, 5]

## Poin jawab ujian lisan
- *Kenapa normalisasi per kondisi?* Karena rezim operasi menggeser level sensor jauh lebih besar daripada degradasi; tanpa itu model harus belajar 6 baseline sekaligus dan tren tertutup.
- *Kenapa tidak memilih sensor dari korelasi dengan RUL?* Ada 2 mode kerusakan dengan arah berlawanan, sehingga korelasi global menipu (s11). Kriteria white noise tidak memakai label, jadi juga bebas kebocoran.
- *Kenapa ffill, bukan interpolasi?* Interpolasi memakai nilai sesudahnya, padahal saat prediksi nilai masa depan belum ada.
- *Kenapa cap 150?* Sesuai definisi soal, dan EDA menunjukkan sensor datar saat RUL > ~150.
