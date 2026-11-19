# Task 2b–c – Label RUL & sliding window

Kode: `kode/task2/data.py` (`add_rul_label`), `kode/task2/windows.py`. Uji: `python -m kode.task2.test_leakage`.

| Keputusan | Alasan |
|---|---|
| Label = `siklus_maks − siklus`, lalu `min(·, 150)` (grafik 09) | Di train, siklus terakhir = saat rusak. Cap 150 sesuai soal, dan sensor datar saat RUL > ~150 (EDA #5) |
| Satu sampel per (mesin, siklus t); input = baris t−W+1..t **mesin itu saja**; target = RUL di t | Tidak ada informasi sesudah t; meniru kondisi test (prediksi di siklus terakhir yang diketahui) |
| W = 50 | Panjang histori test terpendek, jadi window test tidak pernah ber-padding |
| Histori < W: pad kiri dengan 0 + channel **mask** (1 = data asli) | Semua panjang histori ikut dilatih (17,4% window train ber-padding); model tahu bagian mana yang padding |
| Fitur per langkah: 14 sensor + umur (siklus/300) + mask = 16 | Umur mesin adalah informasi masa lalu yang sah |
| Split **per mesin** (5-fold, seed 42) | Baris satu mesin sangat berkorelasi; split per baris = kebocoran, val terlalu optimis |
| Preprocessor di-fit hanya pada mesin train di tiap fold | Statistik normalisasi tidak boleh melihat mesin val |
| Validasi "seperti test": potong histori tiap mesin val di 5 titik acak (≥ 50 siklus, RUL ≥ 1), evaluasi siklus terakhir saja | Distribusi evaluasi sama dengan test tersembunyi |

Ukuran: 254.174 window train (penuh), 300 window test, tensor (W=50, F=16).

## Uji anti-kebocoran (semua lulus, sintetis + 60 mesin asli)
1. Label: RUL=0 di siklus terakhir, cap 150, tidak naik.
2. **Kausal end-to-end**: mengacak semua baris sesudah t (nilai + NaN) → window di t identik (setelah preprocessing).
3. Isolasi mesin: mengubah mesin lain tidak mengubah fitur mesin ini.
4. Fit hanya di train: mengubah mesin val tidak mengubah statistik normalisasi.
5. Split: tidak ada mesin di train dan val sekaligus; tiap mesin di tepat satu fold val.
6. Window test berakhir di siklus terakhir; padding & mask benar.

**Uji mutasi** (memastikan tes bisa gagal): median spike berpusat (melihat masa depan) dan bfill
sengaja disuntikkan → keduanya terdeteksi oleh uji #2.
