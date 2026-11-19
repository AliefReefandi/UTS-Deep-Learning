# Hasil Eksperimen Task 3 – Clustering Multimodal (Gambar Mikrostruktur + Spektrum)

## 1. Deskripsi

36.000 sampel material **tanpa label**: citra mikrostruktur 64×64 dan spektrum komposisi 128 kanal,
diambil dengan dua mikroskop (M1 21.636, M2 14.364). Tujuan: mengelompokkan sampel menjadi jenis
material. Output: `cluster.csv` (`id_sampel,cluster`; integer 0–5) di root repo dan salinannya di `prediksi/`.

Reproduksi: `python run_all.py --tasks 3` (atau `python -m kode.task3.main`).
- 3a–3b (EDA + SimCLR) dijalankan di **Kaggle GPU** (`kode/colab_task3.ipynb`): gambar 150 epoch dalam 14,2 menit, spektrum 100 epoch dalam 1,7 menit.
- 3c–3f dijalankan **lokal di CPU** dari embedding hasil Kaggle: `python -m kode.task3.main --no-eda --no-ssl`, sekitar 5 menit.
- Lingkungan run lokal 3c–3f: Python 3.14.7, torch 2.15.0+cpu, scikit-learn 1.9.1, numpy 2.5.5, umap-learn 0.6.0, seed 42.
- Versi library di Kaggle tidak tercatat, karena sel pengemasan terakhir notebook gagal sebelum menyimpan `env_info` (notebook yang dipakai masih versi sebelum perbaikan env_info).

| Bagian | Isi | Folder / berkas |
|---|---|---|
| a | EDA distorsi gambar & spektrum | `eda/` (lihat `eda/RINGKASAN.md`) |
| b | Representation learning SimCLR per modalitas | `ssl/` |
| c | Fusion (FusionAE vs concat) + ablasi satu modalitas | `clustering/` grafik 16 |
| d | KMeans/GMM vs deep clustering IDEC | `clustering/` grafik 15 |
| e | Pemilihan jumlah cluster | `clustering/` grafik 14 |
| f | UMAP/t-SNE, profil cluster, cluster sulit, heterogenitas | `clustering/` grafik 17–21 |

## 2. Representation learning (3b): SimCLR per modalitas

Encoder dilatih tanpa label dengan loss kontrastif NT-Xent (τ = 0,2). Dua *view* augmentasi dari sampel
yang sama harus berdekatan, sampel lain berjauhan. Augmentasinya diturunkan dari temuan EDA:
- **Gambar:** rotasi bebas + flip, skala ±15%, gamma, blur, noise, standardisasi per gambar.
- **Spektrum:** shift ±7 kanal, stretch ±3%, sisa baseline, noise, spike.

| Encoder | Parameter | Dimensi embedding | Loss awal → akhir |
|---|---:|---:|---|
| Gambar: ResNet kecil | 1,23 jt | 256 | 5,98 → 3,51 |
| Spektrum: 1D-CNN | 0,24 jt | 128 | 5,99 → 2,55 |

Diagnostik representasi (`ssl/diagnostik_representasi.json`, grafik 13) dibandingkan dengan fitur
klasik (PSD radial untuk gambar, PCA untuk spektrum):

| | Gambar: PSD radial | Gambar: SimCLR | Spektrum: PCA | Spektrum: SimCLR |
|---|---:|---:|---:|---:|
| Retrieval view-ke-view top-1 (↑) | 0,016 | **0,367** | 0,018 | **0,796** |
| Rank efektif | 14,4 | 70,3 | 17,6 | 37,6 |
| AUC tebak mikroskop (ideal 0,5) | 0,88 | **0,99** | 0,50 | 0,50 |

Kesepakatan cluster gambar vs spektrum (AMI; k = 3 / 6 / 10): klasik 0,71 / 0,53 / 0,45; SimCLR
0,54 / **0,80** / 0,61. Kedua modalitas SimCLR paling sepakat di k = 6, yang menjadi petunjuk awal
untuk 3e.

**Masalah yang dibawa ke 3c:** embedding gambar SimCLR hampir sempurna menebak mikroskop (AUC 0,99).
Augmentasi noise/blur tidak cukup menghapus jejak alat. Spektrum bebas dari efek mikroskop (AUC 0,50).

## 3. Fusion (3c)

1. **Blok per modalitas:** normalisasi L2, lalu PCA sampai 95% varians, lalu skala ke total varians 1.
   Hasilnya: gambar 16 dimensi, spektrum 14 dimensi. Dengan begitu kedua modalitas berbobot sama,
   walaupun dimensi aslinya berbeda (256 vs 128).
2. **Koreksi mikroskop:** blok gambar dikurangi rata-ratanya per mikroskop (centering per mikroskop).
   Koreksi ini hanya aman jika komposisi material tidak bergantung pada mikroskop. Jika bergantung,
   centering juga ikut menghapus perbedaan material yang asli. Karena itu syaratnya diuji dulu pada
   spektrum, yang bebas mikroskop: Cramér's V antara cluster spektrum (k = 3, 4, 6, 8) dan mikroskop
   maksimum **0,001**, jauh di bawah ambang 0,10. Koreksi dipakai. Hasilnya, semua metode di bawah
   punya Cramér's V terhadap mikroskop sebesar 0,00.
3. **FusionAE** (model deep untuk fusion): autoencoder MLP 30 → 256 → 128 → **16** dengan
   **modality dropout**. Saat training, salah satu blok kadang dinolkan (p = 0,25 per blok), dan model
   harus merekonstruksinya dari modalitas lain. Ini memaksa latent memuat informasi yang **dimiliki
   bersama maupun yang unik** per modalitas.
   - Alasan tidak memakai fusion kontrastif ala CLIP: CLIP hanya mempertahankan informasi yang ada di
     kedua modalitas. Padahal ablasi menunjukkan ada informasi yang hanya ada di gambar.
   - Rekonstruksi penuh: MSE 0,0003.
   - R² merekonstruksi satu blok dari modalitas lain: gambar dari spektrum 0,61, spektrum dari gambar 0,67.
     Artinya kedua modalitas saling memprediksi, tetapi tidak sepenuhnya.

### Ablasi satu modalitas (`clustering/tabel_ablasi_modalitas.csv`, grafik 16)

KMeans dengan k = 6 dijalankan pada tiap representasi, lalu dibandingkan dengan cluster final.
"Pasangan tak terpisah" = pasangan cluster final yang tidak bisa dibedakan oleh representasi itu
(akurasi kNN a-vs-b < 0,75).

| Representasi | Dim | AMI vs final | ARI vs final | Akurasi kNN label final | Pasangan tak terpisah |
|---|---:|---:|---:|---:|---|
| Gambar saja | 16 | 0,999 | 0,999 | 1,000 | – |
| Spektrum saja | 14 | 0,801 | 0,687 | 0,772 | **2–5, 3–4** (akurasi 0,48–0,52 ≈ menebak) |
| Concat | 30 | 1,000 | 1,000 | 1,000 | – |
| FusionAE | 16 | 1,000 | 1,000 | 1,000 | – |

Yang hilang jika hanya memakai spektrum: dua pasangan material dengan spektrum yang **identik**
(c2 bintik terang vs c5 pulau gelap; c3 garis lurus kasar vs c4 garis bergelombang). Pasangan ini hanya
bisa dibedakan dari tekstur.

Pada k = 6, gambar saja sudah cukup (ARI 0,999), jadi di tingkat ini spektrum tidak menambah partisi
baru. Spektrum tetap berguna untuk dua hal:
- **Konfirmasi independen:** spektrum bebas mikroskop dan memisahkan semua pasangan lain dengan akurasi 1,00.
- **Satu-satunya sumber** faktor jarak puncak di cluster 0 (bagian 6).

## 4. Pemilihan k (3e) – `clustering/tabel_pemilihan_k.csv`, grafik 14

Pada latent FusionAE, untuk k = 2–15 dihitung silhouette, Davies-Bouldin (DB), Calinski-Harabasz (CH),
stabilitas bootstrap (ARI antar-subsampel 80%, 5 kali) dan BIC GMM.

**Aturan yang ditetapkan sebelum melihat hasil:**
1. Kandidat = k dengan stabilitas ≥ 0,80.
2. Dari kandidat, pilih jumlah peringkat (silhouette + DB + CH) terkecil.
3. Jika seri, pilih k yang lebih kecil.

| k | Silhouette | DB | CH | Stabilitas | Peringkat |
|---|---:|---:|---:|---:|---:|
| 4 | 0,413 | 1,153 | 18.790 | 0,920 | 13 |
| 5 | 0,464 | 0,907 | 20.632 | 0,967 | 6 |
| **6** | **0,472** | **0,871** | **20.859** | 0,963 | **3** |
| 7 | 0,424 | 1,054 | 19.488 | 0,874 | 9 |
| 8 | 0,410 | 1,078 | 18.594 | 0,959 | 14 |

**k = 6:** silhouette, DB dan CH sama-sama terbaik di k = 6, dan stabilitasnya 0,96.
- BIC GMM terus turun sampai k = 15, jadi tidak memberi titik siku yang jelas. BIC hanya dipakai sebagai informasi.
- Pemeriksaan silang: KMeans pada **gambar saja** juga memilih k = 6 (silhouette 0,65 vs 0,59 di k = 7).

## 5. Clustering (3d) – `clustering/tabel_metode.csv`, grafik 15

Semua metode memakai k = 6 dan 3 seed.

| Metode | Jenis | Silhouette (ruang sendiri) | Silhouette (latent FusionAE) | Stabilitas antar-seed (ARI) | Cramér's V mikroskop | Cluster terkecil |
|---|---|---:|---:|---:|---:|---:|
| KMeans (concat) | klasik | 0,511 | 0,472 | 1,000 | 0,00 | 10,2% |
| KMeans (FusionAE) | klasik | 0,472 | 0,472 | 1,000 | 0,00 | 10,2% |
| GMM kovarians penuh (FusionAE) | klasik | 0,378 | 0,378 | 0,838 | 0,00 | 10,2% |
| **IDEC (FusionAE)** | **deep** | **0,702** | 0,472 | 1,000 | 0,00 | 10,2% |

**IDEC** (Guo et al. 2017) = DEC + rekonstruksi:
- Encoder FusionAE dan pusat cluster dilatih bersama dengan loss KL(P‖Q) + 0,1 × rekonstruksi.
- Soft assignment memakai distribusi Student-t.
- Berhenti jika < 0,1% label berubah antar-epoch.

**Hasil jujur:** IDEC berhenti setelah **satu epoch** tanpa satu pun label berubah, dan labelnya
**identik** dengan KMeans (ARI 1,00; kesepakatan antar-metode di `tabel_kesepakatan_metode.csv`).
- Penyebabnya, keenam cluster sudah terpisah jelas di latent FusionAE, sehingga inisialisasi KMeans sudah stabil.
- IDEC hanya merapatkan cluster di ruangnya sendiri (silhouette 0,47 → 0,70). Ini berguna untuk
  visualisasi, tetapi tidak mengubah partisi.
- GMM paling berbeda (ARI 0,78 vs yang lain) dan paling tidak stabil antar-seed (0,84). Cluster
  terbesarnya lebih kecil (24,1% vs 27,9%), jadi sebagian sampel cluster terbesar dibagi ke cluster
  lain. Kovarians penuh punya jauh lebih banyak parameter, sehingga hasilnya lebih bergantung pada inisialisasi.
- **Metode final: IDEC (FusionAE)**, sesuai aturan bahwa model utama harus deep learning. Pengaman
  `pick_final` (cluster terkecil ≥ 0,5%, tanda tidak collapse) terpenuhi.

## 6. Hasil, visualisasi, dan cluster sulit (3f)

### Profil cluster final (grafik 18; UMAP/t-SNE grafik 17)

| Cluster | n | % M2 | Tekstur | Puncak spektrum (kanal) | Kelompok spektrum |
|---|---:|---:|---|---|---|
| c0 | 10.043 | 40,0 | **campuran:** butiran poligonal + berawan (lihat di bawah) | 20 / 40–44 / 85 | B |
| c1 | 8.685 | 39,1 | garis halus (λ ≈ 5 px) | 15 / 55 / 75 | A |
| c2 | 4.979 | 40,2 | bintik/sel terang | 30 / 62 / 100 | C |
| c3 | 4.340 | 40,0 | garis lurus kasar (λ ≈ 11 px) | 35 / 70 / 110 | D |
| c4 | 4.297 | 40,0 | garis kasar bergelombang | 35 / 70 / 110 | D |
| c5 | 3.656 | 40,7 | pulau gelap di latar terang | 30 / 62 / 100 | C |

- Persentase M2 sama di semua cluster (39–41%, rata-rata data 39,9%). Cluster tidak mengikuti alat.
- Hanya ada 4 pola spektrum (A–D) untuk 6 cluster. Ini sejalan dengan EDA 3a, yang menemukan pola
  3 puncak hanya membentuk sedikit kelompok.

### Pasangan sulit dipisahkan (`clustering/tabel_pasangan_sulit.csv`, grafik 19)

Tumpang tindih kNN (15 tetangga, ruang concat) antar-cluster final 0% untuk semua pasangan. Di ruang
gabungan tidak ada batas yang kabur. Karena itu pasangan diurutkan menurut akurasi pembedaan dari satu
modalitas:

| Pasangan | Akurasi gambar saja | Akurasi spektrum saja | Artinya |
|---|---:|---:|---|
| c3 – c4 | 1,00 | 0,48 | spektrum identik, hanya tekstur yang membedakan |
| c2 – c5 | 1,00 | 0,52 | spektrum identik, hanya tekstur yang membedakan |
| c0 – c1 | 1,00 | 1,00 | (pasangan berikutnya; mudah dipisahkan) |

### Cluster 0 heterogen: dua faktor independen (`clustering/` grafik 20–21, `ringkasan_clustering.json` → `heterogenitas`)

Kriteria k global mengukur keterpisahan rata-rata antar-cluster, sehingga tidak melihat struktur halus
di dalam satu cluster. Pemeriksaan di dalam cluster (`kode/task3/heterogen.py`) menemukan bahwa
**c0 adalah grid 2 × 2 dari dua faktor yang saling independen**:

1. **Faktor tekstur** (dari blok gambar): **butiran poligonal** (5.203 sampel) vs **berawan** (4.840 sampel).
   - Ditemukan oleh refinement blok gambar (bagian 7).
   - Akurasi menebak faktor ini dari blok gambar 0,998; dari blok spektrum 0,51 (menebak).
2. **Faktor spektrum: jarak puncak 1–2** = **20 kanal** (puncak 2 di ~40; 5.933 sampel) vs **24 kanal**
   (puncak 2 di ~44; 4.110 sampel).
   - Diuji bimodal per cluster dengan GMM 1D. Syaratnya: ΔBIC > 10, jarak antar-mode ≥ 2 std, dan komponen kecil ≥ 10%.
   - **Hanya c0 yang bimodal:** ΔBIC 4.488, jarak antar-mode 4,3 std. Lima cluster lain ΔBIC negatif, jarak 1,1–1,5 std.
   - Akurasi menebak faktor ini dari blok spektrum 0,98; dari blok gambar 0,59.

Tabel silang (berbentuk 2×2 karena kedua faktor saling silang):

| | jarak 20 kanal | jarak 24 kanal |
|---|---:|---:|
| butiran poligonal | 3.058 | 2.145 |
| berawan | 2.875 | 1.965 |

- Cramér's V antara tekstur dan spektrum = **0,00**: kedua faktor benar-benar independen.
- Terhadap mikroskop: faktor spektrum V = 0,00; faktor tekstur V = **0,098** (M2 35% di butiran vs 45% di berawan).
  Angka ini tepat di bawah ambang 0,10, jadi keterkaitan kecil dengan mikroskop tidak bisa sepenuhnya dikesampingkan.

Kenapa c0 tetap satu cluster di final:
- Kedua faktor tidak cukup besar untuk mengalahkan kriteria global (bagian 4).
- Tanpa label tidak bisa dipastikan mana yang menandai jenis material dan mana yang variasi di dalam satu material.
- Independensi sempurna kedua faktor juga tidak lazim untuk dua sifat yang sama-sama menentukan material.

**Koreksi interpretasi.** Pada pemeriksaan awal, dua "pulau" c0 di UMAP/t-SNE (grafik 17) kami kira
sebagai dua tekstur. Pemeriksaan lanjutan menunjukkan pulau itu ditentukan oleh **faktor spektrum**
(akurasi menebak pulau dari blok spektrum 1,00; dari blok gambar 0,58). Pemecahan tekstur justru
membelah **tiap** pulau menjadi dua (grafik 20, panel tengah).

## 7. Analisis alternatif (bukan hasil final)

**k = 7: pecah tekstur c0** (`cluster_alternatif_k7.csv`, `clustering/tabel_refinement.csv`).
`cluster.refine_by_block` menjalankan KMeans(k+1) pada blok gambar. Pemecahan diterima hanya jika:
- hasilnya pemecahan **murni**: tepat satu cluster terbagi dua, dan semua cluster baru ≥ 99% berasal dari satu cluster lama;
- pemecahannya tidak mengikuti mikroskop (Cramér's V < 0,10).

Langkah ini diuji pada data sintetis di `test_cluster.test_refinement`.

| Langkah | Kemurnian | Cluster dipecah | Ukuran | V mikroskop | Diterima |
|---|---:|---|---|---:|---|
| k 6 → 7 | 0,999 | c0 | 5.203 / 4.840 | 0,098 | ya |
| k 7 → 8 | 0,999 | c1 (garis halus) | 7.005 / 1.680 | **0,373** (M2 30% vs 76%) | tidak, pemecahan menurut alat |

Hasil k = 7 **tidak** dipakai sebagai final, karena dua alasan:
1. k = 7 bertentangan dengan aturan pemilihan k yang ditetapkan sebelum melihat hasil.
2. V = 0,098 terlalu dekat dengan ambang.

Berkas ini disimpan sebagai pembanding. Hasil final dapat diganti dengan `--refine`.

**k = 9 (tekstur × spektrum di c0)** juga mungkin: empat sel di tabel silang di atas. Tidak dipakai
karena membutuhkan aturan refinement baru pada blok spektrum, yang berarti keputusan lain yang dibuat
setelah melihat hasil.

## 8. Keterbatasan

- **Embedding gambar membawa jejak mikroskop** (AUC 0,99). Jejak ini ditangani di tahap fusion
  (centering per mikroskop), bukan di encoder. Encoder yang invarian terhadap alat (mis. adversarial
  terhadap mikroskop) lebih bersih, tetapi lebih rumit dilatih.
- **IDEC tidak mengubah partisi** dibanding KMeans. Pada data yang cluster-nya sudah terpisah jelas,
  manfaat deep clustering terbatas pada representasi yang lebih rapat.
- **Validasi tanpa label:** semua kesimpulan berdasar ukuran internal, konsistensi antar-modalitas,
  dan inspeksi visual. Kemungkinan c0 berisi 2 atau 4 material tidak dapat dipastikan tanpa label.

## 9. Berkas

| Berkas | Isi |
|---|---|
| `../../cluster.csv`, `../../prediksi/cluster.csv` | prediksi final k = 6 (lolos `python -m kode.validasi_csv`) |
| `cluster_alternatif_k7.csv` | pembanding k = 7 (bukan prediksi) |
| `ssl/` | kurva loss (11), contoh augmentasi (12), diagnostik (13), info training, log Kaggle |
| `clustering/14–21_*.png` | grafik 3c–3f |
| `clustering/tabel_*.csv`, `ringkasan_clustering.json` | semua angka di README ini |
| `../../models/task3_simclr_{gambar,spektrum}.pt` | encoder SimCLR (Kaggle) |
| `../../models/task3_fusion_idec.pt` | FusionAE + pusat cluster IDEC |

## 10. Poin jawab ujian lisan

- **Kenapa SimCLR, bukan autoencoder biasa?** Autoencoder piksel harus merekonstruksi semua detail,
  termasuk orientasi, brightness dan noise mikroskop. SimCLR dengan augmentasi dari EDA diajari
  langsung untuk mengabaikan faktor-faktor itu. Retrieval view-ke-view 0,37 (gambar) dan 0,80
  (spektrum), dibanding 0,02 untuk fitur klasik.
- **Kenapa koreksi mikroskop di fusion, dan kapan aman?** Centering hanya aman jika proporsi material
  sama di kedua mikroskop. Syarat ini diuji pada spektrum, yang bebas alat: Cramér's V 0,001. Jika
  syaratnya gagal, centering otomatis tidak dipakai (diuji di `test_cluster.test_microscope_check`).
- **Kenapa modality dropout?** Agar latent menyimpan informasi unik per modalitas, bukan hanya yang
  dimiliki bersama. Ablasi membuktikan pentingnya: dua pasangan cluster hanya bisa dibedakan oleh gambar.
- **IDEC tidak lebih baik dari KMeans?** Untuk partisi ini, ya, labelnya identik. Itu dilaporkan apa
  adanya. IDEC dipilih karena aturan "model utama deep", dan pengaman collapse terpenuhi.
- **Bagaimana memilih k tanpa label?** Aturan ditetapkan sebelum melihat hasil: stabilitas ≥ 0,8, lalu
  peringkat gabungan silhouette, DB dan CH. Semuanya sepakat di k = 6, dan gambar saja juga memilih 6.
- **Kenapa tidak k = 7, padahal ada dua tekstur di c0?** Karena itu keputusan yang dibuat setelah
  melihat hasil. Selain itu ada faktor spektrum yang independen dari tekstur, sehingga pilihan "yang
  benar" (6, 7, atau 9) tidak bisa dipastikan tanpa label. Analisis alternatifnya disertakan.
- **Kesalahan yang kami koreksi:** pulau UMAP awalnya kami kira tekstur, ternyata jarak puncak
  spektrum. Pelajarannya, struktur di proyeksi 2D harus diverifikasi di ruang aslinya per modalitas
  sebelum dijadikan bukti.
