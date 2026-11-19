# UTS Deep Learning

**Multimodal Classification, Remaining Useful Life (RUL) Regression, and Multimodal Clustering**

Repository ini berisi implementasi dan hasil pengerjaan Ujian Tengah Semester mata kuliah Deep Learning.

## Daftar Task

| Task | Deskripsi | Status |
|---|---|---|
| Task 1 | Multimodal Classification – Product Defects | Selesai |
| Task 2 | Time-Series Regression – Remaining Useful Life (RUL) | Selesai |
| Task 3 | Multimodal Clustering | Selesai |

## Struktur Repository

```text
UTS-Deep-Learning/
├── README.md
├── kontribusi.md
├── laporan.pdf                     # laporan akhir (sumber: laporan/laporan.md)
├── run_all.py                      # satu perintah untuk mereproduksi semua task
├── prediksi_klasifikasi.csv        # prediksi Task 1 (dinilai otomatis)
├── prediksi_rul.csv                # prediksi Task 2 (dinilai otomatis)
├── cluster.csv                     # prediksi Task 3 (dinilai otomatis)
├── kode/
│   ├── 01_Task1_Classification.ipynb
│   ├── colab_task2_task3.ipynb     # Colab: Task 2 (+ Task 3)
│   ├── colab_task3.ipynb           # Colab/Kaggle: Task 3
│   ├── requirements.txt
│   ├── validasi_csv.py             # cek format ketiga CSV prediksi
│   ├── common/                     # argumen, seed, path, util, gaya grafik
│   ├── task1/                      # menjalankan notebook Task 1 lewat nbconvert
│   ├── task2/                      # EDA, label & window, model, evaluasi, uncertainty, tes anti-kebocoran
│   └── task3/                      # EDA, SimCLR, fusion, pemilihan k, clustering, visualisasi, tes
├── models/
│   ├── feature_fusion_final.pth                  # Task 1
│   ├── task2_lstm_fold{0-4}_best.pt (+ .json)    # Task 2: ensemble 5 fold + statistik normalisasi
│   └── task3_simclr_{gambar,spektrum}.pt, task3_fusion_idec.pt
├── prediksi/                       # salinan CSV prediksi
├── results/
│   ├── task1/                      # grafik & README Task 1
│   ├── task2/                      # eda/, windows/, models/, uncertainty/ + README.md
│   └── task3/                      # eda/, ssl/, clustering/ + README.md
└── laporan/                        # sumber laporan (Markdown), persiapan ujian lisan
```

`data/` (dataset) dan `outputs/` (checkpoint & artefak antara) tidak di-commit.

## Teknologi

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Google Colab
- SciPy, umap-learn (Task 2–3); Kaggle GPU (Task 3b)

## Cara Reproduksi

```bash
pip install -r kode/requirements.txt
python run_all.py                    # semua task; prediksi ditulis ke root lalu divalidasi
python run_all.py --tasks 2 3        # hanya Task 2 dan 3
python run_all.py --debug            # subset kecil untuk uji cepat di lokal (CPU)
python -m kode.validasi_csv          # cek format ketiga CSV prediksi
```

- Dataset diletakkan di `data/` (atau `--data_dir /content/data`). Output antara di `outputs/` (atau `--out_dir`).
- Seed 42. Versi library dicatat otomatis di `<out_dir>/env_info.json`.
- Task 1 dijalankan dari notebook Alief (nbconvert, di Colab).
- Training penuh Task 2 (~10 menit) dan Task 3b (SimCLR, ~16 menit) memerlukan GPU. Notebook Colab/Kaggle ada di `kode/`.
- Task 3c–f berjalan di CPU: `python -m kode.task3.main --no-eda --no-ssl`. Jika embedding belum ada, embedding
  dihitung ulang dari `models/task3_simclr_*.pt` (total ~19 menit di CPU). Hasilnya identik dengan `cluster.csv`
  yang di-commit (ARI 1,00; diuji pada clone bersih).

Tes:

```bash
python -m kode.task2.test_leakage    # 6 uji anti-kebocoran + uji mutasi
python -m kode.task2.test_models
python -m kode.task3.test_ssl
python -m kode.task3.test_cluster
```

## Hasil Task 1

Model terbaik berdasarkan rata-rata tiga random seed adalah **Feature Fusion**.

| Model | Mean Macro-F1 |
|---|---:|
| Feature Fusion | 0.6591 |
| Decision Fusion | 0.6409 |
| Sensor-only | 0.4339 |
| Image-only | 0.3123 |

## Hasil Task 2

Prediksi RUL pada siklus terakhir 300 mesin test (cap 150). Validasi 5-fold per mesin, meniru kondisi test.
Detail: `results/task2/README.md`.

| Model | RMSE | Skor asimetris / 300 mesin |
|---|---:|---:|
| **LSTM** (dipakai, ensemble 5 fold) | **19,14** | **4.673** |
| TCN | 19,51 | 5.226 |
| Transformer | 19,75 | 6.170 |
| Baseline gradient boosting (non-DL) | 19,83 | 6.654 |

Interval prediksi (MC dropout + kalibrasi conformal) mencakup 89,9% pada target 90%. Status 300 mesin
test: 64 servis segera, 75 dijadwalkan, 161 aman.

## Hasil Task 3

36.000 sampel tanpa label (citra mikrostruktur + spektrum). Detail: `results/task3/README.md`.

- **Representasi:** SimCLR per modalitas. Retrieval view-ke-view 0,37 (gambar) dan 0,80 (spektrum), vs 0,02 untuk fitur klasik.
- **Fusion:** FusionAE dengan modality dropout; efek mikroskop dikoreksi setelah lolos uji (Cramér's V 0,001).
- **Jumlah cluster:** **k = 6**. Silhouette, Davies-Bouldin dan Calinski-Harabasz sepakat, stabilitas bootstrap 0,96.
- **Model final:** IDEC (deep clustering) pada latent FusionAE.
- **Temuan:**
  - Spektrum saja tidak dapat memisahkan dua pasangan cluster (spektrumnya identik).
  - Cluster 0 berisi dua faktor independen: tekstur dan jarak puncak spektrum.
  - Analisis alternatif k = 7 tersedia di `results/task3/cluster_alternatif_k7.csv`.

## Laporan dan Kontribusi

- Laporan: `laporan.pdf` (sumber `laporan/laporan.md`)
- Kontribusi anggota dan penggunaan AI: `kontribusi.md`
