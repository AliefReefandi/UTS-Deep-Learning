# Task 1 – Multimodal Classification

## 1. Deskripsi
Task 1 bertujuan mengklasifikasikan cacat produk berdasarkan dua jenis input, yaitu citra produk dan sinyal sensor mesin.

Kelas yang digunakan:
- gelembung
- gores
- kontaminasi
- las_dingin
- noda
- normal
- retak

## 2. Model
Model yang dikembangkan menggunakan pendekatan multimodal dengan dua encoder:
- CNN sebagai image encoder.
- 1D-CNN sebagai sensor encoder.

Dua strategi fusion dibandingkan:
- Decision Fusion
- Feature Fusion

Sebagai pembanding, dilakukan eksperimen Image-only dan Sensor-only.

## 3. Hasil Eksperimen

| Model | Mean Macro-F1 | Std |
|---|---:|---:|
| Feature Fusion | 0.6591 | 0.0126 |
| Decision Fusion | 0.6409 | 0.0242 |
| Sensor-only | 0.4339 | 0.0058 |
| Image-only | 0.3123 | 0.0170 |

Feature Fusion menghasilkan performa rata-rata terbaik berdasarkan tiga random seed.

## 4. Robustness dan Evaluasi Tambahan

| Eksperimen | Macro-F1 |
|---|---:|
| Tanpa Noise | 0.6821 |
| Label Noise 10% | 0.6090 |
| Label Noise 20% | 0.5483 |
| Complete Modality | 0.6821 |
| Missing Image | 0.3507 |
| Missing Sensor | 0.3012 |

Evaluasi domain shift dari lini A ke lini B menghasilkan Macro-F1 sebesar 0.1970, sedangkan model yang dilatih menggunakan lini A dan B memperoleh Macro-F1 sebesar 0.5869 pada validasi lini B.

## 5. Visualisasi
- `Confusion Matrix.png`: distribusi prediksi dan kesalahan klasifikasi.
- `learning_curves.png`: perubahan training loss, validation loss, dan Macro-F1.
- `comparison_3_seeds.png`: perbandingan model berdasarkan tiga random seed.
- `label_noise_comparison.png`: pengaruh label noise.
- `domain_shift_comparison.png`: pengaruh perbedaan lini produksi.
- `missing_modality_comparison.png`: pengaruh hilangnya salah satu modality.

## 6. File
- Notebook: `kode/01_Task1_Classification.ipynb`
- Model: `models/feature_fusion_final.pth`
- Prediksi: `prediksi/prediksi_klasifikasi.csv`
- Visualisasi: folder `results/task1/`
