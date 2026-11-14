
# Hasil Eksperimen Task 1 – Multimodal Classification

## 1. Deskripsi

Task 1 bertujuan mengklasifikasikan kondisi produk ke dalam tujuh kelas berdasarkan dua jenis input, yaitu citra produk dan sinyal sensor mesin.

Kelas yang digunakan:
- gelembung
- gores
- kontaminasi
- las_dingin
- noda
- normal
- retak

## 2. Perbandingan Model

Evaluasi dilakukan menggunakan Validation Macro-F1 dengan tiga random seed.

| Model | Mean Macro-F1 | Std Macro-F1 |
|---|---:|---:|
| Feature Fusion | 0.6591 | 0.0126 |
| Decision Fusion | 0.6409 | 0.0242 |
| Sensor-only | 0.4339 | 0.0058 |
| Image-only | 0.3123 | 0.0170 |

Feature Fusion menghasilkan rata-rata Macro-F1 tertinggi, yaitu 0.6591.

## 3. Eksperimen Tambahan

### Label Noise

| Kondisi | Validation Macro-F1 |
|---|---:|
| Tanpa Noise | 0.6821 |
| Noise 10% | 0.6090 |
| Noise 20% | 0.5483 |

Performa model menurun ketika persentase label noise meningkat.

### Domain Shift

| Kondisi | Validation Macro-F1 |
|---|---:|
| Training Lini A + B, Evaluasi Lini B | 0.5869 |
| Training Lini A, Evaluasi Lini B | 0.1970 |

Hasil menunjukkan bahwa penggunaan data dari kedua lini produksi membantu meningkatkan generalisasi model terhadap Lini B.

### Missing Modality

| Kondisi | Validation Macro-F1 |
|---|---:|
| Complete Modality | 0.6821 |
| Missing Image | 0.3507 |
| Missing Sensor | 0.3012 |

Performa menurun ketika salah satu modality tidak tersedia.

## 4. Sanity Check

Eksperimen pengacakan label menghasilkan Macro-F1 sebesar 0.1372, lebih rendah dibandingkan baseline sebesar 0.6821.

Hal ini menunjukkan bahwa model baseline mempelajari pola yang berkaitan dengan label, bukan sekadar menghasilkan prediksi secara acak.

## 5. Visualisasi

Visualisasi hasil eksperimen tersedia pada folder ini:

- `Confusion Matrix.png`
- `learning_curves.png`
- `comparison_3_seeds.png`
- `label_noise_comparison.png`
- `domain_shift_comparison.png`
- `missing_modality_comparison.png`

## 6. Model dan Prediksi

- Model terbaik: `models/feature_fusion_final.pth`
- Notebook: `kode/01_Task1_Classification.ipynb`
- Hasil prediksi: `prediksi/prediksi_klasifikasi.csv`
