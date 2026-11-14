# UTS Deep Learning

**Multimodal Classification, Remaining Useful Life (RUL) Regression, and Multimodal Clustering**

Repository ini berisi implementasi dan hasil pengerjaan Ujian Tengah Semester mata kuliah Deep Learning.

## Daftar Task

| Task | Deskripsi | Status |
|---|---|---|
| Task 1 | Multimodal Classification – Product Defects | Selesai |
| Task 2 | Time-Series Regression – Remaining Useful Life (RUL) | Belum dikerjakan |
| Task 3 | Multimodal Clustering | Belum dikerjakan |

## Struktur Repository

```text
UTS-Deep-Learning/
├── README.md
├── kode/
│   ├── README.md
│   └── 01_Task1_Classification.ipynb
├── models/
│   ├── README.md
│   └── feature_fusion_final.pth
├── prediksi/
│   └── prediksi_klasifikasi.csv
└── results/
    └── task1/
        ├── README.md
        ├── Confusion Matrix.png
        ├── learning_curves.png
        ├── comparison_3_seeds.png
        ├── label_noise_comparison.png
        ├── domain_shift_comparison.png
        └── missing_modality_comparison.png
```

## Teknologi

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Google Colab

## Hasil Task 1

Model terbaik berdasarkan rata-rata tiga random seed adalah **Feature Fusion**.

| Model | Mean Macro-F1 |
|---|---:|
| Feature Fusion | 0.6591 |
| Decision Fusion | 0.6409
