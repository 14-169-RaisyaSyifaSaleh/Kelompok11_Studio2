# Studio 2 — Klasifikasi Sampah · Kelompok 11

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| **No** | **Nama Lengkap** | **NIM** | **Kontribusi Utama** |
| ------ | ---------------- | ------- | ------------------- |
| 1 | Raisya Syifa Saleh | 123140169 | Training model, eksperimen, dan optimasi model |
| 2 | Diva Amelia Saputri | 123140212 | Persiapan submission, evaluasi model, dokumentasi, dan pengelolaan repository |

## Ringkasan Pendekatan

- **Arsitektur / backbone**: ResNet-34 (`torchvision.models.resnet34`, sekitar 21,8 juta parameter), menggunakan pretrained weights ImageNet `IMAGENET1K_V1`. Layer `fc` diganti menjadi `Linear` dengan 5 kelas (`botol_plastik`, `gelas_plastik`, `kertas_kardus`, `saset_kemasan`, `styrofoam`). Submission menggunakan satu model. Fungsi `get_model()` membangun arsitektur ResNet-34, sedangkan bobot model dimuat dari `model.pth` oleh `evaluate.py`.
- **Strategi pelatihan**:
  - Pelatihan dilakukan selama 12 epoch menggunakan GPU Tesla T4.
  - Optimizer: AdamW dengan learning rate awal `1e-4` dan weight decay `1e-4`.
  - Loss function: `CrossEntropyLoss` dengan label smoothing `0.05`.
  - Scheduler: `CosineAnnealingLR` dengan `T_max=12` dan `eta_min=1e-6`.
  - Batch size: 16.
  - Pemilihan checkpoint dilakukan berdasarkan Macro-F1 validasi tertinggi. Model terbaik diperoleh pada epoch 8 dengan Macro-F1 sebesar 0,9428 dan disimpan ke `submission/model.pth`.
  - Seed yang digunakan adalah 42 untuk membantu menjaga konsistensi eksperimen.
- **Data tambahan**: tidak menggunakan data tambahan. Dataset terdiri dari 2.753 gambar dalam 5 kelas.
- **Ukuran input dan preprocessing**:
  - Input gambar berukuran 224 × 224 piksel.
  - Normalisasi menggunakan mean `[0.485, 0.456, 0.406]` dan standard deviation `[0.229, 0.224, 0.225]` sesuai normalisasi ImageNet.
  - Augmentasi training meliputi `RandomResizedCrop`, `RandomHorizontalFlip`, `RandomRotation`, `RandomAffine`, dan `ColorJitter`.
  - Transformasi validasi menggunakan fungsi `get_transform()` dari `submission/model.py`.
- **Test-Time Augmentation (TTA)**: hasil eksperimen TTA pada ResNet-18 tidak digunakan sebagai hasil ResNet-34. Evaluasi model final menggunakan konfigurasi submission tanpa menambahkan TTA secara terpisah.

## Hasil

| **Putaran** | **Macro-F1 (split validasi lokal)** | **Macro-F1 (diumumkan asisten)** |
| ----------- | ----------------------------------- | -------------------------------- |
| Preview 1 (14 Okt) | 0,9428* | |

\* Nilai 0,9428 merupakan hasil evaluasi lokal ResNet-34, bukan nilai resmi yang diumumkan asisten. Nilai resmi belum dicantumkan karena belum tersedia dalam informasi eksperimen ini.

Evaluasi lokal dilakukan menggunakan `evaluate.py` pada `local_split/val/`, yang terdiri dari 413 gambar. Hasil evaluasi menunjukkan Macro-F1 sebesar **0,9428** dan akurasi sebesar **0,9443**.

### Detail Hasil per Kelas

| **Kelas** | **F1-score** | **Jumlah gambar validasi** |
| --------- | -----------: | -------------------------: |
| `botol_plastik` | 0,9673 | 78 |
| `gelas_plastik` | 0,9259 | 78 |
| `kertas_kardus` | 0,9245 | 55 |
| `saset_kemasan` | 0,9360 | 101 |
| `styrofoam` | 0,9604 | 101 |
| **Macro-F1** | **0,9428** | **413** |

### Catatan Eksperimen

- **Pembagian data**: dataset dibagi menjadi 2.340 gambar untuk training dan 413 gambar untuk validasi lokal. Pembagian menggunakan metode stratified dengan seed 42 agar proporsi kelas tetap terjaga.
- **Distribusi data validasi**: `botol_plastik` sebanyak 78 gambar, `gelas_plastik` sebanyak 78 gambar, `kertas_kardus` sebanyak 55 gambar, `saset_kemasan` sebanyak 101 gambar, dan `styrofoam` sebanyak 101 gambar.
- **Proses training**: Macro-F1 validasi meningkat dari 0,8754 pada epoch 1 menjadi 0,9428 pada epoch 8. Setelah itu, skor berfluktuasi hingga mencapai 0,9383 pada epoch 12.
- **Perkembangan loss**: train loss turun dari 0,7366 pada epoch 1 menjadi 0,2560 pada epoch 12. Sementara itu, validation loss terendah tercatat sebesar 0,3936 pada epoch 10.
- **Pemilihan model terbaik**: checkpoint dipilih berdasarkan Macro-F1 validasi tertinggi. Epoch 8 menghasilkan Macro-F1 sebesar 0,9428 dengan akurasi 0,9443.
- **Analisis kesalahan**: berdasarkan confusion matrix, kesalahan prediksi masih ditemukan pada beberapa kelas. Pada kelas `botol_plastik`, terdapat 2 gambar yang diprediksi sebagai `gelas_plastik` dan 2 gambar sebagai `saset_kemasan`. Pada kelas `gelas_plastik`, terdapat masing-masing 1 gambar yang diprediksi sebagai `kertas_kardus`, `saset_kemasan`, dan `styrofoam`. Pada kelas `kertas_kardus`, kesalahan terbanyak terjadi ketika 3 gambar diprediksi sebagai `gelas_plastik`. Pada kelas `saset_kemasan`, beberapa gambar salah diklasifikasikan sebagai `botol_plastik`, `gelas_plastik`, `kertas_kardus`, dan `styrofoam`. Pada kelas `styrofoam`, terdapat 2 gambar yang diprediksi sebagai `gelas_plastik` dan 2 gambar sebagai `saset_kemasan`.
- **Kinerja per kelas**: F1-score tertinggi diperoleh kelas `botol_plastik` sebesar 0,9673, sedangkan F1-score terendah diperoleh kelas `kertas_kardus` sebesar 0,9245. Meskipun menjadi kelas dengan F1-score terendah, hasilnya masih menunjukkan performa yang baik pada validasi lokal.
- **Waktu training**: total waktu berdasarkan log sekitar 27 menit 40 detik. Epoch pertama berlangsung selama 1.214 detik, sedangkan epoch berikutnya berkisar antara 38,6–41,8 detik.
- **Anggaran waktu inferensi**: pengukuran waktu inferensi pada CPU belum dicantumkan untuk ResNet-34. Evaluasi lokal yang tercatat dijalankan menggunakan GPU (`device=cuda`). Batas waktu evaluasi submission adalah 15 menit untuk keseluruhan data evaluasi.
- **Eksperimen lanjutan**: beberapa pendekatan yang dapat diuji selanjutnya meliputi penambahan data, penggunaan backbone lain, perubahan strategi regularisasi, ensemble model, dan sampler penyeimbang kelas.

## Cara Reproduksi

Pelatihan dilakukan melalui notebook `Studio2_123140169_123140212.ipynb` menggunakan Google Colab dengan GPU Tesla T4.

Lingkungan eksperimen yang tercatat:
- Python 3.13.16
- PyTorch 2.11.0+cu130
- torchvision 0.26.0+cu130

Instal dependensi dengan perintah berikut:

```bash
pip install -r submission/requirements.txt
```

Langkah reproduksi:

1. Letakkan dataset pada direktori yang sesuai dengan konfigurasi `DATASET_DIR` di notebook.
2. Pastikan dataset telah dibagi menjadi folder `local_split/train/` dan `local_split/val/`.
3. Jalankan sel notebook untuk memuat dataset, melakukan augmentasi, membangun model ResNet-34, melatih model, dan mengevaluasi hasilnya.
4. Model terbaik disimpan pada `submission/model.pth` berdasarkan Macro-F1 validasi tertinggi.
5. Pastikan `submission/model.py` menggunakan arsitektur ResNet-34 dan transformasi inferensi yang sesuai dengan model yang dilatih.
6. Jalankan evaluasi lokal menggunakan perintah berikut apabila sesuai dengan argumen yang tersedia pada `evaluate.py`:

```bash
python evaluate.py --submission ./submission --data ./local_split/val
```

**Catatan reproduktibilitas:** seed 42 digunakan pada proses pengacakan data dan konfigurasi random yang tercantum pada notebook. Hasil training dapat berbeda antar-run bergantung pada lingkungan komputasi dan determinisme operasi yang digunakan.

## Struktur Repositori

```text
Repo-Kelompok11-ProyekStudio02/
├── submission/
│   ├── model.py                       # Arsitektur ResNet-34 dan transformasi inferensi
│   ├── model.pth                      # Bobot model terbaik
│   └── requirements.txt               # Dependensi model
├── evaluate.py                        # Skrip evaluasi
└── Studio2_123140169_123140212.ipynb  # Notebook training dan evaluasi
```
