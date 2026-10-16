# Studio 2 — Klasifikasi Sampah · Kelompok 11

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| **No** | **Nama Lengkap**    | **NIM**   | **Kontibusi Utama** |
| :----: | :------------------ | :-------: | :------------------ |
|   1    | Raisya Syifa Saleh  | 123140169 | On Process          |
|   2    | Diva Amelia Saputri | 123140212 | On Process          |

## Ringkasan Pendekatan

- **Arsitektur / backbone**: ResNet-18 (`torchvision.models.resnet18`, ± 11,2 juta parameter), pretrained ImageNet (`IMAGENET1K_V1`). Layer `fc` diganti menjadi `Linear` dengan 5 kelas (`botol_plastik`, `gelas_plastik`, `kertas_kardus`, `saset_kemasan`, `styrofoam`). Submission berupa satu model; `get_model()` hanya membangun arsitektur (`weights=None`) dan bobot dimuat dari `model.pth` oleh `evaluate.py`.
- **Strategi pelatihan**:
  - Fine-tuning penuh (tanpa freezing), 10 epoch, batch 16, presisi penuh (fp32) pada GPU Tesla T4 (± 23,7 menit; epoch 1 ± 17,4 menit karena pembacaan pertama data dari Google Drive, epoch berikutnya ± 42 detik).
  - Optimizer AdamW dengan learning rate `3e-4` untuk seluruh parameter (weight decay default AdamW). Loss `CrossEntropyLoss` tanpa label smoothing.
  - Tanpa scheduler (learning rate konstan) dan tanpa gradient clipping.
  - Pemilihan checkpoint: bobot disimpan ke `submission/model.pth` setiap kali Macro-F1 validasi naik atau sama (`>=`) dengan yang terbaik sebelumnya, sehingga yang tersimpan adalah bobot epoch 5 (Macro-F1 0,8626).
  - Tanpa penyeimbangan kelas (rasio kelas terbesar/terkecil hanya 1,84).
  - Augmentasi: `RandomResizedCrop` (224, scale 0,7–1,0) dan `RandomHorizontalFlip`.
- **Data tambahan**: tidak ada. Hanya dataset resmi (2.769 gambar, 5 kelas; semuanya berformat `.jpg`).
- **Ukuran input & preprocessing**:
  - Input **224×224**, normalisasi mean/std ImageNet (`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`).
  - Training: gambar dibaca langsung dari `local_split/train/` lalu melewati pipeline augmentasi di atas.
  - Inferensi (`get_transform()`): `Resize(256)` → `CenterCrop(224)` → `ToTensor` → `Normalize`.
  - **TTA**: tidak dipakai di submission. Rata-rata probabilitas dari gambar asli dan horizontal flip hanya diuji di notebook (lihat catatan eksperimen).

## Hasil

| **Putaran**        | **Macro-F1 (split validasi lokal)** | **Macro-F1 (diumumkan asisten)** |
| :----------------- | :---------------------------------: | :------------------------------: |
| Preview 1 (14 Okt) |               0,8626                |                                  |

Nilai lokal diukur dengan `evaluate.py` pada `local_split/val/` (416 gambar, split stratified 85/15, `seed=42`), dengan pengaturan submission (224 px, tanpa TTA).

Detail per kelas (`evaluate.py`, `local_split/val/`):

| **Kelas**         | **F1**     | **Jumlah gambar** |
| :---------------- | :--------: | :---------------: |
| `botol_plastik`   |   0,8630   |        78         |
| `gelas_plastik`   |   0,8284   |        81         |
| `kertas_kardus`   |   0,8522   |        55         |
| `saset_kemasan`   |   0,8646   |        101        |
| `styrofoam`       |   0,9048   |        101        |
| **Macro-F1**      | **0,8626** |        416        |

Macro-F1 menurut resolusi uji dan TTA (`local_split/val/`, 416 gambar):

| **Resolusi** | **Tanpa TTA** | **TTA flip** |
| :----------: | :-----------: | :----------: |
|   224 px     |    0,8626     |    0,8661    |
|   256 px     |    0,8535     |    0,8625    |
|   288 px     |    0,8475     |    0,8512    |

Catatan eksperimen singkat (apa yang dicoba, apa yang berhasil/gagal):

- **Split data**: 2.353 gambar untuk train dan 416 untuk validasi lokal (stratified, proporsi kelas terjaga). Rinciannya: botol_plastik 445/78, gelas_plastik 456/81, kertas_kardus 310/55, saset_kemasan 572/101, styrofoam 570/101 (train/val).
- **Proses training**: Macro-F1 validasi naik dari 0,8320 pada epoch 1 menjadi 0,8626 pada epoch 5 (puncak), lalu berfluktuasi (0,8505 pada epoch 6; 0,8036 pada epoch 8; 0,8609 pada epoch 9; 0,7797 pada epoch 10). Train loss turun dari 0,6739 menjadi 0,1348, sedangkan skor validasi naik-turun. Karena checkpoint dipilih berdasarkan validasi lokal yang sama, skor 0,8626 cenderung sedikit optimistis.
- **Resolusi uji dan TTA** (pada model yang sama): resolusi 256 dan 288 memakai `Resize(round(res×256/224))` lalu `CenterCrop(res)`. Hasilnya ada di tabel resolusi dan TTA di atas.
- Resolusi 224 px terbaik, sedangkan resolusi lebih besar justru menurun karena model dilatih pada 224 px. TTA flip memberi kenaikan kecil di 224 px (+0,0035; akurasi 0,8654 → 0,8678), tetapi tidak diterapkan di submission karena `evaluate.py` resmi tidak memakai TTA. Submission memakai **224 px tanpa TTA**.
- **Selisih notebook vs `evaluate.py`**: tidak ada. Hasil hitungan di notebook dan `evaluate.py` sama persis, yaitu 0,8626 (akurasi 0,8654).
- **Analisis kesalahan**: kelas terlemah adalah `gelas_plastik` (F1 0,8284). Kesalahan terbanyak: `saset_kemasan` diprediksi `gelas_plastik` (8 gambar), `gelas_plastik` diprediksi `styrofoam` (7 gambar), `saset_kemasan` diprediksi `kertas_kardus` (6 gambar), dan `botol_plastik` diprediksi `gelas_plastik` (6 gambar). Kelas terkuat adalah `styrofoam` (F1 0,9048).
- **Catatan checkpoint**: `submission/model.pth` berisi bobot epoch 5 (Macro-F1 0,8626), sedangkan salinan cadangan di Google Drive (`outputs/model.pth`) berisi bobot epoch 10 yang lebih rendah (Macro-F1 0,7797) dan tidak dipakai untuk submission.
- **Anggaran waktu inferensi**: pada CPU 2 thread (Colab), praproses ± 6 ms dan forward ± 57 ms per gambar, setara ± 0,8 menit untuk 740 gambar (batas 15 menit). Dengan TTA flip sekitar 1,5 menit.
- **Belum dicoba**: data tambahan, backbone yang lebih besar/modern, scheduler learning rate, regularisasi (label smoothing, Mixup/CutMix, drop path), EMA bobot, ensemble beberapa backbone, dan sampler penyeimbang kelas.

## Cara Reproduksi

Pelatihan dilakukan lewat notebook `Studio2_123140169_123140212.ipynb` (dijalankan di Google Colab, GPU Tesla T4; Python 3.13, torch 2.11.0, torchvision 0.26.0).

```bash
pip install -r submission/requirements.txt        # torch, torchvision, pillow
```

1. Letakkan dataset resmi di Google Drive pada `MyDrive/Studio2_Kelompok11/dataset_mahasiswa/train/<nama_kelas>/` (atau ubah `DATASET_DIR` di notebook).
2. Jalankan seluruh sel notebook secara berurutan. Notebook akan membagi data menjadi `local_split/train/` dan `local_split/val/`, membuat `submission/model.py`, melatih model, menyimpan `submission/model.pth` (checkpoint dengan Macro-F1 validasi terbaik), menyalin cadangan bobot ke `dataset_mahasiswa/outputs/`, lalu mengevaluasi (F1 per kelas, resolusi/TTA, waktu inferensi, dan `evaluate.py`).
3. Evaluasi ulang secara mandiri:

```bash
python evaluate.py --submission ./submission --data ./local_split/val
```

Catatan: `seed=42` dipakai pada pembagian data, tetapi proses training tidak dikunci seed-nya sehingga hasil antar-run bisa berbeda (run sebelumnya mencapai 0,8698 pada epoch 7).

## Struktur Repositori
```
Repo-Kelompok11-ProyekStudio02
├── 
├── submission/
│  ├──model.py                    # get_model() dan get_transform()
│  ├──model.pth                   # state_dict model terbaik (epoch 5)
│  ├──requirements.txt            # torch, torchvision, pillow
├──evaluate.py                    # salinan skrip evaluasi resmi
├──Studio2_123140169_123140212.ipynb    # notebook pembagian data, pelatihan, evaluasi, dan ekspor submission

```
