# Experiment log

Satu run = satu baris. Ganti satu variabel per run. Seed & split dikunci.
Test set cuma dibuka sekali, di Hari 8.

Patokan: tebak acak 4 kelas = 0.25. Akurasi manusia di FER2013 (7 kelas) ~0.65.

| # | Tanggal | Arsitektur | Image size | LR | Optimizer | Scheduler | Batch | Augmentation | Class weight | Epoch | Val acc | Catatan |
|---|---------|-----------|-----------|-----|-----------|-----------|-------|--------------|--------------|-------|---------|---------|
| 0 | 10 Sep | TinyCNN (2 conv, 10.468 par) | 48 | 1e-3 | Adam | tanpa | 64 | tanpa | tanpa | 5 | **0.588** | baseline. train 0.597 — nyaris nempel, jadi underfitting, bukan overfitting. loss valid masih turun, belum konvergen. |
| 1 | 10 Sep | TinyCNN (sama) | 48 | **1e-2** | Adam | tanpa | 64 | tanpa | tanpa | 2 | 0.373 | lr 10x baseline. macet: loss diem di 1.339 (≈ ln 4 = 1.386), akurasi gak gerak antar epoch. bukan lambat — kelewatan. |
| 2 | 10 Sep | **DeepCNN** (3 conv, 111.108 par) | 48 | 1e-3 | Adam | tanpa | 64 | tanpa | tanpa | 5 | **0.678** | +9 poin dari baseline, 10x parameter, 17 detik. train 0.694 vs valid 0.678 — gap mulai kebuka dikit tapi masih sehat. loss valid mulai mendatar di epoch 5 (0.819 → 0.813). |

## Hasil akhir (test set — isi Hari 8)

| Metrik | Nilai |
|--------|-------|
| Accuracy | |
| Macro F1 | |
| Latency / frame | |

Kelas yang dibuang dari mapping + alasannya:
