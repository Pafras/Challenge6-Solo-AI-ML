# Experiment log

Satu run = satu baris. Ganti satu variabel per run. Seed & split dikunci.
Test set cuma dibuka sekali, di Hari 8.

Patokan: tebak acak 4 kelas = 0.25. Akurasi manusia di FER2013 (7 kelas) ~0.65.

| # | Tanggal | Arsitektur | Image size | LR | Optimizer | Scheduler | Batch | Augmentation | Class weight | Epoch | Val acc | Catatan |
|---|---------|-----------|-----------|-----|-----------|-----------|-------|--------------|--------------|-------|---------|---------|
| 0 | 10 Sep | TinyCNN (2 conv, 10.468 par) | 48 | 1e-3 | Adam | tanpa | 64 | tanpa | tanpa | 5 | **0.588** | baseline. train 0.597 — nyaris nempel, jadi underfitting, bukan overfitting. loss valid masih turun, belum konvergen. |

## Hasil akhir (test set — isi Hari 8)

| Metrik | Nilai |
|--------|-------|
| Accuracy | |
| Macro F1 | |
| Latency / frame | |

Kelas yang dibuang dari mapping + alasannya:
