# Experiment log

Satu run = satu baris. Ganti satu variabel per run. Seed & split dikunci.
Test set cuma dibuka sekali, di Hari 8.

Patokan: tebak acak 4 kelas = 0.25. Akurasi manusia di FER2013 (7 kelas) ~0.65.

| # | Tanggal | Arsitektur | Image size | LR | Optimizer | Scheduler | Batch | Augmentation | Class weight | Epoch | Val acc | Catatan |
|---|---------|-----------|-----------|-----|-----------|-----------|-------|--------------|--------------|-------|---------|---------|
| 0 | 10 Sep | TinyCNN (2 conv, 10.468 par) | 48 | 1e-3 | Adam | tanpa | 64 | tanpa | tanpa | 5 | **0.588** | baseline. train 0.597 — nyaris nempel, jadi underfitting, bukan overfitting. loss valid masih turun, belum konvergen. |
| 1 | 10 Sep | TinyCNN (sama) | 48 | **1e-2** | Adam | tanpa | 64 | tanpa | tanpa | 2 | 0.373 | lr 10x baseline. macet: loss diem di 1.339 (≈ ln 4 = 1.386), akurasi gak gerak antar epoch. bukan lambat — kelewatan. |
| 2 | 10 Sep | **DeepCNN** (3 conv, 111.108 par) | 48 | 1e-3 | Adam | tanpa | 64 | tanpa | tanpa | 5 | **0.678** | +9 poin dari baseline, 10x parameter, 17 detik. train 0.694 vs valid 0.678 — gap mulai kebuka dikit tapi masih sehat. loss valid mulai mendatar di epoch 5 (0.819 → 0.813). |
| 3 | 10 Sep | **MobileNetV3-S pretrained** (1,52 jt par) | **224** | 1e-3 | Adam | tanpa | 64 | tanpa | tanpa | 5 | **0.813** | +13,5 poin dari DeepCNN, 198 detik. epoch 1 udah 0.732 — di atas nilai akhir DeepCNN, tanda transfer learning. **overfitting pertama:** loss valid terendah di epoch 3 (0.544) lalu naik, train 0.900 vs valid 0.810 di epoch 5. terbaik di epoch 4. dua variabel berubah (arsitektur + ukuran 224) — gak bisa dipisah di run ini. |
| 4 | 10 Sep | **ResNet18 pretrained** (11,2 jt par) | **224** | 1e-3 | Adam | tanpa | 64 | tanpa | tanpa | 5 | 0.797 | **kalah dari MobileNet** walau 7x parameter & 2,5x lebih lama (498 detik). overfit lebih cepat: loss valid terendah udah di epoch 2 (0.567), terus naik sampai 0.630. train 0.882 vs valid 0.797. |
| 5 | 10 Sep | MobileNetV3-S pretrained | 224 | **1e-4** | Adam | tanpa | 64 | tanpa | tanpa | 5 | 0.789 | nguji dugaan "lr 1e-3 kegedean bikin overfit". **dugaan salah:** overfit tetap, pola sama persis — loss valid terendah epoch 3 (0.580) lalu naik, gap train 0.879 vs valid 0.789. cuma lebih lambat (epoch 1: 0.573 vs 0.732) dan puncaknya lebih rendah. |
| 6 | 10 Sep | MobileNetV3-S pretrained | 224 | 1e-3 | Adam | tanpa | 64 | **flip+rot+jitter** | tanpa | 5 | 0.809 | beda dari #3 cuma augmentation. akurasi nyaris sama (0.809 vs 0.813), tapi **overfitting kerem:** loss valid terendah di epoch 5 (0.534) dan masih turun, sedangkan #3 udah naik lagi sejak epoch 3. gap train–valid 3,6 poin (vs 9). belum konvergen di 5 epoch — butuh run lebih panjang. |

## Temuan Hari 5 — arsitektur

Semua run: lr 1e-3, Adam, batch 64, 5 epoch, seed 42. Cuma arsitektur (dan ukuran input buat yang pretrained) yang beda.

| arsitektur | parameter | input | val acc | waktu | loss valid terendah di epoch |
|---|---|---|---|---|---|
| TinyCNN | 10 rb | 48 | 0.588 | 13 dtk | 5 (masih turun) |
| DeepCNN | 111 rb | 48 | 0.678 | 17 dtk | 5 (mulai datar) |
| MobileNetV3-S pretrained | 1,52 jt | 224 | **0.813** | 198 dtk | 3 |
| ResNet18 pretrained | 11,2 jt | 224 | 0.797 | 498 dtk | 2 |

Yang kelihatan dari data:

- **Pretrained ngalahin from-scratch jauh** — +12 sampai +13,5 poin. Epoch pertama pretrained udah di atas nilai akhir DeepCNN.
- **Lebih besar ≠ lebih bagus.** ResNet18 kalah dari MobileNet padahal 7x parameter.
- **Dua-duanya pretrained overfit cepat** — model kecil masih underfitting di epoch 5, model pretrained udah lewat puncaknya di epoch 2–3.
- **Waktunya jomplang.** MobileNet 15x lebih lama dari DeepCNN, ResNet 38x.

Yang **belum** bisa disimpulkan, dan kenapa:

- **Kemenangan pretrained bisa sebagian dari ukuran 224, bukan cuma bobotnya.** Dua variabel berubah bareng. Run pembanding yang bisa misahin: DeepCNN di 224.
- ~~**lr 1e-3 kemungkinan kegedean buat fine-tuning.**~~ **Diuji di run #5, dugaan salah.** lr 1e-4 overfit dengan pola yang sama persis, cuma lebih lambat. Overfitting-nya bukan dari lr — model pretrained emang gampang hafal 16 ribu foto. Obat yang tersisa: augmentation dan regularisasi (Hari 6).

## Hasil akhir (test set — isi Hari 8)

| Metrik | Nilai |
|--------|-------|
| Accuracy | |
| Macro F1 | |
| Latency / frame | |

Kelas yang dibuang dari mapping + alasannya:
