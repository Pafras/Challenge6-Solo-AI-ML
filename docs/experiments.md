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
| 7 | 10 Sep | MobileNetV3-S pretrained | 224 | 1e-3 | Adam | tanpa | 64 | flip+rot+jitter | tanpa | **15** | **0.816** | **plafon ~0.81.** 3x latihan cuma +0,3 dari #3 — dalam rentang berisik val acc (lompat 2–3 poin antar epoch), jadi setara, bukan lebih bagus. augmentation **nunda** overfit (epoch 3 → 5), gak nyegah: loss valid terendah epoch 5 (0.534), lalu naik ke 0.694, gap train–valid 12,7 poin di epoch 15. akurasi datar tapi loss naik = model makin **yakin sama jawaban salah**. checkpoint: epoch 9 → `models/mobilenet-aug.pt`. |
| 8 | 10 Sep | MobileNetV3-S pretrained | 224 | 1e-3 | **AdamW wd 0.01** | tanpa | 64 | flip+rot+jitter | tanpa | 10 | 0.817 | beda dari #7 cuma optimizer. **nyaris identik:** 0.817 vs 0.816, loss valid terendah tetap epoch 5 (0.527 vs 0.534), train epoch 10 0.900 vs 0.903. wd 0.01 terlalu lemah buat ngerem hafalan. goyangan tetap (epoch 6 anjlok ke 0.762). jalurnya beda di awal (ep1 0.757 vs 0.702) tapi nyampe tempat yang sama → optimizer bukan tuasnya. yang sama di #7 dan #8: lr 1e-3 konstan. |
| 9 | 10 Sep | MobileNetV3-S pretrained | 224 | 1e-3 | AdamW wd 0.01 | **cosine** | 64 | flip+rot+jitter | tanpa | 10 | **0.827** | beda dari #8 cuma scheduler. **goyangan hilang:** epoch 7–10 naik rapat 0.822 → 0.827 (#8 di rentang yang sama lompat 5,5 poin, anjlok ke 0.762 di epoch 6). tembus 0.82 pertama kali, +1 poin dari #8, terbaik di epoch terakhir dan masih naik. **overfit belum sembuh:** train 0.954, loss valid terendah epoch 4 (0.507) lalu naik ke 0.602. checkpoint `models/mobilenet-cosine.pt`. |
| 10 | 10 Sep | MobileNetV3-S pretrained | 224 | 1e-3 | AdamW **wd 0.05** | cosine | 64 | flip+rot+jitter | tanpa | 10 | **0.837** | beda dari #9 cuma weight decay (0.01 → 0.05). **akurasi naik tapi hafalan gak berkurang:** train epoch 10 tetap 0.954, loss valid nyaris kembar (terendah 0.506 ep 4, akhir 0.601 vs 0.602). val acc konsisten ~1 poin di atas #9 di epoch 8–10 (0.834/0.835/0.837), bukan puncak tunggal. 1 poin = 29 foto — kecil, tapi akhir run tenang berkat cosine. checkpoint `models/mobilenet-wd005.pt`. pertama kali grafik ketulis otomatis. |
| 11 | 10 Sep | MobileNetV3-S pretrained | 224 | 1e-3 | AdamW wd 0.05 | cosine | 64 | flip+rot+jitter | **balanced** | 10 | **0.840** | beda dari #10 cuma class weight (angry 1.21, happy 0.67, neutral 0.97, surprise 1.53). total +0,3 — berisik. **per kelas yang berubah:** angry 0.748→0.770, neutral 0.803→0.817, surprise 0.845→0.863, happy 0.908→0.884. selisih kelas terbaik–terlemah 16 → 11 poin: model berhenti pilih kasih ke kelas terbanyak. loss valid akhir 0.571 (vs 0.601). checkpoint `models/mobilenet-cw.pt`. |
| 12 | 10 Sep | MobileNetV3-S pretrained | 224 | 1e-3 | AdamW wd 0.05 | cosine | 64 | flip+rot+jitter · **dropout 0.5** | tanpa | 10 | 0.836 | beda dari #10 cuma dropout kepala 0.2 → 0.5. **praktis gak ngefek:** per kelas nyaris identik (angry 0.741, happy 0.906, neutral 0.803, surprise 0.849), total sama. train epoch 10 0.945 vs 0.954 — rem hafalan ada tapi tipis. checkpoint `models/mobilenet-drop05.pt`. |
| 13 | 10 Sep | MobileNetV3-S pretrained | 224 | 1e-3 | AdamW wd 0.05 | cosine | 64 | flip+rot+jitter · **label smoothing 0.1** | balanced | 10 | 0.839 | beda dari #11 cuma label smoothing. akurasi & per kelas seri (angry 0.766, happy 0.884, neutral 0.813, surprise 0.868). **pertama kali loss valid gak naik lagi:** turun terus sampai epoch 10 (0.490), lebih rendah dari titik terendah run mana pun. **salah tapi >90% yakin: 181 → 56 foto.** yakin rata-rata waktu salah 78,7% → 68,7%, waktu benar 94,2% → 85,0%. hafalan (train 0.950) tetap — yang berubah cara salahnya, bukan jumlahnya. threshold keyakinan di app harus lebih rendah (~0,6–0,7). checkpoint `models/mobilenet-ls.pt`. |

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

### Kesimpulan Hari 5 (Pafras)

**Pretrained menang telak** (+12–13,5 poin), karena bobot ImageNet udah ngerti bentuk dasar gambar — MobileNet di epoch 1 aja udah ngalahin nilai akhir DeepCNN. **Lebih besar gak berarti lebih baik:** ResNet18 kalah dari MobileNet walau 7x parameter, karena mulai menghafal lebih awal (epoch 2 vs 3). **Yang belum bisa disimpulkan:** kemenangan pretrained sebagian mungkin datang dari ukuran input 224 dibanding 48, karena dua hal itu berubah bersamaan.

## Model final wajah — run #11

`models/mobilenet-cw.pt` · MobileNetV3-Small pretrained · 224 · AdamW lr 1e-3 wd 0.05 · cosine · augmentation · class weight · 10 epoch · val 0.840

Run #10, #11 dan #12 seri di akurasi total (0.836–0.840, selisih ~12 foto dari 2.902). Pemilihnya keseimbangan antar kelas: #11 satu-satunya yang ngangkat kelas terlemah (angry 0.748 → 0.770) dan menyempitkan jarak kelas terbaik–terlemah dari 16 ke 11 poin. Di app, keempat ekspresi dipakai sama rata dan masing-masing memicu beat sendiri — ekspresi yang gagal dikenali artinya satu bunyi yang gak pernah main. Ongkosnya: happy turun 0.908 → 0.884, tetap kelas terkuat.

Catatan kejujuran: model ini dipilih pakai val set, jadi angka 0.840 sedikit optimis. Angka yang jujur datang dari test set, Hari 8.

## Hasil akhir (test set — isi Hari 8)

| Metrik | Nilai |
|--------|-------|
| Accuracy | |
| Macro F1 | |
| Latency / frame | |

Kelas yang dibuang dari mapping + alasannya:
