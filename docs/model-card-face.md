# Model Card — Pengenal Ekspresi Wajah

Model yang dipakai di app Emotion-to-Beatbox. Detail tiap run ada di [`experiments.md`](experiments.md), penjelasan metode di [`technical-notes.md`](technical-notes.md).

## Ringkasan

| | |
|---|---|
| Tugas | crop wajah → **angry / happy / neutral / surprise / sad** (urutan output neuron 0–4) |
| Arsitektur | MobileNetV3-Small, bobot awal ImageNet · **~1,52 juta parameter** |
| Model final | **Fine-tune bersih** · `models/mobilenet-finetune-bersih.pt` |
| Angka test | **akurasi 0,752 · macro F1 0,745** (FER2013 test, 6.043 foto, dibuka sekali) |
| Dipakai di app | ya — dikonversi ke Core ML (Hari 9) |
| Tanggal | 14 Sep 2026 |

## Kegunaan

**Dimaksudkan untuk:** memilih pola beat di app berdasarkan ekspresi wajah pengguna, secara real-time, dari webcam MacBook.

**Yang dibaca model adalah gerakan wajah (ekspresi), bukan emosi yang dirasakan.** Senyum bisa muncul saat gugup, cemberut saat fokus; wajah saja tidak cukup untuk memastikan perasaan seseorang (Barrett dkk., 2019).

**Tidak dimaksudkan untuk:** menilai emosi atau kondisi mental seseorang; keputusan apa pun tentang orang (rekrutmen, pengawasan, pendidikan, kesehatan); dipakai pada anak-anak (tidak diuji); merekam atau menganalisis orang tanpa izin.

## Input dan preprocessing

1. Kamera → deteksi wajah dengan Apple Vision (`VNDetectFaceRectanglesRequest`).
2. Kotak wajah **diperlebar dengan margin "fer"**: atas +11% tinggi kotak, kiri-kanan masing-masing +5,5% lebar, bawah tetap — supaya dahi dan alis ikut, seperti crop FER2013.
3. Grayscale → **48×48 dulu** (INTER_AREA) → 224×224 → diulang jadi 3 channel → normalisasi mean/std ImageNet.
4. Output 5 logit → softmax.

**Di app:** voting mayoritas ~10 frame + **threshold keyakinan 0,6**. Pola beat hanya berganti di awal bar, dan hanya kalau prediksinya stabil dan lolos threshold.

Langkah 48×48 dan margin wajib direplikasi persis di Swift; beda preprocessing menurunkan akurasi tanpa error. "Gambar emas" dari uji webcam disimpan untuk mengecek kesamaan Python vs Swift.

## Data latih

**Tahap 1 — +Sad (dari ImageNet ke FER2013):** 20.550 foto FER2013 train, 5 kelas, 48×48 grayscale. AdamW lr 1e-3, weight decay 0,05, cosine annealing, augmentasi (flip, rotasi 10°, kecerahan/kontras), class weight seimbang, label smoothing 0,1, batch 64, 10 epoch.

**Tahap 2 — Fine-tune bersih (dari +Sad ke wajah sungguhan):** FER2013 train **+ 1.200 crop rekaman 4 orang** (5 ekspresi × 60 frame per orang, direkam lewat webcam dengan izin), rekaman diulang 10× supaya tidak tenggelam di antara foto FER. lr 1e-4 (sepersepuluh), 5 epoch; epoch terbaik (1) dipilih pada orang validasi. Dua orang lain ikut direkam tapi dikeluarkan setelah dicek per frame (tertawa saat diminta *sad* dan *neutral*; *angry* yang nyaris datar).

## Data evaluasi

| Set | Isi | Dipakai untuk |
|---|---|---|
| FER valid | 3.626 foto, 15% dari train, diacak per kelas | memilih konfigurasi |
| **FER test** | 6.043 foto | angka final, dibuka sekali |
| orang validasi | 1 orang, 300 frame rekaman, tidak pernah dilatih | mengukur fine-tune, memilih epoch |
| uji webcam live | beberapa sesi, termasuk wajah yang tidak pernah dilatih | kondisi pakai sebenarnya |

## Hasil

**FER2013 test (dibuka sekali):**

| kelas | precision | recall | F1 |
|---|---|---|---|
| angry | 0,701 | 0,648 | 0,674 |
| happy | 0,901 | 0,864 | **0,882** |
| neutral | 0,685 | 0,659 | 0,672 |
| surprise | 0,831 | **0,897** | 0,863 |
| sad | 0,607 | 0,670 | **0,637** |
| **total** | | **akurasi 0,752** | **macro F1 0,745** |

Angka valid-nya 0,762 — test hanya 1 poin di bawah, jadi pemilihan model lewat valid set tidak banyak membuat angkanya optimis. Tebakan acak untuk 5 kelas: 0,20.

**Efek fine-tune pada orang yang tidak pernah dilatih** (frame yang sama persis):

| | sebelum (+Sad) | sesudah |
|---|---|---|
| keseluruhan | 0,660 | **0,767** |
| sad | 0,35 | 0,60 |
| surprise | 0,65 | 0,90 |
| angry | 0,60 | 0,75 |
| neutral | 0,80 | 0,72 |
| FER valid (cek "tidak lupa") | 0,770 | 0,762 |

**Uji webcam live:**
- Margin "fer" dibanding kotak Vision apa adanya: angry 0,31 → 0,80, surprise 0,53 → 0,88 (alis ikut di crop); neutral dan happy turun ke 0,86.
- Wajah Pafras (tidak pernah dilatih), +Sad vs Fine-tune bersih: 0,71 vs 0,69 rata-rata per pose — seri. Variasi cara berpose antar sesi lebih besar dari selisih antar model.
- Threshold 0,6: ~67% frame lolos, ~87% di antaranya benar.
- Latensi: ~22 ms per frame untuk deteksi Vision + model, di Python (MPS). Core ML diukur di Hari 9.

## Kapan model gagal

- **Sad dan neutral saling tertukar.** Di FER, 21% neutral ditebak sad; di webcam kebalikannya — sad ditebak neutral. Sad adalah kelas terlemah di wajah baru.
- **Angry lemah dan sulit diperagakan.** Banyak orang membuat "angry" yang nyaris datar; angry ↔ neutral dan angry → sad adalah kesalahan yang sering.
- **Crop tanpa dahi.** Kotak Vision tanpa margin memotong alis; angry dan surprise anjlok.
- **Kamera dari bawah** (laptop terlalu rendah) dan pencahayaan redup.
- **Wajah tertutup** (tangan di depan mulut, menoleh jauh) — terlihat di rekaman.

## Faktor yang relevan

- Sudut kamera, pencahayaan, jarak ke kamera, margin crop.
- Ekspresi yang diperagakan vs spontan: semua data rekaman adalah ekspresi yang diminta.
- Peserta rekaman termasuk yang berkacamata dan berhijab; efeknya tidak diukur terpisah karena jumlah orangnya kecil.
- Komposisi demografis FER2013 tidak kami ukur.

## Batasan

- **Ekspresi ≠ emosi.** Model tidak tahu perasaan seseorang.
- **Sedikit orang:** fine-tune memakai 4 orang, divalidasi pada 1 orang; epoch dipilih pada orang yang sama, jadi 0,767 sedikit optimis.
- **Satu seed per run** di model wajah; selisih 1–2 poin antar run bisa kebetulan.
- **Uji live berisik:** satu orang di satu ruangan tidak cukup untuk membandingkan dua model.
- FER2013 adalah foto 48×48 grayscale dari internet; wajah webcam berbeda sudut, cahaya, dan ketajamannya.

## Data, privasi, dan etika

- Semua orang yang direkam memberi izin. Rekaman disimpan **hanya di laptop** (`data/own_faces/`, di-ignore git) dan tidak dipublikasikan; dapat dihapus atas permintaan. Rekaman dua orang yang dikeluarkan disimpan terpisah (`data/own_faces_rejected/`) sampai dihapus oleh pemiliknya.
- App memproses kamera secara lokal di perangkat; tidak ada gambar yang dikirim ke mana pun.

## Riwayat model

Label smoothing (4 kelas, valid 0,839) → +Sad (5 kelas, valid 0,770) → **Fine-tune bersih** (final). Label smoothing 4 kelas disimpan sebagai cadangan (`models/mobilenet-ls.pt`).
