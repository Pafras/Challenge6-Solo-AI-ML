# Emotion-to-Beatbox

App macOS yang mengubah **ekspresi wajah jadi beat**. Kamera membaca wajah, model memilih ekspresinya, dan tiap ekspresi memainkan pola beatbox yang berulang mengikuti tempo.

```text
KAMERA → CROP WAJAH → EKSPRESI → POLA BEAT → AUDIO
```

Solo challenge Apple Developer Academy (Bali), 7–17 September 2026. Tujuan utamanya adalah **belajar melatih model dengan benar**: arsitektur, hyperparameter, augmentasi, class balance, dan transfer learning dicoba satu per satu, diukur, lalu dibandingkan. App-nya sengaja dibuat sederhana.

## Isi repo

| Folder | Isi |
|---|---|
| `scripts/` | Semua kode Python: data, training, evaluasi, konversi Core ML, uji webcam |
| `EmotionBeatbox/` | Project Xcode app macOS (SwiftUI + Vision + Core ML + AVFoundation) |
| `audio/` | Tabel pola beat (`patterns.json`) dan 12 sampel bunyi beatbox |
| `requirements.txt` | Dependensi Python, dengan alasan tiap versi yang di-pin |

Dataset, checkpoint (`*.pt`), dan model hasil konversi di `models/` tidak masuk git karena ukurannya. Yang ikut di-commit hanya salinan yang dibundel app di `EmotionBeatbox/EmotionBeatbox/Resources/`.

## Dua model

### 1. Pengenal ekspresi wajah (dipakai di app)

| | |
|---|---|
| Tugas | crop wajah → **angry / happy / neutral / surprise / sad** |
| Arsitektur | MobileNetV3-Small, bobot awal ImageNet, ~1,52 juta parameter |
| Data | FER2013 (5 kelas) + 1.200 crop webcam dari 4 orang yang direkam dengan izin |
| Hasil FER2013 test | **akurasi 0,752 · macro F1 0,745** (test set dibuka sekali, tebakan acak 0,20) |
| Orang yang tidak pernah dilatih | 0,660 → **0,767** setelah fine-tune |

Per kelas di FER2013 test:

| kelas | precision | recall | F1 |
|---|---|---|---|
| angry | 0,701 | 0,648 | 0,674 |
| happy | 0,901 | 0,864 | 0,882 |
| neutral | 0,685 | 0,659 | 0,672 |
| surprise | 0,831 | 0,897 | 0,863 |
| sad | 0,607 | 0,670 | 0,637 |

**Yang dijalankan app: gabungan (late fusion).** CNN di atas digabung dengan Landmark MLP, yang hanya melihat 76 titik wajah dari Vision tanpa piksel sama sekali:

```text
p = 0.55 · p_cnn + 0.45 · p_landmark
```

Keduanya salah di foto yang berbeda, jadi gabungannya lebih baik dari masing-masing: FER valid 0,762 → 0,774, orang baru 0,767 → 0,780. Di webcam live, frame yang lolos threshold tapi salah turun dari 7,7% ke 5,0%. Wajah yang titiknya tidak ditemukan Vision tetap memakai CNN saja.

> Model ini membaca **gerakan wajah**, bukan emosi yang dirasakan. Senyum bisa muncul saat gugup. Jangan dipakai untuk menilai orang.

### 2. Klasifikasi bunyi beatbox (latihan kedua, tidak dipakai di app)

| | |
|---|---|
| Tugas | clip 0,5 detik → **clap / hihats / kick / snare** |
| Arsitektur | AudioCNN 3 blok conv di atas log-mel spectrogram 64×44, 23.780 parameter, dari nol |
| Data latih | Pafras/beatbox-bucket, 4.014 clip dari 125 rekaman, split **per rekaman** |
| Hasil jujur | **0,517 ± 0,038** (3 seed) pada 14 orang awam dari dataset AVP |

Temuan utamanya: 0,99 di data bucket sendiri hanya mengukur bucket itu. Di suara orang lain dengan mic lain, akurasi jatuh ke ~0,52, cuma 4 poin di atas selalu menebak "hihats". Padding nol juga ternyata membuat model curang ("bunyi pendek lalu hening = hihat"), jadi clip diisi noise dari clip itu sendiri. Masalah terbesarnya keragaman data, bukan setelan training.

## Perjalanan eksperimen (model wajah)

Tiap run mengubah satu variabel. Seed dan split tetap, test set hanya dibuka sekali di akhir.

| # | Run | Yang diubah | Hasil (val) |
|---|---|---|---|
| 0 | TinyCNN | baseline, 2 conv | 0,588 |
| 1 | LR kegedean | lr 1e-2 | macet, tidak belajar |
| 2 | DeepCNN | 3 conv | 0,678 |
| 3 | MobileNet | pretrained ImageNet, 224 px | **0,813**, transfer learning menang jauh |
| 4 | ResNet18 | pretrained, 7× lebih besar | 0,797, lebih besar ≠ lebih bagus |
| 5–7 | LR kecil, augmentasi | lr 1e-4, flip/rotasi/kecerahan | overfit tertunda, mentok ~0,81 |
| 8–9 | AdamW, cosine | optimizer, scheduler | goyangan hilang, 0,827 |
| 10–12 | Weight decay, class weight, dropout | wd 0,05, bobot kelas, dropout 0,5 | 0,837–0,840 |
| 13 | Label smoothing | target 0,925 | salah-tapi-yakin 181 → 56 foto |
| 14 | +Sad | 5 kelas | 0,770, sad menyedot neutral |
| 15–16 | SGD | optimizer, lr | 0,827, hafalan jauh lebih sedikit |
| 17 | **Fine-tune bersih** ⭐ | + rekaman 4 orang, lr 1e-4 | orang baru 0,660 → 0,767 |
| 18 | Fine-tune semua | + 2 orang berlabel ragu | lebih jelek: label bersih > jumlah data |
| 19 | Landmark MLP | 76 titik wajah, tanpa piksel | 0,673, −9 poin dari CNN |
| 20 | **Gabungan** ⭐ | CNN + Landmark MLP | 0,774 FER / 0,780 orang baru |

## Cara kerja app

1. **Kamera → wajah.** Vision (`VNDetectFaceRectanglesRequest`) mencari wajah di tiap frame.
2. **Crop dengan margin "fer".** Kotak Vision memotong alis, jadi diperlebar 11% tinggi ke atas dan 5,5% lebar ke kiri-kanan, supaya mirip crop FER2013. Ini mengangkat angry 0,31 → 0,80 di webcam.
3. **Preprocessing sama persis dengan training:** grayscale → **turun ke 48×48 dulu** → naik ke 224×224 → 3 channel → normalisasi ImageNet. Langkah 48 wajib, karena model dilatih dari foto 48 px yang buram, bukan crop webcam yang tajam.
4. **Model.** CNN + Landmark MLP di Core ML, ~13 ms per frame.
5. **Anti-jitter.** Voting mayoritas ~10 frame, threshold keyakinan **0,6**, ekspresi baru harus stabil 0,8 detik, dan pola hanya berganti di awal bar.
6. **Beat.** Tiap ekspresi = satu pola yang berulang (bukan satu bunyi). Ada 5 genre (Techno, EDM, Breakbeat, R&B, Brazil funk), masing-masing dengan tempo dan variasi A/B/C per ekspresi.
7. **Variasi dari kekuatan ekspresi.** Kalau pengguna kalibrasi (tahan wajah datar 2 detik), app mengukur seberapa jauh 76 titik wajah bergerak dari posisi diam, lalu memilih variasi A, B, atau C per bar. Tanpa kalibrasi, variasi berputar A → B → A → C.

Semua bunyi diambil dari dataset beatbox-bucket, tiga take per bunyi yang dimainkan bergantian. Semua pemrosesan kamera terjadi di perangkat, tidak ada gambar yang dikirim ke mana pun.

## Menjalankan

### App

Butuh macOS 27 dan Xcode. Buka `EmotionBeatbox/EmotionBeatbox.xcodeproj`, lalu Run. Model dan sampel sudah dibundel, jadi app jalan tanpa langkah Python apa pun. Izinkan akses kamera saat diminta.

### Python

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

Dataset diletakkan manual (di-ignore git): FER2013 di `data/fer2013/{train,test}/<kelas>/`, beatbox-bucket di `BeatboxAudioDataset/`, AVP v4 di `AVP_Dataset-2/`.

Urutan pipeline model wajah:

| Langkah | Script |
|---|---|
| Cek dataset | `scripts/inspect_dataset.py data/fer2013/train` |
| Buat split valid (seed tetap) | `scripts/make_split.py` |
| Training dari ImageNet | `scripts/train.py --arch mobilenet --classes 5 ...` |
| Rekam wajah & split per orang | `scripts/record_faces.py`, `scripts/make_own_split.py` |
| Fine-tune | `scripts/fine_tune.py` |
| Landmark MLP & gabungan | `scripts/extract_landmarks.py`, `scripts/train_landmarks.py`, `scripts/fuse_landmarks.py` |
| Evaluasi | `scripts/evaluate.py`, `scripts/predictions.py` |
| Konversi Core ML + cek angka | `scripts/convert_coreml.py`, `scripts/convert_landmarks.py` |
| Salin ke app | `scripts/sync_app_assets.py` |
| Uji live di webcam | `scripts/webcam_test.py --fuse` |

Opsi `train.py`: `--arch {tinycnn,deepcnn,resnet18,mobilenet}`, `--lr`, `--epochs`, `--batch-size`, `--seed`, `--augment`, `--optimizer {adam,adamw,sgd}`, `--weight-decay`, `--scheduler {none,cosine}`, `--class-weight`, `--dropout`, `--label-smoothing`, `--classes {4,5}`, `--save`. Ganti opsi = ganti eksperimen, tanpa mengedit kode.

Model audio: `scripts/make_audio_split.py` → `scripts/make_avp_split.py` → `scripts/train_audio.py` → `scripts/evaluate_audio.py`.

## Catatan teknis

- **Konversi Core ML** memakai float16, jadi selisih ~1e-2 pada logit MobileNetV3 dibanding PyTorch itu wajar (float32 cocok sampai ~1e-7). Prediksi yang yakin jauh lebih lebar dari selisih itu.
- **Kesamaan Python ↔ Swift** dicek dengan 43 "crop emas" dari uji webcam: jawaban gabungan cocok 43/43.
- **Vision berubah antar versi macOS.** Setelah upgrade ke macOS 27, kotak wajah bergeser ~9 px dari sebelumnya. Bandingkan hanya di OS yang sama, dan uji ulang live setelah upgrade.
- `torch` di-pin ke 2.7.0 karena itu versi terbaru yang sudah diuji oleh coremltools 9.0.

## Batasan

- Sad dan neutral paling sering tertukar; sad adalah kelas terlemah di wajah baru.
- Angry sulit diperagakan: banyak orang membuat angry yang nyaris datar.
- Fine-tune memakai 4 orang dan divalidasi pada 1 orang, jadi angka orang-baru sedikit optimis.
- Kamera dari bawah, cahaya redup, dan wajah tertutup menurunkan akurasi.
- Model audio belum layak dipakai di luar latihan ini.
