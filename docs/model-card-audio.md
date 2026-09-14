# Model Card — Klasifikasi Bunyi Beatbox

Model kedua proyek Emotion-to-Beatbox. **Tidak dipakai di app**; dibuat sebagai latihan training kedua. Detail tiap run ada di [`experiments.md`](experiments.md), penjelasan metode di [`technical-notes.md`](technical-notes.md).

## Ringkasan

| | |
|---|---|
| Tugas | clip bunyi beatbox 0,5 detik → **clap / hihats / kick / snare** |
| Arsitektur | AudioCNN: 3 blok (conv 3×3 + BatchNorm + ReLU + max-pool), rata-rata global, 1 lapisan linear · **23.780 parameter**, dilatih dari nol |
| Konfigurasi final | **Padding noise (dibenerin)**, 3 seed: `models/audio-a2b.pt`, `audio-a2b-s1.pt`, `audio-a2b-s2.pt` |
| Angka jujur | **0,517 ± 0,038** pada 14 orang yang tidak pernah dipakai (AVP test) |
| Tanggal | 14 Sep 2026 |

## Kegunaan

**Dimaksudkan untuk:** latihan dan perbandingan metode training pada data audio — split yang benar, fitur spectrogram, eksperimen terkontrol, dan pengukuran lintas dataset.

**Tidak dimaksudkan untuk:** dipakai di app atau produk apa pun. Di suara orang awam, model ini hanya sedikit lebih baik dari tebakan paling sederhana.

## Input dan preprocessing

1. Audio → mono, 22.050 Hz.
2. Potong ke 0,5 detik; kalau lebih pendek, **diisi noise** setingkat 10 ms terakhir clip itu sendiri (bukan nol).
3. Log-mel spectrogram **64 × 44** (n_fft 1024, hop 256, top_db 80).
4. Normalisasi dengan statistik train untuk pipeline ini: mean −19,2, std 14,5.

Parameter padding dan normalisasi disimpan di checkpoint, dan `scripts/evaluate_audio.py` membangun ulang pipeline dari situ.

## Data latih

- **Pafras/beatbox-bucket**, split train: **4.014 clip dari 125 rekaman**. Tiap rekaman punya ~32 varian hampir kembar, jadi keragaman sebenarnya jauh lebih kecil dari jumlah clip-nya.
- Split valid dibuat **per rekaman** (20% rekaman per kelas, seed 42): 1.044 clip, 33 rekaman.
- Training: Adam lr 1e-3, batch 32, 15 epoch, tanpa augmentasi tambahan, tanpa class weight. Epoch terbaik dipilih dari avp-valid.

## Data evaluasi

| Set | Isi | Dipakai untuk |
|---|---|---|
| valid bucket | 1.044 clip, 33 rekaman | cek dasar (sudah jenuh: ~0,99) |
| **avp-valid** | AVP, 14 orang, 5.025 clip, mic MacBook | membandingkan eksperimen, memilih epoch |
| **avp-test** | AVP, 14 orang lain, 4.748 clip | angka final, dibuka sekali |
| test bucket | 575 clip, 18 rekaman | dibuka sekali |

AVP (Amateur Vocal Percussion v4, Zenodo) adalah 28 orang awam yang tidak ada di data latih. AVP tidak punya clap; hihat tertutup (`hhc`) dan terbuka (`hho`) dilaporkan terpisah.

## Hasil

**avp-test — 3 seed, rata-rata ± simpangan** (bukan seed terbaik):

| | seed 42 | seed 1 | seed 2 | **rata-rata ± sd** |
|---|---|---|---|---|
| akurasi | 0,500 | 0,561 | 0,492 | **0,517 ± 0,038** |

| label AVP | recall |
|---|---|
| kick (`kd`) | **0,733 ± 0,063** |
| hihat tertutup (`hhc`) | 0,443 ± 0,075 |
| hihat terbuka (`hho`) | 0,437 ± 0,074 |
| snare (`sd`) | 0,427 ± 0,063 |

**Patokan pembanding di avp-test:** selalu menebak "hihats" = 0,473. Model hanya **+4 poin** di atasnya.

**Test bucket:** 0,906 ± 0,111 (0,937 / 0,783 / 0,998). Clap, hihats, snare 1,00 di semua seed; kick 0,81 / 0,34 / 0,99.

## Temuan penting

1. **0,99 di valid bucket hanya mengukur bucket.** Di suara orang lain dengan mic berbeda, akurasi jatuh ke ~0,52.
2. **Padding nol membuat model curang.** Model awal belajar "bunyi pendek lalu hening digital = hihat"; mengganti padding dengan noise di clip yang sama menjatuhkan recall hihat 0,991 → 0,502. Karena itu padding noise dipakai.
3. **Panjang clip membocorkan label.** Menebak dari durasi saja mendapat 0,542 di valid bucket (kelas terbanyak 0,309), tapi 0,229 di AVP.
4. **Satu seed tidak cukup.** Keunggulan "paling seimbang" dari Normalisasi volume di seed 42 tidak terulang di seed 1 dan 2. Setelah 3 seed, dua pipeline seri (0,506 vs 0,508 di avp-valid).
5. **Rata-rata 5 epoch terakhir memprediksi test**: 0,506 di avp-valid vs 0,517 di avp-test. Angka "epoch terbaik" (0,576) optimis 6 poin karena epoch dipilih di set yang sama.
6. **Masalah utamanya keragaman data, bukan setelan.** Padding, statistik normalisasi, dan normalisasi volume masing-masing hanya menggeser beberapa poin.

## Kapan model gagal

- **Hihat dan snare saling tertukar** di suara orang awam: hihat tertutup → kick 30%, hihat terbuka → snare 29%, snare → hihats 30%.
- **Batas kick–snare tidak stabil antar seed.** Di test bucket, satu seed membaca 66% kick sebagai snare; kegagalannya menumpuk di satu rekaman kick yang panjangnya tidak biasa (0,57 detik, di atas 90% kick data latih).
- Bunyi yang lebih panjang atau lebih "bervokal" dari gaya beatbox bersih di data latih.

## Batasan

- Data latih: **125 rekaman dari segelintir performer** dengan gaya bersih. Tidak ada suara orang awam, tidak ada variasi mic.
- Set evaluasi kecil di level rekaman: valid bucket 10 rekaman kick, test bucket 6 rekaman kick, clap di test hanya 2 rekaman.
- Ketidakpastian avp-valid antar orang: ±0,055 (14 orang, skor per orang 0,45–0,78). Selisih di bawah ~5 poin tidak bisa dibedakan.
- Clap tidak bisa dievaluasi di AVP.

## Yang akan dilakukan dengan waktu lebih

- Melatih dengan sebagian orang AVP (7 dari 14 orang avp-valid menjadi data latih), karena itu satu-satunya langkah yang menyerang penyebab utamanya.
- Augmentasi noise ruangan dari rekaman sendiri, SpecAugment.
- MobileNet pretrained pada spectrogram, untuk perbandingan scratch vs pretrained seperti di model wajah.

## Data dan etika

- AVP dipakai hanya untuk evaluasi; lisensinya tercantum di halaman Zenodo AVP v4 dan perlu dicek sebelum hasil ini dipublikasikan di luar challenge.
- Semua data audio disimpan lokal (`BeatboxAudioDataset/`, `AVP_Dataset-2/` di-ignore git).
