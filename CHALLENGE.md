# Guitar Chord Chart Sprint

Solo challenge Academy · 7–18 September 2026
Ingredient: Foundation Models & native AI (3★) · Training models with PyTorch (4★, ganti Create ML) · Integrating models with Core ML (3★)

> Learn how to train a model that listens to my guitar playing and automatically turns it into a chord chart.

Pipeline: `mic (AVAudioEngine) → mel-spectrogram → CNN (PyTorch) → coremltools → Core ML on-device → chord sequence → chord chart UI → Foundation Models komentar progression`

Konteks lengkap + riwayat pivot: lihat `CLAUDE.md`.

**Asumsi yang masih bisa diubah:** vocabulary mulai dari 5–6 open chord; dataset public dulu, rekaman sendiri buat test set.

> **Cara pakai bareng Claude Code:** minta Claude Code mencentang tugas (mis. "centang tugas coremltools di hari 6"), lalu `python3 build_tracker.py` untuk regenerate `tracker.html`.

## Risiko utama

- **Preprocessing parity.** Mel-spectrogram di Swift harus persis sama dengan training di PyTorch. Beda window / hop / normalisasi = akurasi anjlok tanpa error apa pun. Penyebab kegagalan nomor satu.
- **Segmentasi sequence.** Chord chart butuh tahu *kapan* chord ganti, bukan cuma chord apa. Sliding window + dedupe berurutan; onset detection kalau perlu.
- **Chord mirip.** C vs Am vs Em beda tipis secara harmonik. Confusion matrix wajib dilihat, bukan cuma akurasi total.
- **Domain gap.** Dataset public direkam beda mic dan ruangan dari iPhone kamu.

---

## Engage

### Hari 1 — Senin, 7 Sep
`Foundation Models`

- [ ] Kunci scope: chord chart dari rekaman, 5–6 open chord
- [ ] Tulis 1 paragraf problem statement / use-case
- [ ] Coba Foundation Models framework — 1 prompt sederhana
- [ ] Setup env Python: PyTorch, torchaudio, coremltools

## Investigate

### Hari 2 — Selasa, 8 Sep
`PyTorch`

- [ ] Pilih & verifikasi dataset chord (cek lisensi + jumlah sample per kelas)
- [ ] Load 1 file audio, render mel-spectrogram, cek bentuk tensor
- [ ] Kunci parameter audio: sample rate, durasi window, n_mels, hop length
- [ ] Riset batasan nyata Foundation Models (bukan cuma dokumentasi)

### Hari 3 — Rabu, 9 Sep
`PyTorch`

- [ ] Bangun Dataset + DataLoader + preprocessing pipeline
- [ ] Sanity check: CNN kecil, overfit 1 batch dulu
- [ ] Training kecil 2 kelas, pastikan loss turun end-to-end
- [ ] Putuskan: lanjut apa adanya, atau ubah fitur/arsitektur

## Act — Minggu 1

### Hari 4 — Kamis, 10 Sep
`PyTorch`

- [ ] Finalisasi 5–6 chord yang dipakai
- [ ] Split dataset train/val/test, cek balance per kelas
- [ ] Tambah augmentation: noise, gain, time shift
- [ ] Training penuh, catat baseline akurasi

### Hari 5 — Jumat, 11 Sep
`PyTorch`

- [ ] Analisis confusion matrix
- [ ] Catat chord yang sering ketuker + kenapa
- [ ] Tuning: learning rate / arsitektur / data untuk kelas lemah
- [ ] Simpan checkpoint v1 dan v2, bandingkan

### Hari 6 — Sabtu, 12 Sep *(buffer)*
`Core ML`

- [ ] Convert PyTorch → Core ML pakai coremltools
- [ ] Verifikasi output Core ML sama dengan PyTorch (bandingkan numerik)
- [ ] Setup project SwiftUI baru + mic permission + AVAudioEngine

### Hari 7 — Minggu, 13 Sep *(buffer)*
`Core ML`

- [ ] Replikasi mel-spectrogram di Swift, cocokkan dengan output PyTorch
- [ ] Jalankan prediksi per window, print chord stream ke console
- [ ] Dedupe chord berurutan jadi sequence (C C C G G → C G)

## Act — Minggu 2

### Hari 8 — Senin, 14 Sep
`Core ML`

- [ ] UI rekam session: start / stop, indikator listening
- [ ] Render chord chart dari sequence hasil deteksi
- [ ] Confidence threshold + handle window tanpa suara

### Hari 9 — Selasa, 15 Sep
`Foundation Models`

- [ ] Integrasi Foundation Models: progression → komentar natural
- [ ] Test beberapa progression (I-V-vi-IV, 12-bar, dll)
- [ ] Rapikan alur end-to-end: rekam → chart → komentar

### Hari 10 — Rabu, 16 Sep
`Core ML`

- [ ] Test di iPhone fisik, bukan simulator
- [ ] Rekam test set sendiri, ukur akurasi di kondisi nyata
- [ ] Perbaiki bug: mic gain, noise ruangan, jarak gitar

## Cooldown

### Hari 11 — Kamis, 17 Sep

- [ ] Rekam video demo (30–60 detik)
- [ ] Tulis model card: dataset, arsitektur, metrik, keterbatasan
- [ ] Kumpulkan screenshot & catatan proses

### Hari 12 — Jumat, 18 Sep

- [ ] Review & polish terakhir
- [ ] Submit ke Academy
- [ ] Refleksi: apa yang dipelajari, apa langkah berikutnya
