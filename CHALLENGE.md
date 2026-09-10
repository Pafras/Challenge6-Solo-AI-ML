# Emotion-to-Beatbox Sprint

Solo challenge Academy · 7–17 September 2026 · **submit Kamis 17 Sep**
Ingredient: Training models with PyTorch (inti, **2 model**) · Integrating models with Core ML (jalur wajib ke app) · Foundation Models & native AI (opsional, kalau sempat)

> Ekspresi wajah jadi alat musik. Muka user ngendaliin beatbox, terus ada quest yang minta user niruin urutan ekspresi buat ngerecreate satu beat.

**Tujuan belajar utama: ngerti cara training model, dengan banyak opsi yang beneran diutak-atik dan dibandingin** — arsitektur, hyperparameter, augmentation, class balance, transfer learning. Bukan sekadar dapet satu model yang jalan. Aplikasi macOS tetap dibuat karena itu bagian dari challenge, tapi sengaja dibikin simple.

Pipeline: `webcam → Vision face detect → CNN ekspresi (PyTorch) → coremltools → Core ML on-device → emosi → beat token (K/H/S) → AVFoundation audio → quest → score`

Konteks lengkap + riwayat pivot: lihat `CLAUDE.md`. Spec teknis penuh: `docs/spec-emotion-beatbox.md`. Hasil tiap eksperimen training: `docs/experiments.md`.

**Dua model** (arahan mentor, 10 Sep):
1. **Ekspresi wajah** — gambar 48×48 → 4 emosi. Dipakai di app.
2. **Klasifikasi audio beatbox** — suara → kick/hihats/snare/clap. Latihan training kedua, **tidak dipakai di app** (app gak punya input mic). Deliverable-nya model card + tabel eksperimen.

**Pembagian hari:** Hari 4–6 training model wajah. Hari 7–8 training model audio + finalisasi dua-duanya. Hari 9–10 app macOS (2 hari, simple). Hari 11 submit.

**Asumsi yang masih bisa diubah:** mulai 4 kelas emosi (happy/neutral/angry/surprise); dataset public (FER2013 dkk); beat generator rule-based, ML beat generator dibuang.

**Catatan jadwal:** Hari 1–2 kepakai arah lama (guitar chord), deadline maju ke Kamis 17 Sep. Rencana 12 hari di spec dipadatkan jadi 9 hari, Hari 3–11.

> **Cara pakai bareng Claude Code:** minta Claude Code mencentang tugas (mis. "centang tugas coremltools di hari 9"), lalu `python3 build_tracker.py` untuk regenerate `tracker.html`.

## Aturan eksperimen

- Satu run = satu baris di `docs/experiments.md`. Config, metrik, catatan. Kalau nggak dicatat, nggak kehitung.
- Ganti satu variabel per run. Ganti dua, nggak akan tau mana yang ngefek.
- Seed dikunci, split dikunci. Kalau dua-duanya berubah tiap run, angkanya nggak bisa dibandingin.
- Test set dibuka sekali di akhir. Semua tuning pakai val set.

## Risiko utama

- **Ngoprek tanpa nyatet.** Risiko nomor satu buat tujuan belajar ini. 20 run tanpa tabel hasil = nggak belajar apa-apa, cuma sibuk.
- **Prediksi kedip-kedip.** Frame-by-frame prediksi loncat-loncat walau muka diem. Temporal smoothing (majority vote + confidence threshold) wajib.
- **Preprocessing parity.** Crop, resize, normalisasi di Swift harus persis sama dengan training PyTorch. Beda dikit = akurasi anjlok tanpa error apa pun.
- **Emosi mirip.** Angry vs disgust, fear vs surprise gampang ketuker. Lihat confusion matrix, bukan cuma akurasi total. Kelas yang jelek dibuang dari mapping.
- **App nyedot waktu training.** App cuma dapat 2 hari. Kalau melar, potong quest, jangan potong hari training.
- **Hari terakhir gak punya buffer.** Hari 11 itu hari submit, bukan hari kerja. Model card didraft Hari 8, demo direkam Hari 10 sebagai cadangan. Jangan numpuk apa pun di Kamis.

---

## Investigate

### Hari 3 — Rabu, 9 Sep
`PyTorch`

- [x] Setup env: venv, PyTorch, torchvision, opencv, requirements.txt
- [x] Pilih & download dataset ekspresi (FER2013 / RAF-DB) — cek lisensi + jumlah sample per kelas
- [x] Bikin Dataset + DataLoader, render 1 batch buat verifikasi label bener
- [ ] Kunci 4 kelas emosi (tes di depan kamera dulu, mana yang bisa dipasang on-demand) + mapping ke K/H/S
- [x] Kunci seed + split train/val/test, simpan split-nya ke file biar konsisten antar run

## Act — Minggu 1

### Hari 4 — Kamis, 10 Sep
`PyTorch`

- [x] Tulis training loop sendiri: forward, loss, backward, step, eval per epoch
- [x] Sanity check: overfit 1 batch sampai loss mendekati nol
- [x] Run baseline: CNN kecil, setting default, catat akurasi val
- [x] Bikin `docs/experiments.md`, isi baris pertama = baseline
- [x] Bikin script train yang baca config, biar ganti opsi nggak perlu edit kode

### Hari 5 — Jumat, 11 Sep
`PyTorch` · wajah

- [x] Eksperimen arsitektur: CNN kecil vs CNN lebih dalam
- [x] Eksperimen arsitektur: ResNet18 pretrained (fine-tune)
- [x] Eksperimen arsitektur: MobileNet pretrained (fine-tune)
- [x] Bandingin akurasi vs jumlah parameter vs waktu training, catat semua
- [x] Simpulin: from-scratch vs transfer learning, menang mana dan kenapa

### Hari 6 — Sabtu, 12 Sep
`PyTorch` · wajah

- [ ] Eksperimen learning rate: 3 nilai, lihat kurva loss-nya
- [ ] Eksperimen optimizer: SGD+momentum vs Adam vs AdamW
- [ ] Eksperimen augmentation: tanpa vs flip vs flip+rotate+brightness
- [x] Eksperimen class weight buat imbalance 2.28x
- [x] Pilih config final model wajah, catat di tabel

### Hari 7 — Minggu, 13 Sep
`PyTorch` · audio

- [ ] Parse label dari nama file (`kick-050-a-6.wav` → `kick`), cek jumlah per kelas
- [ ] Split train/valid/test sendiri, seed dikunci (dataset ini gak punya split)
- [ ] Audio → mel-spectrogram pakai torchaudio, kunci parameternya (n_mels, hop, durasi)
- [ ] Render beberapa spectrogram, lihat apa kick dan hihat beda secara kasat mata
- [ ] Dataset + DataLoader + overfit 1 batch

### Hari 8 — Senin, 14 Sep
`PyTorch` · dua-duanya

- [ ] Training penuh model audio, 2–3 eksperimen (arsitektur / n_mels / augmentation)
- [ ] Buka test set model audio, confusion matrix
- [ ] Buka test set model wajah — sekali ini aja. Accuracy, precision, recall, F1
- [ ] Pipeline realtime Python wajah: webcam → face detect → model → emosi + smoothing
- [ ] Draft dua model card selagi angkanya masih anget

## Act — Minggu 2

### Hari 9 — Selasa, 15 Sep
`Core ML`

- [ ] Convert PyTorch → Core ML pakai coremltools
- [ ] Verifikasi output Core ML sama dengan PyTorch di input yang sama (bandingin numerik)
- [ ] Setup project SwiftUI macOS + camera permission
- [ ] Vision face detection + Core ML prediksi, emosi live ke layar

### Hari 10 — Rabu, 16 Sep
`Core ML`

- [ ] Siapin sample audio: kick.wav, snare.wav, hihat.wav
- [ ] Audio engine: BPM clock + sequencer token → suara, loop mulus
- [ ] Mapping emosi → beat pattern (rule-based, tabel biasa)
- [ ] Ekspresi ganti → pattern ganti di step berikutnya, bukan restart loop
- [ ] **Rekam demo video cadangan sore ini** — free mode aja udah cukup
- [ ] *(stretch)* Quest: 1–2 target sequence hardcoded + score sequence match

## Cooldown

### Hari 11 — Kamis, 17 Sep *(submit)*

- [ ] Polish seadanya: benerin yang paling kerasa ganggu, jangan nambah fitur
- [ ] Finalisasi model card + rapikan `docs/experiments.md` jadi cerita
- [ ] Rekam demo final (kalau yang Rabu udah cukup, pakai itu)
- [ ] Submit ke Academy + refleksi

## Opsional (cuma kalau semua di atas kelar)

Tiga pintu masuk ke generator beat yang sama. Realtime kamera itu P0; dua di
bawahnya cuma ngeganti sumber emosinya, generator dan audio engine gak
disentuh sama sekali.

- [ ] **Input gambar** — pilih foto dari galeri, lewat pipeline yang sama (~30 menit)
- [ ] **Input teks + Foundation Models** — kalimat → emosi → beat (~2–3 jam). Ini yang ngasih ingredient 3★ pekerjaan beneran, bukan tempelan komentar.
- [ ] Quest + scoring lengkap (timing accuracy, 3 level kesulitan)
- [ ] Foundation Models: sequence + score → komentar natural
- [ ] Emotion intensity buat ngendaliin BPM / density
