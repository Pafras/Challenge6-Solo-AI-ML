# Emotion-to-Beatbox Sprint

Solo challenge Academy · 7–17 September 2026 · **submit Kamis 17 Sep**
Ingredient: Training models with PyTorch (inti) · Integrating models with Core ML (jalur wajib ke app) · Foundation Models & native AI (opsional, kalau sempat)

> Ekspresi wajah jadi alat musik. Muka user ngendaliin beatbox, terus ada quest yang minta user niruin urutan ekspresi buat ngerecreate satu beat.

**Tujuan belajar utama: ngerti cara training model, dengan banyak opsi yang beneran diutak-atik dan dibandingin** — arsitektur, hyperparameter, augmentation, class balance, transfer learning. Bukan sekadar dapet satu model yang jalan. Aplikasi macOS tetap dibuat karena itu bagian dari challenge, tapi sengaja dibikin simple.

Pipeline: `webcam → Vision face detect → CNN ekspresi (PyTorch) → coremltools → Core ML on-device → emosi → beat token (K/H/S) → AVFoundation audio → quest → score`

Konteks lengkap + riwayat pivot: lihat `CLAUDE.md`. Spec teknis penuh: `docs/spec-emotion-beatbox.md`. Hasil tiap eksperimen training: `docs/experiments.md`.

**Pembagian hari:** Hari 3–8 training (6 hari, inti). Hari 9–10 app macOS (2 hari, simple). Hari 11 submit. Deadline maju sehari dari rencana awal — yang dipotong app, bukan training.

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

- [ ] Setup env: venv, PyTorch, torchvision, opencv, requirements.txt
- [ ] Pilih & download dataset ekspresi (FER2013 / RAF-DB) — cek lisensi + jumlah sample per kelas
- [ ] Bikin Dataset + DataLoader, render 1 batch buat verifikasi label bener
- [ ] Kunci 4 kelas emosi (tes di depan kamera dulu, mana yang bisa dipasang on-demand) + mapping ke K/H/S
- [ ] Kunci seed + split train/val/test, simpan split-nya ke file biar konsisten antar run

## Act — Minggu 1

### Hari 4 — Kamis, 10 Sep
`PyTorch`

- [ ] Tulis training loop sendiri: forward, loss, backward, step, eval per epoch
- [ ] Sanity check: overfit 1 batch sampai loss mendekati nol
- [ ] Run baseline: CNN kecil, setting default, catat akurasi val
- [ ] Bikin `docs/experiments.md`, isi baris pertama = baseline
- [ ] Bikin script train yang baca config, biar ganti opsi nggak perlu edit kode

### Hari 5 — Jumat, 11 Sep
`PyTorch`

- [ ] Eksperimen arsitektur: CNN kecil vs CNN lebih dalam
- [ ] Eksperimen arsitektur: ResNet18 pretrained (fine-tune)
- [ ] Eksperimen arsitektur: MobileNet pretrained (fine-tune)
- [ ] Bandingin akurasi vs jumlah parameter vs waktu training, catat semua
- [ ] Simpulin: from-scratch vs transfer learning, menang mana dan kenapa

### Hari 6 — Sabtu, 12 Sep
`PyTorch`

- [ ] Eksperimen learning rate: 3 nilai, lihat kurva loss-nya
- [ ] Eksperimen optimizer: SGD+momentum vs Adam vs AdamW
- [ ] Eksperimen scheduler: tanpa scheduler vs StepLR vs CosineAnnealing
- [ ] Eksperimen batch size + efeknya ke lr
- [ ] Catat semua, tandai kombinasi terbaik sejauh ini

### Hari 7 — Minggu, 13 Sep
`PyTorch`

- [ ] Eksperimen augmentation: tanpa augment vs flip vs flip+rotate+brightness
- [ ] Eksperimen class imbalance: tanpa penanganan vs class weight vs oversample
- [ ] Eksperimen image size (48 vs 96 vs 224) — akurasi naik seberapa, latency naik seberapa
- [ ] Eksperimen regularisasi: dropout / weight decay / early stopping
- [ ] Update tabel eksperimen, lihat pola mana yang konsisten

### Hari 8 — Senin, 14 Sep
`PyTorch`

- [ ] Pilih config final dari tabel, training penuh sekali lagi
- [ ] Buka test set — sekali ini aja. Catat accuracy, precision, recall, F1
- [ ] Confusion matrix: kelas mana yang ketuker, buang dari mapping kalau parah
- [ ] Pipeline realtime Python: webcam → face detect → model → emosi + temporal smoothing
- [ ] Ukur latency per frame, pastikan cukup buat realtime
- [ ] Draft model card selagi angkanya masih anget (dataset, arsitektur, metrik, keterbatasan)

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

- [ ] Quest + scoring lengkap (timing accuracy, 3 level kesulitan)
- [ ] Foundation Models: sequence + score → komentar natural
- [ ] Emotion intensity buat ngendaliin BPM / density
