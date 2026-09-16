# Whiteboard (5 menit) — Training Model AI Itu Kayak Tim F1

---

## 1. Training itu apa? *(1 menit)*

**Gambar:**
```
  🧑‍🔧 ENGINEER (kita)          🏎️ PEMBALAP (model AI)
  milih komponen mobil    →   latihan lap demi lap, betulin sendiri
```

**Omongin:**
> "Misalnya kita mau AI yang bisa bedain foto kucing sama anjing. Kita
> kasih ribuan foto plus jawabannya. AI-nya nebak, dikasih tahu salah
> atau bener, terus betulin caranya sedikit. Diulang ribuan kali. Itu
> training."
>
> "Kayak pembalap baru: tiap lap salah dikit, betulin sendiri, lama-lama
> hafal sirkuitnya."
>
> "Tapi mobilnya kayak apa, itu bukan pembalap yang nentuin. Itu tugas
> engineer, yaitu kita."

---

## 2. Komponen mobilnya *(1,5 menit)*

**Gambar:** mobil + 5 panah.

| komponen | di training AI | kalau gak pas |
|---|---|---|
| ⚙️ Mesin | ukuran & jenis model | kekecilan gak kuat, kegedean berat |
| 🪽 Aerodinamika | rem biar AI gak ngapal | kurang liar, kebanyakan lambat |
| 🎯 Setir | seberapa besar AI berubah tiap salah (learning rate) | terlalu sensitif spin, terlalu berat telat |
| 🛞 Ban | variasi data latihan | cuma ban kering, panik pas hujan |
| ⛽ Bensin | data | oplosan (jawaban salah) = mesin rusak |

> "Mobil cuma secepat komponennya yang paling gak pas."

---

## 3. Contoh: kalau satu komponen salah *(1 menit)*

**Gambar:** tiga mobil kecil, tiap mobil satu masalah.

**🎯 Setir terlalu sensitif**
> Tiap koreksi kebesaran, jadi mobil spin terus. AI-nya gak pernah
> stabil. Kalau setirnya terlalu berat, belajarnya lama banget.

**⚙️ Mesin kegedean**
> Cepat banget hafal sirkuit latihan, tapi pas ketemu sirkuit baru
> lambat. AI-nya ngapal foto latihan, bukan paham bedanya kucing sama
> anjing.

**⛽ Bensin oplosan**
> Kalau banyak foto kucing yang dikasih label "anjing", mesin sebagus apa
> pun hasilnya tetap jelek.
> **Data yang bersih lebih penting dari data yang banyak.**

---

## 4. Gimana nentuin yang pas? *(1,5 menit)*

**Gambar:**
```
  🏁 LATIHAN          ⏱️ KUALIFIKASI         🏆 RACE
  buat belajar        buat ngukur            sekali, di akhir
```

> "Waktu latihan gak bisa dipercaya, karena pembalap udah hafal. Yang
> dihitung kualifikasi: foto yang gak pernah dipakai latihan. Terus ganti
> **satu komponen per sesi**, biar ketahuan mana yang bikin cepat."

| gejala | ganti |
|---|---|
| lambat di latihan & kualifikasi | ⚙️ mesin lebih kuat |
| cepat di latihan, lambat di kualifikasi (ngapal) | 🛞 ban, 🪽 aero |
| waktu loncat-loncat | 🎯 setir |

**Penutup:**
> "Tim F1 gak nebak setup. Ganti satu, ukur, catat, ulang. Training AI
> juga gitu. Pembalapnya belajar sendiri, tapi yang bikin dia menang itu
> kerja engineer."

---

## Kalau ditanya

**"Kok AI bisa salah?"** Sama kayak pembalap. Dia cuma bisa sebagus
latihan dan mobilnya. Kalau fotonya buram atau ambigu, manusia aja bisa
salah.

**"Bisa pakai mesin orang lain?"** Bisa, dan sering. Namanya transfer
learning: pakai model yang udah dilatih jutaan foto, tinggal disesuaiin.
Kayak tim kecil F1 beli mesin dari Mercedes.

---

## Papan

```
┌──────────────┬──────────────────┐
│ 1. engineer  │ 2. mobil +       │
│    vs        │    5 komponen    │
│    pembalap  │                  │
├──────────────┼──────────────────┤
│ 3. 3 mobil   │ 4. latihan/kuali/│
│    bermasalah│    race + gejala │
└──────────────┴──────────────────┘
```
