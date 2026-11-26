# Hasil Glukosa 18 — uji keanggotaan pra-latih langsung dan celah LVH per model (protokol §F langkah 16), 26 November 2026

Sumber angka: keluaran `glukosa_replikasi/uji_keanggotaan.py` (venv-ecgfm, GPU; bootstrap 2000, benih
20261120) dan `glukosa_replikasi/celah_lvh.py` (venv-gpu; prediksi tersimpan langkah 15), keduanya pada kode
komit c254d67 tanpa perubahan. Protokol dicatat di komit 8d30713 sebelum kode ditulis; kode dikomit dan diuji asap pada
data sintetis sebelum skor apa pun dihitung pada data nyata. Skor per rekaman ada di
`data/replikasi/keanggotaan_b20261120.npz`, di luar repo dan tidak dibuka di sesi asisten; yang keluar
hanya agregat. Bagian "Keluaran skrip" disambung langsung dari log; jangan diketik ulang.

**Status: PRA-SPESIFIKASI untuk M1 dan kontrol negatif** (aturan di komit bertanggal repositori kerja, bukan rilis
publik). Semua bacaan lain deskriptif: AUROC keanggotaan, D per bagian, kosinus segmen (langkah 16e), dan celah LVH
(langkah 16f, pasca-hasil atas telaah internal 8).

## 0. Catatan pembacaan

- **Kontrol negatif LOLOS.** D plasebo antara dua paruh acak pasien tidak dilihat −0,0009 [−0,0101; 0,0083]; selang
  memuat nol, jadi M1 dibaca.
- **M1: keanggotaan pra-latih TIDAK terkonfirmasi.** Kehilangan InfoNCE ECG-FM pretrained rata-rata 0,9118 [0,9075;
  0,9163] pada pasien dilihat (22.058) dan 0,9173 [0,9125; 0,9223] pada pasien tidak dilihat (22.058); D = 0,0055
  [−0,0008; 0,0122], batas bawah −0,0008 ≤ 0. Arah titik sesuai keanggotaan (rekaman tidak dilihat sedikit lebih sulit
  direkonstruksi), besarnya ±0,6 % dari rerata kehilangan, tetapi selangnya memuat nol. `valid` dan `test` terpisah
  sejalan: 0,0054 [−0,0026; 0,0138] dan 0,0056 [−0,0022; 0,0137].
- **Akibat menurut aturan (c):** naskah menyatakan batas 0,002 bersyarat pada daftar split yang menandai keanggotaan
  pra-latih, dengan dua penjelasan yang tidak dapat dipisahkan oleh M1: daftar tidak cocok dengan pra-latih, atau
  keanggotaan tidak meninggalkan jejak yang terdeteksi pada kehilangan InfoNCE. Judul dan abstrak memuat syarat itu.
- **AUROC keanggotaan** dari −kehilangan 0,504 [0,499; 0,510]: kehilangan per rekaman praktis tidak membedakan anggota
  dari bukan anggota.
- **Kosinus segmen (sekunder deskriptif, pendekatan suku CMSC):** dilihat − tidak dilihat 0,0020 [0,0017; 0,0023];
  selangnya jauh dari nol. Dua segmen 5 s dari rekaman pasien dilihat lebih mirip satu sama lain di ruang token ECG-FM
  daripada pada pasien tidak dilihat. Ini satu-satunya jejak keanggotaan yang terlihat, dan muncul pada suku yang
  memang dilatih untuk menarik segmen pasien yang sama berdekatan. Bacaan ini condong ke "daftar menandai keanggotaan",
  tetapi protokol membuatnya deskriptif: tidak ada plasebo kosinus yang ditetapkan sebelumnya, rerata kosinus per
  kelompok tidak dicetak sehingga besarnya relatif terhadap sebarannya tidak diketahui, dan aturan M1 tidak berubah
  karenanya. Naskah melaporkannya di samping M1, tidak sebagai pengganti.
- **Celah LVH per model, subset pasien di luar kohort (langkah 16f; 42.585 dilihat dengan 1.649 positif, 10.828 tidak
  dilihat dengan 394 positif):** ECG-JEPA 0,004 [−0,011; 0,022]; ECG-FM pretrained 0,030 [0,005; 0,059]; ECG-FM
  *fine-tuned* 0,023 [0,003; 0,044]. DiD 0,026 di Hasil 17 (= 0,030 − 0,004) berasal dari celah ECG-FM sendiri, bukan
  dari celah negatif pembanding; penjelasan artefak pembanding (telaah internal 8, devil's advocate X6 / methods X4) tidak
  didukung. AUROC LVH ECG-FM pretrained 0,897 dilihat dan 0,867 tidak dilihat, di bawah ECG-JEPA (0,940 dan 0,936),
  jadi tugas ini punya ruang di bawah plafon. Tetap deskriptif dan pasca-hasil: subset ini tidak punya kontrol positif
  yang bekerja pada tugas berlabel (Hasil 17), dan ia satu sel di antara banyak sel sekunder.
- **Tumpang-tindih pasien replikasi (dihitung dari angka Hasil 17, untuk naskah 2.8):** 416.503 − 53.413 = 363.090
  rekaman (87,2 %) berasal dari 109.971 − 29.391 = 80.580 pasien yang juga punya EKG di kohort glukosa.
- **Bacaan untuk naskah:** batas 0,002 tetap hasil replikasi pra-registrasi, tetapi dibaca sebagai batas efek
  keanggotaan hanya dengan syarat daftar split menandai pra-latih; kosinus segmen dilaporkan sebagai petunjuk deskriptif
  bahwa daftar itu cocok. Sinyal LVH di subset pasien di luar kohort naik dari "kemungkinan artefak pembanding" menjadi
  "celah milik ECG-FM sendiri" dan masuk abstrak sebagai batas cakupan klaim.

## 1. Catatan proses

- Dijalankan sekali, 26 Nov 2026 07:41–±08:55, exit 0 untuk kedua skrip. 44.116 rekaman dinilai (satu per pasien);
  tidak ada yang gagal dibaca atau berskor tak terhingga; tidak ada pasien dengan rekaman di `train` dan di
  `valid`/`test`, jadi aturan penempatan pasien ganda di `pilih_unit` tidak terpakai.
- Laju ±5.000 rekaman per 7 menit (6 pekerja pembaca ZIP, GPU RTX 5060). Tidak ada gangguan, jalan ulang, atau
  perubahan kode di antara komit c254d67 dan jalan ini.
- `celah_lvh.py` memakai prediksi di luar lipatan langkah 15 (benih lipatan 20261120) tanpa melatih ulang probe;
  kelima tugas diulang bersama dalam bootstrap agar ulangan yang terpakai sama dengan langkah 15 (2.000 terpakai).

## Keluaran skrip

### uji_keanggotaan.py


Uji keanggotaan langkah 16; benih 20261120; K = 4 undian topeng.

Pasien dengan rekaman di train dan di valid/test: 0.

| Split | Pasien (satu rekaman per pasien) |
| --- | --- |
| train | 22058 |
| valid | 10986 |
| test | 11072 |
07:53 rekaman dinilai 5000
08:00 rekaman dinilai 10000
08:08 rekaman dinilai 15000
08:15 rekaman dinilai 20000
08:23 rekaman dinilai 25000
08:31 rekaman dinilai 30000
08:39 rekaman dinilai 35000
08:46 rekaman dinilai 40000

## Rekaman dibuang

| Kelompok | Gagal dibaca | Skor tak terhingga |
| --- | --- | --- |
| dilihat | 0 | 0 |
| tidak dilihat | 0 | 0 |

## Statistik (bootstrap 2000 ulangan atas pasien, benih 20261120)

| Statistik | Nilai [95 % CI] |
| --- | --- |
| rerata dilihat | 0.9118 [0.9075, 0.9163] |
| rerata tidak dilihat | 0.9173 [0.9125, 0.9223] |
| D | 0.0055 [-0.0008, 0.0122] |
| D valid | 0.0054 [-0.0026, 0.0138] |
| D test | 0.0056 [-0.0022, 0.0137] |
| D plasebo | -0.0009 [-0.0101, 0.0083] |
| AUROC keanggotaan | 0.5041 [0.4990, 0.5095] |
| kosinus segmen | 0.0020 [0.0017, 0.0023] |

## Keputusan praregistrasi

- Kontrol negatif (selang D plasebo memuat 0): LOLOS.
- M1 (batas bawah D > 0): keanggotaan TIDAK terkonfirmasi (batas bawah -0.0008); batas 0,002 bersyarat pada daftar split.

### celah_lvh.py

Unit 416503; dibuang karena embedding tak terhingga 0.

## LVH, pasien di luar kohort: 42585 dilihat (1649 positif), 10828 tidak dilihat (394 positif); ulangan terpakai 2000

| Model | AUROC dilihat | AUROC tidak dilihat | Celah dilihat − tidak dilihat |
| --- | --- | --- | --- |
| ECG-JEPA | 0.940 [0.931, 0.948] | 0.936 [0.920, 0.949] | 0.004 [-0.011, 0.022] |
| ECG-FM pretrained | 0.897 [0.885, 0.908] | 0.867 [0.841, 0.888] | 0.030 [0.005, 0.059] |
| ECG-FM fine-tuned | 0.923 [0.913, 0.932] | 0.900 [0.879, 0.917] | 0.023 [0.003, 0.044] |
