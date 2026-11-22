# Hasil Glukosa 2 — perbandingan kendali obat yang adil di ICU (20 November 2026)

Sumber angka: keluaran `glukosa_m1/jalankan_obat.py --boot 1000` atas
`mimiciv_ecg_glucose_aligned.csv` v1.0.0 (PhysioNet, DOI 10.13026/nts2-vj76, Credentialed Health Data
License 1.5.0) di `data/mimic-iv-ext-ecg-glucose/`. Dijalankan di laptop, 3,6 menit; yang keluar
hanya agregat (hitungan 1–10 dicetak `<11`, metrik tidak dicetak bila positif < 11). Tabel di bawah
bagian 0 disalin apa adanya; jangan diketik ulang. Menjawab ⬜ di `Hasil_Glukosa_1.md` catatan 4.

**Split TEST tidak dipakai sebagai uji terpisah**; seperti kepekaan obat di Hasil 1, ketiga split
digabung dan dinilai dengan validasi silang bergugus pasien (keputusan penulis 20 Nov, rencana §C2).

## Pilihan analisis

- **Masalah di Hasil 1:** subset "ICU konservatif" berbeda dari "ICU non-EXCLUDE" bukan hanya soal obat,
  tetapi juga usabilitas, urutan waktu, dan ukuran. Lagi pula `record_usability` sendiri diturunkan
  dari status obat (FLAG_HIGH_RISK → CAUTION, IDEAL mensyaratkan CLEAN; kamus data), jadi tak bisa
  dipakai sebagai syarat mutu yang netral.
- **Subset setara:** pasangan ICU, `temporal_relationship = ECG_BEFORE_GLUCOSE`, `sqi_category` GOOD
  atau FAIR; semua status obat ikut. Syarat mutu dan urutan waktu sama untuk semua strata.
- **Satu model per tugas** (M0, M1-LR, M1-GB; masukan sama dengan Hasil 1, tanpa kolom obat), validasi
  silang 5 lipatan bergugus pasien × 3 benih, peluang luar-lipatan dirata-rata. AUROC dihitung
  **di dalam** tiap `intervention_status`: CLEAN, FLAG_MODERATE_RISK (insulin basal atau dekstrosa
  saja), FLAG_HIGH_RISK (insulin cepat tanpa dekstrosa), EXCLUDE (insulin + dekstrosa).
- Selisih AUROC CLEAN − tiap status lain, CI dari 1000 bootstrap bergugus pasien atas seluruh subset
  (kedua AUROC pada ulangan yang sama).
- **Perubahan sesudah uji asap (`--boot 20`), sebelum jalan penuh:** rancangan awal juga
  membandingkan CLEAN dengan gabungan "obat aktif (bukan CLEAN)". Uji asap menunjukkan AUROC gabungan
  itu lebih tinggi daripada setiap strata penyusunnya — gabungan strata dengan prevalensi berbeda
  menambah pembedaan antar-strata, jadi perbandingannya tidak adil. Gabungan dibuang; yang tersisa
  perbandingan per strata. Estimasi titik per strata sama persis antara uji asap dan jalan penuh.

## 0. Catatan pembacaan

1. **Hipoglikemia, CLEAN: sinyal hampir tak ada.** M1-GB 0,538 [0,500; 0,575] (224 hipo), M1-LR 0,528
   [0,485; 0,569]. Ini menegaskan Hasil 1 (ICU konservatif 0,532) dengan syarat mutu yang kini setara
   dan sampel 1,7 kali lebih besar.
2. **Hipoglikemia, strata obat:** FLAG_MODERATE_RISK M1-GB 0,593 [0,560; 0,626]; selisih CLEAN − MODERATE
   −0,055 [−0,108; −0,004] (LR −0,060 [−0,115; −0,006]) — batas atas CI nyaris 0. EXCLUDE 0,578,
   selisih −0,040 [−0,117; 0,038], CI memuat 0. FLAG_HIGH_RISK hanya 15 hipo; angkanya tak stabil
   (M0 0,685, M1-LR 0,460) dan tidak ditafsirkan. Dengan 9 perbandingan per tugas, satu selisih yang
   batas CI-nya menyentuh 0 bukan bukti kuat. Pembacaan yang jujur: sinyal hipo pada pasangan bebas
   obat lemah sampai nihil; mungkin lebih tinggi bila obat aktif, belum pasti.
3. **Hiperglikemia berat: perbedaan nyata menurut status obat.** M1-LR CLEAN 0,580 [0,546; 0,614]
   lawan EXCLUDE 0,732 [0,712; 0,752], selisih −0,151 [−0,189; −0,113]; FLAG_HIGH_RISK selisih −0,091
   [−0,143; −0,041]; FLAG_MODERATE_RISK tak berbeda (−0,011 [−0,050; 0,030]).
4. **Tetapi M0 (usia + jenis kelamin) juga bergeser:** CLEAN 0,488 lawan EXCLUDE 0,576, selisih −0,088
   [−0,130; −0,047]. Jadi sebagian selisih adalah campuran kasus (pasien yang diberi insulin + dekstrosa
   berbeda dari yang tidak), bukan kemampuan EKG membaca glukosa. Tambahan dari ciri EKG atas M0
   (M1-LR − M0) kira-kira 0,09 di CLEAN dan 0,16 di EXCLUDE (dihitung dari estimasi titik; tanpa CI).
5. **Akibat:** sinyal glukosa dari ukuran mesin EKG di ICU paling lemah tepat pada pasangan yang
   bebas pengaruh insulin/dekstrosa. Sebagian sinyal pada analisis gabungan kemungkinan menumpang
   konteks pengobatan dan keparahan penyakit. Ini memperkuat perlunya analisis kepekaan kendali obat
   di naskah, dan menjadi pembanding wajib untuk M2/M3: model gelombang harus diuji di strata CLEAN
   yang sama.

**Akibat untuk rencana:** belum ada klaim tertulis yang dibantah (sumbangan §A masih ⬜). Bentuk klaim
yang didukung sejauh ini: garis dasar terbuka + bukti bahwa kinerja bergantung pada status obat. Klaim
"EKG mendeteksi hipoglikemia" tidak didukung oleh ukuran mesin.

## Keluaran skrip

Judul dari skrip diturunkan satu tingkat (`#` ditambah); isi lain apa adanya.

## Kendali obat setara (ICU) — bootstrap 1000 bergugus pasien

### Hipoglikemia (< 70) lawan euglikemia (70–180) — ICU, EKG sebelum glukosa, SQI GOOD/FAIR; satu model validasi silang 5 lipatan bergugus pasien × 3 benih, dinilai per status obat

- subset: 27750 pasangan, 16059 pasien, positif 610 (pasien dengan ≥ 1 positif 528).

| Status obat | Pasangan | Positif | M0 usia + jenis kelamin (LR) | M1 ukuran EKG + M0 (LR) | M1 ukuran EKG + M0 (GB) |
| --- | --- | --- | --- | --- | --- |
| CLEAN | 8825 | 224 | 0.486 [0.446, 0.524] | 0.528 [0.485, 0.569] | 0.538 [0.500, 0.575] |
| FLAG_MODERATE_RISK | 9345 | 275 | 0.474 [0.431, 0.514] | 0.588 [0.553, 0.625] | 0.593 [0.560, 0.626] |
| FLAG_HIGH_RISK | 723 | 15 | 0.685 [0.568, 0.791] | 0.460 [0.307, 0.606] | 0.658 [0.552, 0.770] |
| EXCLUDE | 8857 | 96 | 0.459 [0.391, 0.531] | 0.567 [0.501, 0.635] | 0.578 [0.512, 0.642] |

AUROC [95 % CI] di dalam tiap strata; "—" bila positif < 11.

| CLEAN − status | M0 usia + jenis kelamin (LR) | M1 ukuran EKG + M0 (LR) | M1 ukuran EKG + M0 (GB) |
| --- | --- | --- | --- |
| FLAG_MODERATE_RISK | 0.012 [-0.040, 0.069] | -0.060 [-0.115, -0.006] | -0.055 [-0.108, -0.004] |
| FLAG_HIGH_RISK | -0.199 [-0.310, -0.082] | 0.068 [-0.087, 0.220] | -0.120 [-0.238, -0.010] |
| EXCLUDE | 0.027 [-0.051, 0.104] | -0.039 [-0.120, 0.040] | -0.040 [-0.117, 0.038] |

Selisih AUROC [95 % CI]; bootstrap bergugus pasien atas seluruh subset, kedua AUROC pada ulangan yang sama; "—" bila positif < 11.

### Hiperglikemia berat (> 250) lawan euglikemia (70–180) — ICU, EKG sebelum glukosa, SQI GOOD/FAIR; satu model validasi silang 5 lipatan bergugus pasien × 3 benih, dinilai per status obat

- subset: 29674 pasangan, 16875 pasien, positif 2534 (pasien dengan ≥ 1 positif 1818).

| Status obat | Pasangan | Positif | M0 usia + jenis kelamin (LR) | M1 ukuran EKG + M0 (LR) | M1 ukuran EKG + M0 (GB) |
| --- | --- | --- | --- | --- | --- |
| CLEAN | 8967 | 366 | 0.488 [0.456, 0.522] | 0.580 [0.546, 0.614] | 0.585 [0.551, 0.617] |
| FLAG_MODERATE_RISK | 9870 | 800 | 0.526 [0.503, 0.549] | 0.592 [0.569, 0.616] | 0.594 [0.573, 0.615] |
| FLAG_HIGH_RISK | 1149 | 441 | 0.588 [0.544, 0.627] | 0.671 [0.634, 0.708] | 0.650 [0.614, 0.688] |
| EXCLUDE | 9688 | 927 | 0.576 [0.548, 0.601] | 0.732 [0.712, 0.752] | 0.723 [0.703, 0.745] |

AUROC [95 % CI] di dalam tiap strata; "—" bila positif < 11.

| CLEAN − status | M0 usia + jenis kelamin (LR) | M1 ukuran EKG + M0 (LR) | M1 ukuran EKG + M0 (GB) |
| --- | --- | --- | --- |
| FLAG_MODERATE_RISK | -0.038 [-0.079, -0.001] | -0.011 [-0.050, 0.030] | -0.009 [-0.047, 0.031] |
| FLAG_HIGH_RISK | -0.100 [-0.151, -0.050] | -0.091 [-0.143, -0.041] | -0.065 [-0.113, -0.016] |
| EXCLUDE | -0.088 [-0.130, -0.047] | -0.151 [-0.189, -0.113] | -0.138 [-0.176, -0.099] |

Selisih AUROC [95 % CI]; bootstrap bergugus pasien atas seluruh subset, kedua AUROC pada ulangan yang sama; "—" bila positif < 11.

Waktu jalan: 3.6 menit.
