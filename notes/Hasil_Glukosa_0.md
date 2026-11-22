# Hasil Glukosa 0 — hitung ulang label MIMIC-IV-Ext-ECG-Glucose (20 November 2026)

Sumber angka: keluaran `glukosa_hitung_label.py` atas `mimiciv_ecg_glucose_aligned.csv`
v1.0.0 (PhysioNet, DOI 10.13026/nts2-vj76, Credentialed Health Data License 1.5.0), diunduh penulis
20 Nov 2026 ke `data/mimic-iv-ext-ecg-glucose/` (di luar repo; kelima berkas cocok dengan
`SHA256SUMS.txt`). Skrip dijalankan di laptop; yang keluar hanya agregat, hitungan 1–10 dicetak `<11`.
Tidak ada pemodelan. Teruskan tabel di bawah apa adanya; jangan diketik ulang.

## 0. Catatan pembacaan

- Ukuran berkas (437.671 pasangan, 131.771 pasien, 376.648 studi EKG) **sama dengan README**. Tidak
  ada pasien di dua split. Kelas dihitung ulang dari `glucose_mg_dl` sama persis dengan
  `glycemic_class` (0 beda).
- Hipoglikemia (< 70 mg/dL) ±2 % dari pasangan hipo + euglikemia di setiap subset. Uji non-EXCLUDE:
  706 pasangan hipo dari 512 pasien — cukup untuk AUROC/AUPRC dengan CI bergugus pasien.
- **Subset ICU dengan obat terverifikasi bersih (ICU + konservatif) terlalu kecil untuk dievaluasi
  di split TEST: < 11 pasangan hipo.** VALIDATION 15. Klaim "kendali perancu obat pada subset ICU"
  (rencana A, opsi 1) tidak bisa diuji pada pembagian bawaan. Pilihan di rencana §A (⬜ keputusan
  penulis).
- ICU non-EXCLUDE masih layak sebagai subset: uji 90 hipo / 76 pasien (CI akan lebar).
- `intervention_status` bukan CLEAN juga muncul di luar ICU (107 EXCLUDE, 434 FLAG_HIGH_RISK,
  754 FLAG_MODERATE_RISK). README menyebut anotasi hanya dari `inputevents` ICU; dugaan: pencocokan
  per `subject_id` menangkap infus ICU yang jendelanya melewati batas keluar ICU. ⬜ belum diperiksa
  di SQL pipeline.

## 1. Ukuran berkas

| Baris (pasangan) | Pasien | Studi EKG unik |
| --- | --- | --- |
| 437671 | 131771 | 376648 |

Pasien di lebih dari satu split: 0.

## 2. Cek label

Kelas dihitung ulang dari `glucose_mg_dl` (batas bawah [54, 70, 181, 251] mg/dL) berbeda dari `glycemic_class` pada 0 baris. Rentang glukosa 30–700 mg/dL; glukosa kosong 0.

## 3. Kelas glikemik per split (semua baris)

| kelas | TRAIN | VALIDATION | TEST | Jumlah |
| --- | --- | --- | --- | --- |
| SEVERE_HYPO | 1297 | 158 | 159 | 1614 |
| HYPO | 4285 | 562 | 567 | 5414 |
| EUGLYCEMIC | 293249 | 37450 | 36057 | 366756 |
| HYPER | 31660 | 3936 | 3733 | 39329 |
| SEVERE_HYPER | 19485 | 2612 | 2461 | 24558 |

## 4. Kelas glikemik ICU vs luar ICU (non-EXCLUDE)

| kelas | ICU | luar ICU | Jumlah |
| --- | --- | --- | --- |
| SEVERE_HYPO | 282 | 1281 | 1563 |
| HYPO | 704 | 4580 | 5284 |
| EUGLYCEMIC | 47839 | 302564 | 350403 |
| HYPER | 7365 | 28189 | 35554 |
| SEVERE_HYPER | 2951 | 19749 | 22700 |

## 5. Sasaran utama: hipoglikemia (< 70 mg/dL) vs euglikemia (70–180 mg/dL)

| Subset | Split | n hipo | n eu | % hipo | Pasien dengan ≥ 1 hipo |
| --- | --- | --- | --- | --- | --- |
| semua non-EXCLUDE | TRAIN | 5445 | 280233 | 1.91 | 4043 |
| semua non-EXCLUDE | VALIDATION | 696 | 35773 | 1.91 | 536 |
| semua non-EXCLUDE | TEST | 706 | 34397 | 2.01 | 512 |
| konservatif (anjuran README) | TRAIN | 715 | 41909 | 1.68 | 690 |
| konservatif (anjuran README) | VALIDATION | 84 | 5293 | 1.56 | 83 |
| konservatif (anjuran README) | TEST | 98 | 5346 | 1.80 | 95 |
| ICU, non-EXCLUDE | TRAIN | 794 | 38203 | 2.04 | 666 |
| ICU, non-EXCLUDE | VALIDATION | 102 | 5026 | 1.99 | 88 |
| ICU, non-EXCLUDE | TEST | 90 | 4610 | 1.91 | 76 |
| ICU, konservatif (obat terverifikasi) | TRAIN | 111 | 3549 | 3.03 | 107 |
| ICU, konservatif (obat terverifikasi) | VALIDATION | 15 | 503 | 2.90 | 14 |
| ICU, konservatif (obat terverifikasi) | TEST | <11 | 454 | — | <11 |

## 6. intervention_status ICU vs luar ICU (semua baris)

| intervention_status | ICU | luar ICU | Jumlah |
| --- | --- | --- | --- |
| CLEAN | 29766 | 355778 | 385544 |
| EXCLUDE | 21183 | 107 | 21290 |
| FLAG_HIGH_RISK | 4291 | 434 | 4725 |
| FLAG_MODERATE_RISK | 25358 | 754 | 26112 |
