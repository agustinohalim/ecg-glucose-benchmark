# Hasil Glukosa 1 — M0/M1 dari ukuran mesin EKG (20 November 2026)

Sumber angka: keluaran `glukosa_m1/jalankan.py --boot 1000` atas
`mimiciv_ecg_glucose_aligned.csv` v1.0.0 (PhysioNet, DOI 10.13026/nts2-vj76, Credentialed Health Data
License 1.5.0) di `data/mimic-iv-ext-ecg-glucose/`. Dijalankan di laptop, 10,8 menit; yang keluar
hanya agregat (hitungan 1–10 dicetak `<11`, metrik tidak dicetak bila positif < 11). Tabel di bawah
bagian 0 disalin apa adanya; jangan diketik ulang.

**Split TEST tidak disentuh.** Analisis utama: latih TRAIN, nilai VALIDATION. Kepekaan kendali obat:
validasi silang bergugus pasien atas subset ICU, gabungan ketiga split (keputusan penulis 20 Nov,
rencana §C2).

## Pilihan analisis (ditetapkan sebelum skrip dijalankan)

- Masukan M0: `age`, jenis kelamin. M1: M0 + 13 ukuran mesin (HR, RR, QRS, QT, PR, JT, sumbu P/QRS/T,
  empat QTc). Tidak ada kolom turunan glukosa, SQI, atau obat sebagai masukan.
- Kode "tidak terukur" |nilai| ≥ 10000 dan RR ≤ 0 → hilang. LR: pangkas di kuantil 0,5/99,5 % data
  latih, isi median, standarkan, C = 1. GB: `HistGradientBoostingClassifier` (laju 0,05, ≤ 500
  iterasi, henti dini 10 %), menangani nilai hilang sendiri. Tidak ada penalaan hiperparameter.
- Pasangan non-EXCLUDE. Hipo = SEVERE_HYPO + HYPO; eu = EUGLYCEMIC; hiper berat = SEVERE_HYPER; tiga
  kelas: < 70 / 70–180 / > 180 (sama dengan Palanchuk 2026).
- CI: persentil 2,5/97,5 dari 1000 bootstrap bergugus pasien. Kemiringan kalibrasi, O/E, Brier, Se @
  Sp 90 % tanpa CI.
- Uji asap `--boot 20` dijalankan lebih dulu untuk memeriksa skrip; tidak ada yang diubah sesudahnya
  (estimasi titik sama persis).

## 0. Catatan pembacaan

1. **Hipoglikemia: sinyal lemah.** VALIDATION, 696 hipo / 536 pasien: M1-GB AUROC 0,579
   [0,554; 0,603], M1-LR 0,554 [0,528; 0,579], M0 0,519 [0,497; 0,545]. AUPRC 0,024–0,025 lawan
   prevalensi 1,91 %. Se @ Sp 90 % 0,14. Pola sama pada selisih ≤ 60 menit (GB 0,598), SQI GOOD
   (0,587), dan konservatif (0,583, CI 0,521–0,647).
2. **Hiperglikemia berat lebih kuat.** M1-GB 0,669 [0,648; 0,688], M1-LR 0,641, M0 0,545. Pada subset
   konservatif 0,703 [0,675; 0,733].
3. **Tiga kelas, AUROC makro:** M1-GB 0,610 [0,598; 0,621], M1-LR 0,588. Palanchuk 2026 melaporkan
   0,632 (0,610–0,652) dengan rancangan lain (±5 menit, satu sadapan, split CardioLab, ciri berbeda);
   tidak dibandingkan langsung.
4. **Kepekaan kendali obat (ICU), hipoglikemia:** ICU non-EXCLUDE (986 hipo) M1-LR 0,600
   [0,581; 0,620], M1-GB 0,601. ICU + obat terverifikasi bersih (135 hipo / 130 pasien) M1-LR 0,514
   [0,463; 0,564], M1-GB 0,532 [0,488; 0,575] — **CI memuat 0,5**. Sinyal hipo di ICU tidak terlihat
   lagi bila pasangan dengan insulin/dekstrosa aktif dibuang. **Belum boleh disimpulkan sebagai efek
   obat:** subset konservatif juga mensyaratkan usabilitas IDEAL/USABLE dan EKG sebelum glukosa, dan
   jauh lebih kecil. ⬜ Perbandingan yang adil: ICU dengan syarat usabilitas + urutan waktu yang sama,
   dipisah hanya menurut `intervention_status` (CLEAN lawan bukan CLEAN).
5. **Kepekaan kendali obat (ICU), hiperglikemia berat:** ICU non-EXCLUDE M1-LR 0,638 [0,627; 0,651];
   ICU konservatif M1-LR 0,591 [0,548; 0,633] — turun, tetapi CI di atas 0,5.
6. **Kalibrasi:** LR terkalibrasi baik (O/E ≈ 1,00, kemiringan 0,94–1,09 di analisis utama); GB terlalu
   yakin untuk hipo (kemiringan 0,62). Di ICU konservatif kemiringan 0,09–0,68 — model nyaris tak
   membedakan.
7. **Koefisien M1-LR hipo** (per 1 SD, deskriptif; ciri saling berkorelasi): QTc Hodges +0,184, usia
   −0,114, HR +0,101 terbesar. Arah QTc dan HR searah dengan respons simpatoadrenal pada hipoglikemia,
   tetapi koefisien bukan bukti mekanisme.
8. Ukuran hilang: PR 14,08 %, sumbu P 15,32 % (umumnya tanpa gelombang P, mis. fibrilasi atrium);
   lainnya < 0,4 %.

**Akibat untuk rencana:** belum ada klaim tertulis yang dibantah (sumbangan di §A masih ⬜). Risiko
"sinyal lemah" di §E terbukti untuk hipoglikemia pada ciri interval. M2/M3 (gelombang) menentukan
apakah ada sinyal di luar ukuran mesin; butuh ZIP MIMIC-IV-ECG (§F langkah 4).

## Keluaran skrip

Judul dari skrip diturunkan satu tingkat (`#` ditambah); isi lain apa adanya.

## M0/M1 — bootstrap 1000 bergugus pasien

### 0. Ukuran mesin EKG yang hilang (non-EXCLUDE, sesudah kode tidak-terukur dibuang)

| Ukuran | Hilang | % hilang |
| --- | --- | --- |
| heart_rate_bpm | <11 | 0.00 |
| rr_interval | <11 | 0.00 |
| qrs_duration_ms | 0 | 0.00 |
| qt_raw_ms | 11 | 0.00 |
| pr_interval_ms | 58498 | 14.08 |
| jt_interval_ms | 11 | 0.00 |
| p_axis | 63646 | 15.32 |
| qrs_axis | 515 | 0.12 |
| t_axis | 1445 | 0.35 |
| qtc_bazett_ms | 12 | 0.00 |
| qtc_fridericia_ms | 12 | 0.00 |
| qtc_framingham_ms | 11 | 0.00 |
| qtc_hodges_ms | 12 | 0.00 |

### Hipoglikemia (< 70) lawan euglikemia (70–180) — latih TRAIN, nilai VALIDATION (non-EXCLUDE)

- TRAIN: 285678 pasangan, 100015 pasien, positif 5445 (pasien dengan ≥ 1 positif 4043).
- VALIDATION: 36469 pasangan, 12572 pasien, positif 696 (pasien dengan ≥ 1 positif 536).

| Model | AUROC [95 % CI] | AUPRC [95 % CI] | Se @ Sp 90 % | Brier | O/E | Kemiringan kalibrasi |
| --- | --- | --- | --- | --- | --- | --- |
| M0 usia + jenis kelamin (LR) | 0.519 [0.497, 0.545] | 0.020 [0.018, 0.023] | 0.118 | 0.019 | 1.002 | 0.939 |
| M1 ukuran EKG + M0 (LR) | 0.554 [0.528, 0.579] | 0.025 [0.021, 0.031] | 0.145 | 0.019 | 1.005 | 0.939 |
| M1 ukuran EKG + M0 (GB) | 0.579 [0.554, 0.603] | 0.024 [0.021, 0.028] | 0.141 | 0.019 | 1.013 | 0.620 |

Prevalensi positif: 1.91 %. CI: bootstrap bergugus pasien.

Kepekaan (model yang sama, dinilai pada bagian VALIDATION), AUROC [95 % CI]:

| Subset | n positif | M0 usia + jenis kelamin (LR) | M1 ukuran EKG + M0 (LR) | M1 ukuran EKG + M0 (GB) |
| --- | --- | --- | --- | --- |
| selisih ≤ 60 menit | 354 | 0.536 [0.504, 0.568] | 0.567 [0.532, 0.600] | 0.598 [0.565, 0.630] |
| SQI GOOD | 584 | 0.530 [0.503, 0.556] | 0.561 [0.535, 0.591] | 0.587 [0.561, 0.614] |
| konservatif (anjuran README) | 84 | 0.498 [0.431, 0.562] | 0.571 [0.502, 0.638] | 0.583 [0.521, 0.647] |

### Hiperglikemia berat (> 250) lawan euglikemia (70–180) — latih TRAIN, nilai VALIDATION (non-EXCLUDE)

- TRAIN: 298202 pasangan, 102061 pasien, positif 17969 (pasien dengan ≥ 1 positif 8375).
- VALIDATION: 38227 pasangan, 12783 pasien, positif 2454 (pasien dengan ≥ 1 positif 1058).

| Model | AUROC [95 % CI] | AUPRC [95 % CI] | Se @ Sp 90 % | Brier | O/E | Kemiringan kalibrasi |
| --- | --- | --- | --- | --- | --- | --- |
| M0 usia + jenis kelamin (LR) | 0.545 [0.518, 0.573] | 0.070 [0.060, 0.084] | 0.086 | 0.060 | 1.066 | 0.862 |
| M1 ukuran EKG + M0 (LR) | 0.641 [0.620, 0.663] | 0.102 [0.088, 0.119] | 0.206 | 0.059 | 1.069 | 1.086 |
| M1 ukuran EKG + M0 (GB) | 0.669 [0.648, 0.688] | 0.117 [0.100, 0.136] | 0.239 | 0.059 | 1.089 | 0.941 |

Prevalensi positif: 6.42 %. CI: bootstrap bergugus pasien.

Kepekaan (model yang sama, dinilai pada bagian VALIDATION), AUROC [95 % CI]:

| Subset | n positif | M0 usia + jenis kelamin (LR) | M1 ukuran EKG + M0 (LR) | M1 ukuran EKG + M0 (GB) |
| --- | --- | --- | --- | --- |
| selisih ≤ 60 menit | 1377 | 0.527 [0.496, 0.555] | 0.643 [0.622, 0.665] | 0.670 [0.651, 0.691] |
| SQI GOOD | 2162 | 0.532 [0.501, 0.560] | 0.654 [0.631, 0.678] | 0.677 [0.657, 0.699] |
| konservatif (anjuran README) | 319 | 0.469 [0.436, 0.503] | 0.670 [0.639, 0.704] | 0.703 [0.675, 0.733] |

### Tiga kelas (< 70 / 70–180 / > 180), sebanding Palanchuk 2026 — latih TRAIN, nilai VALIDATION

VALIDATION: hipo 696, eu 35773, hiper 6036.

| Model | AUROC hipo | AUROC eu | AUROC hiper | AUROC makro |
| --- | --- | --- | --- | --- |
| M0 usia + jenis kelamin (LR) | 0.522 [0.496, 0.545] | 0.517 [0.498, 0.535] | 0.521 [0.501, 0.540] | 0.520 [0.507, 0.534] |
| M1 ukuran EKG + M0 (LR) | 0.547 [0.522, 0.570] | 0.605 [0.594, 0.616] | 0.613 [0.601, 0.625] | 0.588 [0.577, 0.600] |
| M1 ukuran EKG + M0 (GB) | 0.561 [0.539, 0.583] | 0.629 [0.617, 0.642] | 0.639 [0.626, 0.651] | 0.610 [0.598, 0.621] |

### Kepekaan kendali obat — Hipoglikemia (< 70) lawan euglikemia (70–180): validasi silang 5 lipatan bergugus pasien, 3 benih (peluang dirata-rata), gabungan TRAIN/VALIDATION/TEST

**ICU, non-EXCLUDE (obat bisa aktif)**

- subset: 48825 pasangan, 21055 pasien, positif 986 (pasien dengan ≥ 1 positif 830).

| Model | AUROC [95 % CI] | AUPRC [95 % CI] | Se @ Sp 90 % | Brier | O/E | Kemiringan kalibrasi |
| --- | --- | --- | --- | --- | --- | --- |
| M0 usia + jenis kelamin (LR) | 0.527 [0.507, 0.549] | 0.022 [0.020, 0.024] | 0.117 | 0.020 | 1.000 | 0.938 |
| M1 ukuran EKG + M0 (LR) | 0.600 [0.581, 0.620] | 0.029 [0.026, 0.033] | 0.164 | 0.020 | 1.000 | 0.938 |
| M1 ukuran EKG + M0 (GB) | 0.601 [0.583, 0.619] | 0.027 [0.024, 0.030] | 0.145 | 0.020 | 1.050 | 0.636 |

Prevalensi positif: 2.02 %. CI: bootstrap bergugus pasien.

**ICU, konservatif (obat terverifikasi bersih)**

- subset: 4641 pasangan, 4041 pasien, positif 135 (pasien dengan ≥ 1 positif 130).

| Model | AUROC [95 % CI] | AUPRC [95 % CI] | Se @ Sp 90 % | Brier | O/E | Kemiringan kalibrasi |
| --- | --- | --- | --- | --- | --- | --- |
| M0 usia + jenis kelamin (LR) | 0.467 [0.415, 0.518] | 0.027 [0.022, 0.036] | 0.067 | 0.028 | 0.999 | -1.792 |
| M1 ukuran EKG + M0 (LR) | 0.514 [0.463, 0.564] | 0.032 [0.025, 0.043] | 0.119 | 0.028 | 1.000 | 0.136 |
| M1 ukuran EKG + M0 (GB) | 0.532 [0.488, 0.575] | 0.030 [0.025, 0.038] | 0.081 | 0.028 | 1.058 | 0.086 |

Prevalensi positif: 2.91 %. CI: bootstrap bergugus pasien.

### Kepekaan kendali obat — Hiperglikemia berat (> 250) lawan euglikemia (70–180): validasi silang 5 lipatan bergugus pasien, 3 benih (peluang dirata-rata), gabungan TRAIN/VALIDATION/TEST

**ICU, non-EXCLUDE (obat bisa aktif)**

- subset: 50790 pasangan, 21855 pasien, positif 2951 (pasien dengan ≥ 1 positif 2037).

| Model | AUROC [95 % CI] | AUPRC [95 % CI] | Se @ Sp 90 % | Brier | O/E | Kemiringan kalibrasi |
| --- | --- | --- | --- | --- | --- | --- |
| M0 usia + jenis kelamin (LR) | 0.566 [0.552, 0.580] | 0.071 [0.066, 0.078] | 0.148 | 0.055 | 0.999 | 0.887 |
| M1 ukuran EKG + M0 (LR) | 0.638 [0.627, 0.651] | 0.091 [0.085, 0.100] | 0.195 | 0.054 | 1.000 | 0.966 |
| M1 ukuran EKG + M0 (GB) | 0.639 [0.626, 0.652] | 0.093 [0.086, 0.102] | 0.195 | 0.054 | 1.099 | 0.664 |

Prevalensi positif: 5.81 %. CI: bootstrap bergugus pasien.

**ICU, konservatif (obat terverifikasi bersih)**

- subset: 4739 pasangan, 4125 pasien, positif 233 (pasien dengan ≥ 1 positif 207).

| Model | AUROC [95 % CI] | AUPRC [95 % CI] | Se @ Sp 90 % | Brier | O/E | Kemiringan kalibrasi |
| --- | --- | --- | --- | --- | --- | --- |
| M0 usia + jenis kelamin (LR) | 0.539 [0.498, 0.580] | 0.055 [0.046, 0.068] | 0.107 | 0.047 | 1.001 | 0.897 |
| M1 ukuran EKG + M0 (LR) | 0.591 [0.548, 0.633] | 0.073 [0.059, 0.097] | 0.185 | 0.047 | 1.002 | 0.675 |
| M1 ukuran EKG + M0 (GB) | 0.571 [0.526, 0.612] | 0.069 [0.054, 0.093] | 0.129 | 0.047 | 1.204 | 0.370 |

Prevalensi positif: 4.92 %. CI: bootstrap bergugus pasien.

### Koefisien terstandar M1 (LR), hipoglikemia lawan euglikemia, latih TRAIN

| Ciri | Koefisien (per 1 SD) |
| --- | --- |
| qtc_hodges_ms | +0.184 |
| age | -0.114 |
| heart_rate_bpm | +0.101 |
| rr_interval | +0.072 |
| t_axis | +0.066 |
| p_axis | +0.035 |
| qtc_framingham_ms | -0.035 |
| qrs_axis | +0.033 |
| qrs_duration_ms | -0.023 |
| qtc_bazett_ms | +0.017 |
| laki | -0.017 |
| qt_raw_ms | +0.015 |
| pr_interval_ms | +0.013 |
| qtc_fridericia_ms | +0.011 |
| jt_interval_ms | -0.003 |

Waktu jalan: 10.8 menit.
