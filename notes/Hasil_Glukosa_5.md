# Hasil Glukosa 5 — M3 (ECGFounder beku + probe L2) berdampingan dengan M0–M2 (21 November 2026)

Sumber angka: keluaran `glukosa_m3/jalankan_m3.py --boot 1000` atas
`mimiciv_ecg_glucose_aligned.csv` v1.0.0 (PhysioNet, DOI 10.13026/nts2-vj76, Credentialed Health Data
License 1.5.0) dan gelombang MIMIC-IV-ECG v1.0. Dijalankan di laptop, 11,1 menit (probe, CPU) sesudah
embedding 46,8 menit (`glukosa_m3/bangun_embedding.py`, RTX 5060); yang keluar hanya agregat (hitungan
1–10 dicetak `<11`, metrik tidak dicetak bila positif < 11). Tabel di bawah bagian 0 disalin apa adanya;
jangan diketik ulang. Split TEST tidak disentuh.

## Pilihan analisis

- **Model fondasi:** ECGFounder (`12_lead_ECGFounder.pth`, MIT; dilatih pada Harvard-Emory ECG Database,
  bukan MIMIC-IV-ECG, jadi tidak pernah melihat EKG VALIDATION/TEST ini), beku. Bobot, kode Net1D, dan
  pra-proses sama dengan M3a naskah OMI.
- **Embedding:** 376.648 rekaman dibaca ulang dari ZIP dalam bentuk asli 12 sadapan × 500 Hz (bukan
  singgahan M2), pra-proses resmi (notch 50 Hz, pita 0,67–40 Hz, median bergerak 201 sampel, z-score),
  `deep_features` 1.024 dimensi. Semua terhingga. Diuji 21 Nov pada tiga rekaman terbuka: pra-proses
  sama dengan versi OMI (`medfilt`) sampai 5 × 10⁻⁷; embedding sama dengan panggilan model langsung
  sampai ±0,001 (norma ±20). Sampel hilang (1,3 % rekaman) diisi 0 sebelum pra-proses.
- **Probe:** StandardScaler + regresi logistik L2; C dari kisi OMI {10⁻³ … 100} dengan log-loss pada 3
  lipatan dalam StratifiedGroupKFold (OMI: 5; dikurangi karena data latih ±17 kali lebih besar), di
  data latih saja. Nilai C terpilih tidak dicetak (⬜ bila diminta penelaah).
- **Pembanding:** prediksi M0–M2 dibaca dari jalan Hasil 4 (`data/mimic-iv-ext-ecg-glucose/prediksi/`), dicocokkan baris
  demi baris (y dan subject_id identik, diperiksa skrip); hanya M3 yang dilatih. Susunan sama dengan
  Hasil 4; tiga kelas tidak.
- Uji jalan hanya pada data sintetis (embedding acak: M3 ≈ 0,5, sesuai harapan).
- Satu model fondasi saja; rencana menyebut 2–3. Calon lain (ECG-FM, MERL) diduga dilatih-awal
  memakai MIMIC-IV-ECG, termasuk EKG VALIDATION/TEST (dari ingatan asisten, ⬜ belum diverifikasi di
  makalahnya) — belum dipakai sampai data pra-latihnya diperiksa.

## 0. Catatan pembacaan

1. **Hipoglikemia, VALIDATION:** M3 0,614 [0,590; 0,637], **sama dengan M2** (M3 − M2 +0,001
   [−0,023; 0,027]). M3 − M1-GB +0,035 [0,009; 0,060]. Dua jalan gelombang yang sangat berbeda (CNN
   dilatih dari nol dan model fondasi beku + probe linear) sampai di AUROC yang sama.
2. **Hiperglikemia berat, VALIDATION:** M3 0,716, M3 − M2 −0,003 [−0,018; 0,013]; M3 − M1-GB +0,048
   [0,032; 0,064].
3. **Kalibrasi:** M3 baik tanpa kalibrasi ulang (O/E 1,007 dan 1,071; kemiringan 0,73 dan 1,03),
   berbeda dengan M2 (kemiringan ±0,4). M3 juga deterministik dan murah (probe 11 menit di CPU).
4. **ICU bebas obat (CLEAN), hipoglikemia:** M3 0,565 [0,527; 0,602], CI di atas 0,5; M3 − M1-GB +0,027
   [−0,018; 0,070] dan M3 − M2 −0,016 [−0,057; 0,020]. Seperti M2, keunggulan atas ukuran mesin di
   strata ini **belum terbukti**; keunggulan atas usia + jenis kelamin jelas (+0,079 [0,022; 0,137]).
5. **ICU CLEAN, hiperglikemia berat:** M3 0,651; M3 − M1-GB +0,066 [0,033; 0,101]; M3 − M2 +0,016
   [−0,017; 0,048].
6. **Sel lain:** hipo FLAG_MODERATE_RISK M3 − M2 +0,044 [0,008; 0,085] adalah satu dari 24 selisih dan
   tidak ditafsirkan tersendiri; FLAG_HIGH_RISK hipo (15 positif) tidak ditafsirkan.

**Akibat untuk rencana:** klaim yang kini didukung dua metode berbeda: gelombang EKG 10 s menambah
pembedaan kecil di atas ukuran mesin (+0,03–0,05 AUROC) untuk hipo dan hiper berat; CNN latih penuh
tidak lebih baik daripada model fondasi beku + probe linear, dan model fondasi terkalibrasi lebih baik.
Kesamaan AUROC kedua metode **konsisten dengan** batas informasi gelombang untuk tugas ini, tetapi itu
dugaan, bukan temuan. Tetap belum didukung: gelombang mengalahkan ukuran mesin untuk hipoglikemia di ICU
bebas obat. Semua AUROC hipo < 0,65.

## Keluaran skrip

Judul dari skrip diturunkan satu tingkat (`#` ditambah); isi lain apa adanya.

## M0/M1/M2 — pasangan dengan gelombang terbaca

- pasangan: 437671; gelombang terbaca 437671; dibuang 0.

Bootstrap 1000 bergugus pasien.

### Hipoglikemia (< 70) lawan euglikemia (70–180) — latih TRAIN, nilai VALIDATION (non-EXCLUDE)

| Model | AUROC [95 % CI] | AUPRC [95 % CI] | Se @ Sp 90 % | Brier | O/E | Kemiringan kalibrasi |
| --- | --- | --- | --- | --- | --- | --- |
| M0 usia + jenis kelamin (LR) | 0.519 [0.497, 0.545] | 0.020 [0.018, 0.023] | 0.118 | 0.019 | 1.002 | 0.939 |
| M1 ukuran EKG + M0 (LR) | 0.554 [0.528, 0.579] | 0.025 [0.021, 0.031] | 0.145 | 0.019 | 1.005 | 0.939 |
| M1 ukuran EKG + M0 (GB) | 0.579 [0.554, 0.603] | 0.024 [0.021, 0.028] | 0.141 | 0.019 | 1.013 | 0.620 |
| M2 CNN gelombang | 0.613 [0.588, 0.637] | 0.028 [0.025, 0.032] | 0.191 | 0.168 | 0.052 | 0.405 |
| M3 ECGFounder beku + probe L2 | 0.614 [0.590, 0.637] | 0.029 [0.026, 0.035] | 0.187 | 0.019 | 1.007 | 0.727 |

Prevalensi positif: 1.91 %. CI: bootstrap bergugus pasien.

| Bagian | Pasangan | Positif | AUROC M3 ECGFounder beku + probe L2 | M3 − M2 | M3 − M1 (GB) | M3 − M0 |
| --- | --- | --- | --- | --- | --- | --- |
| VALIDATION | 36469 | 696 | 0.614 | 0.001 [-0.023, 0.027] | 0.035 [0.009, 0.060] | 0.095 [0.064, 0.130] |
| VALIDATION, konservatif | 5377 | 84 | 0.644 | -0.022 [-0.081, 0.036] | 0.061 [-0.018, 0.138] | 0.146 [0.054, 0.239] |

Selisih AUROC [95 % CI]; bootstrap bergugus pasien, kedua model pada ulangan yang sama; "—" bila positif < 11.

### Hipoglikemia (< 70) lawan euglikemia (70–180) — ICU, EKG sebelum glukosa, SQI GOOD/FAIR; validasi silang 5 lipatan bergugus pasien × 3 benih, dinilai per status obat

- subset: 27750 pasangan, 16059 pasien, positif 610 (pasien dengan ≥ 1 positif 528).

| Status obat | Pasangan | Positif | M0 usia + jenis kelamin (LR) | M1 ukuran EKG + M0 (LR) | M1 ukuran EKG + M0 (GB) | M2 CNN gelombang | M3 ECGFounder beku + probe L2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CLEAN | 8825 | 224 | 0.486 [0.451, 0.525] | 0.528 [0.487, 0.566] | 0.538 [0.498, 0.578] | 0.581 [0.542, 0.620] | 0.565 [0.527, 0.602] |
| FLAG_MODERATE_RISK | 9345 | 275 | 0.474 [0.433, 0.516] | 0.588 [0.551, 0.624] | 0.593 [0.559, 0.630] | 0.573 [0.535, 0.610] | 0.617 [0.580, 0.654] |
| FLAG_HIGH_RISK | 723 | 15 | 0.685 [0.565, 0.793] | 0.460 [0.320, 0.610] | 0.658 [0.549, 0.773] | 0.622 [0.447, 0.750] | 0.617 [0.499, 0.738] |
| EXCLUDE | 8857 | 96 | 0.459 [0.387, 0.526] | 0.567 [0.495, 0.640] | 0.578 [0.515, 0.641] | 0.619 [0.555, 0.680] | 0.620 [0.561, 0.680] |

AUROC [95 % CI] di dalam tiap strata; "—" bila positif < 11.

| Bagian | Pasangan | Positif | AUROC M3 ECGFounder beku + probe L2 | M3 − M2 | M3 − M1 (GB) | M3 − M0 |
| --- | --- | --- | --- | --- | --- | --- |
| CLEAN | 8825 | 224 | 0.565 | -0.016 [-0.057, 0.020] | 0.027 [-0.018, 0.070] | 0.079 [0.022, 0.137] |
| FLAG_MODERATE_RISK | 9345 | 275 | 0.617 | 0.044 [0.008, 0.085] | 0.024 [-0.015, 0.064] | 0.143 [0.093, 0.193] |
| FLAG_HIGH_RISK | 723 | 15 | 0.617 | -0.005 [-0.164, 0.212] | -0.041 [-0.205, 0.111] | -0.068 [-0.221, 0.110] |
| EXCLUDE | 8857 | 96 | 0.620 | 0.001 [-0.075, 0.075] | 0.042 [-0.032, 0.116] | 0.162 [0.076, 0.250] |

Selisih AUROC [95 % CI]; bootstrap bergugus pasien, kedua model pada ulangan yang sama; "—" bila positif < 11.

### Hiperglikemia berat (> 250) lawan euglikemia (70–180) — latih TRAIN, nilai VALIDATION (non-EXCLUDE)

| Model | AUROC [95 % CI] | AUPRC [95 % CI] | Se @ Sp 90 % | Brier | O/E | Kemiringan kalibrasi |
| --- | --- | --- | --- | --- | --- | --- |
| M0 usia + jenis kelamin (LR) | 0.545 [0.516, 0.573] | 0.070 [0.061, 0.085] | 0.086 | 0.060 | 1.066 | 0.862 |
| M1 ukuran EKG + M0 (LR) | 0.641 [0.618, 0.664] | 0.102 [0.089, 0.119] | 0.206 | 0.059 | 1.069 | 1.086 |
| M1 ukuran EKG + M0 (GB) | 0.669 [0.649, 0.689] | 0.117 [0.101, 0.137] | 0.239 | 0.059 | 1.089 | 0.941 |
| M2 CNN gelombang | 0.719 [0.701, 0.738] | 0.172 [0.146, 0.205] | 0.331 | 0.143 | 0.231 | 0.451 |
| M3 ECGFounder beku + probe L2 | 0.716 [0.697, 0.735] | 0.164 [0.137, 0.197] | 0.315 | 0.057 | 1.071 | 1.029 |

Prevalensi positif: 6.42 %. CI: bootstrap bergugus pasien.

| Bagian | Pasangan | Positif | AUROC M3 ECGFounder beku + probe L2 | M3 − M2 | M3 − M1 (GB) | M3 − M0 |
| --- | --- | --- | --- | --- | --- | --- |
| VALIDATION | 38227 | 2454 | 0.716 | -0.003 [-0.018, 0.013] | 0.048 [0.032, 0.064] | 0.171 [0.141, 0.199] |
| VALIDATION, konservatif | 5612 | 319 | 0.765 | 0.027 [0.001, 0.056] | 0.062 [0.033, 0.092] | 0.296 [0.250, 0.340] |

Selisih AUROC [95 % CI]; bootstrap bergugus pasien, kedua model pada ulangan yang sama; "—" bila positif < 11.

### Hiperglikemia berat (> 250) lawan euglikemia (70–180) — ICU, EKG sebelum glukosa, SQI GOOD/FAIR; validasi silang 5 lipatan bergugus pasien × 3 benih, dinilai per status obat

- subset: 29674 pasangan, 16875 pasien, positif 2534 (pasien dengan ≥ 1 positif 1818).

| Status obat | Pasangan | Positif | M0 usia + jenis kelamin (LR) | M1 ukuran EKG + M0 (LR) | M1 ukuran EKG + M0 (GB) | M2 CNN gelombang | M3 ECGFounder beku + probe L2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CLEAN | 8967 | 366 | 0.488 [0.455, 0.521] | 0.580 [0.545, 0.615] | 0.585 [0.554, 0.616] | 0.635 [0.605, 0.668] | 0.651 [0.620, 0.687] |
| FLAG_MODERATE_RISK | 9870 | 800 | 0.526 [0.501, 0.551] | 0.592 [0.567, 0.614] | 0.594 [0.569, 0.618] | 0.625 [0.602, 0.648] | 0.634 [0.611, 0.655] |
| FLAG_HIGH_RISK | 1149 | 441 | 0.588 [0.545, 0.625] | 0.671 [0.634, 0.706] | 0.650 [0.612, 0.688] | 0.680 [0.643, 0.716] | 0.699 [0.663, 0.732] |
| EXCLUDE | 9688 | 927 | 0.576 [0.548, 0.599] | 0.732 [0.711, 0.754] | 0.723 [0.704, 0.746] | 0.768 [0.749, 0.787] | 0.773 [0.754, 0.790] |

AUROC [95 % CI] di dalam tiap strata; "—" bila positif < 11.

| Bagian | Pasangan | Positif | AUROC M3 ECGFounder beku + probe L2 | M3 − M2 | M3 − M1 (GB) | M3 − M0 |
| --- | --- | --- | --- | --- | --- | --- |
| CLEAN | 8967 | 366 | 0.651 | 0.016 [-0.017, 0.048] | 0.066 [0.033, 0.101] | 0.163 [0.127, 0.202] |
| FLAG_MODERATE_RISK | 9870 | 800 | 0.634 | 0.008 [-0.014, 0.030] | 0.040 [0.016, 0.062] | 0.107 [0.078, 0.138] |
| FLAG_HIGH_RISK | 1149 | 441 | 0.699 | 0.019 [-0.016, 0.050] | 0.049 [0.016, 0.081] | 0.111 [0.068, 0.155] |
| EXCLUDE | 9688 | 927 | 0.773 | 0.005 [-0.015, 0.023] | 0.050 [0.028, 0.069] | 0.197 [0.169, 0.227] |

Selisih AUROC [95 % CI]; bootstrap bergugus pasien, kedua model pada ulangan yang sama; "—" bila positif < 11.


Waktu jalan: 11.1 menit.
