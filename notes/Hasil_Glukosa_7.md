# Hasil Glukosa 7 — M3b (ECG-JEPA beku + probe L2) berdampingan dengan M0–M3 dan M2b (21 November 2026)

Sumber angka: keluaran (A) `glukosa_m3/jalankan_m3.py --model ecgjepa --boot 1000` dan (B)
`glukosa_m3/bandingkan_simpanan.py --boot 1000` atas `mimiciv_ecg_glucose_aligned.csv` v1.0.0
(PhysioNet, DOI 10.13026/nts2-vj76, Credentialed Health Data License 1.5.0) dan gelombang MIMIC-IV-ECG
v1.0. Dijalankan di laptop: embedding 115,3 menit (`glukosa_m3/bangun_embedding.py --model ecgjepa`,
RTX 5060), probe 21,4 menit, perbandingan tersimpan 4,4 menit; yang keluar hanya agregat (hitungan 1–10
dicetak `<11`, metrik tidak dicetak bila positif < 11). Tabel di bawah bagian 0 disambung langsung dari
log skrip; jangan diketik ulang. Split TEST tidak disentuh.

## Pilihan analisis

- **Model fondasi kedua** (keputusan penulis 21 Nov 2026, rencana §F langkah 6c): ECG-JEPA (Kim 2024,
  arXiv 2410.08559; kode `sehunfromdaegu/ECG_JEPA` komit d937ad2, MIT; titik periksa multi-block
  `multiblock_epoch100.pth` dari README). Dilatih-awal pada Chapman/Ningbo dan CODE-15, tanpa MIMIC-IV-ECG
  (bukti di rencana langkah 6c); dipilih sebelum hasil apa pun dilihat. Beku.
- **Embedding:** 376.648 rekaman dibaca ulang dari ZIP (12 sadapan × 500 Hz), pra-proses resmi
  `linear_eval`: mV, sadapan I, II, V1–V6, `scipy.signal.resample` 5.000 → 2.500 sampel (250 Hz), tanpa
  filter dan tanpa normalisasi; sampel hilang diisi 0. `encoder.representation` (rerata token, 768
  dimensi). Semua terhingga. Diuji 21 Nov pada tiga rekaman terbuka: sinyal dan embedding dari jalur ZIP
  sama persis (selisih 0,0) dengan jalur resmi (wfdb `p_signal`); batch sama dengan satu-satu.
- **timm:** tidak dipasang (menarik torchvision). Modul minimal `timm_pengganti` menyediakan DropPath
  (identitas saat eval) dan `trunc_normal_` (hanya inisialisasi); bobot dimuat ketat (85,4 jt parameter),
  jadi keluaran tidak berubah.
- **Probe dan susunan:** sama dengan M3 (Hasil 5): StandardScaler + regresi logistik L2, C dari kisi
  {10⁻³ … 100} dengan log-loss 3 lipatan StratifiedGroupKFold di data latih; analisis utama TRAIN →
  VALIDATION non-EXCLUDE; ICU setara dengan validasi silang 5 lipatan bergugus pasien × 3 benih.
- **Pembanding:** prediksi M0–M3 dibaca dari jalan sebelumnya dan dicocokkan baris demi baris (y dan
  subject_id identik, diperiksa skrip). (A) memakai M2 satu benih; (B) menambah M2b (rerata 3 benih,
  Hasil 6) untuk analisis utama tanpa melatih apa pun. Di ICU M2 sudah rerata 3 benih sejak Hasil 3.
  Di bagian 0, AUROC dan selisih atas M3/M2b diambil dari (B), selisih atas M1-GB dari (A); CI bootstrap (A) dan (B) bisa beda di digit ketiga
  karena set model berbeda (contoh M3b hiper berat [0,728; 0,764] lawan [0,727; 0,764]).

## 0. Catatan pembacaan

1. **Hipoglikemia, VALIDATION:** M3b 0,643 [0,618; 0,666], tertinggi dari semua model. M3b − M3 +0,029
   [0,011; 0,048]; M3b − M2b +0,025 [0,004; 0,047]; M3b − M1-GB +0,064 [0,039; 0,090]. Bagian
   konservatif (84 positif): M3b 0,692, M3b − M1-GB +0,110 [0,037; 0,185]; selisih atas M3 dan M2b
   searah tetapi CI melewati 0.
2. **Hiperglikemia berat, VALIDATION:** M3b 0,747 [0,727; 0,764]. M3b − M3 +0,030 [0,020; 0,042];
   **M3b − M2b +0,006 [−0,009; 0,020], setara**; M3b − M1-GB +0,078 [0,063; 0,094]. Bagian konservatif
   (319 positif): M3b 0,799, M3b − M2b +0,038 [0,010; 0,065].
3. **Kalibrasi:** M3b baik tanpa kalibrasi ulang (O/E 1,002 dan 1,065; kemiringan 1,02 dan 1,09), seperti
   M3 dan berbeda dengan M2/M2b.
4. **ICU bebas obat (CLEAN), hipoglikemia:** M3b 0,584 [0,547; 0,621]; M3b − M1-GB +0,046 [0,001; 0,087],
   batas bawah tepat di atas 0, satu dari 12 selisih ICU hipo, jadi **belum dianggap terbukti**; M3b − M3
   +0,019 [−0,014; 0,052]. Hasil 5: M3 − M1-GB di sel ini +0,027 [−0,018; 0,070].
5. **ICU CLEAN, hiperglikemia berat:** M3b 0,662; M3b − M1-GB +0,076 [0,047; 0,109]; M3b − M3 +0,010
   [−0,014; 0,036].
6. **Sel lain:** FLAG_HIGH_RISK hipo (15 positif) tidak ditafsirkan. Di ICU M3b ≥ M3 di semua strata kecuali
   FLAG_HIGH_RISK hipo; selisih atas M3 dengan CI di atas 0 hanya pada FLAG_MODERATE_RISK dan EXCLUDE
   hiper berat.

**Akibat untuk rencana:**

- Klaim inti menguat. Gelombang EKG 10 s menambah pembedaan di atas ukuran mesin. Dengan model
  gelombang terbaik selisihnya +0,06 AUROC untuk hipo dan +0,08 untuk hiper berat.
- Dugaan Hasil 5 tentang batas informasi gelombang **gugur**. Dugaan itu berbunyi: dua metode sampai di
  AUROC yang sama, jadi mungkin gelombang memang hanya sejauh itu. Metode ketiga ternyata lebih tinggi
  untuk hipo, jadi pilihan model fondasi berpengaruh sekitar 0,03. Dugaan itu tidak dipakai di naskah.
- Semua AUROC hipo tetap < 0,65. Keunggulan atas ukuran mesin pada hipoglikemia di ICU bebas obat masih
  belum terbukti.
- ⬜ **Keputusan penulis sebelum TEST dibuka:** model gelombang utama dan perbandingan utama untuk TEST.
  Pilihan: M3b saja; M3b + M2b; atau ketiganya dengan koreksi. Sekarang tiga model gelombang dinilai di
  VALIDATION. Memilih yang terbaik di VALIDATION lalu hanya melaporkannya di TEST harus ditulis
  sebagai pemilihan.

## Keluaran skrip

Judul dari skrip diturunkan dua tingkat (`##` ditambah); isi lain apa adanya.

## (A) jalankan_m3.py --model ecgjepa

### M0/M1/M2 — pasangan dengan gelombang terbaca

- pasangan: 437671; gelombang terbaca 437671; dibuang 0.

M3b ECG-JEPA beku + probe L2; bootstrap 1000 bergugus pasien.

#### Hipoglikemia (< 70) lawan euglikemia (70–180) — latih TRAIN, nilai VALIDATION (non-EXCLUDE)

| Model | AUROC [95 % CI] | AUPRC [95 % CI] | Se @ Sp 90 % | Brier | O/E | Kemiringan kalibrasi |
| --- | --- | --- | --- | --- | --- | --- |
| M0 usia + jenis kelamin (LR) | 0.519 [0.497, 0.545] | 0.020 [0.018, 0.023] | 0.118 | 0.019 | 1.002 | 0.939 |
| M1 ukuran EKG + M0 (LR) | 0.554 [0.528, 0.579] | 0.025 [0.021, 0.031] | 0.145 | 0.019 | 1.005 | 0.939 |
| M1 ukuran EKG + M0 (GB) | 0.579 [0.554, 0.603] | 0.024 [0.021, 0.028] | 0.141 | 0.019 | 1.013 | 0.620 |
| M2 CNN gelombang | 0.613 [0.588, 0.637] | 0.028 [0.025, 0.032] | 0.191 | 0.168 | 0.052 | 0.405 |
| M3 ECGFounder beku + probe L2 | 0.614 [0.590, 0.637] | 0.029 [0.026, 0.035] | 0.187 | 0.019 | 1.007 | 0.727 |
| M3b ECG-JEPA beku + probe L2 | 0.643 [0.618, 0.666] | 0.036 [0.031, 0.043] | 0.223 | 0.019 | 1.002 | 1.017 |

Prevalensi positif: 1.91 %. CI: bootstrap bergugus pasien.

| Bagian | Pasangan | Positif | AUROC M3b ECG-JEPA beku + probe L2 | M3b − M3 | M3b − M1 (GB) | M3b − M0 |
| --- | --- | --- | --- | --- | --- | --- |
| VALIDATION | 36469 | 696 | 0.643 | 0.029 [0.011, 0.048] | 0.064 [0.039, 0.090] | 0.124 [0.091, 0.157] |
| VALIDATION, konservatif | 5377 | 84 | 0.692 | 0.049 [-0.008, 0.108] | 0.110 [0.037, 0.185] | 0.194 [0.106, 0.290] |

Selisih AUROC [95 % CI]; bootstrap bergugus pasien, kedua model pada ulangan yang sama; "—" bila positif < 11.

#### Hipoglikemia (< 70) lawan euglikemia (70–180) — ICU, EKG sebelum glukosa, SQI GOOD/FAIR; validasi silang 5 lipatan bergugus pasien × 3 benih, dinilai per status obat

- subset: 27750 pasangan, 16059 pasien, positif 610 (pasien dengan ≥ 1 positif 528).

| Status obat | Pasangan | Positif | M0 usia + jenis kelamin (LR) | M1 ukuran EKG + M0 (LR) | M1 ukuran EKG + M0 (GB) | M2 CNN gelombang | M3 ECGFounder beku + probe L2 | M3b ECG-JEPA beku + probe L2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CLEAN | 8825 | 224 | 0.486 [0.451, 0.525] | 0.528 [0.487, 0.566] | 0.538 [0.498, 0.578] | 0.581 [0.542, 0.620] | 0.565 [0.527, 0.602] | 0.584 [0.547, 0.621] |
| FLAG_MODERATE_RISK | 9345 | 275 | 0.474 [0.433, 0.516] | 0.588 [0.551, 0.624] | 0.593 [0.559, 0.630] | 0.573 [0.535, 0.610] | 0.617 [0.580, 0.654] | 0.643 [0.608, 0.678] |
| FLAG_HIGH_RISK | 723 | 15 | 0.685 [0.565, 0.793] | 0.460 [0.320, 0.610] | 0.658 [0.549, 0.773] | 0.622 [0.447, 0.750] | 0.617 [0.499, 0.738] | 0.578 [0.429, 0.719] |
| EXCLUDE | 8857 | 96 | 0.459 [0.387, 0.526] | 0.567 [0.495, 0.640] | 0.578 [0.515, 0.641] | 0.619 [0.555, 0.680] | 0.620 [0.561, 0.680] | 0.654 [0.595, 0.715] |

AUROC [95 % CI] di dalam tiap strata; "—" bila positif < 11.

| Bagian | Pasangan | Positif | AUROC M3b ECG-JEPA beku + probe L2 | M3b − M3 | M3b − M1 (GB) | M3b − M0 |
| --- | --- | --- | --- | --- | --- | --- |
| CLEAN | 8825 | 224 | 0.584 | 0.019 [-0.014, 0.052] | 0.046 [0.001, 0.087] | 0.099 [0.044, 0.152] |
| FLAG_MODERATE_RISK | 9345 | 275 | 0.643 | 0.025 [-0.006, 0.059] | 0.049 [0.011, 0.086] | 0.169 [0.110, 0.228] |
| FLAG_HIGH_RISK | 723 | 15 | 0.578 | -0.039 [-0.229, 0.111] | -0.080 [-0.255, 0.104] | -0.107 [-0.272, 0.047] |
| EXCLUDE | 8857 | 96 | 0.654 | 0.033 [-0.027, 0.091] | 0.076 [0.009, 0.140] | 0.195 [0.105, 0.284] |

Selisih AUROC [95 % CI]; bootstrap bergugus pasien, kedua model pada ulangan yang sama; "—" bila positif < 11.

#### Hiperglikemia berat (> 250) lawan euglikemia (70–180) — latih TRAIN, nilai VALIDATION (non-EXCLUDE)

| Model | AUROC [95 % CI] | AUPRC [95 % CI] | Se @ Sp 90 % | Brier | O/E | Kemiringan kalibrasi |
| --- | --- | --- | --- | --- | --- | --- |
| M0 usia + jenis kelamin (LR) | 0.545 [0.516, 0.573] | 0.070 [0.061, 0.085] | 0.086 | 0.060 | 1.066 | 0.862 |
| M1 ukuran EKG + M0 (LR) | 0.641 [0.618, 0.664] | 0.102 [0.089, 0.119] | 0.206 | 0.059 | 1.069 | 1.086 |
| M1 ukuran EKG + M0 (GB) | 0.669 [0.649, 0.689] | 0.117 [0.101, 0.137] | 0.239 | 0.059 | 1.089 | 0.941 |
| M2 CNN gelombang | 0.719 [0.701, 0.738] | 0.172 [0.146, 0.205] | 0.331 | 0.143 | 0.231 | 0.451 |
| M3 ECGFounder beku + probe L2 | 0.716 [0.697, 0.735] | 0.164 [0.137, 0.197] | 0.315 | 0.057 | 1.071 | 1.029 |
| M3b ECG-JEPA beku + probe L2 | 0.747 [0.728, 0.764] | 0.200 [0.167, 0.241] | 0.381 | 0.056 | 1.065 | 1.086 |

Prevalensi positif: 6.42 %. CI: bootstrap bergugus pasien.

| Bagian | Pasangan | Positif | AUROC M3b ECG-JEPA beku + probe L2 | M3b − M3 | M3b − M1 (GB) | M3b − M0 |
| --- | --- | --- | --- | --- | --- | --- |
| VALIDATION | 38227 | 2454 | 0.747 | 0.030 [0.019, 0.041] | 0.078 [0.063, 0.094] | 0.202 [0.172, 0.228] |
| VALIDATION, konservatif | 5612 | 319 | 0.799 | 0.034 [0.014, 0.052] | 0.096 [0.070, 0.124] | 0.330 [0.286, 0.375] |

Selisih AUROC [95 % CI]; bootstrap bergugus pasien, kedua model pada ulangan yang sama; "—" bila positif < 11.

#### Hiperglikemia berat (> 250) lawan euglikemia (70–180) — ICU, EKG sebelum glukosa, SQI GOOD/FAIR; validasi silang 5 lipatan bergugus pasien × 3 benih, dinilai per status obat

- subset: 29674 pasangan, 16875 pasien, positif 2534 (pasien dengan ≥ 1 positif 1818).

| Status obat | Pasangan | Positif | M0 usia + jenis kelamin (LR) | M1 ukuran EKG + M0 (LR) | M1 ukuran EKG + M0 (GB) | M2 CNN gelombang | M3 ECGFounder beku + probe L2 | M3b ECG-JEPA beku + probe L2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CLEAN | 8967 | 366 | 0.488 [0.455, 0.521] | 0.580 [0.545, 0.615] | 0.585 [0.554, 0.616] | 0.635 [0.605, 0.668] | 0.651 [0.620, 0.687] | 0.662 [0.630, 0.694] |
| FLAG_MODERATE_RISK | 9870 | 800 | 0.526 [0.501, 0.551] | 0.592 [0.567, 0.614] | 0.594 [0.569, 0.618] | 0.625 [0.602, 0.648] | 0.634 [0.611, 0.655] | 0.659 [0.636, 0.681] |
| FLAG_HIGH_RISK | 1149 | 441 | 0.588 [0.545, 0.625] | 0.671 [0.634, 0.706] | 0.650 [0.612, 0.688] | 0.680 [0.643, 0.716] | 0.699 [0.663, 0.732] | 0.711 [0.678, 0.745] |
| EXCLUDE | 9688 | 927 | 0.576 [0.548, 0.599] | 0.732 [0.711, 0.754] | 0.723 [0.704, 0.746] | 0.768 [0.749, 0.787] | 0.773 [0.754, 0.790] | 0.800 [0.782, 0.817] |

AUROC [95 % CI] di dalam tiap strata; "—" bila positif < 11.

| Bagian | Pasangan | Positif | AUROC M3b ECG-JEPA beku + probe L2 | M3b − M3 | M3b − M1 (GB) | M3b − M0 |
| --- | --- | --- | --- | --- | --- | --- |
| CLEAN | 8967 | 366 | 0.662 | 0.010 [-0.014, 0.036] | 0.076 [0.047, 0.109] | 0.173 [0.137, 0.214] |
| FLAG_MODERATE_RISK | 9870 | 800 | 0.659 | 0.026 [0.010, 0.043] | 0.066 [0.044, 0.088] | 0.133 [0.104, 0.164] |
| FLAG_HIGH_RISK | 1149 | 441 | 0.711 | 0.012 [-0.013, 0.037] | 0.061 [0.028, 0.095] | 0.123 [0.080, 0.167] |
| EXCLUDE | 9688 | 927 | 0.800 | 0.027 [0.011, 0.042] | 0.077 [0.055, 0.097] | 0.224 [0.194, 0.255] |

Selisih AUROC [95 % CI]; bootstrap bergugus pasien, kedua model pada ulangan yang sama; "—" bila positif < 11.


Waktu jalan: 21.4 menit.

## (B) bandingkan_simpanan.py

### M0/M1/M2 — pasangan dengan gelombang terbaca

- pasangan: 437671; gelombang terbaca 437671; dibuang 0.

Prediksi tersimpan, tanpa pelatihan; bootstrap 1000 bergugus pasien.

#### Hipoglikemia (< 70) lawan euglikemia (70–180) — latih TRAIN, nilai VALIDATION (non-EXCLUDE)

| Model | AUROC [95 % CI] | AUPRC [95 % CI] | Se @ Sp 90 % | Brier | O/E | Kemiringan kalibrasi |
| --- | --- | --- | --- | --- | --- | --- |
| M0 usia + jenis kelamin (LR) | 0.519 [0.497, 0.545] | 0.020 [0.018, 0.023] | 0.118 | 0.019 | 1.002 | 0.939 |
| M1 ukuran EKG + M0 (LR) | 0.554 [0.528, 0.579] | 0.025 [0.021, 0.031] | 0.145 | 0.019 | 1.005 | 0.939 |
| M1 ukuran EKG + M0 (GB) | 0.579 [0.554, 0.603] | 0.024 [0.021, 0.028] | 0.141 | 0.019 | 1.013 | 0.620 |
| M2 CNN gelombang | 0.613 [0.588, 0.637] | 0.028 [0.025, 0.032] | 0.191 | 0.168 | 0.052 | 0.405 |
| M2 CNN gelombang, rerata 3 benih | 0.618 [0.593, 0.640] | 0.029 [0.026, 0.034] | 0.210 | 0.187 | 0.048 | 0.478 |
| M3 ECGFounder beku + probe L2 | 0.614 [0.590, 0.637] | 0.029 [0.026, 0.035] | 0.187 | 0.019 | 1.007 | 0.727 |
| M3b ECG-JEPA beku + probe L2 | 0.643 [0.618, 0.666] | 0.036 [0.031, 0.043] | 0.223 | 0.019 | 1.002 | 1.017 |

Prevalensi positif: 1.91 %. CI: bootstrap bergugus pasien.

| Bagian | Pasangan | Positif | AUROC M3b ECG-JEPA beku + probe L2 | M3b − M2b | M3b − M3 | M2b − M3 |
| --- | --- | --- | --- | --- | --- | --- |
| VALIDATION | 36469 | 696 | 0.643 | 0.025 [0.004, 0.047] | 0.029 [0.011, 0.048] | 0.004 [-0.020, 0.027] |
| VALIDATION, konservatif | 5377 | 84 | 0.692 | 0.034 [-0.015, 0.084] | 0.049 [-0.008, 0.108] | 0.015 [-0.045, 0.075] |

Selisih AUROC [95 % CI]; bootstrap bergugus pasien, kedua model pada ulangan yang sama; "—" bila positif < 11.

#### Hiperglikemia berat (> 250) lawan euglikemia (70–180) — latih TRAIN, nilai VALIDATION (non-EXCLUDE)

| Model | AUROC [95 % CI] | AUPRC [95 % CI] | Se @ Sp 90 % | Brier | O/E | Kemiringan kalibrasi |
| --- | --- | --- | --- | --- | --- | --- |
| M0 usia + jenis kelamin (LR) | 0.545 [0.518, 0.571] | 0.070 [0.061, 0.083] | 0.086 | 0.060 | 1.066 | 0.862 |
| M1 ukuran EKG + M0 (LR) | 0.641 [0.618, 0.661] | 0.102 [0.089, 0.117] | 0.206 | 0.059 | 1.069 | 1.086 |
| M1 ukuran EKG + M0 (GB) | 0.669 [0.649, 0.688] | 0.117 [0.100, 0.136] | 0.239 | 0.059 | 1.089 | 0.941 |
| M2 CNN gelombang | 0.719 [0.698, 0.737] | 0.172 [0.144, 0.203] | 0.331 | 0.143 | 0.231 | 0.451 |
| M2 CNN gelombang, rerata 3 benih | 0.741 [0.720, 0.759] | 0.198 [0.162, 0.235] | 0.367 | 0.171 | 0.182 | 0.738 |
| M3 ECGFounder beku + probe L2 | 0.716 [0.696, 0.735] | 0.164 [0.136, 0.194] | 0.315 | 0.057 | 1.071 | 1.029 |
| M3b ECG-JEPA beku + probe L2 | 0.747 [0.727, 0.764] | 0.200 [0.166, 0.238] | 0.381 | 0.056 | 1.065 | 1.086 |

Prevalensi positif: 6.42 %. CI: bootstrap bergugus pasien.

| Bagian | Pasangan | Positif | AUROC M3b ECG-JEPA beku + probe L2 | M3b − M2b | M3b − M3 | M2b − M3 |
| --- | --- | --- | --- | --- | --- | --- |
| VALIDATION | 38227 | 2454 | 0.747 | 0.006 [-0.009, 0.020] | 0.030 [0.020, 0.042] | 0.025 [0.009, 0.040] |
| VALIDATION, konservatif | 5612 | 319 | 0.799 | 0.038 [0.010, 0.065] | 0.034 [0.014, 0.055] | -0.004 [-0.030, 0.025] |

Selisih AUROC [95 % CI]; bootstrap bergugus pasien, kedua model pada ulangan yang sama; "—" bila positif < 11.


Waktu jalan: 4.4 menit.
