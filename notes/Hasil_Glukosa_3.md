# Hasil Glukosa 3 — M2 (CNN gelombang) berdampingan dengan M0/M1 (21 November 2026)

Sumber angka: keluaran `glukosa_m2/jalankan_m2.py --boot 1000` (epoch 20) atas
`mimiciv_ecg_glucose_aligned.csv` v1.0.0 (PhysioNet, DOI 10.13026/nts2-vj76, Credentialed Health Data
License 1.5.0) dan gelombang MIMIC-IV-ECG v1.0 (ZIP utuh 36.295.288.053 byte, diunduh 21 Nov 2026,
tidak diekstrak). Dijalankan di laptop, 97,5 menit, venv-gpu (torch 2.14.0+cu130, RTX 5060 Laptop 8 GB);
yang keluar hanya agregat (hitungan 1–10 dicetak `<11`, metrik tidak dicetak bila positif < 11).
Tabel di bawah bagian 0 disalin apa adanya; jangan diketik ulang. Split TEST tidak disentuh.

## Pilihan analisis

- **Singgahan gelombang** (`glukosa_m2/bangun_singgahan.py`, 3,1 menit): 376.648 rekaman unik dibaca
  langsung dari ZIP, checksum WFDB diperiksa. Semua terbaca; 4.807 (1,3 %) punya sampel hilang (diisi 0
  sesudah rerata sadapan dibuang, tetap dipakai); tidak ada yang hilang dari ZIP, checksum salah, atau
  laju/panjang tak sesuai. Disimpan 8 sadapan independen (I, II, V1–V6), 500 → 250 Hz (rerata dua
  sampel), float16 µV.
- **M2:** resep CNN naskah OMI apa adanya (empat blok Conv1d(7) + BatchNorm + ReLU + MaxPool(4), kanal
  32-64-64-128, rerata global, dropout 0,3; Adam 1e-3, 20 epoch tetap, batch 64, bobot kelas seimbang,
  model epoch terakhir), kecuali 8 sadapan masukan. Hanya gelombang; tanpa usia/jenis kelamin. Tidak
  ada penyetelan.
- **Pasangan sama untuk semua model.** Karena tak ada pasangan yang dibuang, estimasi titik M0/M1 sama
  persis dengan Hasil 1 dan 2; CI-nya sedikit berbeda karena urutan bilangan acak bootstrap berubah
  (ada model keempat).
- Susunan analisis = Hasil 1 (latih TRAIN, nilai VALIDATION non-EXCLUDE; dua tugas biner dan tiga
  kelas; satu benih) + Hasil 2 (kendali obat setara di ICU, validasi silang 5 lipatan bergugus pasien
  × 3 benih, AUROC per `intervention_status`).
- Uji jalan `--epoch 1 --boot 20` sebelum jalan penuh; tidak ada perubahan kode di antaranya.

## 0. Catatan pembacaan

1. **Analisis utama, hipoglikemia:** M2 0,614 [0,589; 0,638] lawan M1-GB 0,579 [0,554; 0,603]. CI
   bertumpang; **selisih berpasangan M2 − M1 belum dihitung** (⬜), jadi "M2 lebih baik untuk hipo"
   belum boleh ditulis.
2. **Hiperglikemia berat:** M2 0,717 [0,699; 0,735] lawan M1-GB 0,669 [0,648; 0,688]; CI tidak
   bertumpang. Kemungkinan besar nyata, tetapi selisih berpasangan tetap perlu dihitung sebelum
   diklaim.
3. **Tiga kelas:** AUROC makro M2 0,643 [0,633; 0,653], M1-GB 0,610. Palanchuk 2026 (satu sadapan,
   rancangan lain) 0,632 — sekisaran, tidak langsung sebanding.
4. **ICU bebas obat (CLEAN), hipoglikemia:** M2 0,574 [0,537; 0,615] — CI pertama di strata ini yang
   tidak memuat 0,5 (M1-GB 0,538 [0,497; 0,577]). Sinyal lemah tetapi ada pada pasangan tanpa
   insulin/dekstrosa. Selisih CLEAN − status obat lain untuk M2 tidak berbeda dari 0 (MODERATE +0,011
   [−0,042; 0,069]; EXCLUDE −0,056 [−0,131; 0,017]): pola "hipo hanya terlihat bila obat aktif" dari M1
   tidak muncul pada M2. Selisih M2 − M1 di dalam CLEAN belum dihitung (⬜).
5. **ICU, hiperglikemia berat:** M2 CLEAN 0,646 [0,613; 0,676] lawan EXCLUDE 0,774, selisih −0,129
   [−0,169; −0,093] — ketergantungan pada status obat bertahan, dan M0 juga bergeser (−0,088), jadi
   sebagian tetap campuran kasus seperti di Hasil 2.
6. **Kalibrasi M2 buruk:** O/E 0,051 (hipo) dan 0,288 (hiper berat), kemiringan 0,39–0,42. Penyebab:
   bobot kelas seimbang menggeser peluang ke atas, dan kemiringan < 1 menandakan prediksi terlalu
   ekstrem (pada uji jalan 1 epoch kemiringannya ±1,0, jadi 20 epoch tanpa henti-dini cenderung
   bertambah yakin). Untuk naskah: M2 dinilai pada pembedaan (AUROC/AUPRC); kalibrasi dilaporkan apa
   adanya atau dikalibrasi ulang di VALIDATION — diputuskan sebelum TEST dibuka.
7. **Batas:** analisis utama satu benih; cuDNN tidak deterministik, jadi jalan ulang bisa bergeser
   sedikit di desimal ketiga.

**Akibat untuk rencana:** belum ada klaim tertulis yang dibantah. Yang baru: gelombang memberi sinyal
hipo kecil di ICU bebas obat, yang tidak diberikan ukuran mesin. Sebelum menulis klaim "gelombang
mengalahkan ukuran mesin": tambahkan selisih berpasangan M2 − M1 (analisis utama dan strata CLEAN),
lalu M3.

## Keluaran skrip

Judul dari skrip diturunkan satu tingkat (`#` ditambah); isi lain apa adanya.

## M0/M1/M2 — pasangan dengan gelombang terbaca

- pasangan: 437671; gelombang terbaca 437671; dibuang 0.

Perangkat CNN: cuda; epoch 20; bootstrap 1000 bergugus pasien.

### Hipoglikemia (< 70) lawan euglikemia (70–180) — latih TRAIN, nilai VALIDATION (non-EXCLUDE)

- TRAIN: 285678 pasangan, 100015 pasien, positif 5445 (pasien dengan ≥ 1 positif 4043).
- VALIDATION: 36469 pasangan, 12572 pasien, positif 696 (pasien dengan ≥ 1 positif 536).

| Model | AUROC [95 % CI] | AUPRC [95 % CI] | Se @ Sp 90 % | Brier | O/E | Kemiringan kalibrasi |
| --- | --- | --- | --- | --- | --- | --- |
| M0 usia + jenis kelamin (LR) | 0.519 [0.497, 0.545] | 0.020 [0.018, 0.023] | 0.118 | 0.019 | 1.002 | 0.939 |
| M1 ukuran EKG + M0 (LR) | 0.554 [0.528, 0.579] | 0.025 [0.021, 0.031] | 0.145 | 0.019 | 1.005 | 0.939 |
| M1 ukuran EKG + M0 (GB) | 0.579 [0.554, 0.603] | 0.024 [0.021, 0.028] | 0.141 | 0.019 | 1.013 | 0.620 |
| M2 CNN gelombang | 0.614 [0.589, 0.638] | 0.029 [0.026, 0.034] | 0.188 | 0.172 | 0.051 | 0.392 |

Prevalensi positif: 1.91 %. CI: bootstrap bergugus pasien.

Kepekaan (model yang sama, dinilai pada bagian VALIDATION), AUROC [95 % CI]:

| Subset | n positif | M0 usia + jenis kelamin (LR) | M1 ukuran EKG + M0 (LR) | M1 ukuran EKG + M0 (GB) | M2 CNN gelombang |
| --- | --- | --- | --- | --- | --- |
| selisih ≤ 60 menit | 354 | 0.536 [0.504, 0.568] | 0.567 [0.532, 0.600] | 0.598 [0.565, 0.630] | 0.617 [0.583, 0.648] |
| SQI GOOD | 584 | 0.530 [0.503, 0.556] | 0.561 [0.535, 0.591] | 0.587 [0.561, 0.614] | 0.623 [0.597, 0.649] |
| konservatif (anjuran README) | 84 | 0.498 [0.431, 0.562] | 0.571 [0.502, 0.638] | 0.583 [0.521, 0.647] | 0.666 [0.603, 0.727] |

### Hiperglikemia berat (> 250) lawan euglikemia (70–180) — latih TRAIN, nilai VALIDATION (non-EXCLUDE)

- TRAIN: 298202 pasangan, 102061 pasien, positif 17969 (pasien dengan ≥ 1 positif 8375).
- VALIDATION: 38227 pasangan, 12783 pasien, positif 2454 (pasien dengan ≥ 1 positif 1058).

| Model | AUROC [95 % CI] | AUPRC [95 % CI] | Se @ Sp 90 % | Brier | O/E | Kemiringan kalibrasi |
| --- | --- | --- | --- | --- | --- | --- |
| M0 usia + jenis kelamin (LR) | 0.545 [0.518, 0.573] | 0.070 [0.060, 0.084] | 0.086 | 0.060 | 1.066 | 0.862 |
| M1 ukuran EKG + M0 (LR) | 0.641 [0.620, 0.663] | 0.102 [0.088, 0.119] | 0.206 | 0.059 | 1.069 | 1.086 |
| M1 ukuran EKG + M0 (GB) | 0.669 [0.648, 0.688] | 0.117 [0.100, 0.136] | 0.239 | 0.059 | 1.089 | 0.941 |
| M2 CNN gelombang | 0.717 [0.699, 0.735] | 0.171 [0.143, 0.201] | 0.326 | 0.116 | 0.288 | 0.418 |

Prevalensi positif: 6.42 %. CI: bootstrap bergugus pasien.

Kepekaan (model yang sama, dinilai pada bagian VALIDATION), AUROC [95 % CI]:

| Subset | n positif | M0 usia + jenis kelamin (LR) | M1 ukuran EKG + M0 (LR) | M1 ukuran EKG + M0 (GB) | M2 CNN gelombang |
| --- | --- | --- | --- | --- | --- |
| selisih ≤ 60 menit | 1377 | 0.527 [0.496, 0.555] | 0.643 [0.622, 0.665] | 0.670 [0.651, 0.691] | 0.728 [0.709, 0.748] |
| SQI GOOD | 2162 | 0.532 [0.501, 0.560] | 0.654 [0.631, 0.678] | 0.677 [0.657, 0.699] | 0.730 [0.712, 0.749] |
| konservatif (anjuran README) | 319 | 0.469 [0.436, 0.503] | 0.670 [0.639, 0.704] | 0.703 [0.675, 0.733] | 0.737 [0.705, 0.769] |

### Tiga kelas (< 70 / 70–180 / > 180), sebanding Palanchuk 2026 — latih TRAIN, nilai VALIDATION

VALIDATION: hipo 696, eu 35773, hiper 6036.

| Model | AUROC hipo | AUROC eu | AUROC hiper | AUROC makro |
| --- | --- | --- | --- | --- |
| M0 usia + jenis kelamin (LR) | 0.522 [0.496, 0.545] | 0.517 [0.498, 0.535] | 0.521 [0.501, 0.540] | 0.520 [0.507, 0.534] |
| M1 ukuran EKG + M0 (LR) | 0.547 [0.522, 0.570] | 0.605 [0.594, 0.616] | 0.613 [0.601, 0.625] | 0.588 [0.577, 0.600] |
| M1 ukuran EKG + M0 (GB) | 0.561 [0.539, 0.583] | 0.629 [0.617, 0.642] | 0.639 [0.626, 0.651] | 0.610 [0.598, 0.621] |
| M2 CNN gelombang | 0.596 [0.573, 0.620] | 0.661 [0.649, 0.673] | 0.673 [0.661, 0.683] | 0.643 [0.633, 0.653] |

### Hipoglikemia (< 70) lawan euglikemia (70–180) — ICU, EKG sebelum glukosa, SQI GOOD/FAIR; satu model validasi silang 5 lipatan bergugus pasien × 3 benih, dinilai per status obat

- subset: 27750 pasangan, 16059 pasien, positif 610 (pasien dengan ≥ 1 positif 528).

| Status obat | Pasangan | Positif | M0 usia + jenis kelamin (LR) | M1 ukuran EKG + M0 (LR) | M1 ukuran EKG + M0 (GB) | M2 CNN gelombang |
| --- | --- | --- | --- | --- | --- | --- |
| CLEAN | 8825 | 224 | 0.486 [0.447, 0.522] | 0.528 [0.490, 0.570] | 0.538 [0.497, 0.577] | 0.574 [0.537, 0.615] |
| FLAG_MODERATE_RISK | 9345 | 275 | 0.474 [0.434, 0.515] | 0.588 [0.548, 0.624] | 0.593 [0.558, 0.626] | 0.563 [0.523, 0.601] |
| FLAG_HIGH_RISK | 723 | 15 | 0.685 [0.560, 0.798] | 0.460 [0.305, 0.607] | 0.658 [0.541, 0.769] | 0.584 [0.427, 0.715] |
| EXCLUDE | 8857 | 96 | 0.459 [0.390, 0.532] | 0.567 [0.495, 0.641] | 0.578 [0.516, 0.645] | 0.630 [0.560, 0.691] |

AUROC [95 % CI] di dalam tiap strata; "—" bila positif < 11.

| CLEAN − status | M0 usia + jenis kelamin (LR) | M1 ukuran EKG + M0 (LR) | M1 ukuran EKG + M0 (GB) | M2 CNN gelombang |
| --- | --- | --- | --- | --- |
| FLAG_MODERATE_RISK | 0.012 [-0.050, 0.071] | -0.060 [-0.114, -0.007] | -0.055 [-0.101, -0.002] | 0.011 [-0.042, 0.069] |
| FLAG_HIGH_RISK | -0.199 [-0.311, -0.087] | 0.068 [-0.078, 0.221] | -0.120 [-0.242, 0.004] | -0.010 [-0.145, 0.154] |
| EXCLUDE | 0.027 [-0.054, 0.106] | -0.039 [-0.121, 0.036] | -0.040 [-0.119, 0.034] | -0.056 [-0.131, 0.017] |

Selisih AUROC [95 % CI]; bootstrap bergugus pasien atas seluruh subset, kedua AUROC pada ulangan yang sama; "—" bila positif < 11.

### Hiperglikemia berat (> 250) lawan euglikemia (70–180) — ICU, EKG sebelum glukosa, SQI GOOD/FAIR; satu model validasi silang 5 lipatan bergugus pasien × 3 benih, dinilai per status obat

- subset: 29674 pasangan, 16875 pasien, positif 2534 (pasien dengan ≥ 1 positif 1818).

| Status obat | Pasangan | Positif | M0 usia + jenis kelamin (LR) | M1 ukuran EKG + M0 (LR) | M1 ukuran EKG + M0 (GB) | M2 CNN gelombang |
| --- | --- | --- | --- | --- | --- | --- |
| CLEAN | 8967 | 366 | 0.488 [0.458, 0.522] | 0.580 [0.550, 0.614] | 0.585 [0.554, 0.616] | 0.646 [0.613, 0.676] |
| FLAG_MODERATE_RISK | 9870 | 800 | 0.526 [0.502, 0.549] | 0.592 [0.567, 0.615] | 0.594 [0.571, 0.616] | 0.632 [0.610, 0.653] |
| FLAG_HIGH_RISK | 1149 | 441 | 0.588 [0.543, 0.627] | 0.671 [0.635, 0.706] | 0.650 [0.611, 0.692] | 0.674 [0.639, 0.705] |
| EXCLUDE | 9688 | 927 | 0.576 [0.552, 0.601] | 0.732 [0.710, 0.752] | 0.723 [0.702, 0.744] | 0.774 [0.756, 0.793] |

AUROC [95 % CI] di dalam tiap strata; "—" bila positif < 11.

| CLEAN − status | M0 usia + jenis kelamin (LR) | M1 ukuran EKG + M0 (LR) | M1 ukuran EKG + M0 (GB) | M2 CNN gelombang |
| --- | --- | --- | --- | --- |
| FLAG_MODERATE_RISK | -0.038 [-0.075, 0.001] | -0.011 [-0.053, 0.029] | -0.009 [-0.046, 0.029] | 0.014 [-0.027, 0.050] |
| FLAG_HIGH_RISK | -0.100 [-0.149, -0.048] | -0.091 [-0.145, -0.044] | -0.065 [-0.114, -0.017] | -0.028 [-0.074, 0.020] |
| EXCLUDE | -0.088 [-0.127, -0.045] | -0.151 [-0.194, -0.111] | -0.138 [-0.176, -0.101] | -0.129 [-0.169, -0.093] |

Selisih AUROC [95 % CI]; bootstrap bergugus pasien atas seluruh subset, kedua AUROC pada ulangan yang sama; "—" bila positif < 11.


Waktu jalan: 97.5 menit.
