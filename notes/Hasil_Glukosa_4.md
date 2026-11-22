# Hasil Glukosa 4 — selisih AUROC berpasangan M2 − M1 dan M2 − M0 (21 November 2026)

Sumber angka: keluaran `glukosa_m2/jalankan_selisih.py --boot 1000` (epoch 20) atas
`mimiciv_ecg_glucose_aligned.csv` v1.0.0 (PhysioNet, DOI 10.13026/nts2-vj76, Credentialed Health Data
License 1.5.0) dan singgahan gelombang MIMIC-IV-ECG v1.0 (Hasil_Glukosa_3.md). Dijalankan di laptop,
42,5 menit, venv-gpu (torch 2.14.0+cu130, RTX 5060 Laptop 8 GB); yang keluar hanya agregat (hitungan
1–10 dicetak `<11`, metrik tidak dicetak bila positif < 11). Tabel di bawah bagian 0 disalin apa adanya;
jangan diketik ulang. Split TEST tidak disentuh. Menjawab ⬜ di Hasil 3 catatan 1, 2, 4.

## Pilihan analisis

- Model, pasangan, resep, dan susunan sama dengan Hasil 3 (latih TRAIN–nilai VALIDATION non-EXCLUDE,
  satu benih; ICU setara dengan validasi silang 5 lipatan bergugus pasien × 3 benih). Tiga kelas tidak
  diulang.
- Selisih AUROC M2 − M1-GB, M2 − M1-LR, M2 − M0 pada pasangan yang sama; CI dari 1000 bootstrap
  bergugus pasien, kedua model pada ulangan yang sama. Di ICU selisih dihitung di dalam tiap
  `intervention_status`. Perbandingan utama: M2 − M1-GB (model M1 terkuat) di VALIDATION — urutan
  pertama di `PERBANDINGAN`, ditulis sebelum jalan; baris lain pendukung.
- Baris tambahan "VALIDATION, konservatif" (anjuran README: IDEAL/USABLE + EKG sebelum glukosa + CLEAN).
- CNN dilatih ulang. Prediksi per pasangan disimpan di `data/mimic-iv-ext-ecg-glucose/`
  (drive terenkripsi, di luar repo; tidak dibuka di sesi asisten) untuk perbandingan M3 nanti.
- Uji jalan hanya pada data sintetis (jalur data nyata sama dengan Hasil 3).

## 0. Catatan pembacaan

1. **Hipoglikemia, VALIDATION:** M2 − M1-GB **+0,033 [0,004; 0,066]**; CI tidak memuat 0, tetapi batas
   bawahnya nyaris 0. Terhadap M1-LR +0,058 [0,026; 0,091], terhadap M0 +0,094. Pada subset konservatif
   +0,083 [0,012; 0,159] (84 hipo).
2. **Hiperglikemia berat, VALIDATION:** M2 − M1-GB **+0,051 [0,033; 0,068]**; konservatif +0,034
   [0,002; 0,068]. Ini keunggulan M2 yang paling kokoh.
3. **ICU bebas obat (CLEAN), hipoglikemia:** M2 0,581; M2 − M1-GB +0,043 [−0,007; 0,091] — **CI memuat
   0**, jadi keunggulan atas ukuran mesin di strata ini belum terbukti. Terhadap M1-LR +0,053 [0,004;
   0,106] dan terhadap M0 +0,095 [0,040; 0,148].
4. **ICU CLEAN, hiperglikemia berat:** M2 − M1-GB +0,050 [0,008; 0,091].
5. **Keragaman antar-jalan:** CNN yang sama dilatih ulang memberi AUROC bergeser — VALIDATION hipo 0,613
   (Hasil 3: 0,614), CLEAN hipo 0,581 (0,574), CLEAN hiper berat 0,635 (0,646). Geseran ±0,01 ini
   sebesar sebagian selisih di strata CLEAN. Analisis utama memakai satu benih; sebelum TEST dibuka,
   M2 utama sebaiknya dirata-rata atas beberapa benih (⬜, diputuskan penulis).
6. **Banyak sel:** 24 selisih dicetak; hanya M2 − M1-GB di VALIDATION yang perbandingan utama. Sel
   lain dibaca sebagai pola, bukan bukti tersendiri. FLAG_HIGH_RISK hipo (15 positif) tidak ditafsirkan.

**Akibat untuk rencana:** klaim yang didukung: CNN gelombang menambah pembedaan kecil di atas ukuran
mesin EKG (+0,03 hipo, +0,05 hiper berat, VALIDATION), dan di ICU bebas obat sinyal hipo gelombang
jelas di atas usia + jenis kelamin. Belum didukung: "gelombang lebih baik daripada ukuran mesin untuk
hipoglikemia di ICU bebas obat". Ukuran efek kecil; AUROC hipo tetap di bawah 0,65, jauh dari guna
klinis.

## Keluaran skrip

Judul dari skrip diturunkan satu tingkat (`#` ditambah); isi lain apa adanya.

## M0/M1/M2 — pasangan dengan gelombang terbaca

- pasangan: 437671; gelombang terbaca 437671; dibuang 0.

Epoch 20; bootstrap 1000 bergugus pasien.

### Hipoglikemia (< 70) lawan euglikemia (70–180) — latih TRAIN, nilai VALIDATION (non-EXCLUDE)

| Bagian | Pasangan | Positif | AUROC M2 CNN gelombang | M2 − M1 (GB) | M2 − M1 (LR) | M2 − M0 |
| --- | --- | --- | --- | --- | --- | --- |
| VALIDATION | 36469 | 696 | 0.613 | 0.033 [0.004, 0.066] | 0.058 [0.026, 0.091] | 0.094 [0.056, 0.128] |
| VALIDATION, konservatif | 5377 | 84 | 0.665 | 0.083 [0.012, 0.159] | 0.095 [0.014, 0.180] | 0.167 [0.069, 0.266] |

Selisih AUROC [95 % CI]; bootstrap bergugus pasien, kedua model pada ulangan yang sama; "—" bila positif < 11.

### Hipoglikemia (< 70) lawan euglikemia (70–180) — ICU, EKG sebelum glukosa, SQI GOOD/FAIR; validasi silang 5 lipatan bergugus pasien × 3 benih, selisih di dalam tiap status obat

- subset: 27750 pasangan, 16059 pasien, positif 610 (pasien dengan ≥ 1 positif 528).

| Bagian | Pasangan | Positif | AUROC M2 CNN gelombang | M2 − M1 (GB) | M2 − M1 (LR) | M2 − M0 |
| --- | --- | --- | --- | --- | --- | --- |
| CLEAN | 8825 | 224 | 0.581 | 0.043 [-0.007, 0.091] | 0.053 [0.004, 0.106] | 0.095 [0.040, 0.148] |
| FLAG_MODERATE_RISK | 9345 | 275 | 0.573 | -0.020 [-0.062, 0.028] | -0.015 [-0.061, 0.034] | 0.099 [0.047, 0.149] |
| FLAG_HIGH_RISK | 723 | 15 | 0.622 | -0.036 [-0.259, 0.149] | 0.161 [0.026, 0.293] | -0.063 [-0.207, 0.061] |
| EXCLUDE | 8857 | 96 | 0.619 | 0.041 [-0.048, 0.124] | 0.053 [-0.046, 0.138] | 0.160 [0.064, 0.254] |

Selisih AUROC [95 % CI]; bootstrap bergugus pasien, kedua model pada ulangan yang sama; "—" bila positif < 11.

### Hiperglikemia berat (> 250) lawan euglikemia (70–180) — latih TRAIN, nilai VALIDATION (non-EXCLUDE)

| Bagian | Pasangan | Positif | AUROC M2 CNN gelombang | M2 − M1 (GB) | M2 − M1 (LR) | M2 − M0 |
| --- | --- | --- | --- | --- | --- | --- |
| VALIDATION | 38227 | 2454 | 0.719 | 0.051 [0.033, 0.068] | 0.079 [0.058, 0.100] | 0.174 [0.145, 0.205] |
| VALIDATION, konservatif | 5612 | 319 | 0.737 | 0.034 [0.002, 0.068] | 0.067 [0.032, 0.101] | 0.269 [0.220, 0.318] |

Selisih AUROC [95 % CI]; bootstrap bergugus pasien, kedua model pada ulangan yang sama; "—" bila positif < 11.

### Hiperglikemia berat (> 250) lawan euglikemia (70–180) — ICU, EKG sebelum glukosa, SQI GOOD/FAIR; validasi silang 5 lipatan bergugus pasien × 3 benih, selisih di dalam tiap status obat

- subset: 29674 pasangan, 16875 pasien, positif 2534 (pasien dengan ≥ 1 positif 1818).

| Bagian | Pasangan | Positif | AUROC M2 CNN gelombang | M2 − M1 (GB) | M2 − M1 (LR) | M2 − M0 |
| --- | --- | --- | --- | --- | --- | --- |
| CLEAN | 8967 | 366 | 0.635 | 0.050 [0.008, 0.091] | 0.054 [0.015, 0.095] | 0.147 [0.105, 0.192] |
| FLAG_MODERATE_RISK | 9870 | 800 | 0.625 | 0.031 [0.004, 0.059] | 0.034 [0.004, 0.062] | 0.099 [0.066, 0.129] |
| FLAG_HIGH_RISK | 1149 | 441 | 0.680 | 0.030 [-0.012, 0.072] | 0.009 [-0.029, 0.049] | 0.092 [0.049, 0.137] |
| EXCLUDE | 9688 | 927 | 0.768 | 0.045 [0.020, 0.068] | 0.036 [0.014, 0.060] | 0.192 [0.162, 0.224] |

Selisih AUROC [95 % CI]; bootstrap bergugus pasien, kedua model pada ulangan yang sama; "—" bila positif < 11.


Waktu jalan: 42.5 menit.
