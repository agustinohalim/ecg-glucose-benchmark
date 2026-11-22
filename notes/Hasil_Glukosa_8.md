# Hasil Glukosa 8 — TEST menurut protokol terkunci (21–22 November 2026)

Sumber angka: keluaran `glukosa_test/jalankan_test.py --boot 1000 --epoch 20` atas
`mimiciv_ecg_glucose_aligned.csv` v1.0.0 (PhysioNet, DOI 10.13026/nts2-vj76, Credentialed Health Data License
1.5.0) dan gelombang MIMIC-IV-ECG v1.0. Dijalankan di laptop (RTX 5060), mulai 21 Nov pukul 22.29, selesai
22 Nov pukul 00.04, 94,4 menit; yang keluar hanya agregat (hitungan 1–10 dicetak `<11`, metrik tidak dicetak
bila positif < 11). Tabel di bawah bagian 0 disambung langsung dari log skrip; jangan diketik ulang.

**Ini pembukaan TEST yang pertama dan satu-satunya.** Protokol dikunci sebelumnya di
`Artikel_Glukosa_Rencana.md` §F langkah 7 (komit 45ff2cb). Uji jalan skrip sebelum itu (`--epoch 1 --boot 20`)
memakai data sintetis di folder sementara, bukan data PhysioNet. Galat teknis: tidak ada (stderr kosong).
Prediksi per baris disimpan di luar repo (`data/mimic-iv-ext-ecg-glucose/prediksi/test_*.npz`) dan tidak
dibuka di sesi.

## Pilihan analisis (dari protokol, tidak diubah)

- **Uji utama, satu per tugas:** selisih AUROC M3b (ECG-JEPA beku + probe L2) − M1-GB (ukuran mesin EKG +
  usia + jenis kelamin, gradient boosting) di TEST non-EXCLUDE; CI 95 % bootstrap bergugus pasien (1.000
  ulangan, benih `BENIH[0]`, kedua model pada ulangan yang sama). Didukung bila batas bawah > 0. Dua tugas
  dilaporkan terpisah tanpa koreksi. M3b dipilih karena tertinggi di VALIDATION di antara tiga model
  gelombang (Hasil 7); di naskah ditulis sebagai pemilihan pada VALIDATION.
- **Sekunder, deskriptif:** M2b − M1-GB, M3 − M1-GB, M3b − M2b, M3b − M3; bagian konservatif; tabel metrik
  semua model.
- **Pelatihan:** semua model dilatih ulang pada TRAIN non-EXCLUDE dengan resep dan benih Hasil 3–7;
  prediksi VALIDATION dan TEST dari model yang sama. M2b = rerata tiga benih (20261120–20261122).
- **Kalibrasi ulang M2b:** regresi logistik tanpa penalti pada logit, dipasang di VALIDATION jalan ini,
  diterapkan tanpa perubahan ke TEST. Model lain tidak dikalibrasi ulang.
- **Tidak dinilai di TEST:** ICU (validasi silang atas ketiga split, jadi baris TEST di subset ICU sudah
  terpakai; masuk Limitations) dan tiga kelas.

## 0. Catatan pembacaan

1. **Uji utama, hipoglikemia: didukung.** TEST 35.103 pasangan, 12.429 pasien, 706 positif (512 pasien).
   M3b AUROC 0,643 [0,617; 0,670]; M1-GB 0,577 [0,552; 0,601]; **M3b − M1-GB +0,066 [0,041; 0,093]**.
   Bagian konservatif (98 positif): +0,089 [0,023; 0,149].
2. **Uji utama, hiperglikemia berat: didukung.** TEST 36.674 pasangan, 12.672 pasien, 2.277 positif
   (1.030 pasien). M3b 0,738 [0,723; 0,753]; M1-GB 0,672 [0,655; 0,688]; **M3b − M1-GB +0,066
   [0,052; 0,082]**. Bagian konservatif (315 positif): +0,081 [0,050; 0,111].
3. **TEST lawan VALIDATION:** selisih utama di VALIDATION +0,064 (hipo) dan +0,078 (hiper berat); di TEST
   +0,066 dan +0,066. AUROC M3b hipo sama (0,643 dan 0,643); hiper berat turun 0,009 (0,747 → 0,738),
   sementara M1-GB naik 0,003. Tidak tampak optimisme pemilihan yang berarti pada hipo; pada hiper berat
   selisihnya menyusut 0,012 tetapi tetap jauh di atas 0.
4. **Sekunder (deskriptif):** M3b − M3 +0,025 [0,006; 0,045] dan +0,026 [0,013; 0,040], searah dengan
   VALIDATION. M3b − M2b +0,025 [−0,001; 0,049] dan +0,014 [−0,000; 0,030]: CI menyentuh 0, jadi
   keunggulan M3b atas M2b untuk hipo yang tampak di VALIDATION (+0,025 [0,004; 0,047]) tidak terulang
   dengan CI di atas 0. M2b − M1-GB +0,041 dan +0,052; M3 − M1-GB +0,041 dan +0,040. Ketiga model gelombang
   di atas M1-GB dengan CI di atas 0 pada kedua tugas.
5. **Kalibrasi di TEST:** M3b tanpa kalibrasi ulang O/E 1,066 dan 1,028, kemiringan 1,04 dan 1,02. M2b
   mentah O/E 0,052 dan 0,174. Kalibrasi ulang di VALIDATION: hipo intersep −3,765, kemiringan 0,439; hiper
   berat −2,452, 0,778. Sesudahnya di TEST O/E 1,060 dan 0,966, kemiringan 1,18 dan 0,87. Kemiringan M1-GB
   hipo 0,65 dan M3 hipo 0,75 (prediksi terlalu tersebar).
6. **Cek ulang VALIDATION (protokol c):** M0, M1-LR, M1-GB, M3, M3b sama persis dengan Hasil 5–7 (hipo
   0,519 / 0,554 / 0,579 / 0,614 / 0,643; hiper berat 0,545 / 0,641 / 0,669 / 0,716 / 0,747). M2 per benih
   bergeser karena GPU: hipo 0,595 / 0,606 / 0,594 (Hasil 6: 0,610 / 0,609 / 0,595), hiper berat 0,721 /
   0,719 / 0,711 (Hasil 6: 0,726 / 0,707 / 0,726). Pergeseran terbesar 0,015, sedikit di atas perkiraan
   ±0,01 di protokol; ini tidak menyentuh uji utama (M3b dan M1-GB deterministik). M2b VALIDATION hipo
   0,611 (Hasil 6: 0,618), hiper berat 0,741 (sama).

**Akibat untuk rencana:**

- Klaim inti naskah lolos uji konfirmatori: gelombang EKG 10 s lewat model fondasi beku + probe linear
  menambah pembedaan di atas ukuran mesin EKG + usia + jenis kelamin, +0,07 AUROC pada kedua tugas di TEST.
- Besarnya tetap sederhana. AUROC hipo 0,64 dan AUPRC 0,040 pada prevalensi 2 %: bukan alat skrining; naskah
  harus menulisnya sebagai bukti adanya sinyal, bukan kesiapan klinis.
- Urutan antarmodel gelombang (M3b > M3; M3b lawan M2b) hanya deskriptif. Klaim "model fondasi mengalahkan
  CNN yang dilatih dari awal" tidak didukung TEST dengan CI di atas 0 dan tidak boleh ditulis.
- Sesudah ini tidak ada perubahan model, resep, atau perbandingan (protokol f). Langkah berikut: naskah.

## Keluaran skrip

Judul dari skrip diturunkan dua tingkat (`##` ditambah); isi lain apa adanya.

### M0/M1/M2 — pasangan dengan gelombang terbaca

- pasangan: 437671; gelombang terbaca 437671; dibuang 0.

TEST, protokol §F langkah 7; epoch 20; bootstrap 1000 bergugus pasien.

#### Hipoglikemia (< 70) lawan euglikemia (70–180) — latih TRAIN, nilai TEST (non-EXCLUDE)

- TEST: 35103 pasangan, 12429 pasien, positif 706 (pasien dengan ≥ 1 positif 512).
- VALIDATION M0 usia + jenis kelamin (LR): AUROC 0.519
- VALIDATION M1 ukuran EKG + M0 (LR): AUROC 0.554
- VALIDATION M1 ukuran EKG + M0 (GB): AUROC 0.579
- VALIDATION M3 ECGFounder beku + probe L2: AUROC 0.614
- VALIDATION M3b ECG-JEPA beku + probe L2: AUROC 0.643
- VALIDATION M2 CNN gelombang benih 20261120: AUROC 0.595
- VALIDATION M2 CNN gelombang benih 20261121: AUROC 0.606
- VALIDATION M2 CNN gelombang benih 20261122: AUROC 0.594
- VALIDATION M2 CNN gelombang, rerata 3 benih: AUROC 0.611
- Kalibrasi ulang M2b (dipasang di VALIDATION): intersep -3.765, kemiringan 0.439

| Model | AUROC [95 % CI] | AUPRC [95 % CI] | Se @ Sp 90 % | Brier | O/E | Kemiringan kalibrasi |
| --- | --- | --- | --- | --- | --- | --- |
| M0 usia + jenis kelamin (LR) | 0.539 [0.512, 0.567] | 0.022 [0.019, 0.025] | 0.099 | 0.020 | 1.054 | 0.927 |
| M1 ukuran EKG + M0 (LR) | 0.560 [0.534, 0.584] | 0.025 [0.022, 0.029] | 0.129 | 0.020 | 1.056 | 0.927 |
| M1 ukuran EKG + M0 (GB) | 0.577 [0.552, 0.601] | 0.027 [0.023, 0.031] | 0.147 | 0.020 | 1.068 | 0.653 |
| M3 ECGFounder beku + probe L2 | 0.618 [0.591, 0.643] | 0.032 [0.027, 0.039] | 0.190 | 0.020 | 1.069 | 0.747 |
| M3b ECG-JEPA beku + probe L2 | 0.643 [0.617, 0.670] | 0.040 [0.033, 0.049] | 0.239 | 0.020 | 1.066 | 1.035 |
| M2 CNN gelombang | 0.609 [0.582, 0.635] | 0.032 [0.028, 0.038] | 0.203 | 0.157 | 0.057 | 0.424 |
| M2 CNN gelombang, rerata 3 benih | 0.618 [0.591, 0.644] | 0.033 [0.028, 0.040] | 0.205 | 0.178 | 0.052 | 0.520 |
| M2 CNN gelombang, rerata 3 benih, terkalibrasi ulang | 0.618 [0.591, 0.644] | 0.033 [0.028, 0.040] | 0.205 | 0.020 | 1.060 | 1.182 |

Prevalensi positif: 2.01 %. CI: bootstrap bergugus pasien.

##### Uji utama

| Bagian | Pasangan | Positif | AUROC M3b ECG-JEPA beku + probe L2 | M3b − M1 (GB) |
| --- | --- | --- | --- | --- |
| TEST | 35103 | 706 | 0.643 | 0.066 [0.041, 0.093] |
| TEST, konservatif | 5444 | 98 | 0.669 | 0.089 [0.023, 0.149] |

Selisih AUROC [95 % CI]; bootstrap bergugus pasien, kedua model pada ulangan yang sama; "—" bila positif < 11.

##### Sekunder (deskriptif)

| Bagian | Pasangan | Positif | AUROC M2 CNN gelombang, rerata 3 benih | M2b − M1 (GB) | M3 − M1 (GB) | M3b − M2b | M3b − M3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TEST | 35103 | 706 | 0.618 | 0.041 [0.008, 0.072] | 0.041 [0.014, 0.066] | 0.025 [-0.001, 0.049] | 0.025 [0.006, 0.045] |
| TEST, konservatif | 5444 | 98 | 0.627 | 0.047 [-0.013, 0.107] | 0.088 [0.029, 0.155] | 0.042 [-0.006, 0.092] | 0.001 [-0.045, 0.041] |

Selisih AUROC [95 % CI]; bootstrap bergugus pasien, kedua model pada ulangan yang sama; "—" bila positif < 11.

#### Hiperglikemia berat (> 250) lawan euglikemia (70–180) — latih TRAIN, nilai TEST (non-EXCLUDE)

- TEST: 36674 pasangan, 12672 pasien, positif 2277 (pasien dengan ≥ 1 positif 1030).
- VALIDATION M0 usia + jenis kelamin (LR): AUROC 0.545
- VALIDATION M1 ukuran EKG + M0 (LR): AUROC 0.641
- VALIDATION M1 ukuran EKG + M0 (GB): AUROC 0.669
- VALIDATION M3 ECGFounder beku + probe L2: AUROC 0.716
- VALIDATION M3b ECG-JEPA beku + probe L2: AUROC 0.747
- VALIDATION M2 CNN gelombang benih 20261120: AUROC 0.721
- VALIDATION M2 CNN gelombang benih 20261121: AUROC 0.719
- VALIDATION M2 CNN gelombang benih 20261122: AUROC 0.711
- VALIDATION M2 CNN gelombang, rerata 3 benih: AUROC 0.741
- Kalibrasi ulang M2b (dipasang di VALIDATION): intersep -2.452, kemiringan 0.778

| Model | AUROC [95 % CI] | AUPRC [95 % CI] | Se @ Sp 90 % | Brier | O/E | Kemiringan kalibrasi |
| --- | --- | --- | --- | --- | --- | --- |
| M0 usia + jenis kelamin (LR) | 0.536 [0.511, 0.559] | 0.064 [0.056, 0.073] | 0.051 | 0.058 | 1.029 | 0.873 |
| M1 ukuran EKG + M0 (LR) | 0.629 [0.611, 0.647] | 0.092 [0.082, 0.104] | 0.186 | 0.058 | 1.032 | 0.963 |
| M1 ukuran EKG + M0 (GB) | 0.672 [0.655, 0.688] | 0.111 [0.098, 0.126] | 0.235 | 0.057 | 1.052 | 0.931 |
| M3 ECGFounder beku + probe L2 | 0.711 [0.695, 0.727] | 0.151 [0.133, 0.171] | 0.307 | 0.056 | 1.034 | 0.984 |
| M3b ECG-JEPA beku + probe L2 | 0.738 [0.723, 0.753] | 0.177 [0.155, 0.202] | 0.343 | 0.055 | 1.028 | 1.015 |
| M2 CNN gelombang | 0.719 [0.703, 0.735] | 0.158 [0.139, 0.180] | 0.323 | 0.133 | 0.246 | 0.407 |
| M2 CNN gelombang, rerata 3 benih | 0.723 [0.707, 0.739] | 0.171 [0.150, 0.195] | 0.352 | 0.175 | 0.174 | 0.678 |
| M2 CNN gelombang, rerata 3 benih, terkalibrasi ulang | 0.723 [0.707, 0.739] | 0.171 [0.150, 0.195] | 0.352 | 0.055 | 0.966 | 0.872 |

Prevalensi positif: 6.21 %. CI: bootstrap bergugus pasien.

##### Uji utama

| Bagian | Pasangan | Positif | AUROC M3b ECG-JEPA beku + probe L2 | M3b − M1 (GB) |
| --- | --- | --- | --- | --- |
| TEST | 36674 | 2277 | 0.738 | 0.066 [0.052, 0.082] |
| TEST, konservatif | 5661 | 315 | 0.785 | 0.081 [0.050, 0.111] |

Selisih AUROC [95 % CI]; bootstrap bergugus pasien, kedua model pada ulangan yang sama; "—" bila positif < 11.

##### Sekunder (deskriptif)

| Bagian | Pasangan | Positif | AUROC M2 CNN gelombang, rerata 3 benih | M2b − M1 (GB) | M3 − M1 (GB) | M3b − M2b | M3b − M3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TEST | 36674 | 2277 | 0.723 | 0.052 [0.034, 0.072] | 0.040 [0.024, 0.056] | 0.014 [-0.000, 0.030] | 0.026 [0.013, 0.040] |
| TEST, konservatif | 5661 | 315 | 0.761 | 0.057 [0.020, 0.094] | 0.063 [0.030, 0.092] | 0.024 [-0.002, 0.051] | 0.018 [-0.003, 0.040] |

Selisih AUROC [95 % CI]; bootstrap bergugus pasien, kedua model pada ulangan yang sama; "—" bila positif < 11.


Waktu jalan: 94.4 menit.
