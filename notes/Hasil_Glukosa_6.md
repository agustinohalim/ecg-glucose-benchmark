# Hasil Glukosa 6 — M2 rerata 3 benih (analisis utama) berdampingan dengan M0–M3 (21 November 2026)

Sumber angka: keluaran `glukosa_m2/jalankan_m2_benih.py --boot 1000` atas
`mimiciv_ecg_glucose_aligned.csv` v1.0.0 (PhysioNet, DOI 10.13026/nts2-vj76, Credentialed Health Data
License 1.5.0) dan gelombang MIMIC-IV-ECG v1.0. Dijalankan di laptop (RTX 5060), 78,6 menit; yang keluar
hanya agregat (hitungan 1–10 dicetak `<11`, metrik tidak dicetak bila positif < 11). Tabel di bawah
bagian 0 disalin apa adanya; jangan diketik ulang. Split TEST tidak disentuh.

## Pilihan analisis

- **Keputusan penulis 21 Nov 2026** (rencana §F langkah 6a, sebelum TEST dibuka): M2 analisis utama =
  rerata peluang CNN yang dilatih dengan 3 benih (`BENIH` = 20261120, 20261121, 20261122, sama dengan
  benih validasi silang ICU). Angka satu benih di Hasil 3/4 tetap dilaporkan sebagai kepekaan.
- **Yang dilatih ulang:** hanya M2, resep sama dengan Hasil 3 (20 epoch, batch 64, Adam 10⁻³, bobot kelas
  seimbang, epoch terakhir), latih TRAIN non-EXCLUDE, nilai VALIDATION non-EXCLUDE; peluang VALIDATION
  ketiga model dirata-rata aritmetik. M0, M1, M2 satu benih (Hasil 4) dan M3 (Hasil 5) dibaca dari
  `data/mimic-iv-ext-ecg-glucose/prediksi/` dan dicocokkan baris demi baris (y dan subject_id identik, diperiksa skrip).
- **Hanya analisis utama.** Subset ICU tidak dijalankan ulang: validasi silangnya sudah merata-rata 3
  benih sejak Hasil 3, jadi M2 di tabel ICU Hasil 3–5 sudah rerata 3 benih.
- Label tabel: `M2 CNN gelombang` = satu benih (Hasil 4); `M2 CNN gelombang, rerata 3 benih` = M2b.
- Kalibrasi ulang M2 (langkah 6b) belum diterapkan di sini; kolom kalibrasi M2b masih mentah.

## 0. Catatan pembacaan

1. **Hipoglikemia, VALIDATION:** M2b 0,618 [0,593; 0,640] (satu benih 0,613). M2b − M1-GB +0,039
   [0,010; 0,069] (satu benih +0,033 [0,004; 0,066]); M3 − M2b −0,004 [−0,027; 0,020]: M2b dan M3
   tetap tak terbedakan. Bagian konservatif M2b − M1-GB +0,076 [0,006; 0,145] (84 positif, CI lebar).
2. **Hiperglikemia berat, VALIDATION:** M2b 0,741 [0,720; 0,759], naik +0,022 dari satu benih (0,719).
   M2b − M1-GB +0,073 [0,052; 0,091]. **M3 − M2b −0,025 [−0,040; −0,009]**: rerata 3 CNN kini lebih baik
   daripada ECGFounder beku + probe. Di bagian konservatif tidak terlihat (M3 − M2b +0,004 [−0,025;
   0,030], 319 positif).
3. **Variasi antarbenih:** hipo 0,610 / 0,609 / 0,595; hiper berat 0,726 / 0,707 / 0,726. Rerata lebih
   tinggi daripada benih terbaik (0,618 > 0,610; 0,741 > 0,726), jadi yang terjadi adalah efek ansambel,
   bukan sekadar meredam derau satu benih.
4. **Pelatihan GPU tidak deterministik:** benih 20261120 sudah dijalankan tiga kali — hipo 0,614 (Hasil 3),
   0,613 (Hasil 4), 0,610 (di sini); hiper berat 0,717, 0,719, 0,726. Angka M2 satu benih hanya dapat
   diulang sampai ±0,01 AUROC; ini satu alasan lagi untuk melaporkan M2b.
5. **Kalibrasi M2b mentah buruk** (O/E 0,048 dan 0,182; kemiringan 0,48 dan 0,74), akibat bobot kelas
   seimbang yang menggeser peluang ke atas. Kalibrasi ulang pada VALIDATION (langkah 6b) wajib sebelum
   angka kalibrasi M2 dilaporkan; AUROC tidak berubah olehnya.
6. **Perbandingan M3 − M2b tidak setara biaya:** M2b = tiga jaringan dilatih penuh (±16 menit GPU per
   benih per tugas); M3 = satu probe linear deterministik di atas embedding yang dihitung sekali (47 menit
   GPU, lalu 11 menit CPU untuk semua tugas). Ansambel probe tidak bermakna karena probe deterministik.

**Akibat untuk rencana:** klaim inti tidak berubah dan sedikit menguat — gelombang EKG 10 s menambah
pembedaan di atas ukuran mesin, kini +0,04 (hipo) dan +0,07 (hiper berat) AUROC dengan M2b. Kalimat
Hasil 5 "CNN latih penuh tidak lebih baik daripada model fondasi beku" **hanya bertahan untuk
hipoglikemia**; untuk hiperglikemia berat ansambel 3 CNN unggul 0,025 atas ECGFounder beku. Semua AUROC
hipo tetap < 0,65. Model fondasi kedua (ECG-JEPA, M3b) menyusul di Hasil 7.

## Keluaran skrip

Judul dari skrip diturunkan satu tingkat (`#` ditambah); isi lain apa adanya.

## M0/M1/M2 — pasangan dengan gelombang terbaca

- pasangan: 437671; gelombang terbaca 437671; dibuang 0.

Epoch 20; benih [20261120, 20261121, 20261122]; bootstrap 1000 bergugus pasien.

- benih 20261120: AUROC VALIDATION 0.610
- benih 20261121: AUROC VALIDATION 0.609
- benih 20261122: AUROC VALIDATION 0.595

### Hipoglikemia (< 70) lawan euglikemia (70–180) — latih TRAIN, nilai VALIDATION (non-EXCLUDE)

| Model | AUROC [95 % CI] | AUPRC [95 % CI] | Se @ Sp 90 % | Brier | O/E | Kemiringan kalibrasi |
| --- | --- | --- | --- | --- | --- | --- |
| M0 usia + jenis kelamin (LR) | 0.519 [0.497, 0.545] | 0.020 [0.018, 0.023] | 0.118 | 0.019 | 1.002 | 0.939 |
| M1 ukuran EKG + M0 (LR) | 0.554 [0.528, 0.579] | 0.025 [0.021, 0.031] | 0.145 | 0.019 | 1.005 | 0.939 |
| M1 ukuran EKG + M0 (GB) | 0.579 [0.554, 0.603] | 0.024 [0.021, 0.028] | 0.141 | 0.019 | 1.013 | 0.620 |
| M2 CNN gelombang | 0.613 [0.588, 0.637] | 0.028 [0.025, 0.032] | 0.191 | 0.168 | 0.052 | 0.405 |
| M3 ECGFounder beku + probe L2 | 0.614 [0.590, 0.637] | 0.029 [0.026, 0.035] | 0.187 | 0.019 | 1.007 | 0.727 |
| M2 CNN gelombang, rerata 3 benih | 0.618 [0.593, 0.640] | 0.029 [0.026, 0.034] | 0.210 | 0.187 | 0.048 | 0.478 |

Prevalensi positif: 1.91 %. CI: bootstrap bergugus pasien.

| Bagian | Pasangan | Positif | AUROC M2 CNN gelombang, rerata 3 benih | M2b − M1 (GB) | M2b − M0 | M3 − M2b |
| --- | --- | --- | --- | --- | --- | --- |
| VALIDATION | 36469 | 696 | 0.618 | 0.039 [0.010, 0.069] | 0.099 [0.064, 0.132] | -0.004 [-0.027, 0.020] |
| VALIDATION, konservatif | 5377 | 84 | 0.658 | 0.076 [0.006, 0.145] | 0.160 [0.064, 0.255] | -0.015 [-0.075, 0.045] |

Selisih AUROC [95 % CI]; bootstrap bergugus pasien, kedua model pada ulangan yang sama; "—" bila positif < 11.

- benih 20261120: AUROC VALIDATION 0.726
- benih 20261121: AUROC VALIDATION 0.707
- benih 20261122: AUROC VALIDATION 0.726

### Hiperglikemia berat (> 250) lawan euglikemia (70–180) — latih TRAIN, nilai VALIDATION (non-EXCLUDE)

| Model | AUROC [95 % CI] | AUPRC [95 % CI] | Se @ Sp 90 % | Brier | O/E | Kemiringan kalibrasi |
| --- | --- | --- | --- | --- | --- | --- |
| M0 usia + jenis kelamin (LR) | 0.545 [0.518, 0.571] | 0.070 [0.061, 0.083] | 0.086 | 0.060 | 1.066 | 0.862 |
| M1 ukuran EKG + M0 (LR) | 0.641 [0.618, 0.661] | 0.102 [0.089, 0.117] | 0.206 | 0.059 | 1.069 | 1.086 |
| M1 ukuran EKG + M0 (GB) | 0.669 [0.649, 0.688] | 0.117 [0.100, 0.136] | 0.239 | 0.059 | 1.089 | 0.941 |
| M2 CNN gelombang | 0.719 [0.698, 0.737] | 0.172 [0.144, 0.203] | 0.331 | 0.143 | 0.231 | 0.451 |
| M3 ECGFounder beku + probe L2 | 0.716 [0.696, 0.735] | 0.164 [0.136, 0.194] | 0.315 | 0.057 | 1.071 | 1.029 |
| M2 CNN gelombang, rerata 3 benih | 0.741 [0.720, 0.759] | 0.198 [0.162, 0.235] | 0.367 | 0.171 | 0.182 | 0.738 |

Prevalensi positif: 6.42 %. CI: bootstrap bergugus pasien.

| Bagian | Pasangan | Positif | AUROC M2 CNN gelombang, rerata 3 benih | M2b − M1 (GB) | M2b − M0 | M3 − M2b |
| --- | --- | --- | --- | --- | --- | --- |
| VALIDATION | 38227 | 2454 | 0.741 | 0.073 [0.052, 0.091] | 0.196 [0.167, 0.223] | -0.025 [-0.040, -0.009] |
| VALIDATION, konservatif | 5612 | 319 | 0.761 | 0.058 [0.024, 0.091] | 0.292 [0.246, 0.341] | 0.004 [-0.025, 0.030] |

Selisih AUROC [95 % CI]; bootstrap bergugus pasien, kedua model pada ulangan yang sama; "—" bila positif < 11.


Waktu jalan: 78.6 menit.
