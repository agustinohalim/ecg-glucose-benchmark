# Hasil Glukosa 12 — analisis tambahan uji kontaminasi atas telaah internal 2 (protokol §F langkah 10), 22 November 2026

Sumber angka: keluaran `glukosa_test/jalankan_kontaminasi_tambahan.py` (bootstrap 10.000, permutasi
2.000, bootstrap kepekaan 2.000) atas prediksi tersimpan langkah 9 (`data/mimic-iv-ext-ecg-glucose/prediksi/
kontaminasi_*.npz`, di luar repo; tanpa melatih ulang) dan daftar split ECG-FM (`meta_split_mimic_iv_ecg.csv`,
bowang-lab/ECG-FM komit 3ff744ef). Data: `mimiciv_ecg_glucose_aligned.csv` v1.0.0 (PhysioNet, DOI
10.13026/nts2-vj76, Credentialed Health Data License 1.5.0). Dijalankan sekali di laptop 22 Nov 2026, selesai pukul
18.47, 4,5 menit, venv overlay ECG-FM, keluar tanpa galat; yang keluar hanya agregat. Bagian di bawah "Keluaran
skrip" disambung langsung dari log; jangan diketik ulang.

**Status: PASCA-HASIL dan deskriptif.** Analisis dan aturan tafsirnya dicatat di rencana §F langkah 10 (komit
270fbd1) sesudah Hasil 11 dan telaah internal 2, sebelum skrip ini dijalankan. Tidak mengubah aturan keputusan langkah 9.

## 0. Catatan pembacaan

- **Split ECG-FM acak per pasien.** Kode fairseq-signals 09c7b9ef (`fairseq_signals/utils/splits.py`,
  `grouped_splits()` baris 289–295): `GroupShuffleSplit` per `subject_id`, 0,8/0,1/0,1 pasien, tanpa benih, tanpa
  stratifikasi label. Keseimbangan sesuai: semua |SMD| ≤ 0,061 (usia, laki-laki, ICU, prevalensi positif, pasangan
  per pasien), kedua tugas. Nol pasien berstatus campur.
- **Aturan (a), selisih model tanpa paparan.** Hipo: M1-GB 0,051, p permutasi 0,027 (< 0,05 → menurut aturan, bukti
  beda kelompok di luar kebetulan); ECGFounder 0,011 (p 0,601), ECG-JEPA 0,006 (p 0,797). Hiper berat: M1-GB 0,016
  (p 0,373), ECGFounder 0,029 (p 0,089), ECG-JEPA 0,025 (p 0,115). Satu dari enam uji model tanpa paparan di bawah
  0,05, tanpa koreksi kelipatan; pada split acak, sebaran nol permutasi adalah sebaran yang benar, jadi M1-GB hipo
  dibaca sebagai beda kelompok yang hanya terlihat oleh model ukuran mesin EKG. Untuk kedua model fondasi bersih,
  selisih mentah = "bising" (SD nol 0,022–0,024 hipo, 0,016 hiper).
- **Aturan inkonklusif/terbatasi.** Hipo: DiD utama 0,028 [−0,005; 0,062], batas atas ≥ 0,033 → **inkonklusif**;
  efek minimum terdeteksi 0,048. Hiper berat: 0,006 [−0,018; 0,032], batas atas < 0,033 → **terbatasi** (tipis).
- **Kepekaan.** Hiper berat: VALIDATION saja 0,006 [−0,023; 0,033], TEST saja 0,007 [−0,034; 0,047], tidak dilihat
  = `test` ECG-FM saja 0,018 [−0,020; 0,057]. Batas di bawah 0,033 hanya bertahan pada himpunan gabungan; himpunan
  yang lebih kecil melebar di atasnya. Hipo: semua himpunan mencakup nol, titik 0,023–0,036.
- **Plasebo (DiD antar model bersih).** ECGFounder − ECG-JEPA 0,005 [−0,025; 0,038] hipo, 0,003 [−0,022; 0,030]
  hiper. M1-GB − ECG-JEPA hipo 0,046 [0,001; 0,091] (permutasi p 0,052): besarnya melebihi DiD utama 0,028; DiD yang
  memakai M1-GB sebagai pembanding mewarisi selisih kelompok M1-GB di atas. Plasebo antar model fondasi bersih dekat
  nol.
- **Galat Monte Carlo.** Hipo 10.000 ulangan: batas bawah −0,005 (1.000 ulangan: −0,003), batas atas 0,062
  (0,060); hiper 0,006 [−0,018; 0,032] (1.000: [−0,019; 0,032]). Tafsir langkah 9 tidak berubah. Permutasi DiD utama
  p 0,123 (hipo) dan 0,578 (hiper).
- **Skala logit:** tanda dan kesimpulan sama (hipo 0,118 [−0,023; 0,264], hiper 0,024 [−0,096; 0,147]).
- **Presisi pembanding bersih.** SD nol permutasi selisih M3c saja 0,0237 lawan DiD M3c − M3b 0,0179 (hipo); 0,0174
  lawan 0,0110 (hiper). Pembanding bersih menyempitkan, bukan menghilangkan bias: split acak tidak menyisakan
  pengacau sistematis untuk dihapus.

## Keluaran skrip

### M0/M1/M2 — pasangan dengan gelombang terbaca

- pasangan: 437671; gelombang terbaca 437671; dibuang 0.

Analisis tambahan kontaminasi, protokol §F langkah 10 (pasca-hasil); bootstrap 10000, permutasi 2000.

### Hipoglikemia (< 70) lawan euglikemia (70–180) — VALIDATION ∪ TEST (non-EXCLUDE)

- pasien dengan status dilihat dan tidak dilihat sekaligus: 0

#### Keseimbangan kelompok dilihat lawan tidak dilihat

| Peubah | Rerata dilihat | Rerata tidak dilihat | SMD |
| --- | --- | --- | --- |
| usia | 61.952 | 61.626 | 0.018 |
| laki-laki | 0.495 | 0.526 | -0.061 |
| ICU | 0.134 | 0.143 | -0.024 |
| positif | 0.019 | 0.021 | -0.012 |
| pasangan per pasien | 7.755 | 7.817 | -0.007 |

#### Bootstrap bergugus pasien, 10000 ulangan terpakai

| Jenis | DiD | Nilai [95 % CI] | SD bootstrap | Efek minimum terdeteksi (2,8 SD) | DiD skala logit [95 % CI] |
| --- | --- | --- | --- | --- | --- |
| utama | M3c ECG-FM − M3b ECG-JEPA | 0.028 [-0.005, 0.062] | 0.0172 | 0.048 | 0.118 [-0.023, 0.264] |
| sekunder | M3c ECG-FM − M3 ECGFounder | 0.023 [-0.017, 0.062] | 0.0203 | 0.057 | 0.095 [-0.072, 0.261] |
| sekunder | M3c ECG-FM − M1-GB | -0.018 [-0.062, 0.025] | 0.0223 | 0.062 | -0.067 [-0.248, 0.110] |
| plasebo | M3 ECGFounder − M3b ECG-JEPA | 0.005 [-0.025, 0.038] | 0.0161 | 0.045 | 0.022 [-0.109, 0.160] |
| plasebo | M1-GB − M3b ECG-JEPA | 0.046 [0.001, 0.091] | 0.0229 | 0.064 | 0.185 [0.000, 0.375] |
| plasebo | M1-GB − M3 ECGFounder | 0.040 [-0.006, 0.087] | 0.0236 | 0.066 | 0.162 [-0.027, 0.354] |

Setengah efek utama TEST: 0.033.

#### Uji permutasi paparan per pasien (2000 permutasi)

| Besaran | Teramati | Rerata nol | SD nol | p dua sisi |
| --- | --- | --- | --- | --- |
| selisih M1-GB | 0.051 | 0.0006 | 0.0230 | 0.027 |
| selisih M3 ECGFounder | 0.011 | -0.0001 | 0.0222 | 0.601 |
| selisih M3b ECG-JEPA | 0.006 | -0.0000 | 0.0225 | 0.797 |
| selisih M3c ECG-FM | 0.034 | 0.0005 | 0.0237 | 0.158 |
| DiD M3c ECG-FM − M3b ECG-JEPA | 0.028 | 0.0005 | 0.0179 | 0.123 |
| DiD M3c ECG-FM − M3 ECGFounder | 0.023 | 0.0006 | 0.0213 | 0.286 |
| DiD M3c ECG-FM − M1-GB | -0.018 | -0.0001 | 0.0238 | 0.457 |
| DiD M3 ECGFounder − M3b ECG-JEPA | 0.005 | -0.0001 | 0.0178 | 0.741 |
| DiD M1-GB − M3b ECG-JEPA | 0.046 | 0.0006 | 0.0233 | 0.052 |
| DiD M1-GB − M3 ECGFounder | 0.040 | 0.0007 | 0.0247 | 0.108 |

#### Kepekaan definisi paparan, DiD utama (bootstrap 2000)

| Himpunan | Positif dilihat | Positif tidak dilihat | DiD M3c − M3b [95 % CI] |
| --- | --- | --- | --- |
| tidak dilihat = `test` ECG-FM saja | 1085 | 129 | 0.036 [-0.007, 0.080] |
| VALIDATION saja | 549 | 141 | 0.023 [-0.027, 0.075] |
| TEST saja | 536 | 157 | 0.032 [-0.009, 0.077] |

### Hiperglikemia berat (> 250) lawan euglikemia (70–180) — VALIDATION ∪ TEST (non-EXCLUDE)

- pasien dengan status dilihat dan tidak dilihat sekaligus: 0

#### Keseimbangan kelompok dilihat lawan tidak dilihat

| Peubah | Rerata dilihat | Rerata tidak dilihat | SMD |
| --- | --- | --- | --- |
| usia | 61.870 | 61.539 | 0.019 |
| laki-laki | 0.499 | 0.527 | -0.055 |
| ICU | 0.134 | 0.143 | -0.026 |
| positif | 0.065 | 0.057 | 0.031 |
| pasangan per pasien | 8.212 | 8.453 | -0.023 |

#### Bootstrap bergugus pasien, 10000 ulangan terpakai

| Jenis | DiD | Nilai [95 % CI] | SD bootstrap | Efek minimum terdeteksi (2,8 SD) | DiD skala logit [95 % CI] |
| --- | --- | --- | --- | --- | --- |
| utama | M3c ECG-FM − M3b ECG-JEPA | 0.006 [-0.018, 0.032] | 0.0128 | 0.036 | 0.024 [-0.096, 0.147] |
| sekunder | M3c ECG-FM − M3 ECGFounder | 0.003 [-0.018, 0.024] | 0.0106 | 0.030 | 0.016 [-0.085, 0.117] |
| sekunder | M3c ECG-FM − M1-GB | 0.016 [-0.019, 0.048] | 0.0169 | 0.047 | 0.083 [-0.078, 0.233] |
| plasebo | M3 ECGFounder − M3b ECG-JEPA | 0.003 [-0.022, 0.030] | 0.0134 | 0.037 | 0.008 [-0.116, 0.139] |
| plasebo | M1-GB − M3b ECG-JEPA | -0.009 [-0.037, 0.020] | 0.0146 | 0.041 | -0.059 [-0.190, 0.077] |
| plasebo | M1-GB − M3 ECGFounder | -0.013 [-0.045, 0.022] | 0.0171 | 0.048 | -0.066 [-0.213, 0.091] |

Setengah efek utama TEST: 0.033.

#### Uji permutasi paparan per pasien (2000 permutasi)

| Besaran | Teramati | Rerata nol | SD nol | p dua sisi |
| --- | --- | --- | --- | --- |
| selisih M1-GB | 0.016 | 0.0009 | 0.0168 | 0.373 |
| selisih M3 ECGFounder | 0.029 | 0.0008 | 0.0164 | 0.089 |
| selisih M3b ECG-JEPA | 0.025 | 0.0007 | 0.0155 | 0.115 |
| selisih M3c ECG-FM | 0.031 | 0.0007 | 0.0174 | 0.076 |
| DiD M3c ECG-FM − M3b ECG-JEPA | 0.006 | 0.0000 | 0.0110 | 0.578 |
| DiD M3c ECG-FM − M3 ECGFounder | 0.003 | -0.0000 | 0.0109 | 0.781 |
| DiD M3c ECG-FM − M1-GB | 0.016 | -0.0002 | 0.0145 | 0.281 |
| DiD M3 ECGFounder − M3b ECG-JEPA | 0.003 | 0.0001 | 0.0112 | 0.766 |
| DiD M1-GB − M3b ECG-JEPA | -0.009 | 0.0002 | 0.0142 | 0.493 |
| DiD M1-GB − M3 ECGFounder | -0.013 | 0.0001 | 0.0145 | 0.367 |

#### Kepekaan definisi paparan, DiD utama (bootstrap 2000)

| Himpunan | Positif dilihat | Positif tidak dilihat | DiD M3c − M3b [95 % CI] |
| --- | --- | --- | --- |
| tidak dilihat = `test` ECG-FM saja | 3821 | 409 | 0.018 [-0.020, 0.057] |
| VALIDATION saja | 1996 | 424 | 0.006 [-0.023, 0.033] |
| TEST saja | 1825 | 421 | 0.007 [-0.034, 0.047] |


Waktu jalan: 4.5 menit.
