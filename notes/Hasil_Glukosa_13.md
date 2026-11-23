# Hasil Glukosa 13 — perluasan uji kontaminasi atas telaah internal 3 (protokol §F langkah 11), 22 November 2026

Sumber angka: keluaran `glukosa_perluasan/jalankan_perluasan.py` (komit fe4cf20; bootstrap 2.000, permutasi
1.000, bootstrap kepekaan 1.000). Prediksi luar lipatan disimpan di `data/mimic-iv-ext-ecg-glucose/prediksi/
perluasan_*.npz` (di luar repo). Embedding ECG-FM fine-tuned dibangun dengan `glukosa_m3/bangun_embedding.py --model
ecgfm_ft` dari `mimic_iv_ecg_finetuned.pt` (wanglab/ecg-fm, revisi 584219ea, MIT, SHA-256
f3e07588ee808d7e1e9e5608a997c9132883239518349f0565fd890590617b3d); 376.648 dari 376.648 rekaman, semua terhingga,
150 menit. Daftar split ECG-FM: `meta_split_mimic_iv_ecg.csv` (bowang-lab/ECG-FM komit 3ff744ef). Data glukosa:
`mimiciv_ecg_glucose_aligned.csv` v1.0.0 (PhysioNet, DOI 10.13026/nts2-vj76, Credentialed Health Data License 1.5.0).
Label pernyataan: `machine_measurements.csv` MIMIC-IV-ECG v1.0 (terbuka, ODbL). Dijalankan sekali di laptop 22 Nov
2026, 21.41–23.11, 90,1 menit, venv overlay ECG-FM, keluar tanpa galat; yang keluar hanya agregat. Bagian di bawah
"Keluaran skrip" disambung langsung dari log; jangan diketik ulang. `hitung_pernyataan.py` (penetapan pola label,
komit f240ca7) dipindah dari `glukosa_test/` ke `glukosa_perluasan/` pada komit fe4cf20 tanpa perubahan isi selain
docstring.

**Status: TERDAFTAR SEBELUM HASIL.** Tugas, model, cross-fitting, margin ε = 0,01, aturan vonis, bendera plafon, dan
syarat kontrol positif dicatat di rencana §F langkah 11 (komit f240ca7) sebelum embedding fine-tuned dibangun dan
sebelum model apa pun dinilai pada tugas pernyataan; kode dikomit (fe4cf20) sebelum dijalankan pada data. Ambang
pasca-hasil 0,033 dari Hasil 12 tidak lagi menjadi vonis (langkah 11g).

## 0. Catatan pembacaan

- **Kontrol positif lolos (aturan 11f).** ECG-FM fine-tuned − ECG-JEPA, batas bawah > 0 pada **keempat** tugas yang
  dilabeli saat fine-tuning: fibrilasi atrium 0,002 [0,001; 0,003], LBBB 0,004 [0,002; 0,006], RBBB 0,002 [0,001;
  0,003], AV blok derajat 1 0,003 [0,001; 0,006]; p permutasi 0,001–0,009. Pada LVH, yang bukan label fine-tuning,
  0,002 [−0,002; 0,006], tidak terdeteksi. Rerata lima tugas 0,002 [0,002; 0,004]. Desain mendeteksi paparan label
  yang diketahui, dan hanya pada label yang memang dipaparkan.
- **Efek kontrol positif lebih kecil dari ε pada skala AUROC.** Batas atas kontrol positif pada keempat tugas
  berlabel 0,003–0,006 < 0,01: paparan label langsung pada rekaman yang dinilai menggeser AUROC kurang dari margin
  yang ditetapkan. Tiga dari empat tugas itu berbendera dekat plafon (AUROC ECG-JEPA > 0,97 di kedua kelompok; AV blok
  0,966–0,967, tepat di bawah bendera). Pada skala logit sekunder efeknya jelas: 0,307 / 0,558 / 0,369 / 0,440,
  semua selang di atas nol. Fungsi vonis memeriksa batas bawah lebih dahulu, jadi kontrol positif dibaca "inflasi
  didukung" walau seluruh selangnya juga di bawah ε. Artinya: "di bawah skala peringkat" pada tugas berplafon tidak
  sama dengan "kontaminasi tak berbahaya"; di sana AUROC hampir tidak bisa bergerak.
- **DiD utama ECG-FM pretrained − ECG-JEPA, tugas pernyataan.** Kelimanya "di bawah skala peringkat": AF 0,000
  [−0,001; 0,001], LBBB 0,001 [−0,001; 0,002], RBBB 0,000 [−0,001; 0,001], AV blok 0,002 [−0,000; 0,004], LVH 0,001
  [−0,004; 0,005]. Rerata 0,001 [−0,000; 0,002], di bawah batas bawah rerata kontrol positif (0,002). Skala logit:
  semua selang mencakup nol (titik 0,002–0,086). Sekunder (lawan ECGFounder) dan plasebo (ECGFounder − ECG-JEPA)
  sama, semua di bawah skala.
- **Glukosa, cross-fitting atas ketiga split.** Hipo: DiD utama 0,008 [−0,008; 0,023], **tak terputuskan**; hiper
  berat: −0,007 [−0,017; 0,002], **di bawah skala peringkat**. Selang menyempit dibanding Hasil 12 (hipo [−0,005;
  0,062], hiper [−0,018; 0,032]) karena unit yang dinilai naik dari VALIDATION ∪ TEST dataset ke ketiga split
  dataset dengan prediksi luar lipatan (positif tidak dilihat kini 1.322 hipo, 4.198 hiper). Plasebo hipo −0,005 [−0,020; 0,010], hiper −0,002 [−0,012; 0,009],
  keduanya di bawah skala; plasebo yang gagal di Hasil 12 (model ukuran − ECG-JEPA) tidak termasuk kontras langkah 11.
- **Kontrol positif pada glukosa tidak terdeteksi**: hipo 0,011 [−0,006; 0,029], hiper 0,000 [−0,011; 0,013]. Label
  fine-tuning bukan glukosa, jadi ini tidak menguji kepekaan desain pada glukosa; kepekaan pada glukosa tetap
  tidak tertunjukkan.
- **Kepekaan, tidak dilihat = `test` saja.** Vonis utama tak berubah pada ketujuh tugas. Hiper berat −0,016 [−0,030;
  −0,003]: selang di bawah nol, ECG-FM relatif lebih baik pada rekaman tidak dilihat; vonis tetap "di bawah skala".
  Kontrol positif tetap terdeteksi pada AF, LBBB, AV blok; pada RBBB batas bawah turun ke −0,000.
- **Selisih mentah satu model menyesatkan.** LVH: ECGFounder −0,006 (p 0,049) dan ECG-JEPA −0,005 (p 0,037), dua
  model yang tidak pernah melihat MIMIC-IV-ECG, "lebih baik" pada rekaman tidak dilihat; DiD-nya ≈ 0. Tanpa koreksi
  kelipatan (35 uji selisih per model). Ini alasan kontras dibaca sebagai DiD, bukan selisih satu model.
- **C terpilih** (glukosa dari C_TRAIN; pernyataan dari subsampel lipatan 1) tercetak di log di bawah.

## Keluaran skrip

### M0/M1/M2 — pasangan dengan gelombang terbaca

- pasangan: 437671; gelombang terbaca 437671; dibuang 0.

- EKG unik: 376648; positif per pernyataan: Fibrilasi atrium 37503, LBBB 14208, RBBB 30652, AV blok derajat 1 23053, LVH 36601

21:41 hipo gb: selesai
21:41 hiper_berat gb: selesai
21:42 af gb: selesai
21:42 lbbb gb: selesai
21:42 rbbb gb: selesai
21:42 avb1 gb: selesai
21:43 lvh gb: selesai
21:43 hipo ecgfounder: C 0.001
21:44 hiper_berat ecgfounder: C 0.001
21:45 af ecgfounder: C 0.001
21:46 lbbb ecgfounder: C 0.001
21:47 rbbb ecgfounder: C 0.001
21:48 avb1 ecgfounder: C 0.01
21:48 lvh ecgfounder: C 0.001
21:49 hipo ecgjepa: C 0.001
21:50 hiper_berat ecgjepa: C 0.01
21:51 af ecgjepa: C 0.01
21:52 lbbb ecgjepa: C 0.01
21:54 rbbb ecgjepa: C 0.1
21:55 avb1 ecgjepa: C 0.1
21:57 lvh ecgjepa: C 0.1
21:57 hipo ecgfm: C 0.001
21:58 hiper_berat ecgfm: C 0.001
21:59 af ecgfm: C 0.1
21:59 lbbb ecgfm: C 0.01
22:00 rbbb ecgfm: C 0.1
22:01 avb1 ecgfm: C 0.01
22:01 lvh ecgfm: C 0.01
22:02 hipo ecgfm_ft: C 0.001
22:04 hiper_berat ecgfm_ft: C 0.01
22:05 af ecgfm_ft: C 0.01
22:06 lbbb ecgfm_ft: C 0.01
22:07 rbbb ecgfm_ft: C 0.001
22:08 avb1 ecgfm_ft: C 0.01
22:10 lvh ecgfm_ft: C 0.1

Perluasan uji kontaminasi, protokol §F langkah 11; bootstrap 2000, permutasi 1000.

### Tugas glukosa, cross-fitting atas ketiga split (non-EXCLUDE)

#### Hipoglikemia (< 70) lawan euglikemia (70–180)

| Kelompok | Unit | Pasien | Positif |
| --- | --- | --- | --- |
| dilihat | 282042 | 99560 | 5421 |
| tidak dilihat | 70356 | 24836 | 1322 |

Ulangan bootstrap terpakai: 2000; permutasi: 1000.

| Model | AUROC dilihat | AUROC tidak dilihat | Dilihat − tidak | p permutasi |
| --- | --- | --- | --- | --- |
| Model ukuran (GB) | 0.582 [0.574, 0.592] | 0.572 [0.552, 0.590] | 0.010 [-0.011, 0.032] | 0.344 |
| ECGFounder (bersih) | 0.616 [0.607, 0.625] | 0.617 [0.599, 0.634] | -0.001 [-0.020, 0.019] | 0.969 |
| ECG-JEPA (pembanding) | 0.640 [0.631, 0.648] | 0.635 [0.617, 0.653] | 0.004 [-0.015, 0.025] | 0.691 |
| ECG-FM pretrained (terpapar) | 0.628 [0.619, 0.637] | 0.615 [0.597, 0.632] | 0.013 [-0.007, 0.033] | 0.218 |
| ECG-FM fine-tuned (kontrol positif) | 0.631 [0.622, 0.641] | 0.616 [0.598, 0.635] | 0.016 [-0.005, 0.036] | 0.136 |

| Jenis | DiD | Nilai [95 % CI] | Skala logit [95 % CI] | p permutasi | Vonis (ε = 0,01) |
| --- | --- | --- | --- | --- | --- |
| utama | ECG-FM pretrained (terpapar) − ECG-JEPA (pembanding) | 0.008 [-0.008, 0.023] | 0.035 [-0.034, 0.100] | 0.303 | tak terputuskan |
| sekunder | ECG-FM pretrained (terpapar) − ECGFounder (bersih) | 0.013 [-0.005, 0.031] | 0.055 [-0.021, 0.133] | 0.152 | tak terputuskan |
| plasebo | ECGFounder (bersih) − ECG-JEPA (pembanding) | -0.005 [-0.020, 0.010] | -0.021 [-0.085, 0.041] | 0.560 | di bawah skala peringkat |
| kontrol positif | ECG-FM fine-tuned (kontrol positif) − ECG-JEPA (pembanding) | 0.011 [-0.006, 0.029] | 0.048 [-0.027, 0.124] | 0.151 | tak terputuskan |

Dekat plafon (AUROC ECG-JEPA kedua kelompok > 0.97): tidak.

#### Hiperglikemia berat (> 250) lawan euglikemia (70–180)

| Kelompok | Unit | Pasien | Positif |
| --- | --- | --- | --- |
| dilihat | 294797 | 101593 | 18176 |
| tidak dilihat | 73232 | 25294 | 4198 |

Ulangan bootstrap terpakai: 2000; permutasi: 1000.

| Model | AUROC dilihat | AUROC tidak dilihat | Dilihat − tidak | p permutasi |
| --- | --- | --- | --- | --- |
| Model ukuran (GB) | 0.657 [0.651, 0.664] | 0.661 [0.648, 0.673] | -0.003 [-0.018, 0.011] | 0.610 |
| ECGFounder (bersih) | 0.705 [0.698, 0.711] | 0.696 [0.683, 0.708] | 0.009 [-0.005, 0.023] | 0.244 |
| ECG-JEPA (pembanding) | 0.735 [0.729, 0.741] | 0.724 [0.712, 0.736] | 0.011 [-0.002, 0.024] | 0.131 |
| ECG-FM pretrained (terpapar) | 0.713 [0.706, 0.720] | 0.709 [0.696, 0.723] | 0.004 [-0.011, 0.019] | 0.661 |
| ECG-FM fine-tuned (kontrol positif) | 0.717 [0.711, 0.724] | 0.706 [0.691, 0.720] | 0.011 [-0.004, 0.028] | 0.165 |

| Jenis | DiD | Nilai [95 % CI] | Skala logit [95 % CI] | p permutasi | Vonis (ε = 0,01) |
| --- | --- | --- | --- | --- | --- |
| utama | ECG-FM pretrained (terpapar) − ECG-JEPA (pembanding) | -0.007 [-0.017, 0.002] | -0.037 [-0.086, 0.010] | 0.123 | di bawah skala peringkat |
| sekunder | ECG-FM pretrained (terpapar) − ECGFounder (bersih) | -0.005 [-0.015, 0.004] | -0.026 [-0.074, 0.020] | 0.271 | di bawah skala peringkat |
| plasebo | ECGFounder (bersih) − ECG-JEPA (pembanding) | -0.002 [-0.012, 0.009] | -0.012 [-0.060, 0.039] | 0.710 | di bawah skala peringkat |
| kontrol positif | ECG-FM fine-tuned (kontrol positif) − ECG-JEPA (pembanding) | 0.000 [-0.011, 0.013] | -0.001 [-0.054, 0.061] | 0.979 | tak terputuskan |

Dekat plafon (AUROC ECG-JEPA kedua kelompok > 0.97): tidak.

### Tugas pernyataan mesin, cross-fitting atas semua EKG

#### Fibrilasi atrium

| Kelompok | Unit | Pasien | Positif |
| --- | --- | --- | --- |
| dilihat | 297060 | 104956 | 29276 |
| tidak dilihat | 74114 | 26159 | 7481 |

Ulangan bootstrap terpakai: 2000; permutasi: 1000.

| Model | AUROC dilihat | AUROC tidak dilihat | Dilihat − tidak | p permutasi |
| --- | --- | --- | --- | --- |
| Model ukuran (GB) | 0.980 [0.979, 0.981] | 0.979 [0.977, 0.980] | 0.001 [-0.001, 0.003] | 0.219 |
| ECGFounder (bersih) | 0.988 [0.987, 0.988] | 0.987 [0.986, 0.988] | 0.000 [-0.001, 0.002] | 0.560 |
| ECG-JEPA (pembanding) | 0.987 [0.986, 0.988] | 0.987 [0.986, 0.988] | 0.000 [-0.001, 0.002] | 0.819 |
| ECG-FM pretrained (terpapar) | 0.985 [0.984, 0.986] | 0.985 [0.984, 0.986] | 0.000 [-0.001, 0.002] | 0.796 |
| ECG-FM fine-tuned (kontrol positif) | 0.995 [0.995, 0.995] | 0.993 [0.992, 0.994] | 0.002 [0.001, 0.003] | 0.001 |

| Jenis | DiD | Nilai [95 % CI] | Skala logit [95 % CI] | p permutasi | Vonis (ε = 0,01) |
| --- | --- | --- | --- | --- | --- |
| utama | ECG-FM pretrained (terpapar) − ECG-JEPA (pembanding) | 0.000 [-0.001, 0.001] | 0.002 [-0.068, 0.070] | 0.913 | di bawah skala peringkat |
| sekunder | ECG-FM pretrained (terpapar) − ECGFounder (bersih) | -0.000 [-0.001, 0.001] | -0.019 [-0.093, 0.050] | 0.701 | di bawah skala peringkat |
| plasebo | ECGFounder (bersih) − ECG-JEPA (pembanding) | 0.000 [-0.000, 0.001] | 0.021 [-0.026, 0.071] | 0.395 | di bawah skala peringkat |
| kontrol positif | ECG-FM fine-tuned (kontrol positif) − ECG-JEPA (pembanding) | 0.002 [0.001, 0.003] | 0.307 [0.227, 0.388] | 0.002 | inflasi didukung |

Dekat plafon (AUROC ECG-JEPA kedua kelompok > 0.97): ya.

#### LBBB

| Kelompok | Unit | Pasien | Positif |
| --- | --- | --- | --- |
| dilihat | 297060 | 104956 | 11128 |
| tidak dilihat | 74114 | 26159 | 2919 |

Ulangan bootstrap terpakai: 2000; permutasi: 1000.

| Model | AUROC dilihat | AUROC tidak dilihat | Dilihat − tidak | p permutasi |
| --- | --- | --- | --- | --- |
| Model ukuran (GB) | 0.974 [0.973, 0.976] | 0.975 [0.971, 0.977] | -0.001 [-0.004, 0.003] | 0.753 |
| ECGFounder (bersih) | 0.988 [0.987, 0.989] | 0.988 [0.986, 0.990] | 0.000 [-0.002, 0.003] | 0.717 |
| ECG-JEPA (pembanding) | 0.986 [0.984, 0.987] | 0.986 [0.984, 0.988] | -0.001 [-0.003, 0.002] | 0.687 |
| ECG-FM pretrained (terpapar) | 0.987 [0.986, 0.988] | 0.987 [0.985, 0.989] | 0.000 [-0.002, 0.002] | 0.982 |
| ECG-FM fine-tuned (kontrol positif) | 0.995 [0.995, 0.995] | 0.991 [0.990, 0.993] | 0.003 [0.002, 0.005] | 0.001 |

| Jenis | DiD | Nilai [95 % CI] | Skala logit [95 % CI] | p permutasi | Vonis (ε = 0,01) |
| --- | --- | --- | --- | --- | --- |
| utama | ECG-FM pretrained (terpapar) − ECG-JEPA (pembanding) | 0.001 [-0.001, 0.002] | 0.040 [-0.091, 0.173] | 0.521 | di bawah skala peringkat |
| sekunder | ECG-FM pretrained (terpapar) − ECGFounder (bersih) | -0.000 [-0.002, 0.001] | -0.032 [-0.143, 0.080] | 0.660 | di bawah skala peringkat |
| plasebo | ECGFounder (bersih) − ECG-JEPA (pembanding) | 0.001 [-0.000, 0.002] | 0.072 [-0.029, 0.177] | 0.216 | di bawah skala peringkat |
| kontrol positif | ECG-FM fine-tuned (kontrol positif) − ECG-JEPA (pembanding) | 0.004 [0.002, 0.006] | 0.558 [0.423, 0.691] | 0.001 | inflasi didukung |

Dekat plafon (AUROC ECG-JEPA kedua kelompok > 0.97): ya.

#### RBBB

| Kelompok | Unit | Pasien | Positif |
| --- | --- | --- | --- |
| dilihat | 297060 | 104956 | 24047 |
| tidak dilihat | 74114 | 26159 | 6152 |

Ulangan bootstrap terpakai: 2000; permutasi: 1000.

| Model | AUROC dilihat | AUROC tidak dilihat | Dilihat − tidak | p permutasi |
| --- | --- | --- | --- | --- |
| Model ukuran (GB) | 0.963 [0.961, 0.965] | 0.963 [0.960, 0.967] | -0.000 [-0.004, 0.004] | 0.951 |
| ECGFounder (bersih) | 0.993 [0.993, 0.993] | 0.993 [0.992, 0.994] | 0.000 [-0.001, 0.001] | 0.966 |
| ECG-JEPA (pembanding) | 0.991 [0.990, 0.991] | 0.991 [0.989, 0.992] | -0.000 [-0.001, 0.001] | 0.963 |
| ECG-FM pretrained (terpapar) | 0.987 [0.986, 0.987] | 0.986 [0.985, 0.988] | 0.000 [-0.002, 0.002] | 0.880 |
| ECG-FM fine-tuned (kontrol positif) | 0.996 [0.996, 0.996] | 0.994 [0.993, 0.995] | 0.002 [0.001, 0.003] | 0.001 |

| Jenis | DiD | Nilai [95 % CI] | Skala logit [95 % CI] | p permutasi | Vonis (ε = 0,01) |
| --- | --- | --- | --- | --- | --- |
| utama | ECG-FM pretrained (terpapar) − ECG-JEPA (pembanding) | 0.000 [-0.001, 0.001] | 0.017 [-0.094, 0.118] | 0.810 | di bawah skala peringkat |
| sekunder | ECG-FM pretrained (terpapar) − ECGFounder (bersih) | 0.000 [-0.001, 0.001] | 0.007 [-0.087, 0.107] | 0.869 | di bawah skala peringkat |
| plasebo | ECGFounder (bersih) − ECG-JEPA (pembanding) | 0.000 [-0.001, 0.001] | 0.009 [-0.075, 0.089] | 0.858 | di bawah skala peringkat |
| kontrol positif | ECG-FM fine-tuned (kontrol positif) − ECG-JEPA (pembanding) | 0.002 [0.001, 0.003] | 0.369 [0.268, 0.469] | 0.002 | inflasi didukung |

Dekat plafon (AUROC ECG-JEPA kedua kelompok > 0.97): ya.

#### AV blok derajat 1

| Kelompok | Unit | Pasien | Positif |
| --- | --- | --- | --- |
| dilihat | 297060 | 104956 | 17945 |
| tidak dilihat | 74114 | 26159 | 4752 |

Ulangan bootstrap terpakai: 2000; permutasi: 1000.

| Model | AUROC dilihat | AUROC tidak dilihat | Dilihat − tidak | p permutasi |
| --- | --- | --- | --- | --- |
| Model ukuran (GB) | 0.993 [0.993, 0.994] | 0.994 [0.992, 0.995] | -0.000 [-0.001, 0.001] | 0.727 |
| ECGFounder (bersih) | 0.964 [0.963, 0.965] | 0.965 [0.962, 0.968] | -0.001 [-0.004, 0.002] | 0.476 |
| ECG-JEPA (pembanding) | 0.966 [0.965, 0.967] | 0.967 [0.964, 0.969] | -0.001 [-0.004, 0.002] | 0.497 |
| ECG-FM pretrained (terpapar) | 0.986 [0.985, 0.987] | 0.985 [0.983, 0.987] | 0.001 [-0.001, 0.003] | 0.447 |
| ECG-FM fine-tuned (kontrol positif) | 0.995 [0.995, 0.996] | 0.993 [0.992, 0.994] | 0.002 [0.001, 0.003] | 0.001 |

| Jenis | DiD | Nilai [95 % CI] | Skala logit [95 % CI] | p permutasi | Vonis (ε = 0,01) |
| --- | --- | --- | --- | --- | --- |
| utama | ECG-FM pretrained (terpapar) − ECG-JEPA (pembanding) | 0.002 [-0.000, 0.004] | 0.086 [-0.028, 0.207] | 0.136 | di bawah skala peringkat |
| sekunder | ECG-FM pretrained (terpapar) − ECGFounder (bersih) | 0.002 [-0.000, 0.004] | 0.093 [-0.025, 0.214] | 0.130 | di bawah skala peringkat |
| plasebo | ECGFounder (bersih) − ECG-JEPA (pembanding) | -0.000 [-0.002, 0.002] | -0.007 [-0.062, 0.049] | 0.776 | di bawah skala peringkat |
| kontrol positif | ECG-FM fine-tuned (kontrol positif) − ECG-JEPA (pembanding) | 0.003 [0.001, 0.006] | 0.440 [0.281, 0.599] | 0.009 | inflasi didukung |

Dekat plafon (AUROC ECG-JEPA kedua kelompok > 0.97): tidak.

#### LVH

| Kelompok | Unit | Pasien | Positif |
| --- | --- | --- | --- |
| dilihat | 297060 | 104956 | 28818 |
| tidak dilihat | 74114 | 26159 | 7370 |

Ulangan bootstrap terpakai: 2000; permutasi: 1000.

| Model | AUROC dilihat | AUROC tidak dilihat | Dilihat − tidak | p permutasi |
| --- | --- | --- | --- | --- |
| Model ukuran (GB) | 0.823 [0.819, 0.826] | 0.825 [0.817, 0.833] | -0.003 [-0.012, 0.006] | 0.564 |
| ECGFounder (bersih) | 0.912 [0.909, 0.914] | 0.917 [0.913, 0.921] | -0.006 [-0.010, -0.000] | 0.049 |
| ECG-JEPA (pembanding) | 0.930 [0.928, 0.932] | 0.935 [0.931, 0.939] | -0.005 [-0.009, -0.001] | 0.037 |
| ECG-FM pretrained (terpapar) | 0.880 [0.877, 0.883] | 0.884 [0.878, 0.889] | -0.004 [-0.010, 0.002] | 0.240 |
| ECG-FM fine-tuned (kontrol positif) | 0.904 [0.901, 0.906] | 0.907 [0.903, 0.912] | -0.003 [-0.009, 0.002] | 0.244 |

| Jenis | DiD | Nilai [95 % CI] | Skala logit [95 % CI] | p permutasi | Vonis (ε = 0,01) |
| --- | --- | --- | --- | --- | --- |
| utama | ECG-FM pretrained (terpapar) − ECG-JEPA (pembanding) | 0.001 [-0.004, 0.005] | 0.041 [-0.010, 0.091] | 0.687 | di bawah skala peringkat |
| sekunder | ECG-FM pretrained (terpapar) − ECGFounder (bersih) | 0.001 [-0.004, 0.006] | 0.030 [-0.026, 0.082] | 0.613 | di bawah skala peringkat |
| plasebo | ECGFounder (bersih) − ECG-JEPA (pembanding) | -0.000 [-0.004, 0.003] | 0.011 [-0.038, 0.060] | 0.813 | di bawah skala peringkat |
| kontrol positif | ECG-FM fine-tuned (kontrol positif) − ECG-JEPA (pembanding) | 0.002 [-0.002, 0.006] | 0.042 [-0.008, 0.092] | 0.354 | di bawah skala peringkat |

Dekat plafon (AUROC ECG-JEPA kedua kelompok > 0.97): tidak.

#### Ringkasan: rerata DiD 5 tugas pernyataan (bobot sama)

| Jenis | DiD | Rerata [95 % CI] | Vonis (ε = 0,01) |
| --- | --- | --- | --- |
| utama | ECG-FM pretrained (terpapar) − ECG-JEPA (pembanding) | 0.001 [-0.000, 0.002] | di bawah skala peringkat |
| sekunder | ECG-FM pretrained (terpapar) − ECGFounder (bersih) | 0.001 [-0.001, 0.002] | di bawah skala peringkat |
| plasebo | ECGFounder (bersih) − ECG-JEPA (pembanding) | 0.000 [-0.001, 0.001] | di bawah skala peringkat |
| kontrol positif | ECG-FM fine-tuned (kontrol positif) − ECG-JEPA (pembanding) | 0.002 [0.002, 0.004] | inflasi didukung |

### Kepekaan: tidak dilihat = `test` ECG-FM saja (bootstrap 1000)

| Tugas | DiD utama [95 % CI], vonis | Kontrol positif [95 % CI], vonis |
| --- | --- | --- |
| Hipoglikemia (< 70) lawan euglikemia (70–180) | 0.004 [-0.020, 0.023] tak terputuskan | 0.004 [-0.017, 0.025] tak terputuskan |
| Hiperglikemia berat (> 250) lawan euglikemia (70–180) | -0.016 [-0.030, -0.003] di bawah skala peringkat | -0.000 [-0.017, 0.019] tak terputuskan |
| Fibrilasi atrium | -0.000 [-0.001, 0.001] di bawah skala peringkat | 0.002 [0.001, 0.003] inflasi didukung |
| LBBB | 0.001 [-0.001, 0.003] di bawah skala peringkat | 0.004 [0.001, 0.006] inflasi didukung |
| RBBB | 0.001 [-0.001, 0.003] di bawah skala peringkat | 0.001 [-0.000, 0.002] di bawah skala peringkat |
| AV blok derajat 1 | 0.002 [-0.001, 0.005] di bawah skala peringkat | 0.004 [0.000, 0.007] inflasi didukung |
| LVH | -0.001 [-0.007, 0.004] di bawah skala peringkat | 0.000 [-0.005, 0.005] di bawah skala peringkat |

Waktu jalan: 90.1 menit.
