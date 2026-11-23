# Hasil Glukosa 11 — uji kontaminasi pra-latih ECG-FM (protokol §F langkah 9), 22 November 2026

Sumber angka: keluaran `glukosa_test/jalankan_kontaminasi.py --boot 1000` atas
`mimiciv_ecg_glucose_aligned.csv` v1.0.0 (PhysioNet, DOI 10.13026/nts2-vj76, Credentialed Health Data License 1.5.0),
embedding ECG-FM/ECGFounder/ECG-JEPA dari ZIP MIMIC-IV-ECG v1.0, dan daftar split ECG-FM
(`meta_split_mimic_iv_ecg.csv`, bowang-lab/ECG-FM komit 3ff744ef). Dijalankan sekali di laptop 22 Nov 2026 pukul
12.51, 15,1 menit, venv overlay ECG-FM, keluar tanpa galat; yang keluar hanya agregat (hitungan 1–10 dicetak `<11`,
metrik tidak dicetak bila positif < 11). Prediksi per pasangan di `data/mimic-iv-ext-ecg-glucose/prediksi/
kontaminasi_*.npz` (di luar repo). Bagian di bawah "Keluaran skrip" disambung langsung dari log; jangan diketik ulang.

**Status: menurut protokol, dicatat sebelum embedding ECG-FM dibangun** (rencana §F langkah 9, komit 82bee79;
skrip daa512b; kode dirilis publik sebelum jalan ini, v1.0.0 doi 10.5281/zenodo.24433534). Embedding ECG-FM:
376.648 baris, semuanya terhingga, 160,6 menit. Sebelum estimand dihitung, yang dilihat hanya pemeriksaan teknis.

## 0. Catatan pembacaan

- **Uji utama: inflasi tidak didukung pada kedua tugas.** DiD (M3c lawan M3b) hipoglikemia 0,028 [−0,003; 0,060],
  hiperglikemia berat 0,006 [−0,019; 0,032]; batas bawah ≤ 0. Menurut protokol (d), batas atas dilaporkan: inflasi
  AUROC akibat paparan pra-latih yang masih sesuai dengan data paling besar 0,060 (hipo) dan 0,032 (hiper berat).
  Ini batas atas, bukan bukti tiada efek (protokol g). Hipo: titik 0,028 tidak kecil dibanding interval; hiper
  berat: dekat nol dan sempit.
- **Sekunder, DiD dengan pembanding lain:** lawan M3 (ECGFounder) 0,023 [−0,015; 0,062] dan 0,003 [−0,017; 0,022];
  lawan M1-GB (tanpa pra-latih) −0,018 [−0,060; 0,026] dan 0,016 [−0,019; 0,048]. Tidak ada yang di atas nol.
- **Kelompok dilihat dan tidak dilihat berbeda bahkan untuk model tanpa pra-latih.** M1-GB hipo dilihat − tidak
  0,051 [0,010; 0,099]; hiper berat 0,016. Selisih mentah dilihat − tidak M3c (0,034 dan 0,031) karena itu tidak bisa
  dibaca sebagai inflasi; inilah alasan protokol memakai pembanding bersih (DiD). Kelompok tidak dilihat sedikit
  lebih banyak laki-laki (52,6 lawan 49,5 %) dan ICU (14,3 lawan 13,4 %); usia sama (61,6 lawan 62,0).
- **Model terkontaminasi lebih rendah dari model bersih.** M3c − M3b di VALIDATION ∪ TEST −0,021 [−0,035; −0,008]
  (hipo) dan −0,019 [−0,027; −0,011] (hiper berat); di TEST saja −0,021 [−0,039; −0,004] dan −0,021 [−0,035;
  −0,009]. Pra-latih swaawas pada rekaman MIMIC-IV-ECG yang sama tidak membuat ECG-FM unggul atas ECG-JEPA yang tidak
  pernah melihat data ini. VALIDATION: M3c 0,622 dan 0,731, di antara M3 dan M3b.
- **Pemeriksaan teknis:** AUROC VALIDATION M1-GB, M3, M3b sama persis dengan Hasil 8 (hipo 0,579 / 0,614 / 0,643;
  hiper 0,669 / 0,716 / 0,747), jadi model deterministik dilatih ulang identik. C terpilih M3c 0,001 kedua tugas.
  Bootstrap terpakai 1000 dari 1000 pada kedua tugas. Tak tercantum di daftar split: 6.583 pasangan (989 dan 1.035
  di himpunan nilai), dikeluarkan dari kontras sesuai protokol (b).

## Keluaran skrip

### M0/M1/M2 — pasangan dengan gelombang terbaca

- pasangan: 437671; gelombang terbaca 437671; dibuang 0.

#### Paparan pra-latih ECG-FM (semua pasangan)

| Paparan | Pasangan |
| --- | --- |
| dilihat | 345211 |
| tidak dilihat | 85877 |
| tak tercantum | 6583 |
- embedding ecgfounder: baris 376648, semua terhingga True
- embedding ecgjepa: baris 376648, semua terhingga True
- embedding ecgfm: baris 376648, semua terhingga True

Uji kontaminasi, protokol §F langkah 9; bootstrap 1000 bergugus pasien.

### Hipoglikemia (< 70) lawan euglikemia (70–180) — latih TRAIN, nilai VALIDATION ∪ TEST (non-EXCLUDE)

- VALIDATION M1 ukuran EKG + M0 (GB): AUROC 0.579
- M3 ECGFounder beku + probe L2: C terpilih 0.001
- VALIDATION M3 ECGFounder beku + probe L2: AUROC 0.614
- M3b ECG-JEPA beku + probe L2: C terpilih 0.001
- VALIDATION M3b ECG-JEPA beku + probe L2: AUROC 0.643
- M3c ECG-FM beku + probe L2 (terkontaminasi): C terpilih 0.001
- VALIDATION M3c ECG-FM beku + probe L2 (terkontaminasi): AUROC 0.622

#### Kelompok paparan di VALIDATION ∪ TEST

| Paparan | Pasangan | Pasien | Positif | Usia rerata | Laki-laki % | ICU % |
| --- | --- | --- | --- | --- | --- | --- |
| dilihat | 56373 | 19933 | 1085 | 62.0 | 49.5 | 13.4 |
| tak tercantum | 989 | 824 | 19 | 63.9 | 55.6 | 23.3 |
| tidak dilihat | 14210 | 4938 | 298 | 61.6 | 52.6 | 14.3 |

#### Kontras dilihat lawan tidak dilihat (DiD utama: pembanding M3b)

| Model | AUROC dilihat | AUROC tidak dilihat | Dilihat − tidak |
| --- | --- | --- | --- |
| M1 ukuran EKG + M0 (GB) | 0.587 [0.569, 0.607] | 0.536 [0.494, 0.572] | 0.051 [0.010, 0.099] |
| M3 ECGFounder beku + probe L2 | 0.618 [0.599, 0.638] | 0.607 [0.568, 0.646] | 0.011 [-0.032, 0.053] |
| M3b ECG-JEPA beku + probe L2 | 0.643 [0.623, 0.663] | 0.638 [0.595, 0.674] | 0.006 [-0.036, 0.049] |
| M3c ECG-FM beku + probe L2 (terkontaminasi) | 0.628 [0.609, 0.648] | 0.595 [0.554, 0.630] | 0.034 [-0.010, 0.078] |

| DiD (M3c − pembanding) | Nilai [95 % CI] |
| --- | --- |
| utama: M3b ECG-JEPA beku + probe L2 | 0.028 [-0.003, 0.060] |
| sekunder: M3 ECGFounder beku + probe L2 | 0.023 [-0.015, 0.062] |
| sekunder: M1 ukuran EKG + M0 (GB) | -0.018 [-0.060, 0.026] |

Ulangan bootstrap terpakai: 1000; positif dilihat 1085, tidak dilihat 298.

#### M3c − M3b keseluruhan (sekunder)

| Bagian | Pasangan | Positif | AUROC M3c ECG-FM beku + probe L2 (terkontaminasi) | M3c − M3b |
| --- | --- | --- | --- | --- |
| VALIDATION ∪ TEST | 71572 | 1402 | 0.621 | -0.021 [-0.035, -0.008] |
| TEST | 35103 | 706 | 0.622 | -0.021 [-0.039, -0.004] |

Selisih AUROC [95 % CI]; bootstrap bergugus pasien, kedua model pada ulangan yang sama; "—" bila positif < 11.

### Hiperglikemia berat (> 250) lawan euglikemia (70–180) — latih TRAIN, nilai VALIDATION ∪ TEST (non-EXCLUDE)

- VALIDATION M1 ukuran EKG + M0 (GB): AUROC 0.669
- M3 ECGFounder beku + probe L2: C terpilih 0.001
- VALIDATION M3 ECGFounder beku + probe L2: AUROC 0.716
- M3b ECG-JEPA beku + probe L2: C terpilih 0.01
- VALIDATION M3b ECG-JEPA beku + probe L2: AUROC 0.747
- M3c ECG-FM beku + probe L2 (terkontaminasi): C terpilih 0.001
- VALIDATION M3c ECG-FM beku + probe L2 (terkontaminasi): AUROC 0.731

#### Kelompok paparan di VALIDATION ∪ TEST

| Paparan | Pasangan | Pasien | Positif | Usia rerata | Laki-laki % | ICU % |
| --- | --- | --- | --- | --- | --- | --- |
| dilihat | 59109 | 20306 | 3821 | 61.9 | 49.9 | 13.4 |
| tak tercantum | 1035 | 853 | 65 | 63.7 | 55.9 | 23.1 |
| tidak dilihat | 14757 | 5013 | 845 | 61.5 | 52.7 | 14.3 |

#### Kontras dilihat lawan tidak dilihat (DiD utama: pembanding M3b)

| Model | AUROC dilihat | AUROC tidak dilihat | Dilihat − tidak |
| --- | --- | --- | --- |
| M1 ukuran EKG + M0 (GB) | 0.673 [0.659, 0.688] | 0.657 [0.622, 0.689] | 0.016 [-0.019, 0.054] |
| M3 ECGFounder beku + probe L2 | 0.719 [0.705, 0.735] | 0.691 [0.661, 0.720] | 0.029 [-0.003, 0.062] |
| M3b ECG-JEPA beku + probe L2 | 0.748 [0.734, 0.762] | 0.722 [0.694, 0.751] | 0.025 [-0.007, 0.057] |
| M3c ECG-FM beku + probe L2 (terkontaminasi) | 0.730 [0.715, 0.745] | 0.698 [0.669, 0.728] | 0.031 [-0.001, 0.065] |

| DiD (M3c − pembanding) | Nilai [95 % CI] |
| --- | --- |
| utama: M3b ECG-JEPA beku + probe L2 | 0.006 [-0.019, 0.032] |
| sekunder: M3 ECGFounder beku + probe L2 | 0.003 [-0.017, 0.022] |
| sekunder: M1 ukuran EKG + M0 (GB) | 0.016 [-0.019, 0.048] |

Ulangan bootstrap terpakai: 1000; positif dilihat 3821, tidak dilihat 845.

#### M3c − M3b keseluruhan (sekunder)

| Bagian | Pasangan | Positif | AUROC M3c ECG-FM beku + probe L2 (terkontaminasi) | M3c − M3b |
| --- | --- | --- | --- | --- |
| VALIDATION ∪ TEST | 74901 | 4731 | 0.724 | -0.019 [-0.027, -0.011] |
| TEST | 36674 | 2277 | 0.717 | -0.021 [-0.035, -0.009] |

Selisih AUROC [95 % CI]; bootstrap bergugus pasien, kedua model pada ulangan yang sama; "—" bila positif < 11.


Waktu jalan: 15.1 menit.
