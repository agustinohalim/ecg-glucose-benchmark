# Hasil Glukosa 10 — kepekaan TEST atas telaah internal pertama: dalam pasien, tempat perawatan, waktu EKG (22 November 2026)

Sumber angka: keluaran `glukosa_test/jalankan_kepekaan_panel.py --boot 1000` atas
`mimiciv_ecg_glucose_aligned.csv` v1.0.0 (PhysioNet, DOI 10.13026/nts2-vj76, Credentialed Health Data License 1.5.0)
dan prediksi TEST tersimpan dari Hasil 8 (`data/mimic-iv-ext-ecg-glucose/prediksi/test_*.npz`, di luar repo).
Dijalankan di laptop 22 Nov 2026, 2,0 menit, venv-gpu, tanpa pelatihan; yang keluar hanya agregat (hitungan 1–10
dicetak `<11`, metrik tidak dicetak bila positif < 11). Bagian di bawah "Keluaran skrip" disambung langsung dari log;
jangan diketik ulang.

**Status: sesudah protokol, deskriptif.** Analisis ini ditambahkan sesudah TEST dibaca (Hasil 8) dan sesudah telaah internal pertama, untuk menjawab temuan kritis advokat setan (sinyal antarpasien lawan
keadaan glukosa) dan temuan metode tentang tempat perawatan serta waktu EKG. Tidak mengubah uji utama.

## Pilihan analisis (ditetapkan sebelum dijalankan)

1. **AUROC dalam pasien:** pasien TEST yang punya pasangan positif dan euglikemik; tiap pasangan positif hanya
   dibandingkan dengan pasangan euglikemik pasien yang sama (AUROC bertingkat per pasien, dijumlah atas pasien).
   Bootstrap 1000 atas pasien yang memenuhi syarat.
2. **Tempat perawatan:** tanpa rawat inap (`hadm_id` kosong: IGD atau rawat jalan menurut kamus data), rawat inap di
   luar ICU, ICU (`during_icu_stay`).
3. **Waktu EKG:** `temporal_relationship` (sebelum, sesudah, serentak) dan `abs_offset_minutes` (≤ 30, 31–120,
   121–240).

## 0. Catatan pembacaan

- **Dalam pasien, keunggulan gelombang tidak terbukti dan diskriminasi keduanya lemah.** Hipo (438 pasien, 6.729
  perbandingan): M3b 0,554 [0,511; 0,590], M1-GB 0,521 [0,483; 0,550], selisih 0,033 [−0,018; 0,080]. Hiper berat
  (713 pasien, 14.289 perbandingan): M3b 0,619 [0,590; 0,652], M1-GB 0,595 [0,554; 0,630], selisih 0,024 [−0,005;
  0,052]. Bandingkan AUROC keseluruhan 0,643 dan 0,738: sebagian besar diskriminasi berasal dari perbedaan
  antarpasien. Temuan kritis advokat setan didukung data; judul "detects ... hypo- and hyperglycemia" tidak bisa
  dipertahankan tanpa kualifikasi.
- **Tempat perawatan tidak menghapus selisih:** hipo 0,061 (tanpa rawat inap), 0,060 (bangsal), 0,114 (ICU, 90
  positif); hiper 0,078, 0,030 [0,001; 0,057], 0,084. Semua interval di atas nol. Bangsal hiper paling kecil.
- **Waktu EKG tidak menghapus selisih:** EKG sebelum glukosa 0,065 / 0,070; sesudah 0,068 / 0,060; per jarak 0,056
  sampai 0,081. Tidak ada pola yang menunjukkan selisih berasal dari EKG sesudah pengobatan.
- Pembagian pasangan: hipo 19.274 sebelum, 15.495 sesudah, 334 serentak; hiper 20.220 / 16.091 / 363.

## Keluaran skrip

### M0/M1/M2 — pasangan dengan gelombang terbaca

- pasangan: 437671; gelombang terbaca 437671; dibuang 0.

Kepekaan TEST atas telaah internal pertama, deskriptif, sesudah protokol; bootstrap 1000 bergugus pasien.

### Hipoglikemia (< 70) lawan euglikemia (70–180) — TEST (non-EXCLUDE)

#### AUROC dalam pasien (pasien dengan pasangan positif dan euglikemik: 438)

| Ukuran | Nilai [95 % CI] |
| --- | --- |
| AUROC dalam pasien M3b ECG-JEPA beku + probe L2 | 0.554 [0.511, 0.590] |
| AUROC dalam pasien M1 ukuran EKG + M0 (GB) | 0.521 [0.483, 0.550] |
| M3b − M1 (GB) | 0.033 [-0.018, 0.080] |
| Perbandingan positif–euglikemik dalam pasien | 6729 |

Bootstrap pasien atas pasien yang memenuhi syarat.

#### Per tempat perawatan dan waktu EKG

| Bagian | Pasangan | Positif | AUROC M3b ECG-JEPA beku + probe L2 | M3b − M1 (GB) |
| --- | --- | --- | --- | --- |
| TEST, semua | 35103 | 706 | 0.643 | 0.066 [0.043, 0.091] |
| tanpa rawat inap (IGD/rawat jalan) | 20057 | 367 | 0.635 | 0.061 [0.029, 0.094] |
| rawat inap di luar ICU | 10346 | 249 | 0.645 | 0.060 [0.017, 0.100] |
| ICU | 4700 | 90 | 0.667 | 0.114 [0.035, 0.195] |
| EKG sebelum glukosa | 19274 | 366 | 0.634 | 0.065 [0.030, 0.100] |
| EKG sesudah glukosa | 15495 | 338 | 0.652 | 0.068 [0.033, 0.104] |
| serentak (menit yang sama) | 334 | <11 | — | — |
| jarak ≤ 30 menit | 13167 | 214 | 0.642 | 0.063 [0.020, 0.106] |
| jarak 31–120 menit | 12737 | 271 | 0.630 | 0.056 [0.016, 0.096] |
| jarak 121–240 menit | 9199 | 221 | 0.652 | 0.081 [0.031, 0.123] |

Selisih AUROC [95 % CI]; bootstrap bergugus pasien, kedua model pada ulangan yang sama; "—" bila positif < 11.

### Hiperglikemia berat (> 250) lawan euglikemia (70–180) — TEST (non-EXCLUDE)

#### AUROC dalam pasien (pasien dengan pasangan positif dan euglikemik: 713)

| Ukuran | Nilai [95 % CI] |
| --- | --- |
| AUROC dalam pasien M3b ECG-JEPA beku + probe L2 | 0.619 [0.590, 0.652] |
| AUROC dalam pasien M1 ukuran EKG + M0 (GB) | 0.595 [0.554, 0.630] |
| M3b − M1 (GB) | 0.024 [-0.005, 0.052] |
| Perbandingan positif–euglikemik dalam pasien | 14289 |

Bootstrap pasien atas pasien yang memenuhi syarat.

#### Per tempat perawatan dan waktu EKG

| Bagian | Pasangan | Positif | AUROC M3b ECG-JEPA beku + probe L2 | M3b − M1 (GB) |
| --- | --- | --- | --- | --- |
| TEST, semua | 36674 | 2277 | 0.738 | 0.066 [0.050, 0.081] |
| tanpa rawat inap (IGD/rawat jalan) | 21026 | 1336 | 0.772 | 0.078 [0.061, 0.096] |
| rawat inap di luar ICU | 10753 | 656 | 0.682 | 0.030 [0.001, 0.057] |
| ICU | 4895 | 285 | 0.712 | 0.084 [0.042, 0.133] |
| EKG sebelum glukosa | 20220 | 1312 | 0.752 | 0.070 [0.052, 0.090] |
| EKG sesudah glukosa | 16091 | 934 | 0.717 | 0.060 [0.040, 0.081] |
| serentak (menit yang sama) | 363 | 31 | 0.736 | 0.074 [-0.010, 0.161] |
| jarak ≤ 30 menit | 13839 | 886 | 0.739 | 0.064 [0.044, 0.082] |
| jarak 31–120 menit | 13300 | 834 | 0.752 | 0.072 [0.049, 0.093] |
| jarak 121–240 menit | 9535 | 557 | 0.715 | 0.060 [0.037, 0.085] |

Selisih AUROC [95 % CI]; bootstrap bergugus pasien, kedua model pada ulangan yang sama; "—" bila positif < 11.


Waktu jalan: 2.0 menit.
