# Hasil Glukosa 9 — subkelompok usia dan jenis kelamin di TEST (deskriptif, sesudah protokol; 22 November 2026)

Sumber angka: keluaran `glukosa_test/jalankan_subkelompok.py --boot 1000` atas
`mimiciv_ecg_glucose_aligned.csv` v1.0.0 (PhysioNet, DOI 10.13026/nts2-vj76, Credentialed Health Data License
1.5.0) dan prediksi TEST tersimpan dari jalan Hasil 8 (`test_<tugas>.npz`, di luar repo, tidak dibuka di sesi;
baris dicocokkan dengan y dan subject_id oleh skrip). Dijalankan di laptop 22 Nov 2026, 1,3 menit, tanpa
pelatihan; stderr kosong. Yang keluar hanya agregat (hitungan 1–10 dicetak `<11`, metrik tidak dicetak bila positif
< 11). Uji jalan lebih dulu pada data sintetis. Tabel di bawah bagian 0 disambung langsung dari log skrip.

**Analisis ini di luar protokol TEST (45ff2cb)** dan ditambahkan atas keputusan penulis 22 Nov 2026
(`Tinjauan_Glukosa_2026-11-22.md`) sesudah Hasil 8 dibaca. Tidak ada model, resep, atau uji utama yang berubah;
semua angka di sini deskriptif.

## Pilihan analisis (ditetapkan sebelum dijalankan)

- Perbandingan: M3b − M1-GB (perbandingan utama protokol), per subkelompok, bootstrap bergugus pasien 1.000
  ulangan, benih `BENIH[0]`.
- Kelompok usia: 18–44, 45–64, 65–79, ≥ 80 tahun (kolom `age`). Jenis kelamin dari `gender`.
- Ras tidak dianalisis: kosong bila `hadm_id` kosong (53,5 % pasangan, README).
- Tidak ada uji beda antar-subkelompok (interaksi).

## 0. Catatan pembacaan

1. Baris "TEST, semua" sama dengan uji utama Hasil 8 (0,066 dan 0,066). CI-nya bergeser di desimal ketiga
   (hipo [0,043; 0,091] lawan [0,041; 0,093]; hiper berat [0,050; 0,081] lawan [0,052; 0,082]) karena urutan
   tarikan bootstrap berbeda (di Hasil 8 tabel metrik menarik lebih dulu). Angka naskah untuk uji utama tetap
   dari Hasil 8.
2. **Hipoglikemia:** M3b − M1-GB positif di keenam subkelompok, 0,038–0,093; CI di atas 0 kecuali usia 18–44
   (0,038 [−0,015; 0,089], 128 positif). AUROC M3b terendah pada usia ≥ 80 (0,578).
3. **Hiperglikemia berat:** positif di keenam subkelompok, 0,039–0,101; CI di atas 0 kecuali usia ≥ 80 (0,039
   [−0,004; 0,082], 290 positif). AUROC M3b menurun dengan usia (0,796 → 0,678).
4. Perempuan lawan laki-laki (hipo 0,086 lawan 0,041; hiper berat 0,084 lawan 0,050) dan antar-kelompok usia
   tidak diuji; CI bertumpang. Tidak ada klaim beda antar-subkelompok.

**Akibat untuk naskah:** keunggulan atas ukuran mesin tidak bergantung pada satu kelompok usia atau jenis kelamin;
kinerja lebih rendah pada pasien tertua. Masuk naskah sebagai tabel suplemen + satu kalimat di Hasil, ditandai
"post hoc, descriptive"; 4.7 diubah (subkelompok usia dan jenis kelamin dianalisis, ras tidak).

## Keluaran skrip

Judul dari skrip diturunkan dua tingkat (`##` ditambah); isi lain apa adanya.

### M0/M1/M2 — pasangan dengan gelombang terbaca

- pasangan: 437671; gelombang terbaca 437671; dibuang 0.

Subkelompok TEST, deskriptif, sesudah protokol; bootstrap 1000 bergugus pasien.

#### Hipoglikemia (< 70) lawan euglikemia (70–180) — TEST (non-EXCLUDE), per subkelompok

| Bagian | Pasangan | Positif | AUROC M3b ECG-JEPA beku + probe L2 | M3b − M1 (GB) |
| --- | --- | --- | --- | --- |
| TEST, semua | 35103 | 706 | 0.643 | 0.066 [0.043, 0.091] |
| usia 18–44 | 6251 | 128 | 0.639 | 0.038 [-0.015, 0.089] |
| usia 45–64 | 12208 | 275 | 0.677 | 0.081 [0.042, 0.121] |
| usia 65–79 | 10455 | 196 | 0.634 | 0.066 [0.022, 0.114] |
| usia ≥ 80 | 6189 | 107 | 0.578 | 0.093 [0.012, 0.176] |
| perempuan | 17487 | 390 | 0.651 | 0.086 [0.057, 0.119] |
| laki-laki | 17616 | 316 | 0.632 | 0.041 [0.003, 0.079] |

Selisih AUROC [95 % CI]; bootstrap bergugus pasien, kedua model pada ulangan yang sama; "—" bila positif < 11.

#### Hiperglikemia berat (> 250) lawan euglikemia (70–180) — TEST (non-EXCLUDE), per subkelompok

| Bagian | Pasangan | Positif | AUROC M3b ECG-JEPA beku + probe L2 | M3b − M1 (GB) |
| --- | --- | --- | --- | --- |
| TEST, semua | 36674 | 2277 | 0.738 | 0.066 [0.050, 0.081] |
| usia 18–44 | 6429 | 306 | 0.796 | 0.091 [0.053, 0.137] |
| usia 45–64 | 13039 | 1106 | 0.748 | 0.101 [0.078, 0.125] |
| usia 65–79 | 10834 | 575 | 0.699 | 0.042 [0.013, 0.068] |
| usia ≥ 80 | 6372 | 290 | 0.678 | 0.039 [-0.004, 0.082] |
| perempuan | 18144 | 1047 | 0.755 | 0.084 [0.059, 0.107] |
| laki-laki | 18530 | 1230 | 0.722 | 0.050 [0.031, 0.069] |

Selisih AUROC [95 % CI]; bootstrap bergugus pasien, kedua model pada ulangan yang sama; "—" bila positif < 11.


Waktu jalan: 1.3 menit.
