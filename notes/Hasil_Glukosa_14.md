# Hasil Glukosa 14 — analisis tambahan atas Hasil 13 (protokol §F langkah 12), 23 November 2026

Sumber angka: keluaran `glukosa_perluasan/jalankan_tambahan_perluasan.py --boot 1000 --perm 1000` (komit
4dce7bc) atas prediksi luar lipatan tersimpan langkah 11 (`data/mimic-iv-ext-ecg-glucose/prediksi/perluasan_*.npz`,
di luar repo; tanpa melatih ulang, C sama dengan Hasil 13). Dijalankan sekali di laptop 22–23 Nov 2026, selesai
sekitar pukul 00.20, venv overlay ECG-FM, keluar tanpa galat; yang keluar hanya agregat. Bagian di bawah "Keluaran
skrip" disambung langsung dari log; jangan diketik ulang.

**Status: PASCA-HASIL dan deskriptif.** Dicatat di rencana §F langkah 12 (komit d16e491) sesudah Hasil 13, sebelum
skrip ditulis. Tidak mengubah vonis langkah 11.

## 0. Catatan pembacaan

- **Keseimbangan.** Ketujuh tugas, semua |SMD| ≤ 0,018 (usia, laki-laki, prevalensi, unit per pasien; ICU untuk
  glukosa), jauh di bawah ambang 0,1.
- **Efek minimum terdeteksi (2,8 × SD bootstrap).** DiD utama: hipo 0,023, hiper berat 0,014, AF 0,001, LBBB 0,002,
  RBBB 0,002, AV blok 0,003, LVH 0,006. Kontrol positif: 0,026 / 0,017 / 0,001 / 0,003 / 0,001 / 0,003 / 0,005. MDE
  di bawah ε = 0,01 untuk kelima pernyataan; di atasnya untuk kedua tugas glukosa.
- **Presisi dari pembanding.** SD nol permutasi DiD utama dibanding SD selisih ECG-FM sendiri: rasio 0,62–0,79 pada
  enam tugas (21–38 % lebih sempit), tetapi **1,07 pada AV blok** (DiD lebih lebar), dengan korelasi selisih FM–JEPA
  0,58, terendah dari ketujuh (lainnya 0,70–0,80). DiD lebih sempit hanya bila kovarians > ½ var(selisih JEPA).
- **Kontrol-variat** (b = kov/var di bawah nol, 0,45–1,11): SD 0,0072 / 0,0047 / 0,0005 / 0,0008 / 0,0006 / 0,0009 /
  0,0024, sama atau sedikit lebih sempit dari DiD; titik 0,010 / −0,006 / 0,000 / 0,000 / 0,000 / 0,001 / 0,001,
  sejalan dengan DiD.
- **Cross-fitting lawan pembanding.** SD nol selisih ECG-FM sendiri turun dari 0,0237 (hipo) dan 0,0174 (hiper) pada
  protokol pertama (Hasil 12, VALIDATION ∪ TEST) ke 0,0101 dan 0,0077 dengan cross-fitting atas ketiga split (57 % dan
  56 % lebih sempit); pembanding lalu menambah 21 % dan 38 %. Bootstrap: lebar CI DiD utama hipo 0,063 → 0,031, hiper
  0,051 → 0,019 (Hasil 12 → 13).

## Keluaran skrip

### M0/M1/M2 — pasangan dengan gelombang terbaca

- pasangan: 437671; gelombang terbaca 437671; dibuang 0.

- EKG unik: 376648; positif per pernyataan: Fibrilasi atrium 37503, LBBB 14208, RBBB 30652, AV blok derajat 1 23053, LVH 36601

- hipo ecgfounder: tersimpan, C 0.001
- hiper_berat ecgfounder: tersimpan, C 0.001
- af ecgfounder: tersimpan, C 0.001
- lbbb ecgfounder: tersimpan, C 0.001
- rbbb ecgfounder: tersimpan, C 0.001
- avb1 ecgfounder: tersimpan, C 0.01
- lvh ecgfounder: tersimpan, C 0.001
- hipo ecgjepa: tersimpan, C 0.001
- hiper_berat ecgjepa: tersimpan, C 0.01
- af ecgjepa: tersimpan, C 0.01
- lbbb ecgjepa: tersimpan, C 0.01
- rbbb ecgjepa: tersimpan, C 0.1
- avb1 ecgjepa: tersimpan, C 0.1
- lvh ecgjepa: tersimpan, C 0.1
- hipo ecgfm: tersimpan, C 0.001
- hiper_berat ecgfm: tersimpan, C 0.001
- af ecgfm: tersimpan, C 0.1
- lbbb ecgfm: tersimpan, C 0.01
- rbbb ecgfm: tersimpan, C 0.1
- avb1 ecgfm: tersimpan, C 0.01
- lvh ecgfm: tersimpan, C 0.01
- hipo ecgfm_ft: tersimpan, C 0.001
- hiper_berat ecgfm_ft: tersimpan, C 0.01
- af ecgfm_ft: tersimpan, C 0.01
- lbbb ecgfm_ft: tersimpan, C 0.01
- rbbb ecgfm_ft: tersimpan, C 0.001
- avb1 ecgfm_ft: tersimpan, C 0.01
- lvh ecgfm_ft: tersimpan, C 0.1

Analisis tambahan langkah 12 (pasca-hasil); bootstrap 1000, permutasi 1000.

### Keseimbangan dilihat lawan tidak dilihat (|SMD| < 0.1)

| Tugas | Peubah | Rerata dilihat | Rerata tidak dilihat | SMD |
| --- | --- | --- | --- | --- |
| Hipoglikemia (< 70) lawan euglikemia (70–180) | usia | 61.800 | 61.887 | -0.005 |
| Hipoglikemia (< 70) lawan euglikemia (70–180) | laki-laki | 0.503 | 0.511 | -0.017 |
| Hipoglikemia (< 70) lawan euglikemia (70–180) | positif | 0.019 | 0.019 | 0.003 |
| Hipoglikemia (< 70) lawan euglikemia (70–180) | unit per pasien | 8.012 | 7.863 | 0.014 |
| Hipoglikemia (< 70) lawan euglikemia (70–180) | ICU | 0.135 | 0.137 | -0.005 |
| Hiperglikemia berat (> 250) lawan euglikemia (70–180) | usia | 61.739 | 61.814 | -0.004 |
| Hiperglikemia berat (> 250) lawan euglikemia (70–180) | laki-laki | 0.504 | 0.513 | -0.018 |
| Hiperglikemia berat (> 250) lawan euglikemia (70–180) | positif | 0.062 | 0.057 | 0.018 |
| Hiperglikemia berat (> 250) lawan euglikemia (70–180) | unit per pasien | 8.370 | 8.179 | 0.017 |
| Hiperglikemia berat (> 250) lawan euglikemia (70–180) | ICU | 0.134 | 0.136 | -0.004 |
| Fibrilasi atrium | usia | 61.909 | 61.998 | -0.005 |
| Fibrilasi atrium | laki-laki | 0.500 | 0.508 | -0.016 |
| Fibrilasi atrium | positif | 0.099 | 0.101 | -0.008 |
| Fibrilasi atrium | unit per pasien | 8.726 | 8.595 | 0.011 |
| LBBB | usia | 61.909 | 61.998 | -0.005 |
| LBBB | laki-laki | 0.500 | 0.508 | -0.016 |
| LBBB | positif | 0.037 | 0.039 | -0.010 |
| LBBB | unit per pasien | 8.726 | 8.595 | 0.011 |
| RBBB | usia | 61.909 | 61.998 | -0.005 |
| RBBB | laki-laki | 0.500 | 0.508 | -0.016 |
| RBBB | positif | 0.081 | 0.083 | -0.007 |
| RBBB | unit per pasien | 8.726 | 8.595 | 0.011 |
| AV blok derajat 1 | usia | 61.909 | 61.998 | -0.005 |
| AV blok derajat 1 | laki-laki | 0.500 | 0.508 | -0.016 |
| AV blok derajat 1 | positif | 0.060 | 0.064 | -0.015 |
| AV blok derajat 1 | unit per pasien | 8.726 | 8.595 | 0.011 |
| LVH | usia | 61.909 | 61.998 | -0.005 |
| LVH | laki-laki | 0.500 | 0.508 | -0.016 |
| LVH | positif | 0.097 | 0.099 | -0.008 |
| LVH | unit per pasien | 8.726 | 8.595 | 0.011 |

### Efek minimum terdeteksi (2.8 × SD bootstrap)

| Tugas | SD DiD utama | MDE utama | SD kontrol positif | MDE kontrol positif |
| --- | --- | --- | --- | --- |
| Hipoglikemia (< 70) lawan euglikemia (70–180) | 0.0083 | 0.023 | 0.0092 | 0.026 |
| Hiperglikemia berat (> 250) lawan euglikemia (70–180) | 0.0050 | 0.014 | 0.0061 | 0.017 |
| Fibrilasi atrium | 0.0005 | 0.001 | 0.0005 | 0.001 |
| LBBB | 0.0009 | 0.002 | 0.0009 | 0.003 |
| RBBB | 0.0006 | 0.002 | 0.0005 | 0.001 |
| AV blok derajat 1 | 0.0011 | 0.003 | 0.0012 | 0.003 |
| LVH | 0.0023 | 0.006 | 0.0019 | 0.005 |

### Presisi dari pembanding di bawah nol permutasi

| Tugas | SD selisih ECG-FM | SD DiD utama | Rasio | Korelasi selisih FM–JEPA | b kontrol-variat | Titik kontrol-variat | SD kontrol-variat |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Hipoglikemia (< 70) lawan euglikemia (70–180) | 0.0101 | 0.0080 | 0.79 | 0.70 | 0.68 | 0.010 | 0.0072 |
| Hiperglikemia berat (> 250) lawan euglikemia (70–180) | 0.0077 | 0.0048 | 0.62 | 0.80 | 0.85 | -0.006 | 0.0047 |
| Fibrilasi atrium | 0.0008 | 0.0005 | 0.62 | 0.79 | 0.93 | 0.000 | 0.0005 |
| LBBB | 0.0012 | 0.0009 | 0.69 | 0.78 | 0.73 | 0.000 | 0.0008 |
| RBBB | 0.0009 | 0.0006 | 0.68 | 0.73 | 0.96 | 0.000 | 0.0006 |
| AV blok derajat 1 | 0.0011 | 0.0011 | 1.07 | 0.58 | 0.45 | 0.001 | 0.0009 |
| LVH | 0.0035 | 0.0024 | 0.69 | 0.73 | 1.11 | 0.001 | 0.0024 |
