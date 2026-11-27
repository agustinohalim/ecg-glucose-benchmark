# Hasil Glukosa 20 — audit terkalibrasi: cek manipulasi berpasangan dan plasebo kesesuaian segmen (protokol §F langkah 18), 27 November 2026

Sumber angka: keluaran `glukosa_audit/jalankan_kesesuaian.py` (venv-gpu, GPU) dan `uji_audit.py` (venv-gpu;
bootstrap 2000, benih 20261120), keduanya pada kode komit 6625c48 tanpa perubahan. Protokol dicatat di komit ebb89dc
sebelum kode ditulis; kode dikomit dan diuji asap pada data sintetis sebelum jalan pada data nyata; protokol dan kode
dirilis publik sebagai versi 1.3.0 (DOI 10.5281/zenodo.24582292, 27 Nov 2026 08:39 UTC) sebelum statistik apa pun
dihitung. Kosinus per rekaman model bersih dihitung sebelum rilis itu, sesudah kode dikomit, dan tidak dibuka. Skor
per rekaman ada di `data/replikasi` dan `data/kontrol`, di luar repo dan tidak dibuka di sesi
asisten; yang keluar hanya agregat. Bagian "Keluaran skrip" disambung langsung dari log (peringatan pustaka dibuang);
jangan diketik ulang.

**Status: PRA-REGISTRASI PUBLIK untuk K2, C2, M2.** Semua bacaan lain deskriptif (langkah 18a–b): AUROC keanggotaan
dari −Δℓ dan Δc, semua bacaan ECG-FM+1, SMD Δc, SMD kosinus tak berpasangan, SMD per model bersih, ECG-FM dengan
`valid` dan `test` terpisah, SMD kovariat.

## 0. Catatan pembacaan

- **K2 GAGAL.** Dₚ(ECG-FM+4) = rerata Δℓ ditahan − disuntik = 0,0007 [−0,0039; 0,0049], batas bawah ≤ 0. ECG-FM+1
  0,0007 [−0,0041; 0,0048]. AUROC keanggotaan dari −Δℓ 0,4993 [0,4914; 0,5067]. Membandingkan kehilangan checkpoint
  sesudah dan sebelum suntikan pada rekaman yang sama, pasien disuntik tidak turun kehilangannya lebih dari pasien
  ditahan. **Akibat menurut aturan (18a, 18d):** kehilangan tidak mencatat suntikan sama sekali. Suntikan dibaca sebagai
  dosis yang tidak terlihat pada kehilangan; K1, P1, P2 (Hasil 19) dibaca sebagai bacaan atas paparan yang tidak terlihat
  pada kehilangan, dan naskah tidak menyebut uji kehilangan "buta terhadap anggota yang diketahui" tanpa syarat itu.
- **C2 GAGAL.** Cₚ(ECG-FM+4) = rerata Δc disuntik − ditahan = 0,0002 [−0,0000; 0,0004]; batas bawah tercetak −0,0000,
  yaitu negatif dan lebih kecil dari 0,00005 dalam nilai mutlak, jadi aturan "> 0" tidak terpenuhi. Ini kegagalan
  tipis. ECG-FM+1 0,0001 [−0,0001; 0,0003]. Deskriptif: SMD Δc ECG-FM+4 0,021 [−0,005; 0,048]; AUROC keanggotaan dari
  Δc 0,506 [0,498; 0,514]; SMD kosinus ECG-FM+4 disuntik − ditahan tak berpasangan 0,026 [0,000; 0,051].
- **M2 LOLOS.** SMD kesesuaian segmen dilihat − tidak dilihat: ECG-FM 0,129 [0,111; 0,147], ECG-JEPA 0,006 [−0,012;
  0,024], ECGFounder 0,025 [0,007; 0,044]. Δ = SMD ECG-FM − SMD model bersih: ECG-JEPA 0,123 [0,102; 0,145],
  ECGFounder 0,103 [0,082; 0,125]; batas bawah keduanya > 0. **Akibat menurut aturan (18d):** judul dan abstrak
  menyatakan daftar split didukung oleh jejak khas ECG-FM; batas 0,002 (Hasil 17) dibaca sebagai batas efek
  keanggotaan dengan syarat jejak itu memang keanggotaan; klaim "tidak dapat diaudit dari artefak rilis" dicabut.

**Sekunder deskriptif (tanpa vonis):**

- **Plasebo mentah tidak nol untuk ECGFounder.** SMD ECGFounder 0,025 dengan selang di atas nol: pasien `train` dan
  pasien `valid` ∪ `test` sedikit berbeda dalam kesesuaian segmen bahkan pada model yang tidak pernah melihat
  MIMIC-IV-ECG. ECG-JEPA memuat nol. Selisih ECG-FM kira-kira lima kali plasebo ECGFounder dan dua puluh kali plasebo
  ECG-JEPA. Inilah alasan M2 diukur sebagai Δ, bukan SMD ECG-FM mentah.
- **Tidak bergantung pada satu bagian.** SMD ECG-FM dengan pembanding `valid` saja 0,130 [0,106; 0,155], `test` saja
  0,127 [0,103; 0,154].
- **Kovariat seimbang.** SMD laju jantung 0,012 dan durasi QRS −0,014 antara dilihat dan tidak dilihat (hilang 24 dan
  28 rekaman). Kedua kelompok tidak berbeda pada dua ukuran mesin ini, jadi selisih kesesuaian tidak berasal dari
  perbedaan irama atau lebar QRS yang kasar.
- **Perbandingan dosis.** Pra-latih asli (±240 epoch atas pasien `train`) memberi SMD 0,129; suntikan 4 epoch memberi
  SMD 0,021 berpasangan dan 0,026 tak berpasangan, keduanya dengan batas bawah di sekitar nol. Arahnya sama dan
  besarnya lebih kecil. Ini sejalan dengan jejak yang tumbuh dengan paparan, tetapi dua titik dosis dengan resep latih
  berbeda (ukuran batch 171 lawan 1.026, Hasil 19) tidak cukup untuk kurva dosis; tidak diklaim.
- **Bacaan untuk naskah.** Jejak yang memisahkan anggota daftar adalah kesesuaian segmen, bukan kehilangan. Kehilangan
  tidak memisahkan anggota daftar (M1, Hasil 18), anggota suntikan secara tak berpasangan (K1, Hasil 19), maupun secara
  berpasangan (K2). Jadi naskah menyatakan: daftar didukung oleh jejak khas ECG-FM pada kesesuaian segmen, dan uji
  kehilangan tidak dapat memeriksanya. Syarat "jejak itu memang keanggotaan" tetap: M2 menyingkirkan perbedaan populasi
  yang juga terlihat oleh model bersih, tetapi tidak menyingkirkan perbedaan populasi yang hanya terlihat oleh ECG-FM.
  C2 yang gagal tipis berarti suntikan 4 epoch tidak dapat dipakai sebagai kalibrasi besar jejak itu.

## 1. Catatan proses

- `jalankan_kesesuaian.py` berjalan sekali di jendela cmd terpisah, 27 Nov 14:06–15:09 WIB; terbaca 44.116 dari 44.116.
  10.000 rekaman kedua memakan 45 menit (lawan 6 menit untuk 10.000 pertama) karena GPU dibatasi daya (1.080 dari
  3.090 MHz, alasan "SW power cap") selama laptop memakai baterai; sesudah pengisi daya terpasang, kecepatan pulih.
  Keluaran tidak terpengaruh: forward pass deterministik dan tidak ada yang dijalankan ulang.
- Rilis 1.3.0 (komit publik 8c9fc0e) dibuat sesudah kosinus dihitung dan sebelum `uji_audit.py` dijalankan.
- `uji_audit.py` dijalankan sekali di jendela cmd terpisah, 27 Nov sesudah 08:39 UTC; selesai dalam kurang dari satu
  menit.

## Keluaran skrip

`jalankan_kesesuaian.py`:

```
Kesesuaian segmen model bersih langkah 18b; benih 20261120; rekaman 44116.
14:12 rekaman 10000
14:57 rekaman 20000
15:02 rekaman 30000
15:07 rekaman 40000
Selesai: terbaca 44116 dari 44116.
```

`uji_audit.py --boot 2000`:

```
Audit terkalibrasi langkah 18; benih 20261120.

| Populasi | Kelompok | Pembanding | Dibuang |
| --- | --- | --- | --- |
| (a) tidak dilihat: disuntik / ditahan | 11029 | 11029 | 0 |
| (b) langkah 16: dilihat / tidak dilihat | 22058 | 22058 | 0 |

## Statistik (bootstrap 2000 ulangan atas pasien per populasi, benih 20261120)

| (a) Berpasangan, disuntik lawan ditahan | ECG-FM+4 | ECG-FM+1 |
| --- | --- | --- |
| Dp = rerata Δℓ ditahan − disuntik | 0.0007 [-0.0039, 0.0049] | 0.0007 [-0.0041, 0.0048] |
| Cp = rerata Δc disuntik − ditahan | 0.0002 [-0.0000, 0.0004] | 0.0001 [-0.0001, 0.0003] |
| AUROC keanggotaan dari −Δℓ | 0.4993 [0.4914, 0.5067] | 0.4998 [0.4918, 0.5069] |
| AUROC keanggotaan dari Δc | 0.5057 [0.4982, 0.5135] | 0.4970 [0.4899, 0.5048] |
| SMD Δc disuntik − ditahan | 0.0209 [-0.0046, 0.0476] | 0.0106 [-0.0150, 0.0383] |

SMD kosinus ECG-FM+4 disuntik − ditahan (tak berpasangan): 0.0256 [0.0004, 0.0512]

| (b) Kesesuaian segmen, dilihat − tidak dilihat | Nilai [95 % CI] |
| --- | --- |
| SMD ECG-FM | 0.1285 [0.1107, 0.1474] |
| SMD ECG-JEPA | 0.0058 [-0.0120, 0.0238] |
| SMD ECGFounder | 0.0254 [0.0072, 0.0443] |
| Δ = SMD ECG-FM − SMD ECG-JEPA | 0.1228 [0.1015, 0.1446] |
| Δ = SMD ECG-FM − SMD ECGFounder | 0.1032 [0.0820, 0.1254] |
| SMD ECG-FM, tidak dilihat = valid | 0.1303 [0.1063, 0.1551] |
| SMD ECG-FM, tidak dilihat = test | 0.1268 [0.1032, 0.1541] |

| Kovariat (dilihat − tidak dilihat) | SMD | Hilang |
| --- | --- | --- |
| denyut | 0.012 | 24 |
| qrs_ms | -0.014 | 28 |

## Keputusan praregistrasi

- K2 (batas bawah > 0): TIDAK LOLOS (batas bawah -0.0039); tidak terbukti: latih lanjut meninggalkan jejak pada kehilangan pasien disuntik.
- C2 (batas bawah > 0): TIDAK LOLOS (batas bawah -0.0000); tidak terbukti: latih lanjut menaikkan kesesuaian segmen pasien disuntik.
- M2 (batas bawah Δ > 0 untuk kedua model bersih): LOLOS (ECG-JEPA 0.1015, ECGFounder 0.0820); daftar didukung oleh jejak khas ECG-FM.
```
