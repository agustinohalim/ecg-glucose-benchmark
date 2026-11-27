# Hasil Glukosa 21 — uji ulang jejak pada data segar, plasebo sepadan masukan, inferensi dataset, dan kehilangan berpasangan bertopeng sama (protokol §F langkah 19), 27 November 2026

Sumber angka: keluaran `glukosa_audit2/jalankan_fm_segar.py` (venv ECG-FM, GPU),
`jalankan_bersih_segar.py` (venv-gpu, GPU), dan `uji_audit2.py` (venv-gpu; bootstrap 2000, benih 20261120), ketiganya
pada kode komit bd72e55 tanpa perubahan. Protokol dicatat di komit d259f63 sebelum kode ditulis; kode dikomit dan
diuji asap pada data sintetis sebelum jalan pada data nyata; protokol dan kode dirilis publik sebagai versi 1.4.0 (DOI
10.5281/zenodo.24583858; komit paket 0030055, 27 Nov 16:44 WIB) sebelum bacaan atau statistik apa pun langkah ini
dihitung. Bacaan per rekaman ada di `data/audit2`, di luar repo dan tidak dibuka di sesi asisten; yang keluar
hanya agregat. Bagian "Keluaran skrip" disambung langsung dari log; jangan diketik ulang.

**Status: PRA-REGISTRASI PUBLIK untuk M3, D1, K3.** Semua bacaan lain deskriptif (langkah 19e): SMD per model, SMD
ECG-FM dengan `valid` dan `test` terpisah, AUROC dan TPR pada FPR 1 % per rekaman, AUROC keanggotaan dari −Δℓ, SMD
kovariat, efek terdeteksi minimum.

## 0. Catatan pembacaan

- **M3 GAGAL.** Pada 20.214 EKG kohort segar (10.107 pasien dilihat, 10.107 tidak dilihat; satu EKG per pasien, pasien di
  luar 44.116 pasien langkah 16), SMD kesesuaian segmen dilihat − tidak dilihat: ECG-FM 0,088 [0,060; 0,115], ECG-JEPA
  0,089 [0,062; 0,117], ECGFounder −0,043 [−0,072; −0,015], ECG-FM-acak −0,012 [−0,040; 0,015]. Δ_c terhadap ECG-JEPA
  −0,002 [−0,033; 0,031] (efek terdeteksi minimum 0,046); terhadap ECGFounder 0,131 [0,099; 0,162]; terhadap ECG-FM-acak
  0,100 [0,065; 0,136]. Batas bawah terhadap ECG-JEPA ≤ 0, jadi M3 gagal walau dua rujukan lain lolos. **Akibat menurut
  aturan (19g):** M2 (Hasil 20) dilaporkan tidak direplikasi; judul dan abstrak tidak lagi menyatakan daftar split
  didukung; batas 0,002 (Hasil 17) dinyatakan bersyarat pada daftar; sumbangan menjadi audit negatif: baik kehilangan
  maupun kesesuaian segmen tidak memverifikasi daftar secara terulang.
- **D1 GAGAL.** SMD log-likelihood kepadatan (PCA 64 + GMM 10 disetel pada 10.107 pasien dilihat lain) dilihat − tidak
  dilihat: ECG-FM −0,139 [−0,166; −0,111], ECG-JEPA −0,142 [−0,171; −0,115], ECGFounder −0,142 [−0,169; −0,115]. Δ_DI
  terhadap ECG-JEPA 0,003 [−0,030; 0,039], terhadap ECGFounder 0,004 [−0,031; 0,039] (efek terdeteksi minimum 0,049 dan
  0,050). **Akibat (19g):** dilaporkan sebagai uji yang tidak mendeteksi keanggotaan; tidak berpengaruh pada M3.
- **K3 LOLOS.** Dengan undian topeng dan negatif yang sama di kedua checkpoint (cek determinisme 20 rekaman pertama:
  identik), Dₚ = rerata Δℓ ditahan − disuntik 0,0017 [0,0006; 0,0028] (efek terdeteksi minimum 0,0016) pada 22.058
  pasien (11.029 disuntik, 11.029 ditahan). AUROC keanggotaan dari −Δℓ 0,5105 [0,5033; 0,5181]. **Akibat (19g):**
  kehilangan mencatat suntikan empat epoch bila derau topeng dihapus; klaim "kehilangan tidak dapat mengaudit"
  disempitkan menjadi "uji kehilangan satu rekaman tak berpasangan kurang daya".

**Sekunder deskriptif (tanpa vonis):**

- **Plasebo model bersih tidak stabil antar-populasi.** Pada unit langkah 16/18, ECG-JEPA memberi SMD 0,006 (Hasil 20);
  pada unit segar 0,089, sama dengan ECG-FM. ECGFounder bergeser dari 0,025 ke −0,043. Jadi besar dan arah "plasebo"
  bergantung pada model rujukan dan populasi, dan Δ ECG-FM di Hasil 20 tidak lebih kokoh dari pilihan rujukan itu.
  ECG-FM-acak (sebaran bobot sama, struktur hasil latih hilang) memuat nol, jadi selisih ECG-FM bukan artefak
  arsitektur atau pra-proses; tetapi model bersih terlatih (ECG-JEPA) yang tidak pernah melihat MIMIC-IV-ECG menghasilkan
  selisih yang sama besar.
- **ECG-FM lebih kecil pada data segar.** SMD 0,088 lawan 0,129 di langkah 16/18; dengan pembanding `valid` saja 0,109
  [0,072; 0,147], `test` saja 0,065 [0,031; 0,103].
- **Kovariat segar tidak seimbang pada umur.** SMD umur 0,212, durasi QRS 0,111, laki-laki 0,017, laju jantung 0,017
  (tidak ada yang hilang). Pasien dilihat di unit segar lebih tua dan QRS-nya sedikit lebih lebar. Di langkah 18 laju
  jantung dan QRS seimbang (0,012 dan −0,014; umur tidak diukur di sana). Perbedaan populasi semacam ini dapat
  menjelaskan selisih ECG-JEPA, tetapi tidak diuji di sini; tidak diklaim sebagai penyebab.
- **Arah inferensi dataset berlawanan dengan keanggotaan dan sama di ketiga model.** Rekaman dilihat mendapat
  log-likelihood lebih rendah dari rekaman tidak dilihat, padahal kepadatan disetel pada pasien dilihat; AUROC per
  rekaman 0,457–0,459 untuk ketiganya. Ini bacaan perbedaan populasi yang dilihat semua penyandi, bukan jejak ECG-FM.
- **Daya pisah per rekaman mendekati kebetulan.** AUROC kosinus segar: ECG-FM 0,521, ECG-JEPA 0,531, ECGFounder 0,491,
  ECG-FM-acak 0,499; TPR pada FPR 1 %: 0,010–0,014. Pada 44.116 rekaman langkah 16/18: ECG-FM 0,523 (TPR 0,013),
  ECG-JEPA 0,501, ECGFounder 0,505. Tidak ada jejak yang dapat mengaudit rekaman satu per satu.
- **K3 dan K2.** Menghapus derau topeng mempersempit selang Dₚ ECG-FM+4 dari lebar 0,0088 (K2, [−0,0039; 0,0049]) ke
  0,0022. Titik Dₚ K3 (0,0017) berada di dalam selang K2. Efeknya kecil: AUROC 0,51 berarti kehilangan berpasangan pun
  hampir tidak memisahkan pasien disuntik dari ditahan satu per satu.
- **Bacaan untuk naskah.** Jejak kesesuaian segmen yang memisahkan anggota daftar pada unit langkah 16 tidak bertahan
  sebagai jejak khas ECG-FM pada unit segar: satu penyandi bersih menunjukkan selisih yang sama. Naskah tidak lagi dapat
  menyatakan daftar didukung. Yang tersisa: (i) kehilangan dan kesesuaian segmen tidak memverifikasi daftar secara
  terulang; (ii) kehilangan berpasangan dengan topeng sama mendeteksi empat epoch suntikan yang diketahui, dengan efek
  kecil, sehingga kegagalan K1/K2 dibaca sebagai kurang daya, bukan kebutaan; (iii) batas 0,002 berlaku bila daftar
  benar, dan daftar itu tidak dapat diperiksa dari artefak rilis dengan uji-uji ini.

## 1. Catatan proses

- Rangkaian berjalan sekali di jendela cmd terpisah dengan pengisi daya terpasang, 27 Nov 16:48–19:03 WIB, dirantai
  `&&`: `jalankan_fm_segar.py` 16:48–18:27 (kosinus ECG-FM dan ECG-FM-acak 20.214 dari 20.214; kehilangan bertopeng
  sama 22.058 dari 22.058), `jalankan_bersih_segar.py` 18:27–19:02 (20.214 dari 20.214), `uji_audit2.py` 19:02–19:03.
  Berkas stderr ketiganya kosong. Access violation dari jalan asap pertama tidak muncul.
- Tidak ada yang dijalankan ulang. Pemantau latar di sesi asisten dihentikan sistem karena RAM laptop menipis (±18:00);
  rangkaian di jendela cmd tidak terpengaruh dan selesai sendiri.
- Unit segar: 437.671 pasangan dengan gelombang terbaca; setel 10.107 dilihat; uji 10.107 dilihat, 5.096 tidak dilihat
  `valid`, 5.011 tidak dilihat `test`. Tidak ada rekaman dibuang di ketiga populasi.

## Keluaran skrip

`jalankan_fm_segar.py`:

```
# M0/M1/M2 — pasangan dengan gelombang terbaca

- pasangan: 437671; gelombang terbaca 437671; dibuang 0.

Unit segar langkah 19a; benih 20261120.

| Peran | Dilihat | Tidak dilihat (valid) | Tidak dilihat (test) |
| --- | --- | --- | --- |
| setel | 10107 | 0 | 0 |
| uji | 10107 | 5096 | 5011 |
16:53 kosinus rekaman 5000
16:57 kosinus rekaman 10000
17:01 kosinus rekaman 15000
17:06 kosinus rekaman 20000
Kosinus ECG-FM terbaca 20214 dari 20214.
17:22 kehilangan rekaman 5000
17:41 kehilangan rekaman 10000
17:59 kehilangan rekaman 15000
18:18 kehilangan rekaman 20000
Kehilangan bertopeng sama terbaca 22058 dari 22058; cek determinisme 20 rekaman pertama: identik.
Selesai.
```

`jalankan_bersih_segar.py`:

```
Kesesuaian segmen model bersih langkah 19b; benih 20261120; rekaman 20214.
18:37 rekaman 10000
19:01 rekaman 20000
Selesai: terbaca 20214 dari 20214.
```

`uji_audit2.py --boot 2000`:

```
Langkah 19; benih 20261120.

| Populasi | Kelompok | Pembanding | Dibuang |
| --- | --- | --- | --- |
| segar (b, c, e): dilihat / tidak dilihat | 10107 | 10107 | 0 |
| topeng sama (d): disuntik / ditahan | 11029 | 11029 | 0 |
| langkah 16/18 (e): dilihat / tidak dilihat | 22058 | 22058 | 0 |

## Statistik (bootstrap 2000 ulangan atas pasien per populasi, benih 20261120)

| (b) Kesesuaian segmen segar, dilihat − tidak dilihat | Nilai [95 % CI] | Efek terdeteksi minimum |
| --- | --- | --- |
| SMD ECG-FM | 0.0876 [0.0603, 0.1154] |  |
| SMD ECG-JEPA | 0.0893 [0.0621, 0.1171] |  |
| SMD ECGFounder | -0.0430 [-0.0716, -0.0153] |  |
| SMD ECG-FM-acak | -0.0124 [-0.0403, 0.0150] |  |
| Δ_c = SMD ECG-FM − SMD ECG-JEPA | -0.0016 [-0.0331, 0.0311] | 0.0459 |
| Δ_c = SMD ECG-FM − SMD ECGFounder | 0.1306 [0.0994, 0.1624] | 0.0450 |
| Δ_c = SMD ECG-FM − SMD ECG-FM-acak | 0.1001 [0.0645, 0.1359] | 0.0510 |
| SMD ECG-FM, tidak dilihat = valid | 0.1089 [0.0724, 0.1465] |  |
| SMD ECG-FM, tidak dilihat = test | 0.0654 [0.0312, 0.1032] |  |

| (c) Inferensi dataset, log-likelihood dilihat − tidak dilihat | Nilai [95 % CI] | Efek terdeteksi minimum |
| --- | --- | --- |
| SMD ECG-FM | -0.1386 [-0.1661, -0.1105] |  |
| SMD ECG-JEPA | -0.1419 [-0.1706, -0.1147] |  |
| SMD ECGFounder | -0.1423 [-0.1692, -0.1153] |  |
| Δ_DI = SMD ECG-FM − SMD ECG-JEPA | 0.0033 [-0.0303, 0.0385] | 0.0491 |
| Δ_DI = SMD ECG-FM − SMD ECGFounder | 0.0037 [-0.0311, 0.0392] | 0.0502 |

| (d) Topeng sama, disuntik lawan ditahan | Nilai [95 % CI] | Efek terdeteksi minimum |
| --- | --- | --- |
| Dp = rerata Δℓ ditahan − disuntik | 0.0017 [0.0006, 0.0028] | 0.0016 |
| AUROC keanggotaan dari −Δℓ | 0.5105 [0.5033, 0.5181] |  |

## Deskriptif 19e

| Daya pisah per rekaman (segar, kosinus) | AUROC | TPR pada FPR 1% |
| --- | --- | --- |
| ecgfm | 0.5210 | 0.0131 |
| ecgjepa | 0.5312 | 0.0143 |
| ecgfounder | 0.4908 | 0.0103 |
| ecgfm_acak | 0.4988 | 0.0107 |

| Daya pisah per rekaman (segar, log-likelihood) | AUROC | TPR pada FPR 1% |
| --- | --- | --- |
| di ecgfm | 0.4591 | 0.0070 |
| di ecgjepa | 0.4572 | 0.0081 |
| di ecgfounder | 0.4586 | 0.0096 |

| Daya pisah per rekaman (langkah 16/18, kosinus) | AUROC | TPR pada FPR 1% |
| --- | --- | --- |
| ecgfm | 0.5233 | 0.0126 |
| ecgjepa | 0.5014 | 0.0118 |
| ecgfounder | 0.5049 | 0.0107 |

| Kovariat segar (dilihat − tidak dilihat) | SMD | Hilang |
| --- | --- | --- |
| age | 0.212 | 0 |
| laki | 0.017 | 0 |
| heart_rate_bpm | 0.017 | 0 |
| qrs_duration_ms | 0.111 | 0 |

## Keputusan praregistrasi

- M3 (batas bawah > 0 untuk semua rujukan): TIDAK LOLOS (ECG-JEPA -0.0331, ECGFounder 0.0994, ECG-FM-acak 0.0645); tidak terbukti: dukungan daftar direplikasi pada data segar melawan plasebo sepadan masukan.
- D1 (batas bawah > 0 untuk semua rujukan): TIDAK LOLOS (ECG-JEPA -0.0303, ECGFounder -0.0311); tidak terbukti: inferensi dataset mendukung daftar.
- K3 (batas bawah Dp > 0): LOLOS (batas bawah 0.0006); kehilangan mencatat suntikan bila derau topeng dihapus.
```
