# Hasil Glukosa 22 — uji ulang M3 dan D1 pada unit segar dengan kelayakan sama di kedua kelompok (protokol §F langkah 20, PASCA-HASIL), 27 November 2026

Sumber angka: keluaran `glukosa_audit2/uji_kelayakan_sama.py` (venv-gpu, CPU; bootstrap 2000, benih
20261120) pada kode komit 4938570 tanpa perubahan. Protokol langkah 20 dan kodenya dikomit bersama (4938570) sebelum
skrip dijalankan, sesudah Hasil 21 dan telaah internal 12; rilis publik belum. Skrip hanya membaca simpanan langkah 19 dan
embedding kohort langkah 11 di `data/audit2` dan folder singgahan, di luar repo dan tidak dibuka di sesi
asisten; yang keluar hanya agregat. Bagian "Keluaran skrip" disambung langsung dari log; jangan diketik ulang.

**Status: PASCA-HASIL.** Aturan (gerbang tafsir, M3\*, D1\*, akibat untuk naskah) ditetapkan sebelum statistik langkah
ini dihitung, tetapi sesudah hasil M3 dan D1 langkah 19 diketahui dan sesudah telaah internal 12 menunjuk mekanisme seleksi. Bukan
konfirmatori; tidak dirilis publik sebelum dijalankan.

## 0. Catatan pembacaan

- **Mekanisme seleksi terkonfirmasi.** Di unit uji 19a, 5.400 dari 10.107 pasien dilihat (53 %) punya rekaman di daftar
  split di luar kohort, sedangkan pasien tidak dilihat 0 dari 10.107. Di antara pasien dilihat, yang tak layak lebih tua
  daripada yang layak (SMD umur 0,397) dan QRS-nya lebih lebar (SMD 0,191). Inilah sumber SMD umur 0,212 di Hasil 21.
- **Gerbang tafsir: sebanding.** Pada unit uji layak (4.707 dilihat, 10.107 tidak dilihat, 0 dibuang), SMD dilihat −
  tidak dilihat: umur 0,009, laki −0,023, laju jantung −0,000, QRS 0,004; keempatnya |SMD| < 0,1.
- **M3\* LOLOS.** SMD kesesuaian segmen: ECG-FM 0,081 [0,048; 0,113], ECG-JEPA 0,011 [−0,023; 0,044], ECGFounder 0,010
  [−0,025; 0,043], ECG-FM-acak −0,003 [−0,037; 0,030]. Δ_c terhadap ECG-JEPA 0,070 [0,030; 0,108] (efek terdeteksi
  minimum 0,056), terhadap ECGFounder 0,071 [0,030; 0,111] (0,058), terhadap ECG-FM-acak 0,084 [0,040; 0,126] (0,061).
  SMD ECG-FM dengan tidak dilihat = `valid` 0,102 [0,062; 0,142], = `test` 0,058 [0,019; 0,098]. ECG-JEPA yang di
  Hasil 21 menyamai ECG-FM (0,089) turun ke 0,011 setelah kelompok sebanding, sedangkan ECG-FM hampir tidak bergeser
  (0,088 → 0,081).
- **D1\* GAGAL.** SMD log-likelihood: ECG-FM −0,014 [−0,047; 0,020], ECG-JEPA 0,010 [−0,027; 0,043], ECGFounder −0,013
  [−0,047; 0,022]; Δ_DI −0,024 [−0,064; 0,020] dan −0,002 [−0,045; 0,041]. Pergeseran −0,14 yang sama di ketiga model
  (Hasil 21) hilang setelah kelompok sebanding, jadi itu efek seleksi; inferensi dataset tetap tidak mendeteksi
  keanggotaan.
- **Daya pisah per rekaman (deskriptif):** AUROC kosinus ECG-FM 0,517 (TPR pada FPR 1 % 0,015), ECG-JEPA 0,502,
  ECGFounder 0,505, ECG-FM-acak 0,504; log-likelihood 0,497–0,505. Jejak ini bermakna pada tingkat kelompok, bukan per
  rekaman.
- **Akibat menurut 20(f) (sebanding dan M3\* lolos):** jejak kesesuaian segmen khas ECG-FM muncul lagi pada pasien segar
  yang sebanding, melawan dua model bersih dan ECG-FM berbobot teracak (pasca-hasil, bukan konfirmatori). Naskah harus
  menyebut mekanisme seleksi 19a dan bahwa kegagalan M3/D1 langkah 19 membandingkan kelompok tak sebanding. Naskah
  boleh menyatakan bukti mendukung daftar dengan label pasca-hasil; judul baru perlu persetujuan penulis. D1\*
  dilaporkan di samping tanpa membatalkan M3\*.

## 1. Catatan proses

- 27 Nov malam: telaah internal 12 menunjuk asimetri seleksi; dicek di kode
  `glukosa_audit2/unit_segar.py` dan protokol langkah 16a dan 19a. Protokol langkah 20 dicatat, kode ditulis (fungsi
  kelayakan diuji pada tabel mainan), lalu keduanya dikomit (4938570) sebelum jalan.
- `statistik_audit2.muat_segar` kini juga meneruskan `baris_gelombang` ke populasi agar kelayakan bisa disejajarkan;
  statistik langkah 19 tidak memakai kunci itu, jadi Hasil 21 tidak berubah.
- Dijalankan sekali, keluar dengan kode 0. Stderr hanya berisi satu `ConvergenceWarning` scikit-learn dari
  GaussianMixture (inisialisasi terbaik belum konvergen pada salah satu model D1\*); tidak diulang dengan pengaturan
  lain.

## Keluaran skrip

Langkah 20 (pasca-hasil); benih 20261120.

| Unit segar 19a | Layak | Tak layak (punya rekaman luar kohort) |
| --- | --- | --- |
| uji, dilihat | 4707 | 5400 |
| uji, tidak dilihat | 10107 | 0 |
| setel, dilihat | 4659 | 5448 |

| Kovariat dilihat (tak layak − layak) | SMD | Hilang |
| --- | --- | --- |
| age | 0.397 | 0 |
| laki | 0.075 | 0 |
| heart_rate_bpm | 0.031 | 0 |
| qrs_duration_ms | 0.191 | 0 |

## (b) Keseimbangan unit uji layak: dilihat 4707, tidak dilihat 10107, dibuang (semua unit) 0

| Kovariat layak (dilihat − tidak dilihat) | SMD | Hilang |
| --- | --- | --- |
| age | 0.009 | 0 |
| laki | -0.023 | 0 |
| heart_rate_bpm | -0.000 | 0 |
| qrs_duration_ms | 0.004 | 0 |

Sebanding (|SMD| < 0.1 untuk keempat kovariat): YA

## (c, d) Statistik (bootstrap 2000 ulangan atas pasien, benih 20261120)

| M3*: kesesuaian segmen, dilihat − tidak dilihat | Nilai [95 % CI] | Efek terdeteksi minimum |
| --- | --- | --- |
| SMD ECG-FM | 0.0806 [0.0476, 0.1131] |  |
| SMD ECG-JEPA | 0.0109 [-0.0232, 0.0435] |  |
| SMD ECGFounder | 0.0101 [-0.0246, 0.0429] |  |
| SMD ECG-FM-acak | -0.0030 [-0.0369, 0.0300] |  |
| Δ_c = SMD ECG-FM − SMD ECG-JEPA | 0.0698 [0.0304, 0.1084] | 0.0557 |
| Δ_c = SMD ECG-FM − SMD ECGFounder | 0.0705 [0.0303, 0.1113] | 0.0579 |
| Δ_c = SMD ECG-FM − SMD ECG-FM-acak | 0.0837 [0.0404, 0.1263] | 0.0613 |
| SMD ECG-FM, tidak dilihat = valid | 0.1020 [0.0620, 0.1418] |  |
| SMD ECG-FM, tidak dilihat = test | 0.0584 [0.0190, 0.0976] |  |

| D1*: log-likelihood, dilihat − tidak dilihat | Nilai [95 % CI] | Efek terdeteksi minimum |
| --- | --- | --- |
| SMD ECG-FM | -0.0142 [-0.0471, 0.0195] |  |
| SMD ECG-JEPA | 0.0095 [-0.0267, 0.0434] |  |
| SMD ECGFounder | -0.0127 [-0.0472, 0.0224] |  |
| Δ_DI = SMD ECG-FM − SMD ECG-JEPA | -0.0238 [-0.0643, 0.0199] | 0.0602 |
| Δ_DI = SMD ECG-FM − SMD ECGFounder | -0.0016 [-0.0452, 0.0406] | 0.0613 |

## Deskriptif: daya pisah per rekaman, unit uji layak

| Skor | AUROC | TPR pada FPR 1% |
| --- | --- | --- |
| ecgfm | 0.5171 | 0.0147 |
| ecgjepa | 0.5024 | 0.0106 |
| ecgfounder | 0.5053 | 0.0110 |
| ecgfm_acak | 0.5038 | 0.0081 |
| di ecgfm | 0.4967 | 0.0087 |
| di ecgjepa | 0.5049 | 0.0089 |
| di ecgfounder | 0.5001 | 0.0123 |

## Keputusan langkah 20(f)

- Gerbang tafsir: sebanding.
- M3* (batas bawah > 0 untuk semua rujukan): LOLOS (ECG-JEPA 0.0304, ECGFounder 0.0303, ECG-FM-acak 0.0404).
- D1* (batas bawah > 0 untuk semua rujukan): TIDAK LOLOS (ECG-JEPA -0.0643, ECGFounder -0.0452).
