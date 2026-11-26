# Hasil Glukosa 17 — replikasi konfirmatori pada rekaman di luar kohort (protokol §F langkah 15), 26 November 2026

Sumber angka: keluaran `glukosa_replikasi/siapkan_replikasi.py` dan
`glukosa_replikasi/jalankan_replikasi.py --folder data/replikasi` (bootstrap 2000, benih
20261120; kode tak berubah sejak komit 4241e8a, isinya sama dengan rilis publik v1.1.0, DOI 10.5281/zenodo.24452624).
Tabel unit, embedding, dan prediksi di luar lipatan ada di `data/replikasi/`, di luar repo; yang
keluar hanya agregat. Bagian "Keluaran skrip" disambung langsung dari log; jangan diketik ulang.

**Status: KONFIRMATORI untuk R1 dan R2** (aturan dan kode dipublikasikan di Zenodo sebelum label rekaman ini dihitung
dan sebelum embedding-nya dibangun). Semua bagian lain deskriptif sesuai langkah 15 (f), tanpa vonis tambahan.

## 0. Catatan pembacaan

- **R1 (gerbang kontrol positif) LOLOS.** Rerata DiD ECG-FM *fine-tuned* empat tugas berlabel 0,002 [0,001; 0,003]
  AUROC, batas bawah 0,0013 > 0. Replikasi informatif.
- **R2 (konfirmatori) LOLOS.** Rerata DiD ECG-FM pretrained empat tugas berlabel 0,0004 [−0,0004; 0,0012] AUROC; batas
  atas 0,0012 < 0,002. Inflasi oleh pra-latih tidak terdeteksi (batas bawah < 0). **Klaim "paling besar 0,002 AUROC
  rata-rata pada pernyataan berlabel" terkonfirmasi** pada 416.503 rekaman (109.971 pasien) yang tidak dipakai di
  analisis mana pun sebelumnya.
- **Sekunder, varian utama:** rerata lima pernyataan pretrained 0,001 [−0,001; 0,002]; per tugas kelima batas atas
  ≤ 0,007 (LVH paling lebar); logit rerata empat tugas pretrained 0,011 [−0,038; 0,058], *fine-tuned* 0,354 [0,302;
  0,407]. Protokol melarang rasio atau perbandingan besar kedua paparan, jadi keduanya hanya dilaporkan berdampingan.
- **TEST saja** sejalan: pretrained 0,001 [−0,000; 0,002], *fine-tuned* 0,002 [0,001; 0,003] (rerata empat tugas,
  AUROC). Berbeda dengan kohort (Hasil 13), kontrol positif RBBB di TEST saja di sini terdeteksi (0,002 [0,000; 0,003]).
- **Kepekaan benih lipatan:** ketiga benih (20261120–22) memberi angka identik sampai tiga desimal untuk kedua model.
- **Keseimbangan:** |SMD| ≤ 0,022 untuk prevalensi kelima pola, unit per pasien, denyut jantung, dan durasi QRS.
- **Keselarasan pola dengan label ECG-FM (`siapkan_replikasi.py`):** AF, LBBB, RBBB selisih ≤ 0,002 dari persentase
  label ECG-FM per split; AV blok 1 lebih luas 1,1–1,2 poin persen (pola menangkap lebih banyak dari label ECG-FM).
  Pola tidak diubah, sesuai (b).
- **Subset pasien di luar kohort (53.413 rekaman; 42.585 dilihat, 10.828 tidak dilihat) perlu dilaporkan, bukan
  disembunyikan.** Di subset ini kontrol positif gagal di AUROC (rerata empat tugas *fine-tuned* 0,000 [−0,002;
  0,002]), jadi subset ini tidak cukup kuat untuk tugas berlabel di skala AUROC; di logit *fine-tuned* 0,536 [0,239;
  0,793]. Untuk **LVH**, batas bawah kedua model > 0: pretrained 0,026 [0,005; 0,050], *fine-tuned* 0,019 [0,005;
  0,033]; akibatnya rerata lima pernyataan pretrained 0,006 [0,001; 0,012]. LVH bukan label *fine-tuning*, sehingga
  sinyal yang sama besar pada model *fine-tuned* tidak dapat berasal dari label; kemungkinannya komposisi pasien
  subset kecil ini atau kebetulan di antara banyak sel sekunder. Label "inflasi didukung" di tabel sekunder dicetak
  mekanis oleh fungsi `vonis`; menurut (f) itu bukan vonis. Naskah menyebutnya sebagai temuan sekunder deskriptif.
- **Bacaan untuk naskah:** klaim utama (batas mutlak 0,002 AUROC rata-rata pernyataan berlabel, dengan kontrol positif
  yang terdeteksi) sekarang berdiri atas data baru yang dipra-registrasi. Klaim logit/log-loss "lebih kecil" dari
  Hasil 16 tetap tidak dipakai (rencana: dibuang).

## 1. Catatan proses

- **Unit dan label:** `siapkan_replikasi.py` dijalankan sekali sesudah rilis Zenodo 1.1.0; 416.503 rekaman, 109.971
  pasien, sama dengan hitungan calon sampel di protokol.
- **Embedding** (`glukosa_m3/bangun_embedding.py --singgahan`, dibangun 23–25 Nov 2026 dalam beberapa sesi; skrip
  melanjutkan dari baris yang sudah selesai): ECG-JEPA (venv-gpu), ECG-FM pretrained dan *fine-tuned* (venv overlay
  ECG-FM, 6 pekerja pembaca; 12 pekerja kehabisan RAM *commit*). Ketiganya 416.503/416.503 baris terhingga; tidak ada
  unit yang dibuang. Segmen terakhir ECG-FM pretrained 134.903 baris dalam 26,7 menit; ECG-FM *fine-tuned* seluruhnya
  126,9 menit. Laju sempat turun ±2,5× saat layar laptop mati (Modern Standby membatasi daya GPU); tidak memengaruhi
  isi embedding.
- **Jalan pertama `jalankan_replikasi.py` (25 Nov 2026, mulai 00:38):** semua prediksi di luar lipatan selesai dan
  tersimpan (benih 20261120: lima tugas × tiga model, 00:38–00:55; benih 20261121 dan 20261122: empat tugas × tiga
  model, 01:32–02:09). Pukul 03:21 berkas keluaran tertimpa oleh satu proses baru dari perintah yang sama (asal mula
  proses kedua tidak diketahui; skrip tidak berulang sendiri), lalu proses itu dihentikan oleh pengawas RAM lingkungan kerja
  Code. Keluaran jalan pertama hilang **sebelum dibaca siapa pun**; tidak ada angka DiD, R1, atau R2 yang terlihat.
- **Jalan kedua (25 Nov 23:47 – 26 Nov 00:26, exit 0)** atas keputusan penulis, dengan pengawas RAM lingkungan kerja dimatikan
  (`CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1`). Prediksi diambil dari singgahan jalan pertama (tidak ada probe
  yang dilatih ulang); bootstrap memakai benih tetap, jadi angkanya sama dengan yang akan dicetak jalan pertama. Tidak
  ada perubahan kode, argumen, atau pilihan analisis di antara kedua jalan. Keluaran yang terputus disimpan di
  folder sementara sebagai `glukosa_17_jalankan_terputus.md` (hanya tabel keseimbangan dan baris kemajuan).

## Keluaran skrip

### siapkan_replikasi.py


Unit replikasi: 416503 rekaman, 109971 pasien; pasien di luar kohort 53413 rekaman, 29391 pasien.

| Split | Rekaman | Pasien | Rekaman pasien di luar kohort |
| --- | --- | --- | --- |
| train | 333118 | 87913 | 42585 |
| valid | 41908 | 10986 | 5324 |
| test | 41477 | 11072 | 5504 |

## Prevalensi pola dan label ECG-FM (label_def.csv, persen positif per split)

| Tugas | Split | Positif | Prevalensi pola | Label ECG-FM |
| --- | --- | --- | --- | --- |
| Fibrilasi atrium | train | 33903 | 0.1018 | 0.1010 |
| Fibrilasi atrium | valid | 4409 | 0.1052 | 0.1033 |
| Fibrilasi atrium | test | 4194 | 0.1011 | 0.1029 |
| LBBB | train | 12109 | 0.0364 | 0.0368 |
| LBBB | valid | 1645 | 0.0393 | 0.0394 |
| LBBB | test | 1578 | 0.0380 | 0.0384 |
| RBBB | train | 27663 | 0.0830 | 0.0820 |
| RBBB | valid | 3733 | 0.0891 | 0.0875 |
| RBBB | test | 3363 | 0.0811 | 0.0807 |
| AV blok derajat 1 | train | 27954 | 0.0839 | 0.0728 |
| AV blok derajat 1 | valid | 3811 | 0.0909 | 0.0788 |
| AV blok derajat 1 | test | 3626 | 0.0874 | 0.0760 |
| LVH | train | 28924 | 0.0868 | — |
| LVH | valid | 3525 | 0.0841 | — |
| LVH | test | 3827 | 0.0923 | — |

### jalankan_replikasi.py (jalan kedua)

Unit 416503; dibuang karena embedding tak terhingga 0.

Replikasi langkah 15; bootstrap 2000, benih bootstrap 20261120, benih lipatan 20261120.

## Keseimbangan kelompok

| Peubah | Dilihat | Tidak dilihat | SMD |
| --- | --- | --- | --- |
| positif Fibrilasi atrium | 0.102 | 0.103 | -0.005 |
| positif LBBB | 0.036 | 0.039 | -0.012 |
| positif RBBB | 0.083 | 0.085 | -0.007 |
| positif AV blok derajat 1 | 0.084 | 0.089 | -0.019 |
| positif LVH | 0.087 | 0.088 | -0.005 |
| unit per pasien | 11.994 | 11.989 | 0.000 |
| denyut jantung | 75.432 | 75.239 | 0.010 |
| durasi QRS | 99.899 | 100.415 | -0.022 |
23:47 benih 20261120 af ecgjepa: selesai
23:47 benih 20261120 lbbb ecgjepa: selesai
23:47 benih 20261120 rbbb ecgjepa: selesai
23:47 benih 20261120 avb1 ecgjepa: selesai
23:47 benih 20261120 lvh ecgjepa: selesai
23:47 benih 20261120 af ecgfm: selesai
23:47 benih 20261120 lbbb ecgfm: selesai
23:47 benih 20261120 rbbb ecgfm: selesai
23:47 benih 20261120 avb1 ecgfm: selesai
23:47 benih 20261120 lvh ecgfm: selesai
23:47 benih 20261120 af ecgfm_ft: selesai
23:47 benih 20261120 lbbb ecgfm_ft: selesai
23:47 benih 20261120 rbbb ecgfm_ft: selesai
23:47 benih 20261120 avb1 ecgfm_ft: selesai
23:47 benih 20261120 lvh ecgfm_ft: selesai

## Utama (tidak dilihat = valid ∪ test): 333118 dilihat, 83385 tidak dilihat; ulangan terpakai 2000

| Kelompok | Skala | DiD ECG-FM pretrained | DiD ECG-FM fine-tuned (kontrol positif) |
| --- | --- | --- | --- |
| Fibrilasi atrium | AUROC | -0.000 [-0.001, 0.001] di bawah skala peringkat | 0.002 [0.001, 0.002] inflasi didukung |
| Fibrilasi atrium | logit | -0.004 [-0.071, 0.058] | 0.298 [0.225, 0.372] |
| LBBB | AUROC | 0.000 [-0.001, 0.002] di bawah skala peringkat | 0.003 [0.001, 0.004] inflasi didukung |
| LBBB | logit | 0.024 [-0.110, 0.156] | 0.503 [0.387, 0.619] |
| RBBB | AUROC | 0.001 [-0.001, 0.002] di bawah skala peringkat | 0.001 [0.001, 0.002] inflasi didukung |
| RBBB | logit | 0.027 [-0.071, 0.130] | 0.335 [0.234, 0.425] |
| AV blok derajat 1 | AUROC | 0.001 [-0.001, 0.003] di bawah skala peringkat | 0.003 [0.000, 0.005] inflasi didukung |
| AV blok derajat 1 | logit | -0.003 [-0.102, 0.091] | 0.281 [0.151, 0.413] |
| LVH | AUROC | 0.002 [-0.004, 0.007] di bawah skala peringkat | 0.001 [-0.003, 0.005] di bawah skala peringkat |
| LVH | logit | 0.044 [-0.017, 0.102] | 0.032 [-0.024, 0.086] |
| Rerata empat tugas berlabel | AUROC | 0.000 [-0.000, 0.001] di bawah skala peringkat | 0.002 [0.001, 0.003] inflasi didukung |
| Rerata empat tugas berlabel | logit | 0.011 [-0.038, 0.058] | 0.354 [0.302, 0.407] |
| Rerata lima pernyataan | AUROC | 0.001 [-0.001, 0.002] di bawah skala peringkat | 0.002 [0.001, 0.003] inflasi didukung |
| Rerata lima pernyataan | logit | 0.017 [-0.024, 0.056] | 0.290 [0.247, 0.334] |

## Keputusan praregistrasi

- R1 (kontrol positif, batas bawah rerata DiD fine-tuned > 0): LOLOS (batas bawah 0.0013).
- R2 (batas atas rerata DiD pretrained < 0.002): LOLOS (0.0004 [-0.0004, 0.0012]).
- Inflasi oleh pra-latih terdeteksi (batas bawah > 0): tidak.
- Klaim terkonfirmasi: ya.

## TEST saja: 333118 dilihat, 41477 tidak dilihat; ulangan terpakai 2000

| Kelompok | Skala | DiD ECG-FM pretrained | DiD ECG-FM fine-tuned (kontrol positif) |
| --- | --- | --- | --- |
| Fibrilasi atrium | AUROC | -0.001 [-0.002, 0.000] di bawah skala peringkat | 0.001 [0.000, 0.002] inflasi didukung |
| Fibrilasi atrium | logit | -0.044 [-0.129, 0.040] | 0.256 [0.166, 0.342] |
| LBBB | AUROC | 0.001 [-0.001, 0.003] di bawah skala peringkat | 0.003 [0.001, 0.004] inflasi didukung |
| LBBB | logit | 0.066 [-0.088, 0.226] | 0.536 [0.391, 0.684] |
| RBBB | AUROC | 0.001 [-0.001, 0.003] di bawah skala peringkat | 0.002 [0.000, 0.003] inflasi didukung |
| RBBB | logit | 0.034 [-0.106, 0.177] | 0.422 [0.299, 0.544] |
| AV blok derajat 1 | AUROC | 0.001 [-0.002, 0.004] di bawah skala peringkat | 0.004 [0.001, 0.007] inflasi didukung |
| AV blok derajat 1 | logit | -0.022 [-0.148, 0.099] | 0.288 [0.090, 0.469] |
| LVH | AUROC | 0.002 [-0.004, 0.009] di bawah skala peringkat | 0.002 [-0.003, 0.008] di bawah skala peringkat |
| LVH | logit | 0.060 [-0.015, 0.138] | 0.058 [-0.011, 0.130] |
| Rerata empat tugas berlabel | AUROC | 0.001 [-0.000, 0.002] di bawah skala peringkat | 0.002 [0.001, 0.003] inflasi didukung |
| Rerata empat tugas berlabel | logit | 0.009 [-0.061, 0.075] | 0.375 [0.305, 0.445] |
| Rerata lima pernyataan | AUROC | 0.001 [-0.001, 0.002] di bawah skala peringkat | 0.002 [0.001, 0.004] inflasi didukung |
| Rerata lima pernyataan | logit | 0.019 [-0.040, 0.074] | 0.312 [0.251, 0.369] |

## Pasien di luar kohort: 42585 dilihat, 10828 tidak dilihat; ulangan terpakai 2000

| Kelompok | Skala | DiD ECG-FM pretrained | DiD ECG-FM fine-tuned (kontrol positif) |
| --- | --- | --- | --- |
| Fibrilasi atrium | AUROC | 0.001 [-0.003, 0.006] di bawah skala peringkat | 0.003 [-0.001, 0.007] di bawah skala peringkat |
| Fibrilasi atrium | logit | 0.137 [-0.413, 0.652] | 0.940 [0.173, 1.523] |
| LBBB | AUROC | 0.002 [-0.003, 0.008] di bawah skala peringkat | 0.001 [-0.001, 0.005] di bawah skala peringkat |
| LBBB | logit | 0.299 [-0.465, 0.934] | 0.564 [0.052, 1.155] |
| RBBB | AUROC | 0.004 [-0.001, 0.010] di bawah skala peringkat | -0.002 [-0.004, 0.001] di bawah skala peringkat |
| RBBB | logit | 0.009 [-0.297, 0.316] | 0.105 [-0.145, 0.358] |
| AV blok derajat 1 | AUROC | -0.003 [-0.011, 0.005] di bawah skala peringkat | -0.002 [-0.010, 0.004] di bawah skala peringkat |
| AV blok derajat 1 | logit | 0.018 [-0.512, 0.487] | 0.536 [-0.209, 1.115] |
| LVH | AUROC | 0.026 [0.005, 0.050] inflasi didukung | 0.019 [0.005, 0.033] inflasi didukung |
| LVH | logit | 0.217 [-0.016, 0.453] | 0.210 [0.024, 0.406] |
| Rerata empat tugas berlabel | AUROC | 0.001 [-0.002, 0.004] di bawah skala peringkat | 0.000 [-0.002, 0.002] di bawah skala peringkat |
| Rerata empat tugas berlabel | logit | 0.116 [-0.169, 0.354] | 0.536 [0.239, 0.793] |
| Rerata lima pernyataan | AUROC | 0.006 [0.001, 0.012] inflasi didukung | 0.004 [0.001, 0.007] inflasi didukung |
| Rerata lima pernyataan | logit | 0.136 [-0.109, 0.337] | 0.471 [0.229, 0.676] |
00:09 benih 20261121 af ecgjepa: selesai
00:09 benih 20261121 lbbb ecgjepa: selesai
00:09 benih 20261121 rbbb ecgjepa: selesai
00:09 benih 20261121 avb1 ecgjepa: selesai
00:09 benih 20261121 af ecgfm: selesai
00:09 benih 20261121 lbbb ecgfm: selesai
00:09 benih 20261121 rbbb ecgfm: selesai
00:09 benih 20261121 avb1 ecgfm: selesai
00:09 benih 20261121 af ecgfm_ft: selesai
00:09 benih 20261121 lbbb ecgfm_ft: selesai
00:09 benih 20261121 rbbb ecgfm_ft: selesai
00:09 benih 20261121 avb1 ecgfm_ft: selesai
00:18 benih 20261122 af ecgjepa: selesai
00:18 benih 20261122 lbbb ecgjepa: selesai
00:18 benih 20261122 rbbb ecgjepa: selesai
00:18 benih 20261122 avb1 ecgjepa: selesai
00:18 benih 20261122 af ecgfm: selesai
00:18 benih 20261122 lbbb ecgfm: selesai
00:18 benih 20261122 rbbb ecgfm: selesai
00:18 benih 20261122 avb1 ecgfm: selesai
00:18 benih 20261122 af ecgfm_ft: selesai
00:18 benih 20261122 lbbb ecgfm_ft: selesai
00:18 benih 20261122 rbbb ecgfm_ft: selesai
00:18 benih 20261122 avb1 ecgfm_ft: selesai

## Kepekaan benih lipatan (rerata empat tugas berlabel, AUROC)

| Benih lipatan | DiD ECG-FM pretrained | DiD ECG-FM fine-tuned |
| --- | --- | --- |
| 20261120 | 0.000 [-0.000, 0.001] | 0.002 [0.001, 0.003] |
| 20261121 | 0.000 [-0.000, 0.001] | 0.002 [0.001, 0.003] |
| 20261122 | 0.000 [-0.000, 0.001] | 0.002 [0.001, 0.003] |
