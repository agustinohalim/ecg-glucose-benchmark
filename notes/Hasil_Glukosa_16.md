# Hasil Glukosa 16 — rasio paparan swaawas terhadap efek label saja (protokol §F langkah 14), 23 November 2026

Sumber angka: keluaran `glukosa_perluasan/jalankan_rasio_label.py --boot 2000` (komit 2ab11c8) atas
prediksi luar lipatan tersimpan langkah 11 (`data/mimic-iv-ext-ecg-glucose/prediksi/perluasan_*.npz`, di luar repo;
tanpa melatih ulang). Dijalankan sekali di laptop 23 Nov 2026, venv overlay ECG-FM, keluar tanpa galat (exit 0); yang
keluar hanya agregat. Bagian di bawah "Keluaran skrip" disambung langsung dari log; jangan diketik ulang. Varian utama
memakai ulangan dan benih yang sama dengan langkah 13, jadi DiD utama dan Δ sama dengan Hasil 15.

**Status: PASCA-HASIL dan deskriptif.** Protokol dicatat di rencana §F langkah 14 (komit 506765c) sesudah telaah internal 6,
sebelum skrip ditulis. Tidak mengubah vonis langkah 11.

## 0. Catatan pembacaan

- **Jawaban atas temuan kritis telaah internal 6 (methods M1).** Karena ECG-FM *fine-tuned* berangkat dari encoder pretrained
  yang sama, efek label saja = Δ (bila kedua efek menjumlah), dan rasio yang benar Q = DiD utama / Δ.
- **Rerata empat tugas berlabel, varian utama:** Q = 0,31 [−0,09; 0,91] AUROC, 0,10 [−0,05; 0,26] logit, 0,06
  [−0,14; 0,29] log-loss. Batas bawah Δ − DiD utama: 0,000 AUROC (positif, tipis), 0,251 logit, 0,003 log-loss.
  Menurut aturan langkah 14 paparan swaawas **lebih kecil dari efek label pada ketiga skala**, tetapi di AUROC hanya
  tipis.
- **"Kurang dari separuh" (Hasil 15) gugur:** batas atas Q di AUROC 0,91, bukan 0,48. Klaim itu dibangun atas
  penyebut yang memuat paparan swaawas juga.
- **Tidak kokoh di AUROC.** Rerata lima pernyataan (praspesifikasi langkah 11): AUROC tak terbedakan (batas bawah
  −0,001); logit dan log-loss lebih kecil. Per tugas: keempat tugas berlabel tak terbedakan di AUROC; lebih kecil di
  logit dan log-loss untuk keempatnya. AV blok 1 di AUROC: Q tak stabil (76 ulangan Δ ≤ 0).
- **Kepekaan TEST saja:** rerata empat tugas AUROC tak terbedakan (Q 0,68 [−0,03; 2,77]; batas bawah −0,001); logit
  (Q 0,16 [−0,03; 0,38]) dan log-loss (0,10 [−0,17; 0,43]) tetap lebih kecil; rerata lima pernyataan sama polanya.
  Per tugas TEST saja: AF dan LBBB lebih kecil di logit dan log-loss, RBBB hanya di logit, AV blok 1 tak terbedakan di
  semua skala; Δ AUROC RBBB tak terdeteksi (757 ulangan ≤ 0), sesuai Hasil 13 (kontrol positif RBBB gagal di TEST saja).
- **Batas mutlak (langkah 14 d):** rerata DiD utama empat tugas berlabel 0,001 [−0,000; 0,001] AUROC (varian utama),
  0,001 [−0,000; 0,002] (TEST saja); rerata lima pernyataan 0,001 [−0,000; 0,002].
- **Bacaan untuk naskah:** di AUROC, paparan swaawas dan efek label saja sama-sama ≤ sekitar 0,002–0,003 dan tidak
  dapat dipisahkan dengan andal (hanya satu dari tiga pengelompokan lolos, dan tipis). Di skala yang tidak dimampatkan
  plafon (logit, log-loss) paparan swaawas lebih kecil dari efek label pada rerata empat dan lima tugas, kokoh terhadap
  TEST saja. Klaim yang tahan: batas mutlak + "lebih kecil di logit dan log-loss"; klaim rasio di AUROC tidak.
- Rencana §A ("kurang dari separuh") terbantah; sumbangan baru menunggu keputusan penulis.

## Keluaran skrip

Rasio terhadap efek label saja, langkah 14 (pasca-hasil); bootstrap 2000, benih 20261120.


## Varian utama (tidak dilihat = valid ∪ test): ulangan terpakai 2000

| Kelompok | Skala | DiD utama | Δ (label saja) | Q = DiD utama / Δ | Ulangan Δ ≤ 0 | Δ − DiD utama | Tafsir |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fibrilasi atrium | AUROC | 0.000 [-0.001, 0.001] | 0.002 [0.001, 0.003] | 0.032 [-0.430, 1.132] | 3 | 0.002 [-0.000, 0.003] | tak terbedakan |
| Fibrilasi atrium | logit | 0.002 [-0.068, 0.070] | 0.304 [0.222, 0.386] | 0.008 [-0.198, 0.261] | 0 | 0.302 [0.182, 0.430] | swaawas lebih kecil dari label |
| Fibrilasi atrium | log-loss | 0.000 [-0.001, 0.002] | 0.007 [0.005, 0.009] | 0.060 [-0.192, 0.418] | 0 | 0.006 [0.003, 0.010] | swaawas lebih kecil dari label |
| LBBB | AUROC | 0.001 [-0.001, 0.002] | 0.003 [0.002, 0.005] | 0.164 [-0.307, 1.018] | 0 | 0.003 [-0.000, 0.006] | tak terbedakan |
| LBBB | logit | 0.040 [-0.091, 0.173] | 0.518 [0.388, 0.647] | 0.077 [-0.156, 0.388] | 0 | 0.478 [0.260, 0.695] | swaawas lebih kecil dari label |
| LBBB | log-loss | 0.000 [-0.001, 0.002] | 0.005 [0.003, 0.006] | 0.098 [-0.213, 0.516] | 0 | 0.004 [0.002, 0.007] | swaawas lebih kecil dari label |
| RBBB | AUROC | 0.000 [-0.001, 0.001] | 0.002 [0.000, 0.003] | 0.130 [-0.540, 2.730] | 28 | 0.001 [-0.001, 0.004] | tak terbedakan |
| RBBB | logit | 0.017 [-0.094, 0.118] | 0.352 [0.244, 0.461] | 0.047 [-0.236, 0.426] | 0 | 0.335 [0.144, 0.517] | swaawas lebih kecil dari label |
| RBBB | log-loss | 0.000 [-0.002, 0.003] | 0.006 [0.004, 0.009] | 0.071 [-0.254, 0.627] | 0 | 0.006 [0.002, 0.010] | swaawas lebih kecil dari label |
| AV blok derajat 1 | AUROC | 0.002 [-0.000, 0.004] | 0.001 [-0.000, 0.003] | 1.192 [-6.972, 10.517] | 76 | -0.000 [-0.003, 0.003] | tak terbedakan |
| AV blok derajat 1 | logit | 0.086 [-0.028, 0.207] | 0.353 [0.208, 0.483] | 0.244 [-0.076, 0.728] | 0 | 0.267 [0.062, 0.463] | swaawas lebih kecil dari label |
| AV blok derajat 1 | log-loss | 0.000 [-0.003, 0.003] | 0.005 [0.003, 0.006] | 0.015 [-0.758, 0.795] | 0 | 0.004 [0.001, 0.008] | swaawas lebih kecil dari label |
| Rerata empat tugas berlabel | AUROC | 0.001 [-0.000, 0.001] | 0.002 [0.001, 0.003] | 0.312 [-0.087, 0.912] | 0 | 0.001 [0.000, 0.003] | swaawas lebih kecil dari label |
| Rerata empat tugas berlabel | logit | 0.036 [-0.018, 0.089] | 0.382 [0.322, 0.440] | 0.095 [-0.046, 0.257] | 0 | 0.346 [0.251, 0.439] | swaawas lebih kecil dari label |
| Rerata empat tugas berlabel | log-loss | 0.000 [-0.001, 0.002] | 0.006 [0.004, 0.007] | 0.062 [-0.137, 0.294] | 0 | 0.005 [0.003, 0.007] | swaawas lebih kecil dari label |
| Rerata lima pernyataan | AUROC | 0.001 [-0.000, 0.002] | 0.002 [0.001, 0.003] | 0.380 [-0.201, 1.656] | 0 | 0.001 [-0.001, 0.003] | tak terbedakan |
| Rerata lima pernyataan | logit | 0.037 [-0.007, 0.083] | 0.306 [0.257, 0.352] | 0.122 [-0.021, 0.287] | 0 | 0.269 [0.191, 0.344] | swaawas lebih kecil dari label |
| Rerata lima pernyataan | log-loss | 0.001 [-0.000, 0.002] | 0.004 [0.003, 0.005] | 0.214 [-0.072, 0.563] | 0 | 0.003 [0.002, 0.005] | swaawas lebih kecil dari label |

## Varian TEST saja: ulangan terpakai 2000

| Kelompok | Skala | DiD utama | Δ (label saja) | Q = DiD utama / Δ | Ulangan Δ ≤ 0 | Δ − DiD utama | Tafsir |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fibrilasi atrium | AUROC | -0.000 [-0.001, 0.001] | 0.002 [0.001, 0.004] | -0.053 [-0.549, 1.285] | 5 | 0.002 [-0.000, 0.005] | tak terbedakan |
| Fibrilasi atrium | logit | -0.010 [-0.105, 0.082] | 0.361 [0.250, 0.477] | -0.027 [-0.257, 0.263] | 0 | 0.371 [0.203, 0.543] | swaawas lebih kecil dari label |
| Fibrilasi atrium | log-loss | 0.000 [-0.002, 0.003] | 0.008 [0.005, 0.011] | 0.038 [-0.260, 0.445] | 0 | 0.008 [0.003, 0.012] | swaawas lebih kecil dari label |
| LBBB | AUROC | 0.001 [-0.001, 0.003] | 0.003 [0.001, 0.005] | 0.452 [-0.448, 3.539] | 18 | 0.001 [-0.002, 0.005] | tak terbedakan |
| LBBB | logit | 0.087 [-0.071, 0.238] | 0.458 [0.302, 0.610] | 0.189 [-0.148, 0.635] | 0 | 0.371 [0.129, 0.617] | swaawas lebih kecil dari label |
| LBBB | log-loss | 0.001 [-0.001, 0.003] | 0.004 [0.002, 0.006] | 0.173 [-0.246, 0.949] | 0 | 0.004 [0.000, 0.007] | swaawas lebih kecil dari label |
| RBBB | AUROC | 0.001 [-0.001, 0.003] | 0.000 [-0.002, 0.002] | 3.169 [-16.695, 22.672] | 757 | -0.001 [-0.004, 0.002] | tak terbedakan |
| RBBB | logit | 0.040 [-0.088, 0.176] | 0.289 [0.157, 0.418] | 0.139 [-0.250, 0.940] | 0 | 0.249 [0.010, 0.475] | swaawas lebih kecil dari label |
| RBBB | log-loss | 0.000 [-0.003, 0.003] | 0.005 [0.002, 0.009] | 0.053 [-0.435, 1.124] | 0 | 0.005 [-0.000, 0.010] | tak terbedakan |
| AV blok derajat 1 | AUROC | 0.002 [-0.001, 0.005] | 0.001 [-0.001, 0.003] | 1.803 [-11.856, 21.361] | 196 | -0.001 [-0.005, 0.003] | tak terbedakan |
| AV blok derajat 1 | logit | 0.118 [-0.045, 0.265] | 0.357 [0.172, 0.533] | 0.331 [-0.128, 1.009] | 0 | 0.239 [-0.002, 0.485] | tak terbedakan |
| AV blok derajat 1 | log-loss | 0.001 [-0.003, 0.005] | 0.005 [0.003, 0.008] | 0.167 [-0.679, 1.115] | 0 | 0.004 [-0.000, 0.009] | tak terbedakan |
| Rerata empat tugas berlabel | AUROC | 0.001 [-0.000, 0.002] | 0.002 [0.001, 0.003] | 0.675 [-0.030, 2.770] | 5 | 0.001 [-0.001, 0.002] | tak terbedakan |
| Rerata empat tugas berlabel | logit | 0.059 [-0.013, 0.132] | 0.366 [0.290, 0.439] | 0.161 [-0.033, 0.383] | 0 | 0.307 [0.190, 0.422] | swaawas lebih kecil dari label |
| Rerata empat tugas berlabel | log-loss | 0.001 [-0.001, 0.002] | 0.006 [0.004, 0.007] | 0.096 [-0.174, 0.427] | 0 | 0.005 [0.003, 0.008] | swaawas lebih kecil dari label |
| Rerata lima pernyataan | AUROC | 0.001 [-0.001, 0.002] | 0.002 [0.000, 0.003] | 0.421 [-0.499, 3.090] | 21 | 0.001 [-0.001, 0.003] | tak terbedakan |
| Rerata lima pernyataan | logit | 0.049 [-0.010, 0.108] | 0.294 [0.231, 0.353] | 0.167 [-0.033, 0.400] | 0 | 0.245 [0.146, 0.340] | swaawas lebih kecil dari label |
| Rerata lima pernyataan | log-loss | 0.001 [-0.000, 0.003] | 0.004 [0.003, 0.006] | 0.335 [-0.078, 0.897] | 0 | 0.003 [0.000, 0.005] | swaawas lebih kecil dari label |
