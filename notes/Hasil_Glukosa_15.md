# Hasil Glukosa 15 — kontras langsung paparan swaawas lawan kontrol positif (protokol §F langkah 13), 23 November 2026

Sumber angka: keluaran `glukosa_perluasan/jalankan_kontras_langsung.py --boot 2000` (komit 5a71616) atas
prediksi luar lipatan tersimpan langkah 11 (`data/mimic-iv-ext-ecg-glucose/prediksi/perluasan_*.npz`, di luar repo;
tanpa melatih ulang, C sama dengan Hasil 13). Dijalankan sekali di laptop 23 Nov 2026, venv overlay ECG-FM, keluar
tanpa galat; yang keluar hanya agregat. Bagian di bawah "Keluaran skrip" disambung langsung dari log; jangan diketik
ulang. Ulangan bootstrap memakai benih dan urutan yang sama dengan langkah 11, jadi DiD AUROC dan logit sama dengan
Hasil 13.

**Status: PASCA-HASIL dan deskriptif.** Protokol dicatat di rencana §F langkah 13 (komit 18a1f3c) sesudah telaah internal 4,
sebelum skrip ditulis. Tidak mengubah vonis langkah 11.

## 0. Catatan pembacaan

- **Jawaban atas temuan kritis telaah internal 4.** Rerata empat tugas berlabel (AF, LBBB, RBBB, AV blok 1): Δ = DiD kontrol
  positif − DiD utama 0,002 [0,001; 0,003] pada AUROC, 0,382 [0,322; 0,440] pada logit, 0,006 [0,004; 0,007] pada
  log-loss. Batas bawah > 0 pada ketiga skala: menurut aturan langkah 13, **paparan swaawas lebih kecil dari paparan
  label**.
- **Rasio rerata** (DiD utama / DiD kontrol positif): 0,24 [−0,10; 0,48] AUROC, 0,09 [−0,05; 0,21] logit, 0,06
  [−0,16; 0,23] log-loss. Dibaca: paparan swaawas paling besar sekitar seperlima efek paparan label pada skala logit
  dan log-loss, kurang dari separuhnya pada skala AUROC.
- **Per tugas.** AF, LBBB, RBBB: "lebih kecil" pada ketiga skala. AV blok 1: lebih kecil pada logit (Δ 0,353 [0,208;
  0,483]) dan log-loss (0,005 [0,003; 0,006]), tak terbedakan pada AUROC (0,001 [−0,000; 0,003]); rasio AUROC-nya
  tidak stabil (8 ulangan penyebut ≤ 0), rasio log-loss juga (14). Batas atas rasio per tugas pada logit 0,21–0,42.
- **LVH dan glukosa: tak terbedakan pada semua skala**, sesuai dugaan karena kontrol positif tak terdeteksi di sana
  (LVH bukan label *fine-tuning*; glukosa bukan label). Tanpa tafsir rasio, sesuai protokol.
- **Log-loss hiperglikemia berat:** DiD utama −0,002 [−0,003; −0,000], ECG-FM sedikit lebih baik pada rekaman tidak
  dilihat; arahnya sama dengan DiD AUROC (−0,007) dan kepekaan `test` saja di Hasil 13. Tidak menunjukkan inflasi.
- Log-loss tugas hipoglikemia dicetak tiga desimal sehingga semua nilai tampak 0,000; prevalensi 1,9 % membuat
  log-loss absolut kecil. Bukan galat.

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

Kontras langsung langkah 13 (pasca-hasil); bootstrap 2000, benih 20261120.

- hipo: ulangan terpakai 2000
- hiper_berat: ulangan terpakai 2000
- af, lbbb, rbbb, avb1, lvh: ulangan terpakai 2000

### Selisih langsung Δ = DiD kontrol positif − DiD utama

| Tugas | Skala | DiD utama | DiD kontrol positif | Δ | Tafsir |
| --- | --- | --- | --- | --- | --- |
| Hipoglikemia (< 70) lawan euglikemia (70–180) | AUROC | 0.008 [-0.008, 0.023] | 0.011 [-0.006, 0.029] | 0.003 [-0.015, 0.022] | tak terbedakan |
| Hipoglikemia (< 70) lawan euglikemia (70–180) | logit | 0.035 [-0.034, 0.100] | 0.048 [-0.027, 0.124] | 0.013 [-0.061, 0.092] | tak terbedakan |
| Hipoglikemia (< 70) lawan euglikemia (70–180) | log-loss | 0.000 [-0.000, 0.001] | 0.000 [-0.000, 0.001] | 0.000 [-0.000, 0.001] | tak terbedakan |
| Hiperglikemia berat (> 250) lawan euglikemia (70–180) | AUROC | -0.007 [-0.017, 0.002] | 0.000 [-0.011, 0.013] | 0.007 [-0.003, 0.018] | tak terbedakan |
| Hiperglikemia berat (> 250) lawan euglikemia (70–180) | logit | -0.037 [-0.086, 0.010] | -0.001 [-0.054, 0.061] | 0.036 [-0.013, 0.088] | tak terbedakan |
| Hiperglikemia berat (> 250) lawan euglikemia (70–180) | log-loss | -0.002 [-0.003, -0.000] | -0.000 [-0.002, 0.002] | 0.002 [-0.000, 0.003] | tak terbedakan |
| Fibrilasi atrium | AUROC | 0.000 [-0.001, 0.001] | 0.002 [0.001, 0.003] | 0.002 [0.001, 0.003] | swaawas lebih kecil |
| Fibrilasi atrium | logit | 0.002 [-0.068, 0.070] | 0.307 [0.227, 0.388] | 0.304 [0.222, 0.386] | swaawas lebih kecil |
| Fibrilasi atrium | log-loss | 0.000 [-0.001, 0.002] | 0.007 [0.005, 0.009] | 0.007 [0.005, 0.009] | swaawas lebih kecil |
| LBBB | AUROC | 0.001 [-0.001, 0.002] | 0.004 [0.002, 0.006] | 0.003 [0.002, 0.005] | swaawas lebih kecil |
| LBBB | logit | 0.040 [-0.091, 0.173] | 0.558 [0.423, 0.691] | 0.518 [0.388, 0.647] | swaawas lebih kecil |
| LBBB | log-loss | 0.000 [-0.001, 0.002] | 0.005 [0.003, 0.007] | 0.005 [0.003, 0.006] | swaawas lebih kecil |
| RBBB | AUROC | 0.000 [-0.001, 0.001] | 0.002 [0.001, 0.003] | 0.002 [0.000, 0.003] | swaawas lebih kecil |
| RBBB | logit | 0.017 [-0.094, 0.118] | 0.369 [0.268, 0.469] | 0.352 [0.244, 0.461] | swaawas lebih kecil |
| RBBB | log-loss | 0.000 [-0.002, 0.003] | 0.007 [0.004, 0.009] | 0.006 [0.004, 0.009] | swaawas lebih kecil |
| AV blok derajat 1 | AUROC | 0.002 [-0.000, 0.004] | 0.003 [0.001, 0.006] | 0.001 [-0.000, 0.003] | tak terbedakan |
| AV blok derajat 1 | logit | 0.086 [-0.028, 0.207] | 0.440 [0.281, 0.599] | 0.353 [0.208, 0.483] | swaawas lebih kecil |
| AV blok derajat 1 | log-loss | 0.000 [-0.003, 0.003] | 0.005 [0.001, 0.008] | 0.005 [0.003, 0.006] | swaawas lebih kecil |
| LVH | AUROC | 0.001 [-0.004, 0.005] | 0.002 [-0.002, 0.006] | 0.001 [-0.003, 0.005] | tak terbedakan |
| LVH | logit | 0.041 [-0.010, 0.091] | 0.042 [-0.008, 0.092] | 0.001 [-0.036, 0.038] | tak terbedakan |
| LVH | log-loss | 0.003 [-0.001, 0.007] | 0.003 [-0.000, 0.007] | -0.000 [-0.003, 0.003] | tak terbedakan |

### Rasio DiD utama / DiD kontrol positif (kontrol positif terdeteksi pada skala itu)

| Tugas | Skala | Rasio [95 % CI] | Ulangan penyebut ≤ 0 |
| --- | --- | --- | --- |
| Fibrilasi atrium | AUROC | 0.031 [-0.738, 0.533] | 0 |
| Fibrilasi atrium | logit | 0.008 [-0.247, 0.207] | 0 |
| Fibrilasi atrium | log-loss | 0.056 [-0.238, 0.295] | 0 |
| LBBB | AUROC | 0.141 [-0.444, 0.505] | 0 |
| LBBB | logit | 0.072 [-0.185, 0.280] | 0 |
| LBBB | log-loss | 0.089 [-0.270, 0.341] | 0 |
| RBBB | AUROC | 0.115 [-0.823, 0.816] | 1 |
| RBBB | logit | 0.045 [-0.310, 0.299] | 0 |
| RBBB | log-loss | 0.066 [-0.341, 0.386] | 0 |
| AV blok derajat 1 | AUROC | 0.544 [-0.189, 1.075] | 8 |
| AV blok derajat 1 | logit | 0.196 [-0.083, 0.421] | 0 |
| AV blok derajat 1 | log-loss | 0.014 [-2.474, 0.465] | 14 |

### Rerata empat tugas berlabel (af, lbbb, rbbb, avb1)

| Skala | Rerata DiD utama | Rerata DiD kontrol positif | Rerata Δ | Rasio rerata | Tafsir |
| --- | --- | --- | --- | --- | --- |
| AUROC | 0.001 [-0.000, 0.001] | 0.003 [0.002, 0.004] | 0.002 [0.001, 0.003] | 0.238 [-0.096, 0.477] | swaawas lebih kecil |
| logit | 0.036 [-0.018, 0.089] | 0.418 [0.354, 0.481] | 0.382 [0.322, 0.440] | 0.087 [-0.048, 0.205] | swaawas lebih kecil |
| log-loss | 0.000 [-0.001, 0.002] | 0.006 [0.005, 0.007] | 0.006 [0.004, 0.007] | 0.058 [-0.159, 0.227] | swaawas lebih kecil |
