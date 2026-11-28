# Hasil Glukosa 23 — tugas lepas plafon (protokol §F langkah 21, KONFIRMATORI), 28 November 2026

Sumber angka: keluaran `glukosa_lepas_plafon/jalankan_lepas_plafon.py` (venv-gpu, CPU; bootstrap 2000, benih
lipatan dan bootstrap 20261120) pada kode komit 5fd2dcc tanpa perubahan. Protokol langkah 21 dan kodenya dikomit
bersama (5fd2dcc) dan dirilis publik sebagai versi 1.5.0 (doi 10.5281/zenodo.24599381) sebelum model mana pun dinilai
pada tugas langkah ini. Label tugas dibuat lebih dulu oleh `siapkan_lepas_plafon.py` (hitungan kesetiaan pola,
tercatat di protokol 21a). Skrip membaca embedding tersimpan langkah 15 di folder replikasi, di luar repo dan tidak
dibuka di sesi asisten; yang keluar hanya agregat. Bagian "Keluaran skrip" disambung langsung dari log, kecuali 18 baris
kemajuan per tugas dan model ("… selesai") yang dibuang; jangan diketik ulang.

**Status: KONFIRMATORI** (G21, H21). Semua baris lain deskriptif tanpa koreksi kelipatan.

## 0. Catatan pembacaan

- **Keluarga lepas plafon: infark (AUROC ECG-JEPA 0,856) dan mutu data buruk (0,819).** PVC (0,974), flutter atrium
  (0,953), pacu (0,964) dan irama sinus (0,974) di atas 0,95, jadi di luar keluarga.
- **G21 LOLOS.** Rerata DiD ECG-FM fine-tuned atas keluarga 0,0257 [0,0194; 0,0319], kira-kira sepuluh kali efek
  label di empat tugas berplafon (rerata 0,002 [0,001; 0,003], Hasil 17). Kontrol positif bekerja jelas di tugas ini.
- **H21: di bawah margin.** Rerata DiD ECG-FM pretrained −0,0001 [−0,0077; 0,0078]; batas atas 0,0078 < 0,01. Pra-latih
  pada pasien yang sama tidak menaikkan AUROC secara terukur di tugas yang masih punya ruang, sedangkan fine-tuning
  label pada rekaman yang sama menaikkannya 0,026. Per tugas: infark 0,0011 [−0,0038; 0,0061], mutu −0,0013 [−0,0159;
  0,0143].
- **Kemurnian daftar.** Bila daftar hanya sebagian benar, DiD teramati = efek × selisih laju keanggotaan sejati antara
  kelompok berlabel. Batas atas 0,0078 melewati 0,01 bila selisih itu < 0,776; untuk ambang 0,005 batas atas sudah di
  atasnya pada daftar tepat (selisih 1), jadi batas lepas plafon terbaik yang dapat dinyatakan adalah 0,008 dengan daftar
  tepat dan 0,01 dengan selisih ≥ 0,78.
- **Sekunder (deskriptif).** Tidak dilihat = `test` saja: rerata keluarga 0,0020 [−0,0087; 0,0127] tak terputuskan;
  kontrol positif 0,0237 [0,0158; 0,0317]. Pasien di luar kohort (42.585 lawan 10.828 rekaman): rerata keluarga 0,0206
  [−0,0040; 0,0453] tak terputuskan, kontrol positif 0,0395 [0,0203; 0,0575]; selang lebar, titiknya positif (mutu
  0,035 [−0,010; 0,079]); irama sinus di subset ini 0,0047 [0,0008; 0,0090] (salah satu dari puluhan sel tanpa
  koreksi, tugas berplafon). Di luar keluarga, PVC tak terputuskan 0,0054 [−0,0007; 0,0116]; flutter, pacu dan sinus
  di bawah margin.
- **Akibat menurut 21(h) (G21 lolos, H21 di bawah margin):** klaim "tidak ada inflasi terukur" meluas ke tugas lepas
  plafon dengan kontrol positif yang bekerja; batas lepas plafon (0,008, selisih DiD dengan kontrol positif 0,026) masuk
  abstrak; batas dinyatakan sebagai fungsi kemurnian daftar. Aturan berhenti 21(g) berlaku: tidak ada analisis
  keanggotaan baru. Subset pasien di luar kohort dilaporkan apa adanya (titik positif, tak terputuskan), dengan standar
  yang sama seperti M3\*: deskriptif.

## 1. Catatan proses

- 28 Nov: penulis memilih arah 1 sesudah telaah internal 13. Hitungan kesetiaan pola dijalankan (agregat, tanpa model); protokol
  21 dan kode ditulis dan dikomit (5fd2dcc); rilis 1.5.0 atas pilihan penulis; lalu dijalankan sekali.
- Keluar dengan kode 0; stderr kosong. 18 probe baru (enam tugas × tiga model) dilatih; simpanan prediksi di folder
  replikasi.

## Keluaran skrip

Unit 416503; dibuang karena embedding tak terhingga 0.

Langkah 21; bootstrap 2000, benih lipatan 20261120.

## Seleksi lepas plafon (AUROC ECG-JEPA di luar lipatan, semua unit)

| Tugas | Positif | AUROC ECG-JEPA | Masuk keluarga utama (≤ 0.95) |
| --- | --- | --- | --- |
| infark | 93579 | 0.8559 | ya |
| pvc | 27483 | 0.9740 | tidak |
| flutter | 7359 | 0.9526 | tidak |
| pacu | 17370 | 0.9636 | tidak |
| mutu | 9499 | 0.8185 | ya |
| sinus | 336003 | 0.9738 | tidak |

## Utama (tidak dilihat = valid ∪ test): 333118 dilihat, 83385 tidak dilihat; ulangan terpakai 2000

| Tugas | Skala | DiD ECG-FM pretrained | DiD ECG-FM fine-tuned (kontrol positif) |
| --- | --- | --- | --- |
| infark | AUROC | 0.0011 [-0.0038, 0.0061] di bawah skala peringkat | 0.0067 [0.0015, 0.0119] inflasi didukung |
| infark | logit | 0.0088 [-0.0307, 0.0493] | 0.1060 [0.0556, 0.1546] |
| pvc | AUROC | 0.0054 [-0.0007, 0.0116] tak terputuskan | 0.0037 [0.0017, 0.0055] inflasi didukung |
| pvc | logit | 0.0508 [-0.0421, 0.1451] | 0.2345 [0.1461, 0.3184] |
| flutter | AUROC | -0.0022 [-0.0065, 0.0020] di bawah skala peringkat | 0.0195 [0.0138, 0.0252] inflasi didukung |
| flutter | logit | -0.0553 [-0.1512, 0.0444] | 1.1440 [0.9792, 1.3065] |
| pacu | AUROC | -0.0043 [-0.0106, 0.0012] di bawah skala peringkat | -0.0039 [-0.0108, 0.0025] di bawah skala peringkat |
| pacu | logit | -0.0544 [-0.2366, 0.1066] | 0.4038 [0.1505, 0.6483] |
| mutu | AUROC | -0.0013 [-0.0159, 0.0143] tak terputuskan | 0.0447 [0.0336, 0.0558] inflasi didukung |
| mutu | logit | -0.0044 [-0.0904, 0.0856] | 0.4538 [0.3689, 0.5395] |
| sinus | AUROC | -0.0004 [-0.0025, 0.0015] di bawah skala peringkat | 0.0008 [-0.0013, 0.0028] di bawah skala peringkat |
| sinus | logit | -0.0056 [-0.0877, 0.0682] | 0.1852 [0.0977, 0.2688] |
| Rerata keluarga utama | AUROC | -0.0001 [-0.0077, 0.0078] di bawah skala peringkat | 0.0257 [0.0194, 0.0319] inflasi didukung |
| Rerata keluarga utama | logit | 0.0022 [-0.0453, 0.0503] | 0.2799 [0.2289, 0.3286] |

## TEST saja: 333118 dilihat, 41477 tidak dilihat; ulangan terpakai 2000

| Tugas | Skala | DiD ECG-FM pretrained | DiD ECG-FM fine-tuned (kontrol positif) |
| --- | --- | --- | --- |
| infark | AUROC | -0.0017 [-0.0095, 0.0050] di bawah skala peringkat | 0.0014 [-0.0064, 0.0087] di bawah skala peringkat |
| infark | logit | -0.0136 [-0.0747, 0.0391] | 0.0771 [0.0074, 0.1434] |
| pvc | AUROC | 0.0055 [-0.0019, 0.0137] tak terputuskan | 0.0038 [0.0014, 0.0061] inflasi didukung |
| pvc | logit | 0.0935 [-0.0245, 0.2155] | 0.2150 [0.1072, 0.3222] |
| flutter | AUROC | -0.0026 [-0.0092, 0.0038] di bawah skala peringkat | 0.0218 [0.0135, 0.0296] inflasi didukung |
| flutter | logit | -0.0539 [-0.1921, 0.0858] | 1.2649 [1.0431, 1.4748] |
| pacu | AUROC | -0.0063 [-0.0167, 0.0021] di bawah skala peringkat | -0.0042 [-0.0161, 0.0055] di bawah skala peringkat |
| pacu | logit | -0.1700 [-0.4389, 0.0799] | 0.2125 [-0.1672, 0.5619] |
| mutu | AUROC | 0.0057 [-0.0138, 0.0263] tak terputuskan | 0.0461 [0.0312, 0.0607] inflasi didukung |
| mutu | logit | 0.0319 [-0.0822, 0.1483] | 0.4672 [0.3539, 0.5793] |
| sinus | AUROC | -0.0005 [-0.0032, 0.0019] di bawah skala peringkat | 0.0009 [-0.0020, 0.0035] di bawah skala peringkat |
| sinus | logit | -0.0076 [-0.1112, 0.0880] | 0.1875 [0.0678, 0.3011] |
| Rerata keluarga utama | AUROC | 0.0020 [-0.0087, 0.0127] tak terputuskan | 0.0237 [0.0158, 0.0317] inflasi didukung |
| Rerata keluarga utama | logit | 0.0091 [-0.0559, 0.0749] | 0.2721 [0.2094, 0.3370] |

## Pasien di luar kohort: 42585 dilihat, 10828 tidak dilihat; ulangan terpakai 2000

| Tugas | Skala | DiD ECG-FM pretrained | DiD ECG-FM fine-tuned (kontrol positif) |
| --- | --- | --- | --- |
| infark | AUROC | 0.0063 [-0.0084, 0.0214] tak terputuskan | 0.0028 [-0.0110, 0.0156] tak terputuskan |
| infark | logit | 0.0458 [-0.0766, 0.1672] | 0.1761 [0.0315, 0.3095] |
| pvc | AUROC | 0.0091 [-0.0104, 0.0310] tak terputuskan | 0.0024 [-0.0060, 0.0101] tak terputuskan |
| pvc | logit | -0.0584 [-0.4181, 0.3719] | 0.4085 [-0.0190, 0.8341] |
| flutter | AUROC | -0.0016 [-0.0217, 0.0201] tak terputuskan | 0.0296 [0.0020, 0.0590] inflasi didukung |
| flutter | logit | 0.0526 [-0.6590, 0.6867] | 2.0073 [0.3911, 3.1542] |
| pacu | AUROC | -0.0005 [-0.0168, 0.0140] tak terputuskan | -0.0017 [-0.0289, 0.0201] tak terputuskan |
| pacu | logit | -0.0075 [-0.4057, 0.4349] | 0.2479 [-0.9559, 1.3914] |
| mutu | AUROC | 0.0348 [-0.0095, 0.0794] tak terputuskan | 0.0763 [0.0406, 0.1110] inflasi didukung |
| mutu | logit | 0.2484 [-0.0336, 0.5377] | 0.7971 [0.4728, 1.1102] |
| sinus | AUROC | 0.0047 [0.0008, 0.0090] inflasi didukung | 0.0074 [0.0031, 0.0115] inflasi didukung |
| sinus | logit | 0.2889 [0.0654, 0.5145] | 0.2895 [-0.0787, 0.6500] |
| Rerata keluarga utama | AUROC | 0.0206 [-0.0040, 0.0453] tak terputuskan | 0.0395 [0.0203, 0.0575] inflasi didukung |
| Rerata keluarga utama | logit | 0.1471 [-0.0169, 0.3035] | 0.4866 [0.3076, 0.6514] |

## Keputusan langkah 21

- Keluarga utama: infark, mutu.
- G21 (batas bawah rerata DiD fine-tuned > 0): LOLOS (0.0194).
- H21 rerata DiD pretrained: -0.000 [-0.008, 0.008], vonis di bawah skala peringkat.
- Batas atas melewati 0.01 bila selisih laju keanggotaan sejati antarkelompok < 0.776.
- Batas atas melewati 0.005 bila selisih laju keanggotaan sejati antarkelompok < 1.551.
