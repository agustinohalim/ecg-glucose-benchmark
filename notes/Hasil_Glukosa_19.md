# Hasil Glukosa 19 — kontrol positif latih lanjut swaawas (protokol §F langkah 17), 27 November 2026

Sumber angka: keluaran `glukosa_kontrol/jalankan_latih.py`, `jalankan_bacaan.py` (venv-ecgfm, GPU), dan
`uji_kontrol.py` (venv-gpu; bootstrap 2000, benih 20261120), semuanya pada kode komit 992a9cb tanpa perubahan.
Protokol dicatat di komit 708b8df sebelum kode ditulis; kode dikomit dan diuji asap pada data sintetis sebelum latih;
protokol dan kode dirilis publik sebagai versi 1.2.0 (DOI 10.5281/zenodo.24549186, 26 Nov 2026 04:22 UTC) sebelum latih
lanjut, embedding, atau skor apa pun. Bobot, embedding, skor, dan prediksi per rekaman ada di `data/kontrol`,
di luar repo dan tidak dibuka di sesi asisten; yang keluar hanya agregat. Bagian "Keluaran skrip" disambung langsung
dari log (baris kemajuan per 100 pembaruan dan per 10.000 rekaman serta peringatan pustaka dibuang); jangan diketik
ulang.

**Status: PRA-REGISTRASI PUBLIK untuk N0, K1, P1, P2.** Semua bacaan lain deskriptif (langkah 17e): semua bacaan
ECG-FM+1, E per tugas, E dengan ECG-JEPA sebagai rujukan, AUROC keanggotaan, kosinus segmen.

## 0. Catatan pembacaan

- **N0 LOLOS.** Celah ECG-JEPA antara paruh disuntik dan paruh ditahan: LVH −0,001 [−0,010; 0,007], rerata lima
  pernyataan −0,001 [−0,004; 0,001]; keduanya memuat nol, jadi pengacakan paruh tidak menimbulkan celah pada model yang
  tidak pernah dilatih ulang. K1, P1, P2 dibaca tanpa catatan ketidakseimbangan.
- **K1 GAGAL.** Dₛ(ECG-FM+4) = rerata kehilangan ditahan − disuntik = 0,0016 [−0,0083; 0,0113], batas bawah ≤ 0. ECG-FM+1
  0,0016 [−0,0082; 0,0112]; ECG-FM asli pada paruh yang sama 0,0009 [−0,0087; 0,0106]. AUROC keanggotaan ECG-FM+4 dari
  −kehilangan 0,4999 [0,4920; 0,5077]. Empat epoch latih swaawas pada pasien yang diketahui tidak membuat kehilangan
  per rekaman mereka lebih rendah secara terdeteksi. **Akibat menurut aturan:** uji kehilangan tidak peka bahkan pada
  keanggotaan yang diketahui dan baru, dan naskah menyatakan kegagalan M1 (Hasil 18) tidak informatif tentang daftar
  split. Batas 0,002 tetap bersyarat pada daftar, dan syarat itu kini tidak dapat diuji dengan uji kehilangan.
- **P1 GAGAL.** E(LVH) = celah ECG-FM+4 − celah ECG-FM asli = −0,004 [−0,008; −0,001]; batas bawah ≤ 0. Paparan swaawas
  dosis ini tidak menaikkan AUROC LVH pasien yang disuntik relatif terhadap yang ditahan. Arahnya negatif dan selangnya
  tidak memuat nol; ini tidak diramalkan protokol dan dibaca deskriptif (lihat di bawah). Pembanding yang ditetapkan,
  celah LVH luar kohort 0,030 (Hasil 18), tidak dihasilkan ulang oleh dosis ini.
- **P2 GAGAL.** E(rerata empat tugas berlabel) = 0,001 [−0,000; 0,001] (batas bawah −0,0001 di log). Menurut aturan,
  **batas atas 0,001 dilaporkan sebagai besar efek swaawas langsung di plafon** pada dosis ini.
- **Dosis.** Paparan ini 4 lintasan atas rekaman pasien disuntik (1.948 pembaruan, laju tetap 5e-5 dari checkpoint),
  sedangkan pra-latih asli ±240 epoch. Protokol sudah menyatakan bahwa kegagalan pada dosis ini tidak membatasi efek
  paparan asli; ia membatasi efek paparan baru yang singkat.
- **Kehilangan latih datar.** Rerata per langkah bertopeng tiap epoch 1,2423 / 1,2422 / 1,2440 / 1,2431 (suku
  kontrastif 1,1715 / 1,1712 / 1,1729 / 1,1723). Laju belajar tidak nol (5e-5, penjadwal `fixed` di konfigurasi
  checkpoint), jadi model memang diperbarui; separuh batch adalah putar-ulang dari `train` yang sudah konvergen. Datarnya
  kurva sejalan dengan K1: tidak ada penurunan kehilangan yang khas pada rekaman disuntik.

**Sekunder deskriptif (tanpa vonis):**

- **E(LVH) negatif.** Celah LVH ECG-FM asli antara paruh acak 0,008 [−0,005; 0,022], ECG-FM+1 0,007, ECG-FM+4 0,004;
  E ECG-FM+1 −0,002 [−0,004; 0,001]. Celah awal yang (tidak bermakna) positif menyusut setelah latih lanjut. E dengan
  ECG-JEPA sebagai rujukan 0,005 [−0,003; 0,014]. Tidak ada mekanisme yang diuji; sel ini satu dari tujuh baris E
  ECG-FM+4 dan tidak dibaca sebagai paparan yang merugikan.
- **E per tugas lain (ECG-FM+4):** AF 0,000, LBBB 0,001 [−0,000; 0,003], RBBB −0,000, AV blok derajat 1 0,001
  [−0,000; 0,002]; semua |E| ≤ 0,001 dengan batas atas ≤ 0,003. Rujukan ECG-JEPA: LBBB 0,002 [0,000; 0,005], sisanya
  memuat nol.
- **Kosinus segmen ECG-FM+4 (disuntik − ditahan):** 0,0004 [0,0000; 0,0009]. Arah sesuai suku CMSC (dua segmen satu
  rekaman pasien disuntik sedikit lebih mirip), besarnya seperlima dari selisih dilihat − tidak dilihat pada ECG-FM asli
  di langkah 16 (0,0020 [0,0017; 0,0023]). Perbandingan dua bacaan ini deskriptif: populasi berbeda (paruh acak pasien
  tidak dilihat lawan dilihat − tidak dilihat), dan langkah 16 tidak punya plasebo kosinus. Bacaan yang konsisten
  dengan keduanya: suku CMSC meninggalkan jejak kecil yang bertambah dengan dosis, dan uji kehilangan tidak menangkap
  jejak itu.
- **Bacaan untuk naskah (pilihan penulis 27 Nov, "A"):** sumbangan dibingkai ulang dari "batas efek kontaminasi" ke
  "apakah kontaminasi pra-latih swaawas dapat diaudit": (i) DiD observasional bersyarat pada daftar split (Hasil 13,
  17); (ii) uji keanggotaan berbasis kehilangan buta bahkan pada anggota yang diketahui (Hasil 18 M1 + K1 di sini);
  (iii) paparan baru 4 epoch tidak menggeser AUROC probe beku (P1, P2; batas atas 0,001 di plafon). Celah LVH luar
  kohort 0,030 tetap tidak terjelaskan: kontaminasi ±240 epoch atau perbedaan campuran pasien.

## 1. Catatan proses

- **Jalan pertama dihentikan, tidak dipakai.** Dimulai 26 Nov 20:54 sebagai proses latar sesi asisten; ±21:35, sekitar
  pembaruan 167.450 di epoch 1, pengawas RAM lingkungan kerja menghentikan proses itu saat sesi diam (bukan galat
  skrip). Belum ada checkpoint yang tersimpan, jadi tidak ada yang dilanjutkan: folder keluaran kosong.
- **Jalan yang dipakai** dimulai ulang dari awal (pembaruan 167.280, epoch 1, benih sama) 26 Nov 21:44 di jendela cmd
  terpisah. Satu-satunya beda argumen: `--pekerja 4` untuk latih dan bacaan (bawaan 8 dan 6), untuk menekan RAM. Jumlah
  pekerja tidak mengubah hasil: `baca_berurutan` menyerahkan rekaman dalam urutan masukan, dan semua undian acak (urutan
  epoch, topeng sadapan, topeng waktu) dibuat di proses utama. Kehilangan pada pembaruan 167.300 adalah 1,2469 di jalan
  pertama dan 1,2468 di jalan kedua (selisih dari nondeterminisme GPU).
- Waktu: latih 21:44–05:44 (epoch ±2 jam, 487 pembaruan per epoch, ±14,4 detik per pembaruan); embedding ECG-FM+4
  05:44–06:24, skor 06:24–07:18; embedding ECG-FM+1 07:18–07:40, skor 07:40–08:11; `uji_kontrol.py` 08:11–08:19. Exit 0.
- Tidak ada rekaman gagal dibaca di latih (0 per epoch), embedding tak terhingga 0 dari 83.385 per dosis, skor terbaca
  22.058 dari 22.058 per dosis; bootstrap memakai 2.000 dari 2.000 ulangan.
- **Ukuran batch dibanding pra-latih asli (diperiksa sesudah hasil, 27 Nov):** konfigurasi checkpoint mencatat
  `batch_size` 171 per perangkat, `distributed_world_size` 3, dan `update_freq` 2, jadi satu pembaruan asli memuat
  1.026 sampel; 167.280 pembaruan dalam 240 epoch selesai (iterator di epoch 241) = ±697 pembaruan per epoch × 1.026
  ≈ 715 ribu sampel, sesuai ukuran `train` MIMIC-IV-ECG plus PhysioNet 2021. Protokol langkah 17 memakai 171 rekaman per
  pembaruan (tidak memperhitungkan tiga perangkat dan `update_freq`), jadi batch latih lanjut seperenam batch asli
  dengan laju belajar yang sama. Akibatnya bobot per rekaman dalam satu pembaruan enam kali lebih besar daripada di
  pra-latih asli, dan 4 epoch di sini = 1.948 pembaruan. Penyimpangan ini dicatat di naskah; arah dugaannya
  memperbesar, bukan memperkecil, jejak per rekaman, tetapi tidak diuji.
- RAM proses latih stabil ±15,9 GB (dipantau 21:55–23:05); berkas halaman Windows sempat tumbuh ke 28,9 GB dan menyusut
  ke 21,6 GB setelah checkpoint epoch 1; ruang disk tidak pernah di bawah 9,9 GB.

## Keluaran skrip


Kontrol positif swaawas langkah 17; benih 20261120; batch 9 × 19 rekaman.

| Kelompok latih | Pasien | Rekaman |
| --- | --- | --- |
| disuntik (tidak dilihat) | 11029 | 41574 |
| putar-ulang (train) | 27029 | 41574 |

Mulai dari pembaruan 167280, epoch 1.

23:34 epoch 1 selesai: pembaruan 167767; gagal dibaca 0; rerata per langkah bertopeng: loss 1.2423, loss_0 1.1715, loss_1 0.0622, loss_2 0.0056, loss_3 0.0030

01:37 epoch 2 selesai: pembaruan 168254; gagal dibaca 0; rerata per langkah bertopeng: loss 1.2422, loss_0 1.1712, loss_1 0.0622, loss_2 0.0057, loss_3 0.0031

03:41 epoch 3 selesai: pembaruan 168741; gagal dibaca 0; rerata per langkah bertopeng: loss 1.2440, loss_0 1.1729, loss_1 0.0622, loss_2 0.0058, loss_3 0.0032

05:44 epoch 4 selesai: pembaruan 169228; gagal dibaca 0; rerata per langkah bertopeng: loss 1.2431, loss_0 1.1723, loss_1 0.0622, loss_2 0.0058, loss_3 0.0029

Dosis epoch 4: embedding terhingga 83385/83385; skor terbaca 22058/22058.
Dosis epoch 1: embedding terhingga 83385/83385; skor terbaca 22058/22058.

Kontrol positif swaawas langkah 17; benih 20261120.

| Dibuang | Rekaman |
| --- | --- |
| embedding tak terhingga | 0 |
| skor tak terbaca | 0 |

| Hitungan | Disuntik | Ditahan |
| --- | --- | --- |
| rekaman | 41574 | 41811 |
| pasien dengan skor kehilangan | 11029 | 11029 |
| positif Fibrilasi atrium | 4197 | 4406 |
| positif LBBB | 1607 | 1616 |
| positif RBBB | 3708 | 3388 |
| positif AV blok derajat 1 | 3749 | 3688 |
| positif LVH | 3676 | 3676 |

## Statistik (bootstrap 2000 ulangan atas pasien, terpakai 2000, benih 20261120)

| Tugas | Celah ECG-JEPA | Celah ECG-FM asli | Celah ECG-FM+1 | Celah ECG-FM+4 |
| --- | --- | --- | --- | --- |
| Fibrilasi atrium | -0.002 [-0.004, -0.000] | -0.001 [-0.004, 0.001] | -0.001 [-0.003, 0.001] | -0.001 [-0.003, 0.001] |
| LBBB | -0.002 [-0.007, 0.001] | -0.001 [-0.006, 0.003] | -0.001 [-0.006, 0.003] | -0.000 [-0.004, 0.004] |
| RBBB | 0.001 [-0.002, 0.004] | 0.000 [-0.004, 0.005] | 0.001 [-0.003, 0.005] | 0.000 [-0.004, 0.004] |
| AV blok derajat 1 | -0.000 [-0.006, 0.005] | -0.000 [-0.003, 0.003] | 0.000 [-0.003, 0.004] | 0.001 [-0.002, 0.004] |
| LVH | -0.001 [-0.010, 0.007] | 0.008 [-0.005, 0.022] | 0.007 [-0.007, 0.020] | 0.004 [-0.008, 0.017] |
| rerata empat berlabel | -0.001 [-0.003, 0.001] | -0.001 [-0.003, 0.001] | -0.000 [-0.002, 0.002] | 0.000 [-0.002, 0.002] |
| rerata lima pernyataan | -0.001 [-0.004, 0.001] | 0.001 [-0.002, 0.005] | 0.001 [-0.002, 0.004] | 0.001 [-0.002, 0.004] |

| Tugas | E ECG-FM+4 (utama) | E ECG-FM+1 | E ECG-FM+4 lawan ECG-JEPA |
| --- | --- | --- | --- |
| Fibrilasi atrium | 0.000 [-0.001, 0.001] | 0.000 [-0.001, 0.001] | 0.001 [-0.000, 0.003] |
| LBBB | 0.001 [-0.000, 0.003] | 0.000 [-0.001, 0.001] | 0.002 [0.000, 0.005] |
| RBBB | -0.000 [-0.002, 0.002] | 0.001 [-0.001, 0.003] | -0.001 [-0.003, 0.002] |
| AV blok derajat 1 | 0.001 [-0.000, 0.002] | 0.000 [-0.001, 0.001] | 0.001 [-0.002, 0.006] |
| LVH | -0.004 [-0.008, -0.001] | -0.002 [-0.004, 0.001] | 0.005 [-0.003, 0.014] |
| rerata empat berlabel | 0.001 [-0.000, 0.001] | 0.000 [-0.000, 0.001] | 0.001 [-0.000, 0.003] |
| rerata lima pernyataan | -0.000 [-0.001, 0.001] | -0.000 [-0.001, 0.001] | 0.002 [-0.000, 0.004] |

| Bacaan kehilangan | Nilai [95 % CI] |
| --- | --- |
| Ds ECG-FM asli (ditahan − disuntik) | 0.0009 [-0.0087, 0.0106] |
| Ds ECG-FM+1 (ditahan − disuntik) | 0.0016 [-0.0082, 0.0112] |
| Ds ECG-FM+4 (ditahan − disuntik) | 0.0016 [-0.0083, 0.0113] |
| AUROC keanggotaan ECG-FM+4 | 0.4999 [0.4920, 0.5077] |
| Kosinus segmen ECG-FM+4 (disuntik − ditahan) | 0.0004 [0.0000, 0.0009] |

## Keputusan praregistrasi

- N0 (celah ECG-JEPA LVH dan rerata lima memuat 0): LOLOS.
- K1 (batas bawah Ds plus4 > 0): TIDAK LOLOS (batas bawah -0.0083); tidak terbukti: uji kehilangan peka pada anggota yang diketahui.
- P1 (batas bawah E plus4 lvh > 0): TIDAK LOLOS (batas bawah -0.0076); tidak terbukti: paparan swaawas terdeteksi pada LVH.
- P2 (batas bawah E plus4 rerata empat berlabel > 0): TIDAK LOLOS (batas bawah -0.0001); tidak terbukti: paparan swaawas terdeteksi di plafon.
keluar 0  
