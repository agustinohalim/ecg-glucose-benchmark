# Analysis plan (excerpt)

Section C2 (dataset facts that shaped the design) and work steps 6, 7 and 9 of section F of the authors' working plan,
in Indonesian. Step 7 is the TEST protocol, recorded before the TEST split was opened; step 9 is the pretraining
contamination protocol, recorded before any ECG-FM embedding was computed. Each protocol was recorded in a dated
commit of the authors' private working repository; the bold **Hasil N** paragraphs inside a step were appended after
the corresponding run. The public timestamp is this release: version 1.0.0 precedes the run of step 9. Sections on
journal choice, manuscript outline and internal review are omitted. File names `Hasil_Glukosa_N.md` refer to `notes/`.

## C2. Fakta dataset yang mengubah rancangan (README + kamus data v1.0.0, dibaca 20 Nov 2026)

- **Kendali perancu obat hanya berlaku di ICU.** Insulin/dekstrosa dianotasi dari `inputevents` ICU
  saja; 80.598 pasangan (18,4 %) di ICU. Di luar ICU `intervention_status = CLEAN` berarti "tidak
  dianotasi". Klaim "dengan kendali perancu pengobatan" hanya sah pada subset ICU — sumbangan di A harus
  dibatasi ke sana atau diturunkan.
- **Pembagian per pasien sudah tersedia** (`split`, 80/10/10, hash `subject_id`). Pakai itu, jangan buat
  sendiri; CI tetap bootstrap bergugus pasien. 22 % pasangan berbagi satu EKG (hingga 12 glukosa per EKG).
- **Kolom bocor, dilarang jadi masukan:** `glycemic_class`, `lab_flag`, `glucose_delta_mg_dl`,
  `glucose_rate_mg_dl_per_hr`, `glucose_trajectory`, `glucose_z_score`, `glucose_minmax_norm`, dan
  semua statistik tingkat pasien (`patient_*_glucose`, `glucose_cv_percent`, `time_*_range_pct`).
- **M1 bisa jalan tanpa gelombang:** ukuran mesin EKG (RR, QRS, QT, PR, JT, sumbu, empat QTc, SQI) sudah
  ada di CSV. Kode "tidak terukur" (29999, 32767, 65535; turunannya mis. −29799) → |nilai| ≥ 10000
  dianggap hilang.
- **Gelombang:** 376.648 studi EKG unik, dirujuk lewat `waveform_path` ke MIMIC-IV-ECG. Perlu ZIP
  penuh (33,8 GB); ekstrak hanya studi yang dirujuk.
- Usabilitas: IDEAL 1,0 %, USABLE 26,5 %, CAUTION 67,4 %, EXCLUDE 5,1 %. Subset konservatif anjuran
  penulis: IDEAL/USABLE + `ECG_BEFORE_GLUCOSE` + `CLEAN`.
- Dua uji glukosa (kimia 50931, gas darah 50809) tak dibedakan di berkas; memulihkannya butuh
  `labevents` MIMIC-IV (DUA terpisah). Glukosa jari (POCT) tidak termasuk.
- Jumlah hipoglikemia per `split`: `Hasil_Glukosa_0.md` (20 Nov 2026). Hipo ±2 %; uji non-EXCLUDE
  706 hipo / 512 pasien; **ICU + obat terverifikasi bersih: uji < 11 hipo** — klaim kendali obat
  tidak bisa diuji pada pembagian bawaan. **Diputuskan penulis 20 Nov 2026:** validasi silang
  bergugus pasien atas seluruh subset ICU + konservatif (gabungan TRAIN/VALIDATION/TEST, ±130 hipo)
  sebagai analisis kepekaan kendali obat; CI lebar dilaporkan apa adanya.


## F. Langkah 6, 7, 9 (petikan)

6. **Keputusan penulis 21 Nov 2026, sebelum TEST dibuka:**
   (a) M2 analisis utama = rerata peluang 3 benih (`BENIH`, sama dengan validasi silang ICU),
   `glukosa_m2/jalankan_m2_benih.py`; angka satu benih di Hasil 3/4 tetap dilaporkan sebagai kepekaan.
   **Hasil 6 (21 Nov):** M2b VALIDATION hipo 0,618, hiper berat 0,741; M2b − M1-GB +0,039 [0,010; 0,069]
   dan +0,073 [0,052; 0,091]; M3 − M2b −0,004 (hipo) dan −0,025 [−0,040; −0,009] (hiper berat). Benih sama
   pada GPU berbeda sampai ±0,01 antarjalan (tidak deterministik).
   (b) M2 dikalibrasi ulang: regresi logistik pada logit peluang M2 (intersep + kemiringan, tanpa
   penalti), dipasang pada prediksi VALIDATION dari model yang sama lalu diterapkan tanpa perubahan ke
   TEST; kalibrasi TEST dilaporkan mentah dan terkalibrasi ulang; AUROC tidak berubah. M0, M1, M3 tidak
   dikalibrasi ulang (M3 sudah baik di VALIDATION).
   (c) Model fondasi kedua: periksa dulu data pra-latih dari sumber primer; tambah satu bila tanpa
   MIMIC-IV-ECG, bobot terbuka berlisensi riset, muat GPU 8 GB, dan dipakai beku + probe L2 yang sama
   dengan M3. Bila tidak ada yang lolos, ECGFounder saja dan alasannya ditulis di Limitations.
   **Hasil pemeriksaan 21 Nov 2026** (sumber primer: makalah, README, kartu model; tabel 1 Reality
   Check arXiv 2509.25095): memakai MIMIC-IV-ECG dalam pra-latih → **dikeluarkan**: ECG-FM (README
   "Pretrained on MIMIC-IV-ECG v1.0 and PhysioNet 2021"), MERL, ECGFM-KED, HuBERT-ECG (hanya sumber
   sekunder), CLEF, DeepECG-SSL, SuPreME, ECG-Byte. Lolos: **ECG-JEPA** (Kim, arXiv 2410.08559 §4:
   Chapman, Ningbo, CODE-15; MIT; ViT ±85 jt; 250 Hz, 10 s, 8 sadapan I, II, V1–V6) — pilihan pertama
   karena data pra-latihnya lain dari ECGFounder (HEEDB); ST-MEM (CC BY-NC 4.0); ECG-CPC (HEEDB, sama
   dengan ECGFounder). Satu tolok ukur (FOUND-AF arXiv 2608.03597) mencantumkan MIMIC untuk
   "ECG-JEPA", kemungkinan tertukar dengan JEPA Weimann & Conrad. **Diperiksa 21 Nov 2026:** titik periksa
   = tautan "ECG-JEPA (multi-block masking)" di README repo (komit d937ad2), `multiblock_epoch100.pth`;
   `pos_embed` 400 × 768 = 8 sadapan × 50 tambalan × 50 sampel (250 Hz, 10 s); skrip pra-latih resmi
   `pretrain_ECG_JEPA.py` hanya menerima `--data_dir_shao` (Shaoxing/Ningbo) dan `--data_dir_code15`, tak
   ada jalur MIMIC di kode. Varian MIMIC di Lampiran A.1 Kim tidak dirilis di README dan tidak dipakai.
   Pemuatan: bobot dimuat ketat (85,4 jt parameter) oleh `load_encoder` resmi; paket timm (yang menarik
   torchvision) diganti modul minimal `timm_pengganti` (DropPath identitas saat eval, `trunc_normal_`
   hanya untuk inisialisasi) — tidak memengaruhi keluaran. Pra-proses jalur ZIP = jalur resmi
   (wfdb `p_signal`, resample 5.000 → 2.500), selisih 0,0 pada 3 rekaman terbuka. M3b memakai embedding
   `encoder.representation` (768 dimensi) dan probe yang sama dengan M3, `--model ecgjepa`.
   **Hasil 7 (21 Nov):** M3b VALIDATION hipo 0,643, hiper berat 0,747, terkalibrasi baik; M3b − M3 +0,029
   [0,011; 0,048] dan +0,030 [0,020; 0,042]; M3b − M2b +0,025 [0,004; 0,047] dan +0,006 [−0,009; 0,020];
   M3b − M1-GB +0,064 dan +0,078. ICU CLEAN hipo M3b − M1-GB +0,046 [0,001; 0,087] (belum dianggap
   terbukti). Dugaan "batas informasi gelombang" (Hasil 5) gugur.
7. **Protokol TEST, keputusan penulis 21 Nov 2026 sesudah Hasil 7, dicatat sebelum TEST dibuka:**
   (a) **Uji utama, satu per tugas** (hipoglikemia; hiperglikemia berat): selisih AUROC M3b − M1-GB di
   TEST non-EXCLUDE, CI 95 % bootstrap bergugus pasien (1.000 ulangan, benih `BENIH[0]`). Didukung bila
   batas bawah CI > 0. Kedua tugas dilaporkan terpisah tanpa koreksi antar-tugas. M3b dipilih sebagai
   model utama karena tertinggi di VALIDATION di antara tiga model gelombang (Hasil 7); di naskah ditulis
   sebagai pemilihan pada VALIDATION.
   (b) **Sekunder, deskriptif, tanpa klaim konfirmatori:** M2b − M1-GB, M3 − M1-GB, M3b − M2b, M3b − M3;
   bagian konservatif; tabel metrik semua model (M0, M1-LR, M1-GB, M2 satu benih `BENIH[0]`, M2b mentah
   dan terkalibrasi ulang, M3, M3b).
   (c) **Model:** semua dilatih ulang pada TRAIN non-EXCLUDE dengan resep dan benih yang sama dengan
   Hasil 3–7; VALIDATION tidak dipakai untuk pelatihan. Prediksi VALIDATION dan TEST diambil dari model
   yang sama. M0, M1, M3, M3b deterministik (VALIDATION harus sama dengan Hasil 5–7); M2 bergeser ±0,01
   antarjalan karena GPU.
   (d) **Kalibrasi ulang M2b** (keputusan 6b): regresi logistik tanpa penalti pada logit peluang M2b,
   dipasang pada prediksi VALIDATION dari jalan ini, diterapkan tanpa perubahan ke TEST. Model lain tidak
   dikalibrasi ulang.
   (e) **ICU dan tiga kelas tidak dinilai di TEST.** Analisis ICU memakai validasi silang atas gabungan
   ketiga split (keputusan 20 Nov, §C2). Jadi label baris TEST yang termasuk subset ICU sudah dipakai di
   analisis kepekaan itu, termasuk saat Hasil 3–7 dibaca. Ini ditulis di Limitations. Uji utama tidak
   terkena, karena modelnya dilatih hanya pada TRAIN.
   (f) TEST dijalankan sekali (`glukosa_test/jalankan_test.py`). Sesudahnya tidak ada perubahan model,
   resep, atau perbandingan. Galat teknis diperbaiki secara teknis saja dan dicatat di Hasil.
   **Hasil 8 (TEST dibuka sekali, 21–22 Nov, tanpa galat):** uji utama **didukung pada kedua tugas**.
   Hipo: M3b 0,643, M3b − M1-GB +0,066 [0,041; 0,093]. Hiper berat: M3b 0,738, +0,066 [0,052; 0,082].
   Sekunder: M3b − M3 +0,025 dan +0,026 (CI > 0); M3b − M2b +0,025 [−0,001; 0,049] dan +0,014
   [−0,000; 0,030], jadi keunggulan model fondasi atas CNN tidak ditulis sebagai temuan. VALIDATION model
   deterministik sama persis dengan Hasil 5–7; M2 per benih bergeser sampai 0,015 (GPU).
9. **Protokol uji kontaminasi, dicatat 22 Nov 2026 sebelum embedding ECG-FM dibangun (belum ada angka ECG-FM):**
   (a) **Model terkontaminasi M3c:** ECG-FM `mimic_iv_ecg_physionet_pretrained.pt` (wanglab/ecg-fm revisi 584219ea,
   SHA-256 4d0142bc…9b7e), beku, pra-proses resmi (`glukosa_m3/ecgfm.py`), probe sama persis dengan M3/M3b
   (standarisasi + logistik L2, penalti dari {10⁻³ … 100} oleh log-loss CV 3 lipatan bergugus pasien di TRAIN).
   (b) **Paparan:** daftar split penulis ECG-FM (`splits/meta_split_mimic_iv_ecg.csv`, blob 773e5591, komit repo
   3ff744ef). Pasangan dengan EKG berstatus `train` = **dilihat** saat pra-latih; `valid` atau `test` = **tidak
   dilihat** (split per pasien; `valid` hanya untuk loss validasi). EKG yang tidak tercantum dikeluarkan dari kontras
   dan jumlahnya dilaporkan.
   (c) **Himpunan nilai:** VALIDATION ∪ TEST non-EXCLUDE (keduanya tidak dipakai melatih probe), per tugas.
   Digabung demi daya: bagian tidak dilihat ±20 % pasangan. TEST sudah dibuka untuk M3b, tetapi ECG-FM dan kontras
   dilihat/tidak dilihat belum pernah dihitung pada split mana pun.
   (d) **Estimand utama, satu per tugas:** selisih-dalam-selisih
   DiD = [AUROC M3c(dilihat) − AUROC M3c(tidak)] − [AUROC M3b(dilihat) − AUROC M3b(tidak)], dengan M3b (ECG-JEPA,
   bersih) sebagai pembanding yang menyerap perbedaan kelompok pasien. CI 95 % bootstrap bergugus pasien, 1.000
   ulangan, benih `BENIH[0]`, semua model pada ulangan yang sama. **Inflasi didukung bila batas bawah > 0.** Bila
   CI memuat 0, batas atas dilaporkan sebagai batas inflasi yang masih sesuai dengan data. Dua tugas dilaporkan
   terpisah tanpa koreksi.
   (e) **Sekunder, deskriptif:** DiD dengan pembanding M3 (ECGFounder, bersih) dan M1-GB (ukuran mesin, tanpa
   pra-latih); AUROC tiap model per kelompok; M3c − M3b keseluruhan di VALIDATION ∪ TEST dan di TEST saja;
   karakteristik kelompok dilihat/tidak (pasangan, pasien, positif, usia, jenis kelamin, bagian ICU).
   (f) **Tanpa pengembangan atas hasil:** sebelum estimand dihitung, yang boleh dilihat hanya pemeriksaan teknis
   (jumlah baris embedding, nilai terhingga, penalti terpilih dan log-loss CV di TRAIN). Satu skrip
   (`glukosa_test/jalankan_kontaminasi.py`), dijalankan sekali; galat teknis diperbaiki secara teknis dan dicatat.
   (g) Tafsir dibatasi: paparan di sini adalah pra-latih swaawas pada rekaman yang sama, tanpa label glukosa;
   kelompok dilihat dan tidak dilihat berbeda pasien. Hasil nol ditulis sebagai batas atas, bukan bukti tiada efek.
