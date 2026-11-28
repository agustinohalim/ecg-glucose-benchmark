# Analysis plan (excerpt)

Section C2 (dataset facts that shaped the design) and work steps 6, 7 and 9–21 of section F of the authors' working
plan, in Indonesian. Step 7 is the TEST protocol, recorded before the TEST split was opened; step 9 is the pretraining
contamination protocol, recorded before any ECG-FM embedding was computed; step 11 extends it to machine-statement
tasks and a fine-tuned positive control, recorded before those were scored. Steps 10, 12, 13 and 14 were recorded
after results they respond to and are marked PASCA-HASIL (post hoc, descriptive). Step 15 is the confirmatory
replication on recordings outside the glucose cohort; version 1.1.0 is released after step 15 and its code were
committed and before its labels were computed or any embedding of those recordings was built. Step 16 is the direct
membership check, recorded before its code was written and before any loss was scored. Step 17 is the
continued-pretraining positive control; version 1.2.0 is released after step 17 and its code were committed and before
any training, embedding or score of that step. Step 18 is the calibrated audit (paired manipulation check and a
segment-agreement placebo on the clean models); version 1.3.0 is released after step 18 and its code were committed and
before any of its statistics was computed. Step 19 re-tests the segment-agreement trace on fresh cohort recordings
against a permuted-weight ECG-FM, adds dataset inference and a same-mask paired loss; version 1.4.0 is released after
step 19 and its code were committed and before any of its readings or statistics was computed. Step 20 re-reads
step 19 with the same eligibility rule for both groups; it was recorded with its code after the step 19 results and
before its own run, and is marked PASCA-HASIL. Step 21 tests ECG-FM fine-tuning labels that are off the ceiling and
is declared the last analysis of the membership effect; version 1.5.0 is released after step 21 and its code were
committed and before any model was scored on its tasks. Each step was recorded
in a dated commit of the authors' private working repository; the bold **Hasil N** paragraphs inside a step were
appended after the corresponding run. "Telaah internal N" is the authors' N-th internal pre-submission review. Sections
on journal choice, manuscript outline and internal review are omitted. File names `Hasil_Glukosa_N.md` refer to
`notes/`.

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
   **Hasil 11 (22 Nov, dijalankan sekali, tanpa galat; kode v1.0.0 dirilis sebelumnya):** inflasi **tidak
   didukung** pada kedua tugas. DiD hipo 0,028 [−0,003; 0,060], hiper berat 0,006 [−0,019; 0,032]; batas inflasi
   0,060 dan 0,032. M1-GB pun berbeda antarkelompok (hipo 0,051 [0,010; 0,099]), jadi selisih mentah tidak bisa
   dipakai. M3c − M3b −0,021 [−0,035; −0,008] dan −0,019 [−0,027; −0,011]: model terkontaminasi lebih rendah.
10. **Analisis tambahan atas telaah internal 2 (dicatat 22 Nov 2026 sebelum dijalankan; PASCA-HASIL, deskriptif, tidak
   mengubah aturan keputusan langkah 9).** Pilihan penulis "Putaran teks + uji kecil". Satu skrip
   (`glukosa_test/jalankan_kontaminasi_tambahan.py`) memakai prediksi tersimpan langkah 9 (`kontaminasi_*.npz`, tanpa
   melatih ulang; baris dicek sejajar), mencetak agregat saja:
   (a) **Uji permutasi paparan:** label dilihat/tidak diacak ulang per pasien (jumlah pasien dilihat tetap), 2.000
   permutasi, benih `BENIH[0]`; untuk tiap model selisih dilihat − tidak dan tiap DiD: SD sebaran nol dan p dua sisi
   nilai teramati. Menjawab apakah selisih M1-GB 0,051 sesuai dengan kebetulan bila split acak.
   (b) **Keseimbangan kelompok:** beda rerata terstandar (SMD) untuk usia, laki-laki, ICU, prevalensi positif, pasangan
   per pasien.
   (c) **DiD plasebo antar model bersih:** M3 − M3b, M1-GB − M3b, M1-GB − M3 (selisih-dalam-selisih dua model tanpa
   paparan), CI bootstrap.
   (d) **Bootstrap 10.000** untuk DiD utama dan sekunder (galat Monte Carlo batas −0,003).
   (e) **Kepekaan definisi paparan:** tidak dilihat = hanya `test` ECG-FM (tanpa `valid`); dan DiD utama di VALIDATION
   dan TEST terpisah.
   (f) **Efek minimum terdeteksi:** 2,8 × SD bootstrap DiD utama (80 % daya, α 0,05 dua sisi), per tugas.
   (g) **Skala logit:** DiD atas logit(AUROC).
   Aturan tafsir ditetapkan sekarang: (a) p < 0,05 untuk selisih model tanpa paparan = bukti beda kelompok di luar
   kebetulan; bila tidak, "selisih mentah bising" (bukan "terkacau"). Hipo disebut inkonklusif bila batas atas DiD ≥
   setengah efek utama (0,033); hiper berat terbatasi bila batas atas < 0,033.
11. **Perluasan uji kontaminasi atas telaah internal 3 (dicatat 22 Nov 2026 malam, sebelum embedding ECG-FM *fine-tuned*
   dibangun dan sebelum tugas pernyataan atau evaluasi *cross-fitting* dinilai).** Pilihan penulis "Perluas uji".
   Yang sudah dilihat sebelum protokol ini: Hasil 11–12 (glukosa) dan hitungan agregat frekuensi pernyataan mesin
   (`glukosa_test/hitung_pernyataan.py`, tanpa model). Satu skrip analisis, mencetak agregat saja.
   (a) **Model:** ECG-FM pretrained (terpapar swaawas), ECG-JEPA (pembanding utama), ECGFounder (bersih kedua), model
   ukuran (GB, 13 ukuran mesin + usia + jenis kelamin), dan **ECG-FM *fine-tuned*** (`mimic_iv_ecg_finetuned.pt`,
   wanglab/ecg-fm), dilatih dengan label diagnosis rekaman bagian `train` (label_def.csv repo ECG-FM memuat hitungan
   positif per train/valid/test). Embedding *fine-tuned* = rerata token `encoder.extract_features`, pra-proses sama
   dengan pretrained. Semua beku + probe L2 yang sama.
   (b) **Tugas pernyataan mesin** (MIMIC-IV-ECG v1.0 `machine_measurements.csv`, terbuka): unit = EKG unik dengan
   embedding (376.648); positif bila gabungan report_0–17 (huruf kecil) cocok pola, negatif semua EKG lain: AF
   `atrial fibrillation`; LBBB `left bundle branch block|\blbbb\b`; RBBB `right bundle branch block|\brbbb\b`; AV blok
   derajat 1 `(1st|first) degree a-?v block`; LVH `left ventricular hypertrophy|\blvh\b`. Empat yang pertama termasuk
   label *fine-tuned* ECG-FM; LVH tidak. Usia, jenis kelamin dan ukuran mesin per EKG diambil dari pasangan pertama EKG
   itu di berkas glukosa.
   (c) **Evaluasi *cross-fitting*:** semua unit (tugas glukosa: pasangan non-EXCLUDE dari ketiga split dataset; tugas
   pernyataan: EKG) dibagi 5 lipatan per pasien (`GroupKFold` atas pasien teracak, benih `BENIH[0]`); tiap model
   dilatih pada 4 lipatan dan memprediksi lipatan sisanya. C probe: tugas glukosa memakai C terpilih di TRAIN
   (langkah 8/9), tanpa penyetelan ulang; tugas pernyataan memilih C dari {10⁻³, 10⁻², 10⁻¹, 1} lewat 3 lipatan dalam
   per pasien pada subsampel 60.000 unit dari bagian latih lipatan 1, lalu tetap untuk kelima lipatan. GB: setelan
   langkah 8.
   (d) **Estimand:** per tugas, DiD = [AUROC dilihat − tidak dilihat] model − pembanding, atas prediksi di luar lipatan.
   Utama: ECG-FM pretrained − ECG-JEPA. Sekunder: lawan ECGFounder. Plasebo: ECGFounder − ECG-JEPA. Kontrol positif:
   ECG-FM *fine-tuned* − ECG-JEPA. Bootstrap bergugus pasien 2.000 ulangan, permutasi paparan per pasien 1.000; tidak
   dilihat = `valid` ∪ `test`, kepekaan `test` saja; skala logit sekunder.
   (e) **Aturan keputusan (ditetapkan sekarang):** margin ε = 0,01 AUROC, separuh selisih ECG-FM − ECG-JEPA di Hasil 11
   (≈ 0,02), sebagai ukuran inflasi yang dapat membalik peringkat encoder. Per tugas: "inflasi didukung" bila batas
   bawah DiD > 0; "di bawah skala peringkat" bila batas atas < ε; selain itu "tak terputuskan". Ringkasan: rerata DiD
   lima tugas pernyataan (bobot sama) dengan CI dari ulangan bootstrap yang sama, dibaca dengan aturan yang sama.
   Tugas dengan AUROC kedua kelompok > 0,97 untuk pembanding utama ditandai "dekat plafon" (DiD AUROC kecil karena
   plafon; skala logit ikut dibaca).
   (f) **Kontrol positif lolos** bila batas bawah DiD *fine-tuned* − ECG-JEPA > 0 pada ≥ 1 dari empat tugas berlabel.
   Bila tidak lolos, kepekaan desain dinyatakan tidak terbukti dan naskah mengatakannya.
   (g) **Glukosa:** tafsir langkah 9 tetap (inflasi tak didukung, batas atas dilaporkan); ambang 0,033 langkah 10 tidak
   lagi dipakai sebagai vonis. DiD glukosa *cross-fitting* dibaca dengan aturan (e).
12. **Analisis tambahan atas Hasil 13 untuk temuan telaah internal 3 (dicatat 22 Nov 2026 larut malam, sesudah Hasil 13,
   sebelum skrip ditulis; PASCA-HASIL, deskriptif, tidak mengubah vonis langkah 11).** Prediksi tersimpan langkah 11,
   tanpa melatih ulang; benih 20261120.
   (a) **Keseimbangan** per tugas atas semua unit yang terpetakan ke split ECG-FM: SMD usia, laki-laki, ICU (glukosa
   saja; tabel pernyataan tak memuat ICU), prevalensi positif, unit per pasien; dibaca dengan ambang |SMD| < 0,1.
   (b) **Efek minimum terdeteksi** = 2,8 × SD bootstrap (1.000 ulangan bergugus pasien) untuk DiD utama dan kontrol
   positif, per tugas.
   (c) **Presisi dari pembanding:** di bawah nol permutasi (1.000 penempatan ulang per pasien) SD selisih ECG-FM
   sendiri, SD DiD utama, korelasi selisih ECG-FM dengan selisih ECG-JEPA, dan SD penaksir kontrol-variat
   selisih ECG-FM − b · selisih ECG-JEPA dengan b = kov/var di bawah nol; titiknya juga dicetak. Dibaca sebagai
   deskripsi; vonis tetap DiD (b = 1).
13. **Kontras langsung paparan swaawas lawan kontrol positif, atas telaah internal 4 (dicatat 23 Nov 2026, sesudah Hasil 13–14
   dan telaah internal 4, sebelum skrip ditulis; PASCA-HASIL, deskriptif, tidak mengubah vonis langkah 11).** Pilihan penulis
   "Langkah 13 + revisi". Temuan kritis telaah internal 4: margin ε = 0,01 juga diloloskan kontrol positif, dan selisih kedua
   paparan tidak pernah ditaksir. Prediksi tersimpan langkah 11, tanpa melatih ulang; bootstrap bergugus pasien 2.000
   ulangan, benih 20261120, ulangan yang sama untuk semua besaran dalam satu kelompok tugas (dua tugas glukosa
   masing-masing; lima pernyataan bersama). Yang sudah dilihat: semua angka Hasil 13–14; selisih langsung dan rasio
   belum pernah dihitung.
   (a) **Selisih langsung** Δ = DiD kontrol positif − DiD utama = [selisih dilihat − tidak ECG-FM *fine-tuned*] −
   [selisih ECG-FM pretrained]; ECG-JEPA saling hapus. Per tugas, skala AUROC dan logit(AUROC), CI 95 % kuantil.
   Ringkasan: rerata Δ empat tugas berlabel (AF, LBBB, RBBB, AV blok 1), bobot sama.
   (b) **Rasio** R = DiD utama / DiD kontrol positif, hanya untuk tugas tempat batas bawah DiD kontrol positif > 0
   pada skala itu (Hasil 13: empat tugas berlabel pada skala AUROC); CI kuantil bootstrap, ulangan dengan penyebut ≤ 0
   dihitung dan jumlahnya dicetak. Juga rasio rerata empat tugas (rerata DiD utama / rerata DiD kontrol positif).
   (c) **Log-loss** (aturan skor sejati, tak berplafon seperti AUROC): per model, rerata log-loss berbobot di kelompok
   dilihat dan tidak dilihat atas peluang di luar lipatan; DiD log-loss dengan tanda dibalik (tidak dilihat − dilihat,
   sehingga positif = lebih baik pada rekaman dilihat) untuk utama dan kontrol positif, serta Δ seperti (a); CI
   bootstrap yang sama.
   (d) **Tabel logit per tugas:** DiD utama dan kontrol positif pada skala logit dengan CI, ketujuh tugas (sebagian
   sudah ada di Hasil 13; dicetak ulang dari ulangan yang sama agar satu tabel).
   **Aturan tafsir (ditetapkan sekarang):** pada suatu skala, paparan swaawas disebut "lebih kecil dari paparan label"
   bila batas bawah Δ > 0; batas atas R dilaporkan sebagai "paling besar R kali efek paparan label". Tafsir ringkasan
   memakai rerata empat tugas berlabel. Bila batas bawah Δ ≤ 0, naskah menyatakan kedua paparan tidak terbedakan
   pada skala itu. Tugas glukosa dan LVH dilaporkan tanpa tafsir rasio (kontrol positif tak terdeteksi). Satu skrip
   (`glukosa_perluasan/jalankan_kontras_langsung.py`), dijalankan sekali; galat teknis diperbaiki dan dicatat.
14. **Rasio terhadap efek label saja, atas telaah internal 6 (dicatat 23 Nov 2026, sesudah Hasil 15 dan telaah internal 6, sebelum skrip
   ditulis; PASCA-HASIL, deskriptif, tidak mengubah vonis langkah 11).** Pilihan penulis "Langkah 14 + revisi".
   Temuan kritis telaah internal 6 (methods M1): ECG-FM *fine-tuned* berangkat dari encoder pretrained yang sama, sehingga
   rekaman dilihat terpapar dua kali (swaawas + label); DiD kontrol positif = efek swaawas + efek label, dan R langkah
   13 = swaawas / (swaawas + label), bukan swaawas / label. Bila kedua efek menjumlah, efek label saja = Δ langkah 13.
   Yang sudah dilihat: semua angka Hasil 15 (R AUROC rerata 0,24 [−0,10; 0,48]; logit 0,09 [−0,05; 0,21]; log-loss
   0,06 [−0,16; 0,23]); transformasi R/(1 − R) memberi batas atas ≈ 0,92 / 0,27 / 0,30 bila semua ulangan R < 1.
   Data, prediksi, ulangan dan benih sama dengan langkah 13 (2.000 ulangan, benih 20261120, ulangan bersama per
   kelompok tugas).
   (a) **Rasio Q = DiD utama / Δ** (swaawas terhadap label saja), per ulangan, CI 95 % kuantil; jumlah ulangan Δ ≤ 0
   dicetak. Skala AUROC, logit(AUROC), log-loss. Per tugas untuk empat tugas berlabel, rerata empat tugas berlabel
   (rerata DiD utama / rerata Δ), dan rerata lima pernyataan (empat berlabel + LVH) karena rerata lima tugas
   dipraspesifikasi di langkah 11 tetapi belum dilaporkan (methods M5).
   (b) **Aturan "lebih kecil dari paparan label" (methods M2):** batas bawah 95 % dari Δ − DiD utama (= DiD kontrol
   positif − 2 · DiD utama) > 0. Aturan langkah 13 (batas bawah Δ > 0) hanya menunjukkan paparan label ada; tidak
   dipakai lagi untuk perbandingan besar. Tidak ada ambang lain ("separuh" dsb.); naskah melaporkan batas atas Q.
   (c) **Kepekaan TEST saja (devils-advocate X1):** (a) dan (b) diulang dengan tidak dilihat = `test` saja (rekaman
   `valid` dibuang), ulangan dan benih sama.
   (d) **Selisih mutlak:** rerata DiD utama empat tugas berlabel pada skala AUROC beserta CI dicetak ulang agar batas
   mutlak (≈ 0,001) dan rasio berasal dari ulangan yang sama.
   Tafsir: bila batas bawah (b) > 0 pada skala AUROC untuk rerata empat tugas berlabel, naskah menyatakan paparan
   swaawas lebih kecil dari paparan label di skala itu, dengan batas atas Q; bila tidak, keduanya tidak terbedakan
   di skala itu. Hasil utama naskah menjadi batas mutlak; rasio pelengkap. Satu skrip
   (`glukosa_perluasan/jalankan_rasio_label.py`), dijalankan sekali; galat teknis diperbaiki dan dicatat.
15. **Replikasi konfirmatori pada rekaman yang belum pernah dinilai (dicatat 23 Nov 2026 malam, sesudah telaah internal 7,
   sebelum kode replikasi ditulis, sebelum embedding rekaman ini dibangun, dan sebelum label pernyataan rekaman ini
   dihitung).** Pilihan penulis "Replikasi konfirmatori", sampel "Lepas rekaman", pra-registrasi "Zenodo publik":
   protokol ini dan kodenya diunggah sebagai versi baru rekaman Zenodo Glukosa sebelum embedding dibangun (unggahan
   atas konfirmasi penulis). Alasan: telaah internal 5–7 masing-masing menemukan kritis baru pada lapisan perbandingan
   pasca-hasil; replikasi menguji klaim utama (batas mutlak + kontrol positif) pada data baru dengan aturan yang
   ditetapkan sebelum data itu disentuh. Yang sudah dilihat: semua Hasil 1–16; hitungan agregat calon sampel
   (skrip sementara `hitung_replikasi.py`, tanpa label atau model): 416.503 rekaman daftar split ECG-FM di luar
   kohort glukosa (109.971 pasien; train 333.118, valid 41.908, test 41.477), di antaranya 53.413 rekaman dari 29.391
   pasien yang tidak ada di kohort.
   (a) **Data (semua terbuka kecuali daftar pengecualian):** MIMIC-IV-ECG v1.0 (`record_list.csv`,
   `machine_measurements.csv`, gelombang dari ZIP) dan daftar split ECG-FM (`meta_split_mimic_iv_ecg.csv`, commit
   3ff744ef). Unit = rekaman di daftar split yang `path`-nya tidak ada di kohort MIMIC-IV-Ext-ECG-Glucose (kohort hanya
   dipakai sebagai daftar pengecualian jalur dan pasien). Rekaman yang gelombangnya gagal dibaca atau embedding-nya
   tidak terhingga dibuang dan dihitung. Dilihat = `train`; tidak dilihat = `valid` ∪ `test`.
   (b) **Tugas:** lima pernyataan mesin dengan pola **identik** langkah 11b (AF, LBBB, RBBB, AV blok 1, LVH), atas
   gabungan report_0–17 huruf kecil. Pemeriksaan keselarasan (deskriptif, dicetak sebelum model): prevalensi tiap pola
   per split dibanding persentase positif label ECG-FM di `label_def.csv` repo ECG-FM (AF, LBBB, RBBB, 1st degree AV
   block); pola tidak diubah apa pun hasilnya.
   (c) **Model:** ECG-FM pretrained (`mimic_iv_ecg_physionet_pretrained.pt`), ECG-FM *fine-tuned*
   (`mimic_iv_ecg_finetuned.pt`, kontrol positif), ECG-JEPA (pembanding). ECGFounder dan model ukuran tidak dipakai.
   Embedding dengan `glukosa_m3/bangun_embedding.py` dan pra-proses yang sama dengan langkah 11, di folder singgahan
   baru di luar repo.
   (d) **Evaluasi:** probe L2 identik langkah 11c (`glukosa_perluasan/silang_lipatan.py`): 5 lipatan per pasien,
   benih 20261120; C dari {10⁻³, 10⁻², 10⁻¹, 1} lewat 3 lipatan dalam per pasien pada subsampel 60.000 unit bagian latih
   lipatan 1, per tugas dan model. DiD AUROC (dilihat − tidak dilihat) model − ECG-JEPA atas prediksi di luar lipatan;
   bootstrap bergugus pasien 2.000 ulangan, benih 20261120, semua model dan tugas pada ulangan yang sama.
   (e) **Hipotesis dan aturan (ditetapkan sekarang).**
   R1 — gerbang kontrol positif: batas bawah 95 % rerata DiD ECG-FM *fine-tuned* empat tugas berlabel (AF, LBBB, RBBB,
   AV blok 1; bobot sama) > 0. Bila gagal, replikasi dinyatakan tidak informatif dan naskah mengatakannya.
   R2 — konfirmatori utama: batas atas 95 % rerata DiD ECG-FM pretrained empat tugas berlabel (skala AUROC) < 0,002.
   Bila R1 dan R2 lolos: klaim "paling besar 0,002 AUROC rata-rata pada pernyataan berlabel" terkonfirmasi. Bila R2
   gagal dengan batas atas < 0,01: batas dilaporkan apa adanya dan klaim 0,002 tidak terkonfirmasi. Bila batas bawah
   > 0: inflasi oleh pra-latih terdeteksi pada replikasi, dan naskah mengatakannya.
   (f) **Sekunder (deskriptif, tanpa vonis tambahan):** rerata lima pernyataan; per tugas (termasuk LVH); skala
   logit(AUROC); tidak dilihat = `test` saja; subset pasien di luar kohort (53.413 rekaman); kepekaan lipatan: seluruh
   evaluasi rerata empat tugas diulang dengan benih lipatan 20261121 dan 20261122 (C dipilih ulang), dilaporkan sebagai
   rentang tiga nilai; keseimbangan kelompok (SMD prevalensi, unit per pasien, denyut jantung, durasi QRS dari
   `machine_measurements.csv`). Tidak ada rasio atau perbandingan besar antara pra-latih dan *fine-tuning*.
   (g) **Proses:** dua skrip di `glukosa_replikasi/`: `siapkan_replikasi.py` (daftar rekaman + tabel unit di luar repo,
   cetak hitungan dan keselarasan saja) dan `jalankan_replikasi.py` (probe, bootstrap, vonis). Keduanya dikomit dan
   diunggah ke Zenodo bersama protokol ini sebelum embedding dibangun; dijalankan sekali; galat teknis diperbaiki
   dan dicatat di berkas hasil. Keluaran stdout hanya agregat (hitungan 1–10 disamarkan).
   **Catatan proses 23 Nov 2026 (sebelum data replikasi disentuh):** pemeriksa gaya kode menolak satu berkas besar,
   sehingga `jalankan_replikasi.py` dipecah menjadi tiga modul dengan isi yang sama: `prediksi_replikasi.py` (unit,
   embedding, prediksi di luar lipatan), `statistik_replikasi.py` (kontras, bootstrap, rerata DiD), dan
   `jalankan_replikasi.py` (cetak dan vonis R1/R2). Diuji asap pada data sintetis acak (6.000 unit buatan, bukan data
   MIMIC) agar galat teknis tertangkap sebelum dijalankan sekali pada data nyata. Komit 4241e8a.
   **Pra-registrasi publik:** atas konfirmasi penulis, rilis GitHub v1.1.0 (komit publik 277bf02) diarsipkan Zenodo
   sebagai versi 1.1.0, DOI 10.5281/zenodo.24452624, 23 Nov 2026 02:13 UTC — sebelum `siapkan_replikasi.py` dijalankan
   (label rekaman replikasi belum dihitung) dan sebelum embedding replikasi dibangun.
   **Hasil 17 (26 Nov 2026, `Hasil_Glukosa_17.md`):** R1 LOLOS (rerata DiD *fine-tuned* empat tugas berlabel 0,002
   [0,001; 0,003] AUROC); R2 LOLOS (pretrained 0,0004 [−0,0004; 0,0012], batas atas < 0,002); inflasi tidak terdeteksi;
   klaim 0,002 terkonfirmasi. TEST saja dan ketiga benih lipatan sejalan. Sekunder deskriptif: pada subset pasien di
   luar kohort kontrol positif tidak terdeteksi di AUROC, dan LVH memberi DiD > 0 untuk kedua model (pretrained 0,026
   [0,005; 0,050]); dilaporkan di naskah tanpa vonis. Jalan pertama selesai menghitung semua prediksi tetapi berkas
   keluarannya tertimpa sebelum dibaca; jalan kedua dari prediksi tersimpan, benih sama, kode tak berubah (dicatat di
   Hasil 17 §1).

16. **Uji keanggotaan pra-latih langsung (dicatat 26 Nov 2026, sesudah telaah internal 8, sebelum kodenya ditulis dan sebelum
   skor kehilangan apa pun dihitung).** Pilihan penulis "Pre-register + run check". Alasan: kritis konsensus telaah internal 8
   (empat dari lima peran): daftar split tidak pernah diuji sebagai daftar pra-latih; bila pra-latih memakai undian
   pasien lain, DiD swaawas mendekati nol karena konstruksi, sedangkan kontrol positif tetap terdeteksi karena label
   *fine-tuning* berasal dari bagian `train` daftar itu. Yang sudah dilihat: semua Hasil 1–17; belum ada skor
   kehilangan pra-latih yang dihitung.
   (a) **Unit:** tabel unit replikasi langkah 15. Satu rekaman per pasien diundi (benih 20261120). Tidak dilihat =
   semua pasien yang rekamannya di `valid` ∪ `test` (22.058 pasien); dilihat = sampel acak pasien `train` sebanyak
   jumlah yang sama (benih 20261120). Rekaman yang gelombangnya gagal dibaca atau skornya tidak terhingga dibuang dan
   dihitung.
   (b) **Skor:** kehilangan InfoNCE wav2vec 2.0 milik ECG-FM pretrained sendiri (`mimic_iv_ecg_physionet_pretrained.pt`
   dengan kuantisator dan proyeksi bawaan, mode eval), pra-proses sama dengan embedding (12 sadapan, standardisasi per
   sadapan atas 10 s, dua segmen 5 s, tanpa penyamaran sadapan), parameter topeng dan negatif dari konfigurasi
   checkpoint (mask_prob 0,65, panjang 10, 100 negatif dari rekaman yang sama). Skor per rekaman = rerata kehilangan
   atas langkah bertopeng kedua segmen dan K = 4 undian topeng dan negatif (benih torch dan numpy 20261120 sekali di
   awal, urutan rekaman tetap).
   (c) **Aturan M1 (ditetapkan sekarang):** D = rerata skor tidak dilihat − rerata skor dilihat; CI 95 % bootstrap
   2.000 ulangan atas pasien, benih 20261120. Bila batas bawah D > 0, daftar terkonfirmasi menandai keanggotaan
   pra-latih, dan batas 0,002 dibaca sebagai batas efek keanggotaan. Bila tidak, keanggotaan tidak terkonfirmasi:
   naskah menyatakan batas itu bersyarat pada daftar yang menandai keanggotaan pra-latih, dengan dua penjelasan yang
   tidak dapat dipisahkan (daftar tidak cocok dengan pra-latih, atau keanggotaan tidak meninggalkan jejak yang
   terdeteksi pada kehilangan), dan judul serta abstrak memuat syarat itu.
   (d) **Kontrol negatif:** pasien tidak dilihat dibagi dua secara acak (benih 20261120); D plasebo antara kedua
   paruh harus berselang yang memuat nol. Bila tidak, uji dinyatakan bermasalah dan M1 tidak dibaca.
   (e) **Sekunder deskriptif:** AUROC keanggotaan dari −skor dengan CI bootstrap yang sama; D terpisah untuk `valid`
   dan `test`; kesesuaian segmen (kosinus rerata token keluaran kedua segmen tanpa topeng, pendekatan untuk suku CMSC)
   dilihat − tidak dilihat.
   (f) **Pasca-hasil deskriptif (atas telaah internal 8, methods X4 dan devil's advocate X6), tanpa vonis:** pada subset pasien
   di luar kohort, celah AUROC dilihat − tidak dilihat untuk LVH dari tiap model (ECG-FM pretrained, *fine-tuned*,
   ECG-JEPA), dari prediksi tersimpan langkah 15, bootstrap dan benih langkah 15.
   (g) **Proses:** skrip di `glukosa_replikasi/` dikomit sebelum dijalankan; diuji asap pada data sintetis; dijalankan
   sekali; stdout agregat saja. Hasil → `Hasil_Glukosa_18.md`.
   **Catatan proses 26 Nov 2026 (sebelum skor apa pun dihitung pada data nyata):** kode dipecah per tugas:
   `skor_keanggotaan.py` (kehilangan InfoNCE dan kosinus segmen per rekaman), `baca_keanggotaan.py` (gelombang dari ZIP,
   urutan tetap), `hitung_keanggotaan.py` (benih, simpanan skor di luar repo), `statistik_keanggotaan.py` (pemilihan
   unit, paruh plasebo, bootstrap), `uji_keanggotaan.py` (cetak dan vonis M1), dan `celah_lvh.py` (butir f). Rincian
   yang protokol tidak sebut: dalam satu panggilan model, K undian kedua segmen satu rekaman diproses bersama (8 deret),
   sehingga pemotongan jumlah langkah bertopeng ke minimum batch fairseq hanya terjadi di dalam rekaman itu; benih
   dipasang sesudah checkpoint dimuat; pasien yang punya rekaman di `train` dan di `valid`/`test` (bila ada) masuk
   kelompok tidak dilihat dengan rekaman `valid`/`test`-nya; rekaman dengan kosinus tak terhingga juga dibuang. Diuji
   asap pada ZIP WFDB sintetis (466 rekaman acak, bukan data MIMIC; dua jalan memberi skor identik) dan pada folder
   replikasi sintetis langkah 15 (`celah_lvh.py`).
   **Hasil 18 (26 Nov 2026, `Hasil_Glukosa_18.md`):** kontrol negatif LOLOS (D plasebo −0,0009 [−0,0101; 0,0083]); **M1
   TIDAK terkonfirmasi** (D = 0,0055 [−0,0008; 0,0122]; AUROC keanggotaan 0,504 [0,499; 0,510]), jadi batas 0,002
   dinyatakan bersyarat pada daftar split dan judul serta abstrak memuat syarat itu. Sekunder deskriptif: kosinus segmen
   dilihat − tidak dilihat 0,0020 [0,0017; 0,0023], satu-satunya jejak keanggotaan, dilaporkan tanpa mengubah M1. Celah
   LVH subset di luar kohort milik ECG-FM sendiri (pretrained 0,030 [0,005; 0,059], ECG-JEPA 0,004 [−0,011; 0,022]).

17. **Kontrol positif swaawas dengan anggota buatan (dicatat 26 Nov 2026, sesudah telaah internal 9, sebelum kodenya ditulis,
   sebelum model apa pun dilatih lanjut, dan sebelum embedding atau skor baru dihitung).** Pilihan penulis
   "Continued-pretraining control". Alasan: konsensus telaah internal 9 — uji kehilangan M1 tidak pernah diuji pada anggota yang
   diketahui, jadi kegagalannya tidak memisahkan "daftar salah" dari "uji lemah" (devil's advocate X2, domain D2,
   methods M1); satu-satunya kontrol positif adalah paparan label pada tugas di plafon, bukan paparan swaawas pada
   tugas berruang (editor X2, methods M8, domain D4, devil's advocate X5). Yang sudah dilihat: semua Hasil 1–18;
   konfigurasi checkpoint pretrained (kriteria `wav2vec2_with_cmsc`, bobot suku tambahan [0,1; 10], temp 0,1,
   perturbasi `random_leads_masking` p 1,0 dengan peluang per sadapan 0,5; Adam lr 5·10⁻⁵ tetap, betas (0,9; 0,98),
   eps 10⁻⁶, peluruhan bobot 0,01; batch 171 × update_freq 2; iterator di epoch 241, 167.280 pembaruan); laju latih pada
   tensor acak di GPU ini (±30 segmen/detik batch 16 fp32, 3,8 GiB; autocast fp16 gagal di `apply_mask`
   fairseq-signals). Belum ada model terlatih lanjut, embedding, prediksi, atau skor baru.
   (a) **Penugasan acak per pasien:** unit = 22.058 pasien tidak dilihat langkah 16 (`valid` ∪ `test`). Disuntik =
   paruh `paruh_b` langkah 16 (`pilih_unit` + `buat_data` di `statistik_keanggotaan.py`, benih 20261120; 11.029
   pasien); ditahan = sisanya. Paruh ini sudah dipakai sebagai plasebo langkah 16 sehingga garis dasar sebelum
   suntikan tercatat (D kehilangan −0,0009 [−0,0101; 0,0083], Hasil 18); pembagiannya acak dan tidak diubah.
   (b) **Latih lanjut (paparan yang diketahui):** mulai dari `mimic_iv_ecg_physionet_pretrained.pt`. Data latih = semua
   rekaman replikasi (tabel unit langkah 15) milik pasien disuntik, ditambah rekaman `train` replikasi sebanyak yang
   sama yang diundi acak (benih 20261120) sebagai putar-ulang, agar model tidak hanya melihat rekaman suntikan.
   Pra-proses sama dengan embedding (12 sadapan, standardisasi per sadapan atas 10 s, dua segmen 5 s). Objektif =
   kriteria checkpoint sendiri: InfoNCE wav2vec 2.0 + suku tambahan berbobot [0,1; 10] + CMSC dengan pasangan positif =
   dua segmen satu rekaman; perturbasi penyamaran sadapan acak dari konfigurasi tugas checkpoint. Pengoptimal seperti
   checkpoint (Adam, lr tetap 5·10⁻⁵, betas, eps, peluruhan bobot). Batch efektif 342 segmen (171 rekaman × 2 segmen)
   lewat akumulasi gradien dari batch mikro 16 rekaman; negatif CMSC berasal dari batch mikro (penyimpangan dari 171
   karena VRAM GPU 8 GB, dicatat di naskah). fp32. Urutan acak per epoch (benih 20261120). **Empat epoch.**
   Checkpoint disimpan sesudah epoch 1 (dosis rendah, ECG-FM+1) dan epoch 4 (dosis utama, ECG-FM+4), di luar repo.
   Bila suatu suku kriteria atau perturbasi tidak dapat dipanggil dari fairseq-signals, penyimpangan itu ditulis di
   rencana ini sebelum latih dimulai, bukan sesudah hasil.
   (c) **Bacaan:** embedding mode eval dengan ECG-FM+1 dan ECG-FM+4 untuk semua 83.385 rekaman `valid` ∪ `test`
   replikasi, pra-proses langkah 15; ECG-FM pretrained asli dan ECG-JEPA dari singgahan langkah 15. Untuk keempat model,
   probe L2 identik langkah 15 dipasang ulang pada 83.385 unit ini saja (5 lipatan per pasien, benih 20261120; C dari
   {10⁻³, 10⁻², 10⁻¹, 1} lewat 3 lipatan dalam per pasien pada bagian latih lipatan 1, per tugas dan model); lima
   pernyataan langkah 15. Celah g(m, t) = AUROC(disuntik) − AUROC(ditahan) dari prediksi di luar lipatan. Efek paparan
   E(t) = g(ECG-FM+4, t) − g(ECG-FM asli, t): model yang sama sebelum suntikan menjadi pembanding. Kehilangan: skor
   langkah 16b (K = 4, topeng dan negatif sama, benih dipasang sesudah model dimuat) dengan ECG-FM+4 pada satu rekaman
   per pasien undian langkah 16; Dₛ = rerata ditahan − rerata disuntik. Bootstrap bergugus pasien 2.000 ulangan, benih
   20261120, semua bacaan pada ulangan yang sama.
   (d) **Aturan (ditetapkan sekarang):**
   N0 — kontrol negatif pengacakan: celah ECG-JEPA, g(ECG-JEPA, ·), pada LVH dan pada rerata lima pernyataan berselang
   memuat nol. Bila tidak, ketidakseimbangan dilaporkan dan K1, P1, P2 dibaca dengan catatan itu.
   K1 — kepekaan uji kehilangan: batas bawah 95 % Dₛ(ECG-FM+4) > 0. Lolos: uji kehilangan langkah 16 mampu
   mendeteksi keanggotaan yang diketahui pada dosis ini; Dₛ dibandingkan dengan batas atas D langkah 16 (0,0122) secara
   deskriptif, dan naskah tidak menyimpulkan daftar salah, karena dosis ini lebih baru tetapi jauh lebih sedikit
   lintasan (4 lawan ±240 epoch). Gagal: uji kehilangan tidak peka bahkan pada keanggotaan yang diketahui dan baru, dan
   naskah menyatakan kegagalan M1 tidak informatif tentang daftar.
   P1 — kontrol positif swaawas pada tugas berruang: batas bawah 95 % E(LVH) > 0. Lolos: desain mendeteksi paparan
   swaawas pada AUROC LVH; E(LVH) dibandingkan deskriptif dengan celah LVH luar kohort (0,030). Gagal: paparan swaawas
   dosis ini tidak menggeser AUROC LVH secara terdeteksi, dan naskah mengatakannya.
   P2 — tugas di plafon: batas bawah 95 % E(rerata empat tugas berlabel) > 0; batas atasnya dilaporkan sebagai besar
   efek swaawas langsung di plafon.
   Tidak ada koreksi multiplisitas: K1, P1, P2 adalah tiga gerbang kalibrasi terpisah, masing-masing dilaporkan.
   (e) **Sekunder deskriptif (tanpa vonis):** semua bacaan untuk ECG-FM+1; E per tugas; E dengan ECG-JEPA sebagai
   pembanding (DiD); AUROC keanggotaan dari −kehilangan; kosinus segmen ECG-FM+4 disuntik − ditahan; Dₛ ECG-FM+1;
   kehilangan latih per epoch (agregat).
   (f) **Proses:** kode di folder baru `glukosa_kontrol/`, dikomit dan diuji asap pada data sintetis sebelum latih;
   latih, embedding, dan skor dijalankan sekali; galat teknis diperbaiki dan dicatat di berkas hasil; stdout hanya
   agregat (hitungan 1–10 disamarkan); checkpoint, embedding, prediksi, dan skor di `data/` di luar repo (ruang
   disk ±24 GB, jadi tidak ada singgahan gelombang penuh). Pra-registrasi publik (versi Zenodo baru) hanya atas
   konfirmasi penulis, sebelum latih dimulai. Hasil → `Hasil_Glukosa_19.md`.
   **Catatan proses 26 Nov 2026 (sebelum kode ditulis, sesudah membaca kode fairseq-signals dan isi checkpoint; belum
   ada latih):** (1) 171 tidak habis dibagi 16, jadi batch mikro menjadi 9 rekaman (18 segmen) × 19 akumulasi = 171
   rekaman / 342 segmen per pembaruan; negatif CMSC dari 9 rekaman. (2) Checkpoint rilis memuat `last_optimizer_state`
   Adam; momen Adam dilanjutkan dari situ (lebih setia daripada Adam baru), langkah peluruhan bobot terpisah seperti
   Adam fairseq. (3) Suhu Gumbel kuantisator dilanjutkan dari pembaruan 167.280 (`set_num_updates`; 0,866, sama dengan
   log latih checkpoint). (4) Gradien dibagi jumlah langkah bertopeng seluruh batch mikro dalam satu pembaruan, seperti
   pelatih fairseq. (5) Penyamaran sadapan per segmen memakai `RandomLeadsMask` fairseq-signals (numpy); pasangan CMSC
   = `idx` rekaman, sama dengan `CMSCECGDataset`; `dependency` hanya dipakai perturbasi 3KG, jadi tidak berlaku. (6)
   Checkpoint asli fp32 (`fp16` False), latih lanjut juga fp32. (7) Benih torch, numpy, dan random 20261120 dipasang
   sekali sesudah model dimuat; urutan tiap epoch dari satu generator numpy benih 20261120. (8) Sisa rekaman di akhir
   epoch menjadi satu pembaruan terakhir yang lebih kecil, seperti batch terakhir fairseq.
   **Catatan proses kode (26 Nov 2026, sebelum latih pada data nyata):** `glukosa_kontrol/` berisi
   `penugasan_kontrol.py` (paruh disuntik, rekaman latih, unit bacaan), `pelatih_lanjut.py` (model, kriteria, Adam
   lanjutan, langkah mikro, simpan/lanjut beserta keadaan acak), `jalankan_latih.py` (epoch, akumulasi, bobot dosis),
   `jalankan_bacaan.py` (embedding dan skor kehilangan per dosis), `statistik_kontrol.py` (probe, celah, E, Dₛ,
   bootstrap bersama), dan `uji_kontrol.py` (tabel dan vonis N0, K1, P1, P2). Kriteria fairseq-signals butuh konfigurasi
   OmegaConf dan kunci `id` di sampel (diisi indeks rekaman, seperti collater). Diuji asap pada ZIP WFDB sintetis dari uji
   asap langkah 16 (466 rekaman acak, bukan data MIMIC) dengan label dan embedding acak: latih empat epoch berjalan;
   jalan yang dihentikan sesudah epoch 2 lalu dilanjutkan memberi kehilangan epoch 3 dan 4 identik dengan jalan utuh;
   bacaan dan vonis berjalan.
   **Pra-registrasi publik:** atas konfirmasi penulis, rilis GitHub v1.2.0 (komit publik 6f2991c; protokol langkah 16–17,
   kode `glukosa_kontrol/`, Hasil 17–18) diarsipkan Zenodo sebagai versi 1.2.0, DOI 10.5281/zenodo.24549186,
   26 Nov 2026 04:22 UTC — sebelum latih lanjut, embedding, atau skor langkah 17 apa pun.
   **Hasil 19 (27 Nov 2026, `Hasil_Glukosa_19.md`):** N0 LOLOS (celah ECG-JEPA LVH −0,001 [−0,010; 0,007], rerata lima
   −0,001 [−0,004; 0,001]). **K1 GAGAL:** Dₛ(ECG-FM+4) 0,0016 [−0,0083; 0,0113], AUROC keanggotaan 0,4999; uji kehilangan
   buta bahkan pada anggota yang diketahui, jadi kegagalan M1 tidak informatif tentang daftar split. **P1 GAGAL:**
   E(LVH) −0,004 [−0,008; −0,001] (arah negatif, deskriptif). **P2 GAGAL:** E(rerata empat berlabel) 0,001 [−0,000;
   0,001]; batas atas 0,001 dilaporkan sebagai efek swaawas langsung di plafon pada dosis 4 epoch (lawan ±240 epoch
   pra-latih asli). Jalan pertama dihentikan pengawas RAM sebelum checkpoint apa pun dan tidak dipakai; jalan yang dipakai
   dimulai ulang dari awal dengan `--pekerja 4` (tidak mengubah urutan atau undian). Penulis 27 Nov memilih membingkai
   ulang naskah sebagai audit keterdeteksian kontaminasi (pilihan "A"), lalu telaah internal 10.
   **Telaah internal 10 (27 Nov):** NOT READY, Reject in current form; signifikansi 2 dari
   kelima peran; editor desk reject. Kritis konsensus: kesepakatan segmen memisahkan anggota daftar (0,0020) dan anggota
   suntikan (0,0004) tetapi plasebonya pada model bersih belum dijalankan; "tidak dapat diaudit" bertumpu pada satu uji
   kehilangan tak berpasangan; tidak ada cek manipulasi suntikan. Penulis 27 Nov memilih pra-registrasi langkah 18.
18. **Audit terkalibrasi: cek manipulasi berpasangan dan plasebo kesesuaian segmen (dicatat 27 Nov 2026, sesudah panel
   10, sebelum kodenya ditulis dan sebelum statistik apa pun di bawah dihitung).** Alasan: telaah internal 10 (methods X1–X4,
   devil's advocate X1–X4, domain X1–X3, editor X1–X2). Yang sudah dilihat: semua Hasil 1–19; dari kesepakatan segmen
   hanya selisih rerata ECG-FM dilihat − tidak dilihat (0,0020 [0,0017; 0,0023], Hasil 18) dan ECG-FM+4 disuntik −
   ditahan (0,0004 [0,0000; 0,0009], Hasil 19); dari kehilangan hanya Dₛ tak berpasangan per model. Belum dihitung:
   selisih berpasangan per pasien antara checkpoint, simpangan baku kesesuaian per kelompok, dan kesesuaian segmen model
   bersih apa pun. Skor per rekaman di `data/replikasi` dan `data/kontrol` tetap tidak dibuka
   di sesi asisten; skrip mencetak agregat saja.
   (a) **Cek manipulasi dan uji kehilangan berpasangan (skor tersimpan, tanpa GPU):** unit = 22.058 pasien tidak
   dilihat langkah 16, satu rekaman per pasien, paruh disuntik/ditahan langkah 17. Per pasien Δℓ = kehilangan
   ECG-FM+4 − kehilangan ECG-FM asli pada rekaman yang sama (skor langkah 16 dan 17; undian topeng berbeda karena urutan
   penilaian berbeda, jadi derau topeng masuk selang). Dₚ = rerata Δℓ ditahan − rerata Δℓ disuntik. **K2:** batas
   bawah 95 % Dₚ > 0 → latih lanjut meninggalkan jejak pada kehilangan pasien disuntik, dan uji yang dikalibrasi pada
   checkpoint sebelum suntikan mendeteksi anggota yang diketahui. Gagal → kehilangan tidak mencatat suntikan sama
   sekali; P1, P2, dan K1 dibaca sebagai bacaan atas paparan yang tidak terlihat pada kehilangan, dan naskah tidak
   menyebut uji kehilangan "buta terhadap anggota yang diketahui" tanpa syarat itu. Kesesuaian segmen berpasangan: Δc =
   kosinus ECG-FM+4 − kosinus ECG-FM asli; Cₚ = rerata Δc disuntik − rerata Δc ditahan. **C2:** batas bawah 95 %
   Cₚ > 0. Deskriptif: AUROC keanggotaan dari −Δℓ dan dari Δc, Dₚ dan Cₚ untuk ECG-FM+1, dan rerata kehilangan latih
   tidak dipisah ulang (log latih tidak memisah suntikan dan putar-ulang).
   (b) **Plasebo kesesuaian segmen pada model bersih (GPU):** unit = 44.116 rekaman langkah 16 (22.058 dilihat, 22.058
   tidak dilihat), satu per pasien, urutan `baris_gelombang`. ECGFounder: pra-proses resmi atas 10 s penuh, lalu dua
   paruh 2.500 sampel masing-masing masuk Net1D terpisah; kesesuaian = kosinus dua `deep_features` 1.024 dimensi.
   ECG-JEPA: masukan resmi 10 s tidak dapat dipotong (posisi tetap 50 tambalan), jadi satu forward pass, lalu
   kesesuaian = kosinus rerata token tambalan 1–25 lawan 26–50 atas delapan sadapan. ECG-FM tidak dihitung ulang (skor
   langkah 16). Per model m, SMD_m = (rerata dilihat − rerata tidak dilihat) / simpangan baku gabungan kedua kelompok
   pada sampel penuh (penyebut tetap, bukan per ulangan). Δ_c = SMD_ECG-FM − SMD_c untuk c ∈ {ECG-JEPA, ECGFounder}.
   **M2:** batas bawah 95 % Δ_c > 0 untuk kedua model bersih → daftar didukung: ECG-FM menunjukkan selisih kesesuaian
   segmen dilihat − tidak dilihat yang tidak ditunjukkan model yang tidak pernah melihat MIMIC-IV-ECG. Gagal untuk salah
   satu → daftar tetap tidak terverifikasi. Deskriptif: SMD_c tiap model bersih dengan selangnya (plasebo mentah),
   SMD ECG-FM+4 disuntik − ditahan dan SMD Δc berpasangan (perbandingan dosis), ECG-FM dengan `valid` dan `test`
   terpisah, serta SMD laju jantung dan durasi QRS dari `machine_measurements` antara kelompok bila sudah ada di tabel
   unit replikasi.
   (c) **Statistik:** bootstrap 2.000 ulangan atas pasien (satu unit = satu pasien), benih 20261120, selang kuantil
   95 %, semua statistik (a) dan (b) pada ulangan bersama masing-masing populasinya. Tanpa koreksi multiplisitas: K2,
   C2, M2 gerbang terpisah.
   (d) **Akibat untuk naskah, ditetapkan sekarang:** M2 lolos → judul dan abstrak menyatakan daftar didukung oleh jejak
   khas ECG-FM dan batas 0,002 dibaca sebagai batas efek keanggotaan dengan syarat jejak itu memang keanggotaan; klaim
   "tidak dapat diaudit" dicabut. M2 gagal → klaim "tidak dapat diaudit dari artefak rilis" dipertahankan tetapi
   disempitkan ke uji yang dicoba (kehilangan tak berpasangan, kehilangan berpasangan bila K2 gagal, kesesuaian
   segmen). K2 menentukan apakah suntikan dibaca sebagai paparan yang terekam (lolos) atau dosis yang tidak terlihat
   (gagal). Uji gabungan semua rekaman per pasien ala inferensi dataset tidak dijalankan (butuh menilai ±400 ribu
   rekaman tambahan) dan disebut sebagai batasan.
   (e) **Proses:** kode di folder baru `glukosa_audit/`, dikomit dan diuji asap pada data sintetis sebelum jalan pada
   data nyata; dijalankan sekali di jendela cmd terpisah; hasil ke `Hasil_Glukosa_20.md`.
   **Catatan proses kode (27 Nov 2026, sebelum statistik apa pun):** `glukosa_audit/` berisi `kesesuaian_bersih.py`
   (pembaca dua model, token paruh ECG-JEPA, dua paruh ECGFounder), `jalankan_kesesuaian.py` (kosinus 18b per rekaman),
   `statistik_audit.py` (populasi berpasangan dan plasebo, Dₚ, Cₚ, SMD penyebut tetap, Δ_c) dan `uji_audit.py` (tabel
   dan vonis K2, C2, M2); komit 6625c48. Diuji asap pada ZIP WFDB sintetis dan skor acak langkah 17 (154 rekaman acak,
   bukan data MIMIC; kovariat acak ditambahkan karena tabel unit sintetis tidak memuatnya): rantai penuh berjalan, dan
   simpanan yang sudah ada tidak dihitung ulang. Kosinus 18b pada data nyata dihitung 27 Nov 14:06–15:09 di jendela cmd
   terpisah (terbaca 44.116 dari 44.116; GPU sempat dibatasi daya saat laptop memakai baterai, tanpa akibat pada
   keluaran). Simpanan itu tidak dibuka dan `uji_audit.py` belum dijalankan. Penulis memilih pra-registrasi publik
   (rilis 1.3.0) sebelum statistik apa pun dihitung.
   **Pra-registrasi publik:** atas konfirmasi penulis, rilis GitHub v1.3.0 (komit publik 8c9fc0e; protokol langkah 18,
   kode `glukosa_audit/`, Hasil 19) diarsipkan Zenodo sebagai versi 1.3.0, DOI 10.5281/zenodo.24582292,
   27 Nov 2026 08:39 UTC — sebelum statistik langkah 18 apa pun dihitung.
   **Hasil 20 (27 Nov 2026, `Hasil_Glukosa_20.md`):** **K2 GAGAL:** Dₚ(ECG-FM+4) 0,0007 [−0,0039; 0,0049]; kehilangan
   tidak mencatat suntikan, jadi suntikan dibaca sebagai dosis yang tidak terlihat pada kehilangan. **C2 GAGAL tipis:**
   Cₚ 0,0002 [−0,0000; 0,0004]. **M2 LOLOS:** SMD kesesuaian segmen dilihat − tidak dilihat ECG-FM 0,129 [0,111; 0,147],
   ECG-JEPA 0,006, ECGFounder 0,025; Δ 0,123 [0,102; 0,145] dan 0,103 [0,082; 0,125]. Menurut 18(d): judul dan abstrak
   menyatakan daftar didukung oleh jejak khas ECG-FM; klaim "tidak dapat diaudit" dicabut; batas 0,002 bersyarat pada
   jejak itu sebagai keanggotaan.
   **Naskah ditulis ulang (6936eab), telaah internal 11 (27 Nov):** NOT READY, Reject in
   current form; signifikansi 2 dari kelima peran (sebelas panel berturut ≤ 2); editor desk reject (satu model, batas
   seukuran kontrol positif di plafon, tugas laboratorium tak terputuskan, kesesuaian segmen = prinsip EncoderMI [17]).
   Kritis konsensus: judul menyatakan 0,002 tanpa syarat dan cakupan; "segment agreement … could" audit melampaui akibat
   18(d), standar ganda karena C2 juga gagal; M2 hanya menunjukkan asosiasi daftar–keanggotaan, bukan kecocokan (efek
   bisa 0,002/(π_s − π_u)). Mayor: M2 pada rekaman yang sisi ECG-FM-nya sudah dilihat (usul: ulang pada EKG kohort),
   plasebo tak sepadan masukan (usul: arsitektur ECG-FM berbobot acak), K2 berderau topeng, "uji berbasis kehilangan"
   terlalu umum, suntikan tanpa cek manipulasi lolos, sinyal LVH luar kohort, kanari dan inferensi dataset [19] tidak
   dijalankan/dikutip, satu model terpapar, struktur glukosa-dulu, Gambar 1. Penulis 27 Nov memilih pra-registrasi
   langkah 19.
19. **Uji ulang jejak pada data segar, plasebo sepadan masukan, inferensi dataset, dan kehilangan berpasangan bertopeng
   sama (dicatat 27 Nov 2026, sesudah telaah internal 11, sebelum kodenya ditulis dan sebelum statistik apa pun di bawah
   dihitung).** Alasan: telaah internal 11 (methods X1–X4, X11; devil's advocate X2–X5, X9; domain X1, X4; editor X5). Yang sudah
   dilihat: semua Hasil 1–20. Belum dihitung: kesesuaian segmen model apa pun pada EKG kohort Glukosa; ECG-FM dengan
   bobot teracak; inferensi dataset; skor kehilangan dengan undian topeng yang sama antar-checkpoint. Embedding kohort
   ECG-FM, ECG-JEPA, ECGFounder sudah ada (langkah 11) tetapi hanya pernah dipakai untuk probe. Skor dan embedding per
   rekaman tetap tidak dibuka di sesi asisten; skrip mencetak agregat saja.
   (a) **Unit segar:** EKG kohort Glukosa (tabel EKG tugas pernyataan langkah 11, yang ada di daftar split) dari pasien
   yang **tidak** termasuk 44.116 pasien uji keanggotaan langkah 16; satu EKG per pasien diundi (benih 20261120). Uji:
   semua pasien tidak dilihat (`valid` ∪ `test`) yang tersisa dan pasien dilihat sama banyak diundi; set penyetelan
   kepadatan (b): pasien dilihat lain sama banyak, terpisah dari uji. Rekaman dan pasien ini belum pernah dinilai untuk
   kesesuaian segmen atau keanggotaan oleh model apa pun. Hitungan dicetak; tidak ada ambang n minimum, efek terdeteksi
   minimum (2,8 × simpangan baku bootstrap) dilaporkan deskriptif.
   (b) **M3 — plasebo kesesuaian segmen, data segar, tiga rujukan:** kesesuaian segmen pada unit uji untuk ECG-FM
   (seperti langkah 16: dua segmen 5 s, rerata token keluaran tanpa topeng, kosinus), ECG-JEPA dan ECGFounder (seperti
   langkah 18b), dan **ECG-FM-acak**: arsitektur, pra-proses, dan pengukuran ECG-FM yang persis sama, dengan isi setiap
   tensor bobot checkpoint pretrained diacak urutannya (permutasi per tensor, benih 20261120), sehingga sebaran nilai
   bobot sama tetapi struktur hasil latih hilang. SMD dan Δ_c seperti 18b (penyebut tetap). **M3:** batas bawah 95 %
   Δ_c > 0 untuk **ketiga** rujukan (ECG-JEPA, ECGFounder, ECG-FM-acak).
   (c) **D1 — inferensi dataset ala [19]:** per model (ECG-FM, ECG-JEPA, ECGFounder), dari embedding kohort tersimpan
   langkah 11: standarisasi dan PCA 64 komponen disetel pada set penyetelan, lalu GaussianMixture 10 komponen kovarians
   penuh (benih 20261120) pada set penyetelan; skor = log-likelihood per rekaman uji. SMD dilihat − tidak dilihat
   (penyebut tetap) dan Δ_DI = SMD_ECG-FM − SMD_c. **D1:** batas bawah 95 % Δ_DI > 0 untuk ECG-JEPA dan ECGFounder.
   Kepadatan tidak disetel ulang per ulangan bootstrap.
   (d) **K3 — kehilangan berpasangan dengan topeng sama:** 22.058 pasien tidak dilihat langkah 17 (paruh
   disuntik/ditahan), rekaman yang sama. ECG-FM asli dan ECG-FM+4 dinilai ulang dengan skor kehilangan langkah 16b
   (K = 4), benih torch dan numpy dipasang per rekaman (20261120 + nomor urut rekaman) sebelum tiap penilaian, sehingga
   undian topeng dan negatif sama di kedua checkpoint. Δℓ dan Dₚ seperti 18a. **K3:** batas bawah 95 % Dₚ > 0.
   Deskriptif: AUROC keanggotaan dari −Δℓ; korelasi Δℓ lama (18a) dan baru tidak dihitung (skor lama memakai topeng
   lain). Cek determinisme sebelum jalan nyata: checkpoint asli dinilai dua kali pada data sintetis memberi skor identik.
   (e) **Deskriptif tambahan tanpa vonis:** daya pisah per rekaman jejak (AUROC keanggotaan dari kosinus dan TPR pada
   FPR 1 %) untuk ECG-FM dan tiap rujukan, pada data segar dan pada 44.116 rekaman langkah 16/18; SMD ECG-FM dengan
   `valid` dan `test` terpisah; SMD umur, jenis kelamin, laju jantung, dan durasi QRS antarkelompok uji.
   (f) **Statistik:** bootstrap 2.000 ulangan atas pasien, benih 20261120, selang kuantil 95 %; populasi uji segar
   (b, c, e) dan populasi 22.058 pasien (d) masing-masing dengan ulangan bersama. M3, D1, K3 gerbang terpisah tanpa
   koreksi multiplisitas.
   (g) **Akibat untuk naskah, ditetapkan sekarang:** M3 lolos → dukungan daftar direplikasi pada rekaman dan pasien
   segar dan melawan plasebo sepadan masukan; naskah menyatakan daftar didukung, dengan batas 0,002 dinyatakan beserta
   cakupannya (rerata empat pernyataan berlabel) dan syaratnya (jejak = keanggotaan), dan kata "could audit" tidak
   dipakai. M3 gagal (untuk satu rujukan saja pun) → M2 dilaporkan tidak direplikasi; judul dan abstrak tidak lagi
   menyatakan daftar didukung; batas dinyatakan bersyarat pada daftar; sumbangan menjadi audit negatif: baik kehilangan
   maupun kesesuaian segmen tidak memverifikasi daftar secara terulang. D1 lolos → uji kedua yang terpisah (inferensi
   dataset) mendukung daftar; gagal → dilaporkan sebagai uji yang tidak mendeteksi keanggotaan, tanpa membatalkan M3. K3
   lolos → kehilangan mencatat suntikan bila derau topeng dihapus; klaim "kehilangan tidak dapat mengaudit" disempitkan
   menjadi "uji kehilangan satu rekaman tak berpasangan kurang daya". K3 gagal → pernyataan bahwa kehilangan bertopeng
   ini tidak mencatat empat epoch suntikan dipertahankan, dibatasi pada kehilangan ini (bukan semua uji berbasis
   kehilangan atau MIA terkalibrasi). Judul dalam semua kasus menyebut cakupan batas.
   (h) **Proses:** kode di folder baru `glukosa_audit2/`, dikomit dan diuji asap pada data sintetis sebelum jalan nyata;
   protokol dan kode dirilis publik (versi 1.4.0, atas konfirmasi penulis) sebelum statistik apa pun dihitung;
   dijalankan sekali di jendela cmd terpisah dengan pengisi daya terpasang; hasil ke `Hasil_Glukosa_21.md`.
   **Proses (27 Nov):** kode dikomit (bd72e55) dan diuji asap pada data sintetis (ZIP WFDB acak, kohort dan embedding
   acak; ECG-FM+4 sintetis = bobot asli sehingga Dₚ K3 tepat 0; cek determinisme identik; kontrol positif inferensi
   dataset memberi log-likelihood dilihat lebih tinggi); satu jalan asap pertama berhenti dengan access violation dan
   tidak berulang pada tiga jalan berikutnya (skrip melewati simpanan yang sudah ada, jadi jalan nyata dapat diulang).
   Rilis publik versi 1.4.0 atas "go" penulis, doi 10.5281/zenodo.24583858, sebelum bacaan atau statistik apa pun
   langkah ini dihitung.
   **Hasil 21 (27 Nov 2026, `Hasil_Glukosa_21.md`):** dijalankan sekali 16:48–19:03 WIB, tanpa ulang. **M3 GAGAL:**
   pada 20.214 EKG segar, SMD kesesuaian segmen ECG-FM 0,088 dan ECG-JEPA 0,089; Δ_c terhadap ECG-JEPA −0,002 [−0,033;
   0,031], terhadap ECGFounder 0,131 [0,099; 0,162], terhadap ECG-FM-acak 0,100 [0,065; 0,136]. **D1 GAGAL:** Δ_DI 0,003
   [−0,030; 0,039] dan 0,004 [−0,031; 0,039]; ketiga model memberi SMD log-likelihood ±−0,14 (dilihat lebih rendah).
   **K3 LOLOS:** Dₚ 0,0017 [0,0006; 0,0028], AUROC 0,51. Deskriptif: SMD umur segar 0,212, QRS 0,111. Akibat menurut
   19(g): M2 tidak direplikasi; judul dan abstrak tidak menyatakan daftar didukung; batas 0,002 bersyarat pada daftar;
   sumbangan = audit negatif; klaim kehilangan disempitkan menjadi "uji satu rekaman tak berpasangan kurang daya".
   Naskah ditulis ulang per 19(g) pada 27 Nov malam (judul baru disetujui penulis; 2.13 dan 3.10 + Tabel 8 baru);
   belum ditelaah panel. **Panel 12 (27 Nov malam): NOT READY, Reject,
   signifikansi 2; kritis konsensus: unit segar 19a tak sebanding menurut konstruksi (lihat langkah 20).**

20. **Uji ulang 19b–c dengan kelayakan sama di kedua kelompok (dicatat 27 Nov 2026 malam, sesudah Hasil 21 dan panel
   12, sebelum kodenya ditulis; PASCA-HASIL — aturan ditetapkan sebelum statistik langkah ini dihitung, tetapi sesudah
   hasil M3 dan D1 diketahui, jadi tidak konfirmatori).** Alasan: langkah 16a mengambil **semua** pasien tidak dilihat
   di tabel unit replikasi (pasien yang punya rekaman di daftar split di luar kohort), sedangkan pasien dilihat hanya
   diundi sebagian. Langkah 19a mengecualikan pasien langkah 16, sehingga tidak dilihat segar praktis hanya pasien
   tanpa rekaman luar kohort, sedangkan dilihat segar campuran. Asimetri ini ada di kode (`unit_segar.py`) dan tidak
   terlihat saat langkah 19 dicatat; telaah internal 12 (metode, domain, devil's advocate) menemukannya. Kondisi itu cocok dengan
   SMD umur 0,212 dan SMD log-likelihood −0,14 yang sama di ketiga model. Tidak ada rekaman atau model baru; hanya
   simpanan langkah 19 dan embedding kohort langkah 11 yang dibaca ulang, skrip mencetak agregat saja.
   (a) **Kelayakan:** pasien layak = `subject_id` tidak ada di `unit_replikasi.csv` (tidak punya rekaman di daftar split
   di luar kohort). Diterapkan sama pada dilihat dan tidak dilihat di unit uji 19a, dan pada set penyetelan. Hitungan
   layak/tak layak per kelompok dicetak.
   (b) **Cek keseimbangan (gerbang tafsir):** SMD umur, jenis kelamin, laju jantung, dan durasi QRS dilihat − tidak
   dilihat pada unit uji layak. Sebanding bila |SMD| < 0,1 untuk keempatnya. Deskriptif: SMD kovariat yang sama antara
   pasien dilihat tak layak dan dilihat layak (menguji mekanisme seleksi secara langsung).
   (c) **M3\*:** statistik 19b persis sama (SMD penyebut tetap, Δ_c terhadap ECG-JEPA, ECGFounder, ECG-FM-acak) pada
   unit uji layak. Lolos bila batas bawah 95 % Δ_c > 0 untuk ketiga rujukan.
   (d) **D1\*:** statistik 19c persis sama, kepadatan disetel ulang sekali pada set penyetelan layak. Lolos bila batas
   bawah 95 % Δ_DI > 0 untuk ECG-JEPA dan ECGFounder.
   (e) **Statistik:** bootstrap 2.000 ulangan atas pasien, benih 20261120, selang kuantil 95 %; efek terdeteksi minimum
   2,8 × simpangan baku bootstrap; daya pisah per rekaman seperti 19e. K3 tidak diulang (tidak terkena seleksi ini).
   (f) **Akibat untuk naskah, ditetapkan sekarang:** dalam semua kasus naskah menyebut mekanisme seleksi 19a dan bahwa
   M3/D1 langkah 19 membandingkan kelompok tak sebanding, sehingga kegagalan keduanya tidak lagi dibaca sebagai bukti
   bahwa jejak M2 bukan keanggotaan. Tidak sebanding (|SMD| ≥ 0,1 untuk salah satu kovariat) → uji ulang segar
   dinyatakan tak dapat ditafsirkan; daftar tidak terverifikasi dan tidak terbantah; batas 0,002 tetap bersyarat pada
   daftar. Sebanding dan M3\* lolos → jejak kesesuaian segmen khas ECG-FM muncul lagi pada pasien segar yang sebanding
   (pasca-hasil, bukan konfirmatori); naskah boleh menyatakan bukti mendukung daftar dengan label itu, judul baru perlu
   persetujuan penulis. Sebanding dan M3\* gagal → pembacaan negatif 19(g) dipertahankan, kini pada kelompok
   sebanding. D1\* dilaporkan di samping M3\* tanpa membatalkannya.
   (g) **Proses:** kode `glukosa_audit2/uji_kelayakan_sama.py`, dikomit bersama protokol ini sebelum dijalankan; satu
   kali jalan; hasil ke `Hasil_Glukosa_22.md`. Rilis publik berikutnya hanya atas perintah penulis.
   **Hasil 22 (27 Nov 2026 malam, `Hasil_Glukosa_22.md`, kode 4938570, satu jalan):** 5.400 dari 10.107 pasien
   dilihat segar tak layak, tidak dilihat 0; dilihat tak layak lebih tua (SMD umur 0,397). Unit layak 4.707 dilihat
   lawan 10.107 tidak dilihat, **sebanding** (|SMD| ≤ 0,023). **M3\* LOLOS:** SMD ECG-FM 0,081, ECG-JEPA 0,011,
   ECGFounder 0,010, ECG-FM-acak −0,003; Δ_c 0,070 [0,030; 0,108], 0,071 [0,030; 0,111], 0,084 [0,040; 0,126].
   **D1\* GAGAL** (pergeseran −0,14 Hasil 21 hilang: efek seleksi). Akibat menurut 20(f): jejak khas ECG-FM muncul lagi
   pada pasien segar sebanding (pasca-hasil); naskah menyebut mekanisme seleksi 19a; judul baru menunggu persetujuan
   penulis.
   **Panel 13 (28 Nov 2026), atas naskah ditulis ulang per 20(f) (4f6583e):** NOT
   READY, Reject in current form, signifikansi 2 (semua peran), editor desk reject. Kritis: batas 0,002 perlu
   dinyatakan sebagai fungsi kemurnian daftar. Penyebab penolakan tetap sama sejak telaah internal 11: tugas di plafon dan satu
   model; perbaikan teks tidak menggeser signifikansi (pola 27 Sep – 16 Okt). Keputusan arah ada pada penulis.
   **28 Nov: penulis memilih arah 1 (tugas lepas plafon) → langkah 21.**

21. **Tugas lepas plafon (dicatat 28 Nov 2026, sebelum model mana pun dinilai pada tugas ini).** Menjawab keberatan
   editor telaah internal 11–13: batas 0,002 hanya berlaku di empat tugas berplafon, tempat fine-tuning label pun hampir tak
   terukur. Uji konfirmasi LVH pada pasien baru tidak mungkin: seluruh pasien daftar split sudah dipakai di kohort atau
   replikasi, dan subset pasien-terpisah sudah dibaca (Tabel 3 baris 7).
   (a) **Tugas:** label ECG-FM fine-tuned (label_def.csv) yang belum dinilai, pola di
   `glukosa_lepas_plafon/tugas_lepas_plafon.py`. Kesetiaan pola diperiksa lebih dulu dengan hitungan agregat tanpa
   model (`siapkan_lepas_plafon.py`, 28 Nov): selisih terbesar prevalensi pola lawan label per bagian split infark
   0,25 poin persen, PVC 0,29, flutter atrium 0,14, pacu elektronik 0,28, mutu data buruk 0,38, irama sinus 0,53;
   ditolak takikardia 14,61, bradikardia 2,44, jalur aksesori 10,78, SVT aberan 2,84. Ambang setia 1,5 poin persen
   (selisih terbesar yang sudah diterima di 2.2, AV blok derajat 1, 1,2, dibulatkan ke atas) ditetapkan sesudah hitungan
   itu terlihat; dicatat terus terang. Tidak diikutkan: takikardia ventrikel (0,09 %), AV blok dan blok bifasikular
   (memuat tugas lama). Positif tugas baru yang juga positif tugas lama: 24–54 % (pacu 23,7, flutter 30,9, sinus
   25,1, mutu 36,5, infark 44,8, PVC 54,3).
   (b) **Data dan model:** unit replikasi langkah 15 (416.503 rekaman); embedding tersimpan ECG-JEPA, ECG-FM, ECG-FM
   fine-tuned; probe L2, C dipilih seperti 2.3, lima lipatan pasien benih 20261120; bootstrap 2.000 bergugus pasien
   benih 20261120, semua tugas pada ulangan yang sama. ECGFounder tidak ada di replikasi.
   (c) **Keluarga utama (lepas plafon):** tugas setia dengan AUROC ECG-JEPA di luar lipatan atas semua unit ≤ 0,95,
   dihitung dalam jalan yang sama sebelum DiD dibaca. Keluarga kosong → langkah tidak informatif, H21 tidak diuji.
   (d) **G21 (gerbang kontrol positif):** batas bawah rerata DiD ECG-FM fine-tuned atas keluarga > 0. Gagal → H21
   tidak ditafsirkan.
   (e) **H21:** rerata DiD ECG-FM pretrained atas keluarga dengan aturan 2.7 (ε 0,01): *didukung* bila batas bawah > 0,
   *di bawah margin* bila batas atas < 0,01, *tak terputuskan* selain itu. Batas atasnya dilaporkan sebagai batas lepas
   plafon, dibandingkan dengan DiD kontrol positif.
   (f) **Sekunder deskriptif, tanpa vonis:** per tugas, skala logit, tidak dilihat = `test` saja, pasien di luar
   kohort, dan kemurnian: selisih laju keanggotaan sejati antarkelompok di bawah mana batas atas melewati 0,01 dan
   0,005 (DiD teramati = efek × selisih itu).
   (g) **Aturan berhenti:** langkah 21 adalah analisis terakhir atas efek keanggotaan dan atas daftar split di naskah
   ini. Apa pun hasilnya, tidak ada tugas, aturan kelayakan, atau pembacaan ulang baru; naskah melaporkan hasil ini dan
   menyebut aturan berhenti ini.
   (h) **Akibat untuk naskah:** G21 lolos dan H21 di bawah margin → klaim "tidak ada inflasi terukur" meluas ke tugas
   lepas plafon dengan kontrol positif yang bekerja, dan batas lepas plafon masuk abstrak. G21 lolos dan H21 didukung →
   inflasi terdeteksi pada tugas lepas plafon; judul dijawab "ya" untuk tugas itu, batas 0,002 dibatasi ke tugas
   berplafon. G21 lolos dan tak terputuskan → dilaporkan apa adanya tanpa klaim baru. G21 gagal → tidak informatif.
   Batas dinyatakan sebagai fungsi kemurnian daftar di semua kasus (temuan kritis telaah internal 13).
   (i) **Proses:** protokol ini dan kodenya dikomit bersama sebelum jalan. Rilis publik sebelum jalan (konfirmatori)
   hanya atas perintah penulis; tanpa rilis, statusnya *pre-specified*. Satu kali jalan, hasil ke
   `Hasil_Glukosa_23.md`.
   Rilis publik versi 1.5.0 atas pilihan penulis (28 Nov), doi 10.5281/zenodo.24599381, sebelum model mana pun dinilai
   pada tugas langkah ini; status konfirmatori.
   **Hasil 23 (28 Nov 2026, `Hasil_Glukosa_23.md`, kode 5fd2dcc, satu jalan):** keluarga lepas plafon infark (AUROC
   ECG-JEPA 0,856) dan mutu data buruk (0,819). **G21 LOLOS:** DiD fine-tuned 0,0257 [0,0194; 0,0319], sekitar sepuluh
   kali efek label di tugas berplafon. **H21 di bawah margin:** DiD pretrained −0,0001 [−0,0077; 0,0078]. Batas atas
   melewati 0,01 bila selisih laju keanggotaan sejati < 0,776. Sekunder: `test` saja 0,0020 [−0,0087; 0,0127], pasien di
   luar kohort 0,0206 [−0,0040; 0,0453] (keduanya tak terputuskan). Akibat 21(h): klaim nol meluas ke tugas lepas
   plafon dengan kontrol positif yang bekerja; naskah ditulis ulang; aturan berhenti berlaku.
   **Naskah ditulis ulang per 21(h) (28 Nov):** abstrak memuat batas 0,0012 (plafon) dan 0,008 (lepas plafon) lawan
   efek label 0,026, syarat daftar tepat dan ambang kemurnian 0,78; Hasil disusun menurut klaim (3.1–3.8 DiD, 3.9–3.12
   daftar, 3.13 injeksi); 3.6 dan Tabel 6 baru; 3.7 subset pasien-terpisah dengan standar sama seperti M3\*; 2.6 memuat
   tally enam aturan daftar dan aturan berhenti; pustaka kanari, LiRA, blind baseline ditambahkan di 4.5 ([77]–[79]).
   Temuan telaah internal 13 yang menuntut analisis keanggotaan baru tidak dikerjakan karena 21(g); dicatat di 4.6.
   **telaah internal 14 (28 Nov): kritis: kemurnian daftar tak berbatas bawah.** Penulis memilih opsi 1 (revisi teks) +
   opsi 2 (surel ke pengembang ECG-FM). Opsi 1: kesimpulan
   bersyarat "no inflation above 0.0012 / 0.008 detected, assuming an exact list"; kontras per tugas (infark tak
   terpisah); 3.9–3.13 dipindah ke Suplemen Bagian S4, diganti 3.9/3.10 + Tabel 7 ringkasan; Gambar 1 baru dua panel
   (Gambar 1 lama = S2); M3\* setara subset; ε 0,02/0,005; pustaka [80]–[82]. Uji berpasangan dua paparan dan uji
   heterogenitas subset tidak ditambah (21(g)). Opsi 2: draf `Surat_ECGFM_Manifes.md`, dikirim penulis; manifes yang
   datang hanya dibandingkan dengan daftar (hitungan cocok), bukan analisis keanggotaan baru, kecuali penulis
   memutuskan lain di sini.
   **telaah internal 15 (28 Nov): kritis: batas diperluas ke tolok ukur lain [3] padahal subset luar kohort 0,045.** Revisi
   putaran 2 (a203957): perluasan dibuang, 0,045 dinyatakan (Tabel S21), batas per tugas, Metode 2.10–2.14 ke S5,
   judul pilihan penulis. Penerima surel diisi dari README repo ECG-FM (28 Nov); penulis yang mengirim.
   **telaah internal 16 (28 Nov): NOT READY, Major revision, editor send_to_review, median signifikansi 2 (tiga panel
   berturut), tanpa kritis.** Konsensus menuntut uji berpasangan infark, heterogenitas subset, keseimbangan M2/subset
   dan selang valid − test (methods X1/X4, devil's advocate X6/X7): semuanya dilarang 21(g); membukanya keputusan
   penulis yang ditulis di sini, berlabel pasca-hasil deskriptif dari keluaran tersimpan.
