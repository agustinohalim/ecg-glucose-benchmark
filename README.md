# ecg-glucose-benchmark

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24433533.svg)](https://doi.org/10.5281/zenodo.24433533)

Analysis code, pre-specified protocols and aggregate results for a study of hypoglycemia and severe hyperglycemia
detection from inpatient 12-lead ECGs in MIMIC-IV-Ext-ECG-Glucose v1.0.0 (manuscript in preparation).

The study compares, on the dataset's patient-level split, a demographic baseline (M0: age and sex), the ECG machine
measurements (M1: logistic regression and gradient boosting), a 1-D CNN trained from scratch on the waveforms (M2,
mean of three seeds), and frozen ECG foundation models with an L2 logistic probe: ECGFounder (M3) and ECG-JEPA (M3b),
whose pretraining data do not include MIMIC-IV-ECG, and ECG-FM (M3c), which was pretrained on the `train` part of
the ECG-FM authors' patient split of MIMIC-IV-ECG and is used to measure the effect of pretraining exposure.

**No data from the datasets are included.** The scripts read the releases locally and print aggregate tables only;
counts between 1 and 10 are printed as "<11", and metrics are suppressed when a group has fewer than 11 positives.
Per-pair predictions, waveform caches and embeddings are written outside the repository and are not released.
MIMIC-IV-Ext-ECG-Glucose is available to credentialed PhysioNet users who sign its data use agreement
(https://doi.org/10.13026/nts2-vj76); the waveforms come from MIMIC-IV-ECG v1.0 (https://doi.org/10.13026/4nqg-sb35).

## Pre-specification

Two protocols were recorded in dated commits of the authors' working repository before the data they govern were
looked at; both are reproduced in `plan/Artikel_Glukosa_Rencana_petikan.md`:

- **TEST protocol** (plan step 7, 21 November 2026): one primary comparison per task, M3b − M1-GB AUROC on TEST
  non-EXCLUDE pairs with a patient-cluster bootstrap. TEST was opened once (`notes/Hasil_Glukosa_8.md`).
- **Contamination protocol** (plan step 9, 22 November 2026): difference-in-differences of AUROC between pairs whose
  ECG was in ECG-FM's pretraining split and pairs whose ECG was not, M3c against the clean M3b, on VALIDATION ∪ TEST.
  Version 1.0.0 was released before this test was run; its output is `notes/Hasil_Glukosa_11.md`.
- **Extension** (plan step 11, 22 November 2026, recorded before the fine-tuned embeddings were built and before the
  statement tasks were scored): five machine-statement tasks on all 376,648 cohort ECGs with patient-level
  cross-fitting, and ECG-FM fine-tuned on the `train` labels as a positive control (`notes/Hasil_Glukosa_13.md`).
- **Confirmatory replication** (plan step 15, 23 November 2026): the same design on the 416,503 recordings in ECG-FM's
  split list that are not in the glucose cohort, with two rules fixed in advance — R1, the lower 95 % bound of the
  fine-tuned model's mean difference-in-differences over the four labelled statements is above 0; R2, the upper 95 %
  bound of the pretrained model's is below 0.002 AUROC. Version 1.1.0 (doi 10.5281/zenodo.24452624) was released
  before the labels of these recordings were computed and before any of their embeddings was built. The replication
  was run once; both rules passed (`notes/Hasil_Glukosa_17.md`).
- **Direct membership check** (plan step 16, 26 November 2026, recorded in a dated private commit before its code was
  written and before any loss was scored): ECG-FM's own wav2vec 2.0 contrastive loss on one recording per patient,
  22,058 `valid` ∪ `test` patients against as many `train` patients, rule M1 (lower 95 % bound of unseen − seen > 0)
  with a random-halves negative control. Run once; the negative control passed and M1 was not met
  (`notes/Hasil_Glukosa_18.md`).
- **Continued-pretraining positive control** (plan step 17, 26 November 2026): ECG-FM's self-supervised pretraining
  is continued for four epochs, with its own criterion and optimizer state, on all recordings of a random half of the
  22,058 unseen patients (the negative-control halves of step 16) plus as many `train` recordings, creating known
  members. Rules fixed in advance: N0, the comparator's seen − unseen gap between the halves contains 0; K1, the lower
  95 % bound of the loss difference (held out − injected) of the continued model is above 0; P1, the lower 95 % bound
  of the injected − held-out AUROC gap of the continued model minus that of the original model is above 0 for LVH; P2,
  the same for the mean of the four labelled statements. Version 1.2.0 (doi 10.5281/zenodo.24549186) was released
  before any training, embedding or score of this step. Run once: N0 passed; K1, P1 and P2 were not met
  (`notes/Hasil_Glukosa_19.md`). The note also records a deviation found after the run: continued pretraining used 171
  recordings per update against 1,026 in the original pretraining.
- **Calibrated audit** (plan step 18, 27 November 2026, recorded in a dated private commit before its code was written
  and before any of its statistics was computed): (a) a paired manipulation check on the 22,058 unseen patients, using
  the per-recording scores already stored by steps 16 and 17 — per patient, the loss and segment agreement of the
  continued model minus those of the original model on the same recording; rule K2, the lower 95 % bound of the mean
  loss change of held-out minus injected patients is above 0; rule C2, the lower 95 % bound of the mean
  segment-agreement change of injected minus held-out patients is above 0; (b) a segment-agreement placebo on the two
  models that never saw MIMIC-IV-ECG, on the 44,116 recordings of step 16 (ECGFounder: cosine of the embeddings of the
  two 5 s halves; ECG-JEPA: cosine of the mean tokens of patches 1–25 and 26–50); rule M2, the lower 95 % bound of
  the seen − unseen standardised mean difference of ECG-FM minus that of each clean model is above 0, for both. The
  consequences of each outcome for the manuscript are fixed in the plan (step 18d). **Version 1.3.0 is released before
  any statistic of this step is computed**; the per-recording cosines of the clean models were computed after the code
  was committed and have not been read. `glukosa_audit/` holds the scripts; the verdict script will be run once and its
  output added as a note in a later version.

Analyses recorded after the results they respond to (`Hasil_Glukosa_9.md`, `Hasil_Glukosa_10.md`, `Hasil_Glukosa_12.md`,
`Hasil_Glukosa_14.md` to `Hasil_Glukosa_16.md`, plan steps 10, 12, 13 and 14, the per-model LVH gaps of step 16f in
`Hasil_Glukosa_18.md`, and the ICU analysis in `Hasil_Glukosa_2.md`) are marked as post hoc and descriptive in their
notes.

## Requirements

- Python 3.12.10 with NumPy 2.5.4, pandas 2.3.3, SciPy 1.18.1, scikit-learn 1.9.1 and PyTorch 2.14.0 (CUDA 13.0 build;
  M2 and the embeddings were run on an 8 GB laptop GPU) — `requirements.txt`. WFDB records are read from the
  MIMIC-IV-ECG ZIP by the reader in `glukosa_m2/zip_wfdb.py`, not by the wfdb package.
- ECGFounder: code `NickLJLee/ECGFounder` commit 67367071 in `ext/ECGFounder`, weights `12_lead_ECGFounder.pth`
  (`PKUDigitalHealth/ECGFounder`) in `models/ECGFounder/`.
- ECG-JEPA: code `sehunfromdaegu/ECG_JEPA` commit d937ad2 in `ext/ECG_JEPA`, weights `multiblock_epoch100.pth`
  (linked from that README) in `models/ECG_JEPA/`. `ext/timm_pengganti/` replaces the two `timm` functions ECG-JEPA
  imports, so that timm and torchvision are not needed; both are inactive or overwritten at inference.
- ECG-FM (M3c only): `Jwoo5/fairseq-signals` commit 6ffdec72 installed into a separate environment with omegaconf
  2.3.1, hydra-core 1.3.8, wfdb 5.0.0 (imported by fairseq-signals) and transformers 4.57.6 (< 5); weights
  `mimic_iv_ecg_physionet_pretrained.pt` from Hugging Face `wanglab/ecg-fm` revision 584219ea (SHA-256
  4d0142bcb485eb9f0c7845e0c19ff3463f6ae9d0e458eab69136efe90ceb9b7e) and the split list
  `splits/meta_split_mimic_iv_ecg.csv` from `bowang-lab/ECG-FM` commit 3ff744ef, both in `models/ECG-FM/`. The
  positive control uses `mimic_iv_ecg_finetuned.pt` from the same Hugging Face revision; the replication also reads
  `splits/label_def_mimic_iv_ecg.csv` from the same commit.

Default paths are relative to the repository root: `data/mimic-iv-ext-ecg-glucose/` (the extracted dataset release),
`data/mimic-iv-ecg-1.0.zip` (the MIMIC-IV-ECG ZIP, not extracted), `models/`, `ext/`. Each script also takes the paths
as options. Run from the repository root without `-I`, because the scripts import each other; set
`PYTHONIOENCODING=utf-8` when redirecting output. Seeds are fixed in `glukosa_m1/pemodelan.py` (`BENIH`); CNN training
on GPU is not bit-reproducible (VALIDATION AUROC moved by up to 0.015 between runs, see the notes).

## Layout and order

- `glukosa_hitung_label.py` — label counts per split (`notes/Hasil_Glukosa_0.md`).
- `glukosa_m1/` — loading, leakage-safe columns, M0 and M1, metrics and bootstrap (`Hasil_1`); ICU medication
  sensitivity by patient-grouped cross-validation (`jalankan_obat.py`, `Hasil_2`).
- `glukosa_m2/` — ZIP reader and waveform cache (`zip_wfdb.py`, `bangun_singgahan.py`), CNN (`jaringan.py`,
  `jalankan_m2.py`, `Hasil_3`), paired differences (`jalankan_selisih.py`, `Hasil_4`), three-seed mean
  (`jalankan_m2_benih.py`, `Hasil_6`).
- `glukosa_m3/` — foundation-model loaders (`ecgfounder.py`, `ecgjepa.py`, `ecgfm.py`, listed in `model_fondasi.py`),
  embeddings (`bangun_embedding.py --model ecgfounder|ecgjepa|ecgfm|ecgfm_ft`), probe (`probe.py`), M3 and M3b on VALIDATION
  (`jalankan_m3.py`, `Hasil_5` and `Hasil_7`), check against stored predictions (`bandingkan_simpanan.py`).
- `glukosa_test/` — the TEST run (`jalankan_test.py`, `Hasil_8`), descriptive subgroups (`jalankan_subkelompok.py`,
  `Hasil_9`), descriptive within-patient, care-setting and timing analyses (`jalankan_kepekaan_panel.py`, `Hasil_10`),
  the contamination test (`jalankan_kontaminasi.py`, `Hasil_11`) and its post hoc additions
  (`jalankan_kontaminasi_tambahan.py`, `Hasil_12`).
- `glukosa_perluasan/` — statement labels (`tugas_pernyataan.py`), cross-fitting (`silang_lipatan.py`), the extension
  (`jalankan_perluasan.py`, `Hasil_13`) and its post hoc analyses (`jalankan_tambahan_perluasan.py`,
  `jalankan_kontras_langsung.py`, `jalankan_rasio_label.py`; `Hasil_14` to `Hasil_16`).
- `glukosa_replikasi/` — the confirmatory replication: unit table and labels (`siapkan_replikasi.py`), then, after the
  three embeddings are built with `glukosa_m3/bangun_embedding.py --singgahan <replication folder>`, the analysis
  (`jalankan_replikasi.py`, with `prediksi_replikasi.py` and `statistik_replikasi.py`; `Hasil_17`); the membership
  check (`uji_keanggotaan.py`, with `skor_keanggotaan.py`, `baca_keanggotaan.py`, `hitung_keanggotaan.py` and
  `statistik_keanggotaan.py`) and the per-model LVH gaps (`celah_lvh.py`; both `Hasil_18`); the figure of the
  manuscript (`gambar_replikasi.py`).
- `glukosa_kontrol/` — the continued-pretraining positive control: training (`jalankan_latih.py`, with
  `penugasan_kontrol.py` and `pelatih_lanjut.py`), readings with the continued weights (`jalankan_bacaan.py`), and the
  verdicts (`uji_kontrol.py`, with `statistik_kontrol.py`; `Hasil_19`).
- `glukosa_audit/` — the calibrated audit: segment agreement of the clean models (`jalankan_kesesuaian.py`, with
  `kesesuaian_bersih.py`), then the verdicts (`uji_audit.py`, with `statistik_audit.py`). Not yet run in 1.3.0.
- `notes/` — the working notes `Hasil_Glukosa_0.md` to `Hasil_Glukosa_19.md`, in which every number of the manuscript
  was recorded with the copied script output.
- `plan/` — the excerpt of the analysis plan described above.

Comments, notes and the plan are in Indonesian. Glossary: *hipo* hypoglycemia, *hiper berat* severe hyperglycemia,
*pasangan* ECG–glucose pair, *pasien* patient, *latih* train, *nilai* evaluate, *selisih* difference, *benih* seed,
*bergugus pasien* patient-clustered, *singgahan* cache, *gelombang* waveform, *sadapan* lead, *penanam* embedder,
*model fondasi* foundation model, *paparan* exposure, *dilihat* / *tidak dilihat* seen / unseen in pretraining,
*kepekaan* sensitivity analysis, *pernyataan* machine statement, *lipatan* fold, *perluasan* extension, *replikasi*
replication, *kontrol positif* positive control, *swaawas* self-supervised, *keanggotaan* membership, *kehilangan*
loss, *latih lanjut* continued pretraining, *disuntik* / *ditahan* injected / held out, *kesesuaian segmen* segment
agreement, *model bersih* clean model, *plasebo* placebo, *berpasangan* paired, *subkelompok* subgroup, *kalibrasi ulang* recalibration, *ukuran mesin* machine
measurements, *rawat* care, *keputusan penulis* author decision, *telaah internal* internal review. Notes refer to an
earlier study of the same authors (the "OMI" study) whose CNN recipe and probe grid were reused; its code is public at
https://github.com/agustinohalim/omi-parent-label-baseline. Notes also refer to planning files that are not part of
this repository.

## Licence

Code: MIT. Notes and plan: CC BY 4.0. Aggregate results derived from MIMIC-IV-Ext-ECG-Glucose and MIMIC-IV-ECG; cite
both datasets and PhysioNet when using them.
