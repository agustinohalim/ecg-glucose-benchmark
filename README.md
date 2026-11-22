# ecg-glucose-benchmark

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
  **Version 1.0.0 is released before this test was run**; `glukosa_test/jalankan_kontaminasi.py` is the script that
  will be run once, and its output will be added as `notes/Hasil_Glukosa_11.md` in a later version.

Analyses recorded after a protocol (`Hasil_Glukosa_9.md`, `Hasil_Glukosa_10.md`, and the ICU analysis in
`Hasil_Glukosa_2.md`) are marked as descriptive in their notes.

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
  `splits/meta_split_mimic_iv_ecg.csv` from `bowang-lab/ECG-FM` commit 3ff744ef, both in `models/ECG-FM/`.

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
  embeddings (`bangun_embedding.py --model ecgfounder|ecgjepa|ecgfm`), probe (`probe.py`), M3 and M3b on VALIDATION
  (`jalankan_m3.py`, `Hasil_5` and `Hasil_7`), check against stored predictions (`bandingkan_simpanan.py`).
- `glukosa_test/` — the TEST run (`jalankan_test.py`, `Hasil_8`), descriptive subgroups (`jalankan_subkelompok.py`,
  `Hasil_9`), descriptive within-patient, care-setting and timing analyses (`jalankan_kepekaan_panel.py`, `Hasil_10`),
  and the contamination test (`jalankan_kontaminasi.py`, not yet run in 1.0.0).
- `notes/` — the working notes `Hasil_Glukosa_0.md` to `Hasil_Glukosa_10.md`, in which every number of the manuscript
  was recorded with the copied script output.
- `plan/` — the excerpt of the analysis plan described above.

Comments, notes and the plan are in Indonesian. Glossary: *hipo* hypoglycemia, *hiper berat* severe hyperglycemia,
*pasangan* ECG–glucose pair, *pasien* patient, *latih* train, *nilai* evaluate, *selisih* difference, *benih* seed,
*bergugus pasien* patient-clustered, *singgahan* cache, *gelombang* waveform, *sadapan* lead, *penanam* embedder,
*model fondasi* foundation model, *paparan* exposure, *dilihat* / *tidak dilihat* seen / unseen in pretraining,
*kepekaan* sensitivity analysis, *subkelompok* subgroup, *kalibrasi ulang* recalibration, *ukuran mesin* machine
measurements, *rawat* care, *keputusan penulis* author decision, *telaah internal* internal review. Notes refer to an
earlier study of the same authors (the "OMI" study) whose CNN recipe and probe grid were reused; its code is public at
https://github.com/agustinohalim/omi-parent-label-baseline. Notes also refer to planning files that are not part of
this repository.

## Licence

Code: MIT. Notes and plan: CC BY 4.0. Aggregate results derived from MIMIC-IV-Ext-ECG-Glucose and MIMIC-IV-ECG; cite
both datasets and PhysioNet when using them.
