"""Unit replikasi konfirmatori Glukosa menurut protokol Artikel_Glukosa_Rencana.md §F langkah 15 (komit d24c095):
rekaman di daftar split ECG-FM yang jalurnya tidak ada di kohort MIMIC-IV-Ext-ECG-Glucose, dengan label lima pernyataan
mesin (pola identik langkah 11b), denyut jantung dan durasi QRS dari machine_measurements.csv, serta tanda pasien di
luar kohort. Menulis gelombang_jalur.txt (masukan glukosa_m3/bangun_embedding.py) dan unit_replikasi.csv ke folder
di luar repo. Yang dicetak hanya hitungan per split, prevalensi pola, dan keselarasan dengan label_def.csv ECG-FM.

Pakai venv-gpu (cetak memuat scikit-learn), dari akar repo:
    python glukosa_replikasi/siapkan_replikasi.py [--keluaran data/replikasi]
"""
import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

AKAR_EKG = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(AKAR_EKG / d) for d in ("glukosa_perluasan", "glukosa_m2", "glukosa_m1")]
from cetak import tabel, tampil_hitungan  # noqa: E402
from singgahan import BERKAS_JALUR  # noqa: E402
from tugas_pernyataan import BERKAS_UKURAN, NAMA_TUGAS, POLA_TUGAS, tambah_label  # noqa: E402

AWALAN = "mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/"
BERKAS_UNIT = "unit_replikasi.csv"
LABEL_ECGFM = {"af": "Atrial fibrillation", "lbbb": "Left bundle branch block", "rbbb": "Right bundle branch block",
               "avb1": "1st degree atrioventricular block"}
SPLIT = ("train", "valid", "test")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    argumen = baca_argumen()
    unit = susun_unit(argumen)
    unit = tambah_label(unit, argumen.zip)
    unit = tambah_ukuran(unit, argumen.zip)
    folder = Path(argumen.keluaran)
    folder.mkdir(parents=True, exist_ok=True)
    folder.joinpath(BERKAS_JALUR).write_text("\n".join(unit.waveform_path) + "\n", encoding="ascii")
    unit.to_csv(folder / BERKAS_UNIT, index=False)
    cetak_ringkasan(unit, pd.read_csv(argumen.label_def).set_index("name"))


def baca_argumen():
    pengurai = argparse.ArgumentParser()
    pengurai.add_argument("--zip", default="data/mimic-iv-ecg-1.0.zip")
    pengurai.add_argument("--split", default="models/ECG-FM/meta_split_mimic_iv_ecg.csv")
    pengurai.add_argument("--label_def", default="models/ECG-FM/label_def_mimic_iv_ecg.csv")
    pengurai.add_argument("--kohort", default="data/mimic-iv-ext-ecg-glucose/mimiciv_ecg_glucose_aligned.csv")
    pengurai.add_argument("--keluaran", default="data/replikasi")
    return pengurai.parse_args()


def susun_unit(argumen):
    with zipfile.ZipFile(argumen.zip) as arsip, arsip.open(AWALAN + "record_list.csv") as berkas:
        rekaman = pd.read_csv(berkas, usecols=["subject_id", "study_id", "path"])
    split = pd.read_csv(argumen.split, usecols=["source_path", "split"]).set_index("source_path").split
    kohort = pd.read_csv(argumen.kohort, usecols=["subject_id", "waveform_path"])
    rekaman["split_ecgfm"] = rekaman.path.map(split)
    unit = rekaman[rekaman.split_ecgfm.notna() & ~rekaman.path.isin(set(kohort.waveform_path))]
    unit = unit.rename(columns={"path": "waveform_path"}).sort_values("waveform_path").reset_index(drop=True)
    unit["lepas_pasien"] = ~unit.subject_id.isin(set(kohort.subject_id))
    unit["baris_gelombang"] = np.arange(len(unit))
    return unit


def tambah_ukuran(unit, arsip):
    kolom = ["study_id", "rr_interval", "qrs_onset", "qrs_end"]
    with zipfile.ZipFile(arsip) as zip_ecg, zip_ecg.open(BERKAS_UKURAN) as berkas:
        ukuran = pd.read_csv(berkas, usecols=kolom).drop_duplicates("study_id").set_index("study_id")
    rr = ukuran.rr_interval.where(ukuran.rr_interval.between(200, 3000))
    qrs = (ukuran.qrs_end - ukuran.qrs_onset).where(lambda d: d.between(40, 250))
    unit["denyut"] = unit.study_id.map(60000 / rr).values
    unit["qrs_ms"] = unit.study_id.map(qrs).values
    return unit


def cetak_ringkasan(unit, label_def):
    print(f"\nUnit replikasi: {len(unit)} rekaman, {unit.subject_id.nunique()} pasien; pasien di luar kohort "
          f"{int(unit.lepas_pasien.sum())} rekaman, {unit.subject_id[unit.lepas_pasien].nunique()} pasien.\n")
    tabel(["Split", "Rekaman", "Pasien", "Rekaman pasien di luar kohort"], [baris_split(unit, s) for s in SPLIT])
    print("\n## Prevalensi pola dan label ECG-FM (label_def.csv, persen positif per split)\n")
    tabel(["Tugas", "Split", "Positif", "Prevalensi pola", "Label ECG-FM"],
          [baris_prevalensi(unit, label_def, kode, s) for kode in POLA_TUGAS for s in SPLIT])


def baris_split(unit, split):
    bagian = unit[unit.split_ecgfm == split]
    return [split, tampil_hitungan(len(bagian)), tampil_hitungan(bagian.subject_id.nunique()),
            tampil_hitungan(bagian.lepas_pasien.sum())]


def baris_prevalensi(unit, label_def, kode, split):
    bagian = unit[unit.split_ecgfm == split]
    label = LABEL_ECGFM.get(kode)
    acuan = f"{label_def.loc[label, f'pos_percent_{split}']:.4f}" if label else "—"
    return [NAMA_TUGAS[kode], split, tampil_hitungan(bagian[f"y_{kode}"].sum()), f"{bagian[f'y_{kode}'].mean():.4f}",
            acuan]


if __name__ == "__main__":
    main()
