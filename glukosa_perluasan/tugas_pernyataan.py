"""Tugas pernyataan mesin protokol §F langkah 11b: satu baris per EKG ber-embedding, label dari machine_measurements.csv
MIMIC-IV-ECG v1.0 (berkas terbuka). Pola ditetapkan dari hitungan agregat (hitung_pernyataan.py) sebelum model dinilai."""
import zipfile

import pandas as pd

from muat import DEMOGRAFI, UKURAN_EKG

BERKAS_UKURAN = "mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/machine_measurements.csv"
KOLOM_LAPORAN = [f"report_{i}" for i in range(18)]
POLA_TUGAS = {
    "af": r"atrial fibrillation",
    "lbbb": r"left bundle branch block|\blbbb\b",
    "rbbb": r"right bundle branch block|\brbbb\b",
    "avb1": r"(?:1st|first) degree a-?v block",
    "lvh": r"left ventricular hypertrophy|\blvh\b",
}
NAMA_TUGAS = {"af": "Fibrilasi atrium", "lbbb": "LBBB", "rbbb": "RBBB", "avb1": "AV blok derajat 1", "lvh": "LVH"}
LABEL_FINETUNED = {"af", "lbbb", "rbbb", "avb1"}
KOLOM_EKG = ["subject_id", "baris_gelombang", "waveform_path", "split_ecgfm", *DEMOGRAFI, *UKURAN_EKG]


def tabel_ekg(pasangan):
    ekg = pasangan.drop_duplicates("baris_gelombang").sort_values("baris_gelombang")[KOLOM_EKG]
    ekg = ekg.reset_index(drop=True)
    ekg["study_id"] = ekg.waveform_path.str.rsplit("/", n=1).str[1].astype(int)
    return ekg


def tambah_label(ekg, arsip):
    with zipfile.ZipFile(arsip) as zip_ecg, zip_ecg.open(BERKAS_UKURAN) as berkas:
        laporan = pd.read_csv(berkas, usecols=["study_id"] + KOLOM_LAPORAN, dtype=str)
    teks = laporan.set_index(laporan.study_id.astype(int))[KOLOM_LAPORAN].fillna("").agg(" ".join, axis=1).str.lower()
    teks_ekg = ekg.study_id.map(teks).fillna("")
    for kode, pola in POLA_TUGAS.items():
        ekg[f"y_{kode}"] = teks_ekg.str.contains(pola, regex=True).astype(int).values
    return ekg


def bagian_tugas(ekg, kode):
    bagian = ekg.copy()
    bagian["y"] = ekg[f"y_{kode}"].values
    return bagian
