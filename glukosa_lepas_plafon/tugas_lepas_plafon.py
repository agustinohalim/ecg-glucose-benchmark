"""Tugas pernyataan baru langkah 21 Glukosa (protokol Artikel_Glukosa_Rencana.md §F langkah 21): label ECG-FM fine-tuned
yang belum dinilai, dengan pola atas pernyataan mesin machine_measurements.csv MIMIC-IV-ECG v1.0. Pola ditetapkan
sebelum model mana pun dinilai pada tugas ini."""
import zipfile

import pandas as pd

BERKAS_UKURAN = "mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/machine_measurements.csv"
KOLOM_LAPORAN = [f"report_{i}" for i in range(18)]
POLA_BARU = {
    "infark": r"infarct",
    "pvc": r"\bpvcs?\b|premature ventricular|ventricular premature|ventricular extrasystole",
    "flutter": r"atrial flutter",
    "pacu": r"\bpac(?:ed|ing|emaker)",
    "mutu": r"data quality|unsuitable for analysis|lead reversal|artifact|noise",
    "takikardia": r"tachycardia",
    "bradikardia": r"bradycardia",
    "sinus": r"\bsinus\b",
    "jalur_aksesori": r"short pr|pre-?excitation|\bwpw\b|wolff",
    "svt_aberan": r"aberran",
}
LABEL_ECGFM_BARU = {
    "infark": "Infarction", "pvc": "Premature ventricular contraction", "flutter": "Atrial flutter",
    "pacu": "Electronic pacemaker", "mutu": "Poor data quality", "takikardia": "Tachycardia",
    "bradikardia": "Bradycardia", "sinus": "Sinus rhythm", "jalur_aksesori": "Accessory pathway conduction",
    "svt_aberan": "Supraventricular tachycardia with aberrancy",
}
TUGAS_LAMA = ("af", "lbbb", "rbbb", "avb1", "lvh")


def teks_laporan(arsip):
    with zipfile.ZipFile(arsip) as zip_ecg, zip_ecg.open(BERKAS_UKURAN) as berkas:
        laporan = pd.read_csv(berkas, usecols=["study_id"] + KOLOM_LAPORAN, dtype=str)
    return laporan.set_index(laporan.study_id.astype(int))[KOLOM_LAPORAN].fillna("").agg(" ".join, axis=1).str.lower()


def tambah_label_baru(unit, arsip):
    teks = unit.study_id.map(teks_laporan(arsip)).fillna("")
    label = {f"y_{kode}": teks.str.contains(pola, regex=True).astype(int).values for kode, pola in POLA_BARU.items()}
    return unit.assign(**label)
