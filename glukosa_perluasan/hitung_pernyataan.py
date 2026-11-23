"""Frekuensi pernyataan mesin MIMIC-IV-ECG v1.0 (berkas terbuka machine_measurements.csv) pada EKG yang punya embedding,
untuk menetapkan pola label tugas pernyataan protokol §F langkah 11 sebelum model apa pun dinilai. Mencetak agregat
saja: hitungan per pernyataan (1–10 dicetak <11) dan hitungan per pola kandidat.

Dari akar repo:
    python glukosa_perluasan/hitung_pernyataan.py
"""
import argparse
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

BERKAS_UKURAN = "mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/machine_measurements.csv"
KOLOM_LAPORAN = [f"report_{i}" for i in range(18)]
AMBANG_SEL_KECIL = 11
POLA_KANDIDAT = {
    "fibrilasi atrium": r"atrial fibrillation",
    "flutter atrium": r"atrial flutter",
    "LBBB": r"left bundle branch block|\blbbb\b",
    "RBBB": r"right bundle branch block|\brbbb\b",
    "AV blok derajat 1": r"(1st|first) degree a-?v block",
    "LVH": r"left ventricular hypertrophy|\blvh\b",
    "sinus bradikardia": r"sinus bradycardia",
    "sinus takikardia": r"sinus tachycardia",
    "QT memanjang": r"prolonged qt",
    "infark lama": r"infarct",
    "irama pacu": r"pac(ed|ing|emaker)",
}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    argumen = baca_argumen()
    laporan = muat_laporan(argumen.zip, studi_terpakai(argumen.jalur))
    print(f"EKG dengan embedding: {len(laporan)}\n")
    cetak_pola(laporan)
    cetak_pernyataan_teratas(laporan, argumen.teratas)


def baca_argumen():
    pengurai = argparse.ArgumentParser()
    pengurai.add_argument("--zip", default="data/mimic-iv-ecg-1.0.zip")
    pengurai.add_argument("--jalur", default="data/mimic-iv-ext-ecg-glucose/gelombang/gelombang_jalur.txt")
    pengurai.add_argument("--teratas", type=int, default=60)
    return pengurai.parse_args()


def studi_terpakai(jalur):
    return {int(baris.rsplit("/", 1)[1]) for baris in Path(jalur).read_text().split()}


def muat_laporan(arsip, studi):
    with zipfile.ZipFile(arsip) as zip_ecg, zip_ecg.open(BERKAS_UKURAN) as berkas:
        ukuran = pd.read_csv(berkas, usecols=["study_id"] + KOLOM_LAPORAN, dtype=str)
    ukuran = ukuran[ukuran.study_id.astype(int).isin(studi)]
    return ukuran.set_index("study_id")[KOLOM_LAPORAN]


def tampil(hitungan):
    return "<11" if 0 < hitungan < AMBANG_SEL_KECIL else str(hitungan)


def cetak_pola(laporan):
    teks = laporan.fillna("").agg(" ".join, axis=1).str.lower()
    print("| Pola kandidat | EKG cocok |\n| --- | --- |")
    for nama, pola in POLA_KANDIDAT.items():
        print(f"| {nama} (`{pola}`) | {tampil(int(teks.str.contains(pola, regex=True).sum()))} |")
    print()


def cetak_pernyataan_teratas(laporan, teratas):
    pernyataan = laporan.stack().str.strip().str.lower()
    hitungan = pernyataan[pernyataan.ne("")].value_counts().head(teratas)
    print("| Pernyataan | EKG |\n| --- | --- |")
    for teks, jumlah in hitungan.items():
        print(f"| {re.sub(r'[|]', '/', teks)} | {tampil(int(jumlah))} |")


if __name__ == "__main__":
    main()
