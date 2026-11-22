"""Membangun singgahan gelombang M2 dari ZIP MIMIC-IV-ECG v1.0 untuk semua `waveform_path` unik di
MIMIC-IV-Ext-ECG-Glucose v1.0.0 (Artikel_Glukosa_Rencana.md §F langkah 5). ZIP tidak diekstrak.

Keluaran di luar repo (folder --keluar): gelombang_250hz.npy (n × 8 × 2500, float16 µV), gelombang_status.npy,
gelombang_jalur.txt. Bila terputus, jalankan ulang perintah yang sama; baris yang sudah selesai dilewati. Yang dicetak
hanya hitungan agregat (1–10 dicetak "<11").

Pakai (dari akar repo, tanpa -I agar modul saudara terbaca):
    python glukosa_m2/bangun_singgahan.py [--zip data/mimic-iv-ecg-1.0.zip]
        [--csv data/mimic-iv-ext-ecg-glucose/mimiciv_ecg_glucose_aligned.csv]
        [--keluar data/mimic-iv-ext-ecg-glucose/gelombang] [--pekerja 8]
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from singgahan import BELUM, BERKAS_SINYAL, BERKAS_STATUS, NAMA_STATUS, Potongan, kerjakan_potongan, siapkan_berkas
from zip_wfdb import indeks_anggota, nama_anggota

UKURAN_POTONGAN = 2000
SEL_KECIL = 11


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    argumen = baca_argumen()
    mulai = time.time()
    jalur_rekaman = sorted(pd.read_csv(argumen.csv, usecols=["waveform_path"]).waveform_path.dropna().unique())
    keluar = Path(argumen.keluar)
    status = siapkan_berkas(keluar, jalur_rekaman)
    sisa = np.flatnonzero(status == BELUM)
    print(f"Rekaman unik: {len(jalur_rekaman)}; belum diolah: {len(sisa)}.", flush=True)
    if len(sisa):
        indeks = indeks_anggota(argumen.zip, [jalur_rekaman[i] for i in sisa])
        print(f"Indeks ZIP siap ({(time.time() - mulai) / 60:.1f} menit).", flush=True)
        olah_semua(susun_potongan(argumen, jalur_rekaman, sisa, indeks), keluar, status, argumen.pekerja)
    cetak_ringkasan(status)
    print(f"\nWaktu jalan: {(time.time() - mulai) / 60:.1f} menit.")


def baca_argumen():
    pengurai = argparse.ArgumentParser()
    pengurai.add_argument("--zip", default="data/mimic-iv-ecg-1.0.zip")
    pengurai.add_argument("--csv", default="data/mimic-iv-ext-ecg-glucose/mimiciv_ecg_glucose_aligned.csv")
    pengurai.add_argument("--keluar", default="data/mimic-iv-ext-ecg-glucose/gelombang")
    pengurai.add_argument("--pekerja", type=int, default=8)
    return pengurai.parse_args()


def susun_potongan(argumen, jalur_rekaman, sisa, indeks):
    jalur_sinyal = str(Path(argumen.keluar) / BERKAS_SINYAL)
    tugas = [(int(i), jalur_rekaman[i], lokasi_rekaman(indeks, jalur_rekaman[i])) for i in sisa]
    return [Potongan(argumen.zip, jalur_sinyal, tugas[k:k + UKURAN_POTONGAN])
            for k in range(0, len(tugas), UKURAN_POTONGAN)]


def lokasi_rekaman(indeks, jalur):
    nama = [nama_anggota(jalur, akhiran) for akhiran in ("hea", "dat")]
    return {n: indeks[n] for n in nama if n in indeks}


def olah_semua(daftar_potongan, keluar, status, pekerja):
    with ProcessPoolExecutor(pekerja) as kolam:
        tugas = [kolam.submit(kerjakan_potongan, p) for p in daftar_potongan]
        for selesai, hasil in enumerate(as_completed(tugas), 1):
            for baris, nilai in hasil.result():
                status[baris] = nilai
            np.save(keluar / BERKAS_STATUS, status)
            if selesai % 10 == 0 or selesai == len(tugas):
                print(f"{time.strftime('%H:%M')} potongan {selesai}/{len(tugas)}", flush=True)


def cetak_ringkasan(status):
    print("\n| Status | Rekaman |\n| --- | --- |")
    for kode, nama in NAMA_STATUS.items():
        n = int((status == kode).sum())
        print(f"| {nama} | {f'<{SEL_KECIL}' if 0 < n < SEL_KECIL else n} |")
    if (status == BELUM).any():
        print(f"| belum diolah | {int((status == BELUM).sum())} |")


if __name__ == "__main__":
    main()
