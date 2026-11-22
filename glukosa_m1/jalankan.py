"""M0 (usia + jenis kelamin) dan M1 (ukuran mesin EKG + usia + jenis kelamin) untuk artikel Glukosa.

Rancangan: Artikel_Glukosa_Rencana.md §A dan §C2 (disetujui penulis 20 Nov 2026). Data credentialed
MIMIC-IV-Ext-ECG-Glucose v1.0.0 di data/mimic-iv-ext-ecg-glucose/ (di luar repo). Dijalankan di laptop;
keluarannya hanya agregat dan disalin apa adanya ke Hasil_Glukosa_1.md.

Analisis utama: latih di split TRAIN, nilai di split VALIDATION (pasangan non-EXCLUDE). Split TEST tidak
disentuh; dipakai sekali di akhir untuk semua model. Kepekaan kendali obat: validasi silang 5 lipatan
bergugus pasien atas subset ICU (non-EXCLUDE, dan ICU + konservatif), gabungan ketiga split.

Pakai (dari akar repo; tanpa -I agar modul saudara di folder ini terbaca):
    python glukosa_m1/jalankan.py [--boot 1000] [--folder data/mimic-iv-ext-ecg-glucose]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

from analisis import analisis_tiga_kelas, analisis_utama, cetak_koefisien, cetak_ukuran_hilang, kepekaan_obat_icu
from muat import BERKAS, TUGAS_BINER, muat_pasangan
from pemodelan import BENIH
from ukuran_kinerja import Bootstrap


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    argumen = baca_argumen()
    mulai = time.time()
    pasangan = muat_pasangan(Path(argumen.folder) / BERKAS)
    bootstrap = Bootstrap(argumen.boot, np.random.default_rng(BENIH[0]))
    print(f"# M0/M1 — bootstrap {argumen.boot} bergugus pasien\n")
    cetak_ukuran_hilang(pasangan)
    for tugas in TUGAS_BINER:
        analisis_utama(pasangan, tugas, bootstrap)
    analisis_tiga_kelas(pasangan, bootstrap)
    for tugas in TUGAS_BINER:
        kepekaan_obat_icu(pasangan, tugas, bootstrap)
    cetak_koefisien(pasangan)
    print(f"\nWaktu jalan: {(time.time() - mulai) / 60:.1f} menit.")


def baca_argumen():
    pengurai = argparse.ArgumentParser()
    pengurai.add_argument("--boot", type=int, default=1000)
    pengurai.add_argument("--folder", default="data/mimic-iv-ext-ecg-glucose")
    return pengurai.parse_args()


if __name__ == "__main__":
    main()
