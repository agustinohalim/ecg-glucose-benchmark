"""Perbandingan kendali obat yang adil di ICU untuk artikel Glukosa (Hasil_Glukosa_2.md).

Subset: pasangan ICU dengan EKG sebelum glukosa dan SQI GOOD/FAIR, semua status obat, gabungan ketiga split.
Satu model per tugas (validasi silang bergugus pasien), lalu AUROC dihitung terpisah per intervention_status,
sehingga syarat mutu dan urutan waktu sama dan yang berbeda hanya status obat. record_usability tidak dipakai
karena label itu sendiri bergantung pada status obat.

Pakai (dari akar repo; tanpa -I agar modul saudara di folder ini terbaca):
    python glukosa_m1/jalankan_obat.py [--boot 1000] [--folder data/mimic-iv-ext-ecg-glucose]
"""
import sys
import time
from pathlib import Path

import numpy as np

from analisis_obat import kendali_obat_setara
from jalankan import baca_argumen
from muat import BERKAS, TUGAS_BINER, muat_pasangan
from pemodelan import BENIH
from ukuran_kinerja import Bootstrap


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    argumen = baca_argumen()
    mulai = time.time()
    pasangan = muat_pasangan(Path(argumen.folder) / BERKAS)
    bootstrap = Bootstrap(argumen.boot, np.random.default_rng(BENIH[0]))
    print(f"# Kendali obat setara (ICU) — bootstrap {argumen.boot} bergugus pasien\n")
    for tugas in TUGAS_BINER:
        kendali_obat_setara(pasangan, tugas, bootstrap)
    print(f"Waktu jalan: {(time.time() - mulai) / 60:.1f} menit.")


if __name__ == "__main__":
    main()
