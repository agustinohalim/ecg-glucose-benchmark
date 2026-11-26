"""Penugasan kontrol positif swaawas Glukosa (protokol §F langkah 17a–b): pasien disuntik = paruh `paruh_b` langkah 16 di
antara pasien tidak dilihat (undian `pilih_unit` + `buat_data` yang sama, benih 20261120); rekaman latih lanjut = semua
rekaman replikasi milik pasien disuntik ditambah rekaman `train` sama banyak sebagai putar-ulang; unit bacaan = semua
rekaman `valid` ∪ `test` replikasi dengan tanda disuntik. Pemanggil memasang sys.path ke folder glukosa_* lain."""
from pathlib import Path

import numpy as np
import pandas as pd

from statistik_keanggotaan import BENIH, DILIHAT, buat_data, pilih_unit

FOLDER_REPLIKASI = Path("data/replikasi")
FOLDER_KONTROL = Path("data/kontrol")
BERKAS_UNIT = "unit_replikasi.csv"
KOLOM_PILIH = ["subject_id", "waveform_path", "split_ecgfm", "baris_gelombang"]


def muat_unit(folder=FOLDER_REPLIKASI):
    return pd.read_csv(Path(folder) / BERKAS_UNIT)


def pilihan_satu_per_pasien(unit):
    pilihan = pilih_unit(unit[KOLOM_PILIH])
    paruh_b = buat_data(pilihan.assign(kehilangan=np.nan, kosinus=np.nan)).paruh_b
    return pilihan.assign(disuntik=paruh_b)


def pasien_disuntik(pilihan):
    return frozenset(pilihan.subject_id[pilihan.disuntik])


def unit_bacaan(unit, disuntik):
    bagian = unit[unit.split_ecgfm.ne(DILIHAT)].sort_values("baris_gelombang").reset_index(drop=True)
    return bagian.assign(disuntik=bagian.subject_id.isin(disuntik))


def rekaman_latih(unit, disuntik):
    suntikan = unit[unit.subject_id.isin(disuntik)]
    latih = unit[unit.split_ecgfm.eq(DILIHAT)]
    pilih = np.sort(np.random.default_rng(BENIH).choice(len(latih), len(suntikan), replace=False))
    gabungan = pd.concat([suntikan.assign(disuntik=True), latih.iloc[pilih].assign(disuntik=False)])
    return gabungan.sort_values("baris_gelombang").reset_index(drop=True)
