"""Kelayakan langkah 20 Glukosa (protokol Artikel_Glukosa_Rencana.md §F langkah 20a): pasien layak = tanpa rekaman di
daftar split di luar kohort (`subject_id` tidak ada di unit_replikasi.csv), diterapkan sama pada kedua kelompok."""
import numpy as np
import pandas as pd

from statistik_audit import Populasi


def tandai_layak(unit, berkas_replikasi):
    pasien_luar = pd.read_csv(berkas_replikasi, usecols=["subject_id"]).subject_id.unique()
    return unit.assign(layak=~unit.subject_id.isin(pasien_luar))


def unit_untuk_muat(unit):
    return unit[unit.peran.eq("uji") | unit.layak]


def rekaman_layak(pop, unit):
    return np.isin(pop.nilai["baris_gelombang"].astype(np.int64), unit.baris_gelombang[unit.layak])


def bagian(pop, pilih):
    return Populasi({k: v[pilih] for k, v in pop.nilai.items()}, pop.kelompok[pilih], pop.split[pilih], pop.dibuang)


def dilihat_menurut_kelayakan(pop, layak):
    dilihat = bagian(pop, pop.kelompok)
    return Populasi(dilihat.nilai, ~layak[pop.kelompok], dilihat.split, dilihat.dibuang)
