"""Kontras dan statistik replikasi Glukosa (protokol langkah 15): AUROC dilihat lawan tidak dilihat per tugas, ulangan
bootstrap bergugus pasien yang dipakai bersama oleh semua tugas, dan rerata DiD (model − ECG-JEPA) pada skala AUROC atau
logit dengan selang kuantil 95 %."""
from dataclasses import dataclass

import numpy as np

from kontras_berbobot import auroc_per_kelompok, buat_kontras, did_dari, logit
from kontras_perluasan import KUANTIL, ulangan_bersama
from model_fondasi import ECGJEPA
from prediksi_replikasi import MODEL

BENIH_BOOT = 20261120
SKALA = {"AUROC": lambda x: x, "logit": logit}


@dataclass
class Sampel:
    kontras: dict
    titik: list
    ulangan: list


def sampel(konteks, peluang, pilih, daftar_tugas):
    bagian = konteks.unit[pilih].reset_index(drop=True)
    kontras = {}
    for kode in daftar_tugas:
        tugas = bagian.assign(y=bagian[f"y_{kode}"].values)
        kontras[kode] = buat_kontras(tugas, {m.kode: peluang[kode][m.kode][pilih] for m in MODEL})
    daftar = list(kontras.values())
    titik = [auroc_per_kelompok(k, np.ones(len(k.y))) for k in daftar]
    ulangan = ulangan_bersama(daftar, konteks.boot, np.random.default_rng(BENIH_BOOT))
    return Sampel(kontras, titik, ulangan)


def statistik(sampel_, fungsi):
    nilai = fungsi(sampel_.titik)
    bawah, atas = np.quantile([fungsi(u) for u in sampel_.ulangan], KUANTIL)
    return nilai, bawah, atas


def rerata_did(anggota, semua_tugas, model, skala):
    indeks = [semua_tugas.index(k) for k in anggota]
    return lambda auroc: np.mean([did_dari(auroc[i], model, ECGJEPA.kode, SKALA[skala]) for i in indeks])
