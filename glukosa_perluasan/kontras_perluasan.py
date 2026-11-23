"""Hitungan kontras protokol §F langkah 11d–e: ulangan bootstrap bergugus pasien yang dipakai bersama oleh beberapa tugas
atas unit yang sama, sebaran nol permutasi paparan per pasien, selang kuantil, dan vonis menurut margin ε."""
import numpy as np

from cetak import SEL_KECIL
from kontras_berbobot import auroc_per_kelompok, selisih_per_model

EPSILON = 0.01
KUANTIL = [0.025, 0.975]


def vonis(bawah, atas):
    if bawah > 0:
        return "inflasi didukung"
    if atas < EPSILON:
        return "di bawah skala peringkat"
    return "tak terputuskan"


def ulangan_bersama(daftar_kontras, n, rng):
    pasien, n_pasien = daftar_kontras[0].pasien, daftar_kontras[0].n_pasien
    hasil = []
    for _ in range(n):
        bobot = np.bincount(rng.integers(0, n_pasien, n_pasien), minlength=n_pasien)[pasien].astype(float)
        auroc = [auroc_per_kelompok(k, bobot) for k in daftar_kontras]
        if not any(np.isnan(list(a.values())).any() for a in auroc):
            hasil.append(auroc)
    return hasil


def positif_cukup(kontras):
    return min(kontras.y[kontras.dilihat].sum(), kontras.y[~kontras.dilihat].sum()) >= SEL_KECIL


def p_permutasi(teramati, nol):
    return (1 + np.sum(np.abs(nol - nol.mean()) >= abs(teramati - nol.mean()))) / (1 + len(nol))


def sebaran_permutasi(kontras, n, rng):
    dilihat_pasien = np.zeros(kontras.n_pasien, bool)
    dilihat_pasien[kontras.pasien[kontras.dilihat]] = True
    satu = np.ones(len(kontras.y))
    nol = [selisih_per_model(kontras, satu, rng.permutation(dilihat_pasien)[kontras.pasien]) for _ in range(n)]
    return selisih_per_model(kontras, satu, kontras.dilihat), {m: np.array([s[m] for s in nol]) for m in nol[0]}


def selang(ulangan, fungsi):
    return np.quantile([fungsi(u) for u in ulangan], KUANTIL)
