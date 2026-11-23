"""AUROC dilihat lawan tidak dilihat sebagai statistik Mann-Whitney berbobot, untuk bootstrap bergugus pasien (bobot =
cacah pasien terambil) dan permutasi paparan tanpa menyalin baris; setara roc_auc_score pada baris terulang."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

DILIHAT, TIDAK = "dilihat", "tidak"


@dataclass
class Kontras:
    y: np.ndarray
    pasien: np.ndarray
    dilihat: np.ndarray
    peringkat: dict

    @property
    def n_pasien(self):
        return int(self.pasien.max()) + 1


def buat_kontras(bagian, peluang):
    kode_pasien = pd.factorize(bagian.subject_id.values)[0]
    peringkat = {m: np.unique(p, return_inverse=True)[1] for m, p in peluang.items()}
    return Kontras(bagian.y.values.astype(float), kode_pasien, bagian.split_ecgfm.values == "train", peringkat)


def auroc_berbobot(peringkat, y, bobot):
    n_unik = peringkat.max() + 1
    negatif = np.bincount(peringkat, weights=bobot * (1 - y), minlength=n_unik)
    positif = np.bincount(peringkat, weights=bobot * y, minlength=n_unik)
    di_bawah = np.cumsum(negatif) - negatif
    total_positif, total_negatif = positif.sum(), negatif.sum()
    if total_positif == 0 or total_negatif == 0:
        return np.nan
    return float(np.sum(positif * (di_bawah + 0.5 * negatif)) / (total_positif * total_negatif))


def selisih_per_model(kontras, bobot, dilihat):
    return {m: auroc_berbobot(r, kontras.y, bobot * dilihat) - auroc_berbobot(r, kontras.y, bobot * ~dilihat)
            for m, r in kontras.peringkat.items()}


def auroc_per_kelompok(kontras, bobot):
    return {(m, g): auroc_berbobot(r, kontras.y, bobot * pilih) for m, r in kontras.peringkat.items()
            for g, pilih in ((DILIHAT, kontras.dilihat), (TIDAK, ~kontras.dilihat))}


def ulangan_bootstrap(kontras, n, rng):
    hasil = []
    for _ in range(n):
        cacah = np.bincount(rng.integers(0, kontras.n_pasien, kontras.n_pasien), minlength=kontras.n_pasien)
        auroc = auroc_per_kelompok(kontras, cacah[kontras.pasien].astype(float))
        if not np.isnan(list(auroc.values())).any():
            hasil.append(auroc)
    return hasil


def logit(x):
    return np.log(x / (1 - x))


def did_dari(auroc, a, b, ubah=lambda x: x):
    return ((ubah(auroc[(a, DILIHAT)]) - ubah(auroc[(a, TIDAK)]))
            - (ubah(auroc[(b, DILIHAT)]) - ubah(auroc[(b, TIDAK)])))
