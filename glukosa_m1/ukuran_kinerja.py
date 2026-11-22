from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

SPESIFISITAS_TETAP = 0.9


@dataclass
class Penilaian:
    y: np.ndarray
    gugus: np.ndarray
    peluang: dict

    def bagian(self, pilih):
        return Penilaian(self.y[pilih], self.gugus[pilih], {m: p[pilih] for m, p in self.peluang.items()})


@dataclass
class Bootstrap:
    n: int
    rng: np.random.Generator


def metrik_lengkap(y, p):
    ambang = np.quantile(p[y == 0], SPESIFISITAS_TETAP)
    return {**metrik_terurut(y, p), "Se@Sp90": np.mean(p[y == 1] > ambang), "Brier": np.mean((p - y) ** 2),
            "O/E": y.mean() / p.mean(), "kemiringan": kemiringan_kalibrasi(y, p)}


def metrik_terurut(y, p):
    return {"AUROC": roc_auc_score(y, p), "AUPRC": average_precision_score(y, p)}


def auroc_satu_lawan_sisa(y, P):
    hasil = {nama: roc_auc_score(y == k, P[:, k]) for k, nama in enumerate(["hipo", "eu", "hiper"])}
    hasil["makro"] = np.mean(list(hasil.values()))
    return hasil


def kemiringan_kalibrasi(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    return LogisticRegression(C=1e10, max_iter=1000).fit(logit, y).coef_[0, 0]


def selang_bootstrap(penilaian, fungsi, bootstrap):
    y = penilaian.y
    ulangan = {m: [] for m in penilaian.peluang}
    for indeks in indeks_bootstrap(penilaian.gugus, bootstrap):
        if len(np.unique(y[indeks])) < len(np.unique(y)):
            continue
        for m, p in penilaian.peluang.items():
            ulangan[m].append(fungsi(y[indeks], p[indeks]))
    return {m: {k: np.quantile([u[k] for u in hasil], [0.025, 0.975]) for k in hasil[0]}
            for m, hasil in ulangan.items()}


def indeks_bootstrap(gugus, bootstrap):
    urut = np.argsort(gugus, kind="stable")
    terurut = gugus[urut]
    awal = np.flatnonzero(np.r_[True, terurut[1:] != terurut[:-1]])
    panjang = np.diff(np.r_[awal, len(gugus)])
    for _ in range(bootstrap.n):
        pilih = bootstrap.rng.integers(0, len(awal), len(awal))
        mulai, lebar = awal[pilih], panjang[pilih]
        geser = np.repeat(mulai - np.r_[0, np.cumsum(lebar)[:-1]], lebar)
        yield urut[geser + np.arange(lebar.sum())]
