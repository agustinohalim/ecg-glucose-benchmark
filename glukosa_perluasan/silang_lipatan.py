"""Evaluasi cross-fitting protokol §F langkah 11c: 5 lipatan per pasien, tiap model dilatih pada 4 lipatan dan memprediksi
lipatan sisanya. Prediksi di luar lipatan disimpan per tugas dan model di luar repo, sehingga jalan yang terputus
dilanjutkan tanpa melatih ulang; berkas prediksi tidak dibuka di sesi asisten."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold

from pemodelan import buat_gb
from probe import pipa

N_LIPATAN = 5
KISI_C_SUBSAMPEL = [1e-3, 1e-2, 1e-1, 1]
UKURAN_SUBSAMPEL = 60000
LIPATAN_DALAM = 3


@dataclass
class DataLipatan:
    X: np.ndarray
    y: np.ndarray
    subject_id: np.ndarray
    lipatan: np.ndarray


def lipatan_pasien(subject_id, benih):
    pasien, kode = np.unique(subject_id, return_inverse=True)
    urutan = np.random.default_rng(benih).permutation(len(pasien))
    return (urutan % N_LIPATAN)[kode]


def pilih_C_subsampel(data, benih):
    latih = np.flatnonzero(data.lipatan != 0)
    pilih = np.sort(np.random.default_rng(benih).choice(latih, min(UKURAN_SUBSAMPEL, len(latih)), replace=False))
    X, y = data.X[pilih], data.y[pilih]
    pembagi = StratifiedGroupKFold(LIPATAN_DALAM, shuffle=True, random_state=benih)
    bagi = list(pembagi.split(X, y, data.subject_id[pilih]))
    skor = {C: np.mean([log_loss(y[uji], pipa(C).fit(X[a], y[a]).predict_proba(X[uji]), labels=[0, 1])
                        for a, uji in bagi])
            for C in KISI_C_SUBSAMPEL}
    return min(skor, key=skor.get)


def luar_lipatan(buat_model, data):
    peluang = np.empty(len(data.y))
    for k in range(N_LIPATAN):
        uji = data.lipatan == k
        peluang[uji] = buat_model().fit(data.X[~uji], data.y[~uji]).predict_proba(data.X[uji])[:, 1]
    return peluang


def luar_lipatan_probe(data, C):
    return luar_lipatan(lambda: pipa(C), data)


def luar_lipatan_gb(data, benih):
    return luar_lipatan(lambda: buat_gb(benih), data)


def jalur_simpanan(folder, tugas, model):
    return Path(folder) / f"perluasan_{tugas}_{model}.npz"


def muat_simpanan(jalur, bagian):
    if not jalur.exists():
        return None
    tersimpan = np.load(jalur)
    if not (np.array_equal(tersimpan["y"], bagian.y.values)
            and np.array_equal(tersimpan["subject_id"], bagian.subject_id.values)):
        raise SystemExit(f"Baris {jalur.name} tidak sejajar dengan unit sekarang.")
    return tersimpan["peluang"], float(tersimpan["C"])


def simpan(jalur, bagian, peluang, C):
    np.savez_compressed(jalur, y=bagian.y.values, subject_id=bagian.subject_id.values, peluang=peluang, C=C)
