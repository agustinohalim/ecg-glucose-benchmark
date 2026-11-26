"""Statistik uji keanggotaan (protokol langkah 16a, c–e): satu rekaman per pasien, semua pasien valid ∪ test lawan sampel
acak pasien train sama banyak, paruh plasebo di antara pasien tidak dilihat, dan statistik berbobot per ulangan
bootstrap (satu unit = satu pasien, jadi ulangan atas unit = ulangan atas pasien) dengan selang kuantil 95 %."""
from dataclasses import dataclass

import numpy as np

from kontras_berbobot import auroc_berbobot

BENIH = 20261120
KUANTIL = [0.025, 0.975]
DILIHAT = "train"


@dataclass(frozen=True)
class DataUji:
    kehilangan: np.ndarray
    kosinus: np.ndarray
    dilihat: np.ndarray
    split: np.ndarray
    paruh_b: np.ndarray


def pasien_dua_kelompok(unit):
    per_pasien = unit.groupby("subject_id").split_ecgfm.agg(lambda s: s.eq(DILIHAT).any() and s.ne(DILIHAT).any())
    return int(per_pasien.sum())


def pilih_unit(unit):
    rng = np.random.default_rng(BENIH)
    pasien_tidak = set(unit.subject_id[unit.split_ecgfm.ne(DILIHAT)])
    calon = unit[unit.subject_id.isin(pasien_tidak) == unit.split_ecgfm.ne(DILIHAT)]
    satu = calon.iloc[rng.permutation(len(calon))].drop_duplicates("subject_id")
    pasien_dilihat = np.sort(satu.subject_id[satu.split_ecgfm.eq(DILIHAT)].to_numpy())
    terpilih = set(rng.choice(pasien_dilihat, len(pasien_tidak), replace=False))
    pilihan = satu[satu.subject_id.isin(terpilih) | satu.split_ecgfm.ne(DILIHAT)]
    return pilihan.assign(dilihat=pilihan.split_ecgfm.eq(DILIHAT)).sort_values("baris_gelombang").reset_index(drop=True)


def buat_data(bagian):
    tidak = np.flatnonzero(~bagian.dilihat.to_numpy())
    paruh_b = np.zeros(len(bagian), bool)
    paruh_b[np.random.default_rng(BENIH).permutation(tidak)[:len(tidak) // 2]] = True
    return DataUji(bagian.kehilangan.to_numpy(), bagian.kosinus.to_numpy(), bagian.dilihat.to_numpy(),
                   bagian.split_ecgfm.to_numpy(), paruh_b)


def statistik_uji(data):
    tidak, skor = ~data.dilihat, data.kehilangan
    peringkat = np.unique(-skor, return_inverse=True)[1]
    return {
        "rerata dilihat": lambda w: np.average(skor, weights=w * data.dilihat),
        "rerata tidak dilihat": lambda w: np.average(skor, weights=w * tidak),
        "D": lambda w: selisih_rerata(skor, w * tidak, w * data.dilihat),
        "D valid": lambda w: selisih_rerata(skor, w * (data.split == "valid"), w * data.dilihat),
        "D test": lambda w: selisih_rerata(skor, w * (data.split == "test"), w * data.dilihat),
        "D plasebo": lambda w: selisih_rerata(skor, w * data.paruh_b, w * (tidak & ~data.paruh_b)),
        "AUROC keanggotaan": lambda w: auroc_berbobot(peringkat, data.dilihat.astype(float), w),
        "kosinus segmen": lambda w: selisih_rerata(data.kosinus, w * data.dilihat, w * tidak),
    }


def selisih_rerata(nilai, bobot_a, bobot_b):
    return np.average(nilai, weights=bobot_a) - np.average(nilai, weights=bobot_b)


def bootstrap(statistik, n_unit, n_ulangan):
    rng = np.random.default_rng(BENIH)
    ulangan = {nama: [] for nama in statistik}
    for _ in range(n_ulangan):
        bobot = np.bincount(rng.integers(0, n_unit, n_unit), minlength=n_unit).astype(float)
        for nama, fungsi in statistik.items():
            ulangan[nama].append(fungsi(bobot))
    satu = np.ones(n_unit)
    return {nama: (fungsi(satu), *np.quantile(ulangan[nama], KUANTIL)) for nama, fungsi in statistik.items()}
