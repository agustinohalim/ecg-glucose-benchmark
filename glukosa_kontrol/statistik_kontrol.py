"""Statistik kontrol positif swaawas Glukosa (protokol §F langkah 17c–e): probe L2 langkah 15 dipasang ulang pada rekaman
`valid` ∪ `test` replikasi untuk ECG-JEPA, ECG-FM asli, ECG-FM+1, dan ECG-FM+4; celah AUROC g = disuntik − ditahan per
model dan tugas; efek paparan E = celah ECG-FM+k − celah ECG-FM asli; Dₛ kehilangan = rerata ditahan − rerata
disuntik. Semua bacaan dihitung pada ulangan bootstrap bergugus pasien yang sama (2.000, benih 20261120), selang kuantil
95 %. Prediksi per tugas dan model disimpan di luar repo dan tidak dibuka di sesi asisten."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hitung_keanggotaan import BERKAS_SKOR, muat_skor
from kontras_berbobot import DILIHAT, TIDAK, Kontras, auroc_berbobot, auroc_per_kelompok, did_dari
from prediksi_replikasi import prediksi_satu
from silang_lipatan import DataLipatan, lipatan_pasien
from statistik_keanggotaan import BENIH, KUANTIL, selisih_rerata

MODEL = ("ecgjepa", "ecgfm", "plus1", "plus4")
DOSIS = ("plus1", "plus4")
SEMUA = ("af", "lbbb", "rbbb", "avb1", "lvh")
BERLABEL = SEMUA[:4]
KELOMPOK = {"rerata empat berlabel": BERLABEL, "rerata lima pernyataan": SEMUA}


@dataclass
class DataKontrol:
    kontras: dict
    pasien_skor: np.ndarray
    disuntik_skor: np.ndarray
    kehilangan: dict
    kosinus: np.ndarray
    peringkat_kehilangan: np.ndarray
    n_pasien: int
    dibuang: dict


def muat_data(bacaan, tidak, folder_replikasi, folder_kontrol):
    emb = {m: muat_embedding(m, bacaan, folder_replikasi, folder_kontrol) for m in MODEL}
    terhingga = np.logical_and.reduce([np.isfinite(x).all(axis=1) for x in emb.values()])
    skor = muat_kehilangan(tidak, folder_replikasi, folder_kontrol)
    nilai = skor.filter(regex="^(kehilangan|kosinus)_").to_numpy(float)
    sah = skor.terbaca.to_numpy(bool) & np.isfinite(nilai).all(axis=1)
    bagian, skor = bacaan[terhingga].reset_index(drop=True), skor[sah].reset_index(drop=True)
    pasien = pd.Index(np.sort(tidak.subject_id.unique()))
    kode_bacaan = pasien.get_indexer(bagian.subject_id)
    kontras = {t: Kontras(bagian[f"y_{t}"].to_numpy(float), kode_bacaan, bagian.disuntik.to_numpy(),
                          prediksi_kontras(bagian, {m: x[terhingga] for m, x in emb.items()}, t, folder_kontrol))
               for t in SEMUA}
    return DataKontrol(kontras, pasien.get_indexer(skor.subject_id), skor.disuntik.to_numpy(),
                       {m: skor[f"kehilangan_{m}"].to_numpy() for m in ("ecgfm", *DOSIS)}, skor.kosinus_plus4.to_numpy(),
                       np.unique(skor.kehilangan_plus4.to_numpy(), return_inverse=True)[1], len(pasien),
                       {"embedding tak terhingga": int((~terhingga).sum()), "skor tak terbaca": int((~sah).sum())})


def muat_embedding(model, bacaan, folder_replikasi, folder_kontrol):
    if model not in DOSIS:
        return np.asarray(np.load(Path(folder_replikasi) / f"emb_{model}.npy", mmap_mode="r")[bacaan.baris_gelombang])
    tersimpan = np.load(Path(folder_kontrol) / f"emb_{model}.npz")
    if not np.array_equal(tersimpan["baris_gelombang"], bacaan.baris_gelombang.to_numpy()):
        raise SystemExit(f"Embedding {model} tidak sejajar dengan unit bacaan.")
    return tersimpan["emb"]


def prediksi_kontras(bagian, emb, tugas, folder_kontrol):
    simpanan = Path(folder_kontrol) / f"prediksi_kontrol_b{BENIH}"
    simpanan.mkdir(exist_ok=True)
    subjek = bagian.subject_id.to_numpy()
    lipatan = lipatan_pasien(subjek, BENIH)
    y = bagian[f"y_{tugas}"].to_numpy()
    peluang = {m: prediksi_satu(DataLipatan(emb[m], y, subjek, lipatan), simpanan / f"{tugas}_{m}.npz", BENIH)
               for m in MODEL}
    return {m: np.unique(p, return_inverse=True)[1] for m, p in peluang.items()}


def muat_kehilangan(tidak, folder_replikasi, folder_kontrol):
    asli = np.load(Path(folder_replikasi) / BERKAS_SKOR)
    posisi = pd.Index(asli["baris_gelombang"]).get_indexer(tidak.baris_gelombang)
    if (posisi < 0).any():
        raise SystemExit("Skor langkah 16 tidak memuat semua pasien tidak dilihat.")
    skor = tidak[["subject_id", "disuntik"]].assign(terbaca=asli["terbaca"][posisi],
                                                    kehilangan_ecgfm=asli["kehilangan"][posisi])
    for dosis in DOSIS:
        lanjut = muat_skor(Path(folder_kontrol) / f"keanggotaan_{dosis}_b{BENIH}.npz", tidak)
        skor[f"kehilangan_{dosis}"] = lanjut.kehilangan.to_numpy()
        skor[f"kosinus_{dosis}"] = lanjut.kosinus.to_numpy()
        skor["terbaca"] &= lanjut.terbaca.to_numpy()
    return skor.drop(columns="kosinus_plus1")


def bacaan_ulangan(data, cacah):
    auroc = {t: auroc_per_kelompok(k, cacah[k.pasien].astype(float)) for t, k in data.kontras.items()}
    hasil = {}
    for m in MODEL:
        hasil.update(per_tugas_dan_rerata(f"g {m}", auroc, lambda a, m=m: a[(m, DILIHAT)] - a[(m, TIDAK)]))
    for dosis in DOSIS:
        hasil.update(per_tugas_dan_rerata(f"E {dosis}", auroc, lambda a, d=dosis: did_dari(a, d, "ecgfm")))
    hasil.update(per_tugas_dan_rerata("E plus4 lawan ECG-JEPA", auroc, lambda a: did_dari(a, "plus4", "ecgjepa")))
    bobot = cacah[data.pasien_skor].astype(float)
    disuntik, ditahan = bobot * data.disuntik_skor, bobot * ~data.disuntik_skor
    for model, nilai in data.kehilangan.items():
        hasil[f"Ds {model}"] = selisih_rerata(nilai, ditahan, disuntik)
    hasil["AUROC keanggotaan plus4"] = auroc_berbobot(data.peringkat_kehilangan.max() - data.peringkat_kehilangan,
                                                      data.disuntik_skor.astype(float), bobot)
    hasil["kosinus plus4"] = selisih_rerata(data.kosinus, disuntik, ditahan)
    return hasil


def per_tugas_dan_rerata(awalan, auroc, fungsi):
    nilai = {t: fungsi(a) for t, a in auroc.items()}
    hasil = {f"{awalan} {t}": v for t, v in nilai.items()}
    hasil.update({f"{awalan} {nama}": float(np.mean([nilai[t] for t in anggota])) for nama, anggota in KELOMPOK.items()})
    return hasil


def bootstrap(data, n_ulangan):
    rng = np.random.default_rng(BENIH)
    ulangan = []
    for _ in range(n_ulangan):
        nilai = bacaan_ulangan(data, np.bincount(rng.integers(0, data.n_pasien, data.n_pasien), minlength=data.n_pasien))
        if not np.isnan(list(nilai.values())).any():
            ulangan.append(nilai)
    titik = bacaan_ulangan(data, np.ones(data.n_pasien, int))
    selang = {k: np.quantile([u[k] for u in ulangan], KUANTIL) for k in titik}
    return {k: (v, *selang[k]) for k, v in titik.items()}, len(ulangan)
