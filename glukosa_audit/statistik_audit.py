"""Statistik audit terkalibrasi Glukosa (protokol Artikel_Glukosa_Rencana.md §F langkah 18a–c). (a) Pada 22.058 pasien
tidak dilihat: selisih berpasangan per pasien antara ECG-FM+k dan ECG-FM asli pada rekaman yang sama, Dₚ = rerata Δℓ
ditahan − disuntik dan Cₚ = rerata Δc disuntik − ditahan. (b) Pada 44.116 rekaman langkah 16: SMD kesesuaian segmen
dilihat − tidak dilihat per model dengan penyebut tetap (simpangan baku gabungan sampel penuh), dan Δ_c = SMD ECG-FM −
SMD model bersih. Semua statistik dihitung per ulangan bootstrap atas pasien; berkas skor tidak dibuka di sesi asisten."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hitung_keanggotaan import BERKAS_SKOR, muat_skor
from kontras_berbobot import auroc_berbobot
from statistik_keanggotaan import BENIH, selisih_rerata

DOSIS = ("plus4", "plus1")
BERSIH = ("ecgjepa", "ecgfounder")
KOVARIAT = ("denyut", "qrs_ms")


@dataclass(frozen=True)
class Populasi:
    nilai: dict
    kelompok: np.ndarray
    split: np.ndarray
    dibuang: int


def muat_berpasangan(pilihan, folder_replikasi, folder_kontrol):
    asli = muat_skor(Path(folder_replikasi) / BERKAS_SKOR, pilihan)[~pilihan.dilihat.to_numpy()].reset_index(drop=True)
    tidak = pilihan[~pilihan.dilihat].reset_index(drop=True)
    nilai, sah = {}, asli.terbaca.to_numpy(bool)
    for dosis in DOSIS:
        lanjut = muat_skor(Path(folder_kontrol) / f"keanggotaan_{dosis}_b{BENIH}.npz", tidak)
        sah &= lanjut.terbaca.to_numpy(bool)
        nilai[f"dl {dosis}"] = lanjut.kehilangan.to_numpy(float) - asli.kehilangan.to_numpy(float)
        nilai[f"dc {dosis}"] = lanjut.kosinus.to_numpy(float) - asli.kosinus.to_numpy(float)
        nilai[f"kosinus {dosis}"] = lanjut.kosinus.to_numpy(float)
    sah &= np.isfinite(np.column_stack(list(nilai.values()))).all(axis=1)
    return Populasi({k: v[sah] for k, v in nilai.items()}, tidak.disuntik.to_numpy()[sah],
                    tidak.split_ecgfm.to_numpy()[sah], int((~sah).sum()))


def muat_plasebo(pilihan, unit, folder_replikasi, berkas_kesesuaian):
    asli = muat_skor(Path(folder_replikasi) / BERKAS_SKOR, pilihan)
    bersih = np.load(berkas_kesesuaian)
    if not np.array_equal(bersih["baris_gelombang"], pilihan.baris_gelombang.to_numpy()):
        raise SystemExit("Kesesuaian model bersih tidak sejajar dengan pilihan langkah 16.")
    nilai = {"ecgfm": asli.kosinus.to_numpy(float), **{m: bersih[f"kosinus_{m}"].astype(float) for m in BERSIH}}
    sah = asli.terbaca.to_numpy(bool) & bersih["terbaca"] & np.isfinite(np.column_stack(list(nilai.values()))).all(axis=1)
    kovariat = pilihan[["baris_gelombang"]].merge(unit[["baris_gelombang", *KOVARIAT]], how="left")
    nilai.update({k: kovariat[k].to_numpy(float) for k in KOVARIAT})
    return Populasi({k: v[sah] for k, v in nilai.items()}, pilihan.dilihat.to_numpy()[sah],
                    pilihan.split_ecgfm.to_numpy()[sah], int((~sah).sum()))


def statistik_berpasangan(pop):
    disuntik, ditahan = pop.kelompok, ~pop.kelompok
    hasil = {}
    for dosis in DOSIS:
        dl, dc = pop.nilai[f"dl {dosis}"], pop.nilai[f"dc {dosis}"]
        hasil[f"Dp {dosis}"] = lambda w, dl=dl: selisih_rerata(dl, w * ditahan, w * disuntik)
        hasil[f"Cp {dosis}"] = lambda w, dc=dc: selisih_rerata(dc, w * disuntik, w * ditahan)
        hasil[f"AUROC -dl {dosis}"] = auroc_kelompok(-dl, disuntik)
        hasil[f"AUROC dc {dosis}"] = auroc_kelompok(dc, disuntik)
        hasil[f"SMD dc {dosis}"] = smd(dc, disuntik)
    hasil["SMD kosinus plus4"] = smd(pop.nilai["kosinus plus4"], disuntik)
    return hasil


def statistik_plasebo(pop):
    dilihat = pop.kelompok
    hasil = {f"SMD {m}": smd(pop.nilai[m], dilihat) for m in ("ecgfm", *BERSIH)}
    for m in BERSIH:
        hasil[f"Delta {m}"] = lambda w, m=m: hasil["SMD ecgfm"](w) - hasil[f"SMD {m}"](w)
    for bagian in ("valid", "test"):
        hasil[f"SMD ecgfm {bagian}"] = smd(pop.nilai["ecgfm"], dilihat, pop.split == bagian)
    return hasil


def smd(nilai, kelompok, pembanding=None):
    pembanding = ~kelompok if pembanding is None else pembanding
    skala = simpangan_gabungan(nilai, kelompok, pembanding)
    return lambda w: selisih_rerata(nilai, w * kelompok, w * pembanding) / skala


def auroc_kelompok(skor, kelompok):
    peringkat = np.unique(skor, return_inverse=True)[1]
    return lambda w: auroc_berbobot(peringkat, kelompok.astype(float), w)


def simpangan_gabungan(nilai, kelompok, pembanding):
    return float(np.sqrt((np.var(nilai[kelompok]) + np.var(nilai[pembanding])) / 2))


def smd_kovariat(pop):
    hasil = {}
    for kolom in KOVARIAT:
        nilai = pop.nilai[kolom]
        ada = np.isfinite(nilai)
        a, b = nilai[ada & pop.kelompok], nilai[ada & ~pop.kelompok]
        hasil[kolom] = (float((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)), int((~ada).sum()))
    return pd.DataFrame(hasil, index=["smd", "hilang"]).T
