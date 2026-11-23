"""Keseimbangan kelompok dilihat lawan tidak dilihat (beda rerata terstandar) dan cek status paparan per pasien;
mencetak agregat saja."""
import numpy as np

from cetak import tabel, tampil_hitungan


def cetak_status_pasien(bagian):
    campur = bagian.groupby("subject_id").split_ecgfm.apply(lambda s: s.eq("train").nunique() > 1).sum()
    print(f"- pasien dengan status dilihat dan tidak dilihat sekaligus: {tampil_hitungan(campur)}\n")


def smd(a, b):
    sebaran = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return (np.mean(a) - np.mean(b)) / sebaran


def cetak_keseimbangan(bagian):
    dilihat = bagian.split_ecgfm.eq("train").values
    per_pasien = bagian.groupby("subject_id").subject_id.transform("size").values.astype(float)
    peubah = {"usia": bagian.age.values.astype(float), "laki-laki": bagian.laki.values.astype(float),
              "ICU": bagian.di_icu.values.astype(float), "positif": bagian.y.values.astype(float),
              "pasangan per pasien": per_pasien}
    print("### Keseimbangan kelompok dilihat lawan tidak dilihat\n")
    tabel(["Peubah", "Rerata dilihat", "Rerata tidak dilihat", "SMD"],
          [[nama, f"{v[dilihat].mean():.3f}", f"{v[~dilihat].mean():.3f}", f"{smd(v[dilihat], v[~dilihat]):.3f}"]
           for nama, v in peubah.items()])
    print()
