"""Selisih AUROC berpasangan antar-model pada pasangan yang sama; CI dari bootstrap bergugus pasien yang sama untuk
kedua model, sehingga korelasi antar-prediksi ikut terhitung."""
from collections import namedtuple

import numpy as np
from sklearn.metrics import roc_auc_score

from cetak import SEL_KECIL, dengan_selang, tabel, tampil_hitungan
from ukuran_kinerja import indeks_bootstrap

Perbandingan = namedtuple("Perbandingan", ["model_a", "model_b", "label"])


def selisih_titik(y, peluang, perbandingan):
    return {p.label: roc_auc_score(y, peluang[p.model_a]) - roc_auc_score(y, peluang[p.model_b])
            for p in perbandingan}


def selang_selisih_model(penilaian, perbandingan, bootstrap):
    ulangan = {p.label: [] for p in perbandingan}
    for indeks in indeks_bootstrap(penilaian.gugus, bootstrap):
        y = penilaian.y[indeks]
        if not 0 < y.sum() < len(y):
            continue
        peluang = {m: p[indeks] for m, p in penilaian.peluang.items()}
        for label, nilai in selisih_titik(y, peluang, perbandingan).items():
            ulangan[label].append(nilai)
    return {label: np.quantile(u, [0.025, 0.975]) for label, u in ulangan.items()}


def baris_selisih(nama, penilaian, perbandingan, bootstrap):
    n_positif = penilaian.y.sum()
    awal = [nama, tampil_hitungan(len(penilaian.y)), tampil_hitungan(n_positif)]
    if n_positif < SEL_KECIL:
        return [*awal, "—", *(["—"] * len(perbandingan))]
    titik = selisih_titik(penilaian.y, penilaian.peluang, perbandingan)
    selang = selang_selisih_model(penilaian, perbandingan, bootstrap)
    auroc_a = roc_auc_score(penilaian.y, penilaian.peluang[perbandingan[0].model_a])
    return [*awal, f"{auroc_a:.3f}", *(dengan_selang(titik[p.label], selang[p.label]) for p in perbandingan)]


def cetak_tabel_selisih(isi, perbandingan):
    tabel(["Bagian", "Pasangan", "Positif", f"AUROC {perbandingan[0].model_a}", *(p.label for p in perbandingan)],
          isi)
    print("\nSelisih AUROC [95 % CI]; bootstrap bergugus pasien, kedua model pada ulangan yang sama; "
          "\"—\" bila positif < 11.\n")
