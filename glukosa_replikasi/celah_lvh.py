"""Celah AUROC LVH dilihat − tidak dilihat per model pada subset pasien di luar kohort glukosa, menurut protokol
Artikel_Glukosa_Rencana.md §F langkah 16f (pasca-hasil atas telaah internal 8, deskriptif, tanpa vonis): prediksi di luar lipatan
tersimpan dari langkah 15 (benih lipatan 20261120), ulangan bootstrap bergugus pasien dan benihnya sama dengan langkah 15
(kelima tugas diulang bersama agar ulangan yang terpakai identik). stdout hanya agregat.

Pakai lingkungan utama (README), dari akar repo, sesudah jalankan_replikasi.py:
    python -u glukosa_replikasi/celah_lvh.py [--boot 2000]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from prediksi_replikasi import MODEL, muat_konteks
from statistik_replikasi import sampel, statistik
from cetak import dengan_selang, tabel, tampil_hitungan
from kontras_berbobot import DILIHAT, TIDAK
from model_fondasi import ECGFM, ECGFM_FT, ECGJEPA
from silang_lipatan import muat_simpanan
from tugas_pernyataan import POLA_TUGAS

SEMUA = tuple(POLA_TUGAS)
LVH = "lvh"
NAMA_MODEL = {ECGJEPA.kode: "ECG-JEPA", ECGFM.kode: "ECG-FM pretrained", ECGFM_FT.kode: "ECG-FM fine-tuned"}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    argumen = baca_argumen()
    konteks = muat_konteks(Path(argumen.folder), argumen.boot)
    peluang = muat_prediksi(konteks, argumen.benih)
    luar_kohort = konteks.unit.lepas_pasien.values.astype(bool)
    cetak_celah(sampel(konteks, peluang, luar_kohort, SEMUA))


def baca_argumen():
    pengurai = argparse.ArgumentParser()
    pengurai.add_argument("--folder", default="data/replikasi")
    pengurai.add_argument("--boot", type=int, default=2000)
    pengurai.add_argument("--benih", type=int, default=20261120)
    return pengurai.parse_args()


def muat_prediksi(konteks, benih):
    folder = konteks.folder / f"prediksi_b{benih}"
    return {kode: prediksi_tugas(konteks.unit, folder, kode) for kode in SEMUA}


def prediksi_tugas(unit, folder, kode):
    bagian = pd.DataFrame({"y": unit[f"y_{kode}"].values, "subject_id": unit.subject_id.values})
    return {spek.kode: prediksi_tersimpan(folder / f"{kode}_{spek.kode}.npz", bagian) for spek in MODEL}


def prediksi_tersimpan(jalur, bagian):
    tersimpan = muat_simpanan(jalur, bagian)
    if tersimpan is None:
        raise SystemExit(f"Prediksi tersimpan {jalur.name} tidak ada; jalankan langkah 15 dahulu.")
    return tersimpan[0]


def cetak_celah(subset):
    kontras = subset.kontras[LVH]
    positif = kontras.y.astype(bool)
    print(f"\n## LVH, pasien di luar kohort: {tampil_hitungan(kontras.dilihat.sum())} dilihat "
          f"({tampil_hitungan((kontras.dilihat & positif).sum())} positif), {tampil_hitungan((~kontras.dilihat).sum())} "
          f"tidak dilihat ({tampil_hitungan((~kontras.dilihat & positif).sum())} positif); ulangan terpakai "
          f"{len(subset.ulangan)}\n")
    indeks = list(subset.kontras).index(LVH)
    tabel(["Model", "AUROC dilihat", "AUROC tidak dilihat", "Celah dilihat − tidak dilihat"],
          [baris_model(subset, indeks, spek.kode) for spek in MODEL])


def baris_model(subset, indeks, kode):
    fungsi = (lambda a: a[indeks][(kode, DILIHAT)], lambda a: a[indeks][(kode, TIDAK)],
              lambda a: a[indeks][(kode, DILIHAT)] - a[indeks][(kode, TIDAK)])
    return [NAMA_MODEL[kode]] + [dengan_selang(n, (b, a)) for n, b, a in (statistik(subset, f) for f in fungsi)]


if __name__ == "__main__":
    main()
