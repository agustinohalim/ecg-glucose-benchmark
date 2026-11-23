"""Rasio paparan swaawas terhadap efek label saja menurut protokol Artikel_Glukosa_Rencana.md §F langkah 14
(komit 506765c), PASCA-HASIL dan deskriptif: Q = DiD utama / Δ dengan Δ = DiD kontrol positif − DiD utama, aturan
batas bawah Δ − DiD utama > 0, per tugas berlabel, rerata empat tugas berlabel, rerata lima pernyataan, dan kepekaan
TEST saja. Memakai prediksi tersimpan langkah 11; tanpa melatih ulang. Keluaran stdout hanya agregat.

Pakai lingkungan ECG-FM (README), dari akar repo:
    python -u glukosa_perluasan/jalankan_rasio_label.py
        [--boot 2000]
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

AKAR_EKG = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(AKAR_EKG / d) for d in ("glukosa_test", "glukosa_m3", "glukosa_m2", "glukosa_m1")]
from cetak import tabel  # noqa: E402
from cetak_perluasan import FM, FT, JEPA  # noqa: E402
from jalankan_kontras_langsung import BATAS_PELUANG, BERLABEL, MODEL, SKALA, did_skala, ringkas, sel  # noqa: E402
from jalankan_kontras_langsung import ulangan_bersama  # noqa: E402
from jalankan_perluasan import baca_argumen  # noqa: E402
from jalankan_tambahan_perluasan import muat_tugas  # noqa: E402
from kontras_berbobot import buat_kontras  # noqa: E402
from kontras_perluasan import KUANTIL  # noqa: E402
from pemodelan import BENIH  # noqa: E402

VARIAN = {"utama (tidak dilihat = valid ∪ test)": ("valid", "test"), "TEST saja": ("test",)}


@dataclass
class Sampel:
    titik: list
    ulangan: list


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    argumen = baca_argumen()
    _, pernyataan, peluang = muat_tugas(argumen)
    print(f"\nRasio terhadap efek label saja, langkah 14 (pasca-hasil); bootstrap {argumen.boot}, benih {BENIH[0]}.\n")
    for nama_varian, tidak_dilihat in VARIAN.items():
        siap = [siapkan(t, peluang, tidak_dilihat) for t in pernyataan]
        sampel = Sampel([ringkas(k, p, np.ones(len(k.y))) for k, p in siap],
                        ulangan_bersama(siap, argumen.boot, np.random.default_rng(BENIH[0])))
        print(f"\n## Varian {nama_varian}: ulangan terpakai {len(sampel.ulangan)}\n")
        tabel(["Kelompok", "Skala", "DiD utama", "Δ (label saja)", "Q = DiD utama / Δ", "Ulangan Δ ≤ 0",
               "Δ − DiD utama", "Tafsir"],
              [baris(nama, indeks, skala, sampel) for nama, indeks in kelompok_tugas(pernyataan) for skala in SKALA])


def siapkan(tugas, peluang, tidak_dilihat):
    pilih = tugas.bagian.split_ecgfm.isin(("train",) + tidak_dilihat).values
    prob = {m: np.clip(peluang[tugas.kode][m][pilih], BATAS_PELUANG, 1 - BATAS_PELUANG) for m in MODEL}
    return buat_kontras(tugas.bagian[pilih], prob), prob


def kelompok_tugas(pernyataan):
    kode = [t.kode for t in pernyataan]
    berlabel = [kode.index(k) for k in BERLABEL]
    return ([(pernyataan[i].nama, [i]) for i in berlabel]
            + [("Rerata empat tugas berlabel", berlabel), ("Rerata lima pernyataan", list(range(len(kode))))])


def baris(nama, indeks, skala, sampel):
    swaawas = lambda entri: np.mean([did_skala(entri[i], skala, FM, JEPA) for i in indeks])
    label = lambda entri: np.mean([did_skala(entri[i], skala, FT, FM) for i in indeks])
    selisih = lambda entri: label(entri) - swaawas(entri)
    besaran = lambda fungsi: sel(sampel.titik, sampel.ulangan, fungsi)
    label_ulangan = np.array([label(u) for u in sampel.ulangan])
    bawah = np.quantile([selisih(u) for u in sampel.ulangan], KUANTIL[0])
    return [nama, skala, besaran(swaawas), besaran(label), besaran(lambda entri: swaawas(entri) / label(entri)),
            int(np.sum(label_ulangan <= 0)), besaran(selisih),
            "swaawas lebih kecil dari label" if bawah > 0 else "tak terbedakan"]


if __name__ == "__main__":
    main()
