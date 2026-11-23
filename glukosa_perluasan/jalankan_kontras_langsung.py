"""Kontras langsung paparan swaawas lawan kontrol positif menurut protokol Artikel_Glukosa_Rencana.md §F langkah 13
(komit 18a1f3c), PASCA-HASIL dan deskriptif: selisih Δ = DiD kontrol positif − DiD utama, rasio DiD utama / DiD kontrol
positif, dan DiD log-loss, pada skala AUROC, logit(AUROC) dan log-loss, dengan CI bootstrap bergugus pasien bersama.
Memakai prediksi tersimpan langkah 11; tanpa melatih ulang. Keluaran stdout hanya agregat.

Pakai lingkungan ECG-FM (README), dari akar repo:
    python -u glukosa_perluasan/jalankan_kontras_langsung.py
        [--boot 2000]
"""
import sys
from pathlib import Path

import numpy as np

AKAR_EKG = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(AKAR_EKG / d) for d in ("glukosa_test", "glukosa_m3", "glukosa_m2", "glukosa_m1")]
from cetak import dengan_selang, tabel  # noqa: E402
from cetak_perluasan import FM, FT, JEPA  # noqa: E402
from jalankan_perluasan import baca_argumen  # noqa: E402
from jalankan_tambahan_perluasan import muat_tugas  # noqa: E402
from kontras_berbobot import DILIHAT, TIDAK, auroc_per_kelompok, buat_kontras, did_dari, logit  # noqa: E402
from kontras_perluasan import KUANTIL  # noqa: E402
from pemodelan import BENIH  # noqa: E402

MODEL = (JEPA, FM, FT)
SKALA = ("AUROC", "logit", "log-loss")
BERLABEL = ("af", "lbbb", "rbbb", "avb1")
BATAS_PELUANG = 1e-7


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    argumen = baca_argumen()
    glukosa, pernyataan, peluang = muat_tugas(argumen)
    print(f"\nKontras langsung langkah 13 (pasca-hasil); bootstrap {argumen.boot}, benih {BENIH[0]}.\n")
    hasil = []
    for kelompok in [[t] for t in glukosa] + [pernyataan]:
        siap = [siapkan(t, peluang) for t in kelompok]
        ulangan = ulangan_bersama(siap, argumen.boot, np.random.default_rng(BENIH[0]))
        print(f"- {', '.join(t.kode for t in kelompok)}: ulangan terpakai {len(ulangan)}")
        titik = [ringkas(k, p, np.ones(len(k.y))) for k, p in siap]
        hasil += [(t.kode, t.nama, titik[i], [u[i] for u in ulangan]) for i, t in enumerate(kelompok)]
    print("\n## Selisih langsung Δ = DiD kontrol positif − DiD utama\n")
    tabel(["Tugas", "Skala", "DiD utama", "DiD kontrol positif", "Δ", "Tafsir"],
          [baris_tugas(h, s) for h in hasil for s in SKALA])
    print("\n## Rasio DiD utama / DiD kontrol positif (kontrol positif terdeteksi pada skala itu)\n")
    tabel(["Tugas", "Skala", "Rasio [95 % CI]", "Ulangan penyebut ≤ 0"],
          [b for h in hasil for s in SKALA if (b := baris_rasio(h, s))])
    print(f"\n## Rerata empat tugas berlabel ({', '.join(BERLABEL)})\n")
    tabel(["Skala", "Rerata DiD utama", "Rerata DiD kontrol positif", "Rerata Δ", "Rasio rerata", "Tafsir"],
          [baris_ringkasan([h for h in hasil if h[0] in BERLABEL], s) for s in SKALA])


def siapkan(tugas, peluang):
    pilih = tugas.bagian.split_ecgfm.notna().values
    prob = {m: np.clip(peluang[tugas.kode][m][pilih], BATAS_PELUANG, 1 - BATAS_PELUANG) for m in MODEL}
    return buat_kontras(tugas.bagian[pilih], prob), prob


def ringkas(kontras, prob, bobot):
    rugi = {m: -(kontras.y * np.log(p) + (1 - kontras.y) * np.log(1 - p)) for m, p in prob.items()}
    log_loss = {(m, g): np.sum(bobot * pilih * r) / np.sum(bobot * pilih) for m, r in rugi.items()
                for g, pilih in ((DILIHAT, kontras.dilihat), (TIDAK, ~kontras.dilihat))}
    return auroc_per_kelompok(kontras, bobot), log_loss


def ulangan_bersama(siap, n, rng):
    pasien, n_pasien = siap[0][0].pasien, siap[0][0].n_pasien
    hasil = []
    for _ in range(n):
        bobot = np.bincount(rng.integers(0, n_pasien, n_pasien), minlength=n_pasien)[pasien].astype(float)
        satu = [ringkas(k, p, bobot) for k, p in siap]
        if not any(np.isnan(list(a.values())).any() for a, _ in satu):
            hasil.append(satu)
    return hasil


def did_skala(entri, skala, a, b):
    auroc, log_loss = entri
    if skala == "log-loss":
        celah = {m: log_loss[(m, TIDAK)] - log_loss[(m, DILIHAT)] for m in (a, b)}
        return celah[a] - celah[b]
    return did_dari(auroc, a, b, logit if skala == "logit" else (lambda x: x))


def sel(titik, ulangan, fungsi):
    return dengan_selang(fungsi(titik), np.quantile([fungsi(u) for u in ulangan], KUANTIL))


def baris_tugas(hasil, skala):
    _, nama, titik, ulangan = hasil
    delta = lambda e: did_skala(e, skala, FT, FM)
    bawah = np.quantile([delta(u) for u in ulangan], KUANTIL[0])
    return [nama, skala, sel(titik, ulangan, lambda e: did_skala(e, skala, FM, JEPA)),
            sel(titik, ulangan, lambda e: did_skala(e, skala, FT, JEPA)), sel(titik, ulangan, delta),
            "swaawas lebih kecil" if bawah > 0 else "tak terbedakan"]


def baris_rasio(hasil, skala):
    _, nama, titik, ulangan = hasil
    penyebut = np.array([did_skala(u, skala, FT, JEPA) for u in ulangan])
    if np.quantile(penyebut, KUANTIL[0]) <= 0:
        return None
    rasio = lambda e: did_skala(e, skala, FM, JEPA) / did_skala(e, skala, FT, JEPA)
    return [nama, skala, sel(titik, ulangan, rasio), int(np.sum(penyebut <= 0))]


def baris_ringkasan(daftar, skala):
    rerata = lambda entri, a, b: np.mean([did_skala(e, skala, a, b) for e in entri])
    titik = [h[2] for h in daftar]
    ulangan = list(zip(*[h[3] for h in daftar]))
    delta = lambda entri: rerata(entri, FT, JEPA) - rerata(entri, FM, JEPA)
    bawah = np.quantile([delta(u) for u in ulangan], KUANTIL[0])
    return [skala, sel(titik, ulangan, lambda e: rerata(e, FM, JEPA)),
            sel(titik, ulangan, lambda e: rerata(e, FT, JEPA)), sel(titik, ulangan, delta),
            sel(titik, ulangan, lambda e: rerata(e, FM, JEPA) / rerata(e, FT, JEPA)),
            "swaawas lebih kecil" if bawah > 0 else "tak terbedakan"]


if __name__ == "__main__":
    main()
