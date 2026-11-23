"""Analisis tambahan atas Hasil 13 menurut protokol Artikel_Glukosa_Rencana.md §F langkah 12 (komit d16e491), PASCA-HASIL
dan deskriptif: keseimbangan dilihat lawan tidak dilihat per tugas, efek minimum terdeteksi (2,8 × SD bootstrap) DiD
utama dan kontrol positif, dan presisi yang ditambahkan pembanding di bawah nol permutasi (SD, korelasi, penaksir
kontrol-variat). Memakai prediksi tersimpan langkah 11; tanpa melatih ulang. Keluaran stdout hanya agregat.

Pakai lingkungan ECG-FM (README), dari akar repo:
    python glukosa_perluasan/jalankan_tambahan_perluasan.py
        [--boot 1000] [--perm 1000]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AKAR_EKG = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(AKAR_EKG / d) for d in ("glukosa_test", "glukosa_m3", "glukosa_m2", "glukosa_m1")]
from cetak import tabel  # noqa: E402
from cetak_perluasan import FM, FT, JEPA  # noqa: E402
from jalankan_m2 import pasang_gelombang  # noqa: E402
from jalankan_perluasan import baca_argumen, kontras_tugas, susun_tugas  # noqa: E402
from keseimbangan_paparan import smd  # noqa: E402
from kontras_berbobot import did_dari  # noqa: E402
from kontras_perluasan import sebaran_permutasi, ulangan_bersama  # noqa: E402
from muat import BERKAS, muat_pasangan  # noqa: E402
from pemodelan import BENIH  # noqa: E402
from prediksi_perluasan import hitung_peluang  # noqa: E402
from singgahan import Singgahan  # noqa: E402

FAKTOR_MDE = 2.8
AMBANG_SMD = 0.1


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    argumen = baca_argumen()
    glukosa, pernyataan, peluang = muat_tugas(argumen)
    print(f"\nAnalisis tambahan langkah 12 (pasca-hasil); bootstrap {argumen.boot}, permutasi {argumen.perm}.\n")
    print(f"## Keseimbangan dilihat lawan tidak dilihat (|SMD| < {AMBANG_SMD})\n")
    tabel(["Tugas", "Peubah", "Rerata dilihat", "Rerata tidak dilihat", "SMD"],
          [baris for tugas in glukosa + pernyataan for baris in baris_keseimbangan(tugas)])
    kelompok = [[t] for t in glukosa] + [pernyataan]
    print(f"\n## Efek minimum terdeteksi ({FAKTOR_MDE} × SD bootstrap)\n")
    tabel(["Tugas", "SD DiD utama", "MDE utama", "SD kontrol positif", "MDE kontrol positif"],
          [baris for daftar in kelompok for baris in baris_mde(daftar, peluang, argumen.boot)])
    print("\n## Presisi dari pembanding di bawah nol permutasi\n")
    tabel(["Tugas", "SD selisih ECG-FM", "SD DiD utama", "Rasio", "Korelasi selisih FM–JEPA", "b kontrol-variat",
           "Titik kontrol-variat", "SD kontrol-variat"],
          [baris_presisi(t, kontras_tugas(t, peluang), argumen.perm) for t in glukosa + pernyataan])


def muat_tugas(argumen):
    pasangan = pasang_gelombang(muat_pasangan(Path(argumen.folder) / BERKAS, ["waveform_path"]),
                                Singgahan.buka(argumen.singgahan))
    split = pd.read_csv(argumen.split, usecols=["source_path", "split"]).set_index("source_path").split
    pasangan["split_ecgfm"] = pasangan.waveform_path.map(split).values
    glukosa, pernyataan = susun_tugas(pasangan, argumen.zip)
    return glukosa, pernyataan, hitung_peluang(glukosa + pernyataan, argumen.singgahan, argumen.prediksi)


def baris_keseimbangan(tugas):
    bagian = tugas.bagian[tugas.bagian.split_ecgfm.notna()]
    dilihat = bagian.split_ecgfm.eq("train").values
    peubah = {"usia": bagian.age, "laki-laki": bagian.laki, "positif": bagian.y,
              "unit per pasien": bagian.groupby("subject_id").subject_id.transform("size")}
    if "di_icu" in bagian:
        peubah["ICU"] = bagian.di_icu
    return [[tugas.nama, nama, f"{v[dilihat].mean():.3f}", f"{v[~dilihat].mean():.3f}",
             f"{smd(v[dilihat], v[~dilihat]):.3f}"]
            for nama, v in ((n, np.asarray(s, dtype=float)) for n, s in peubah.items())]


def baris_mde(daftar, peluang, n):
    kontras = [kontras_tugas(t, peluang) for t in daftar]
    ulangan = ulangan_bersama(kontras, n, np.random.default_rng(BENIH[0]))
    isi = []
    for k, tugas in enumerate(daftar):
        sd = [np.std([did_dari(u[k], model, JEPA) for u in ulangan], ddof=1) for model in (FM, FT)]
        isi.append([tugas.nama, f"{sd[0]:.4f}", f"{FAKTOR_MDE * sd[0]:.3f}", f"{sd[1]:.4f}",
                    f"{FAKTOR_MDE * sd[1]:.3f}"])
    return isi


def baris_presisi(tugas, kontras, n):
    teramati, nol = sebaran_permutasi(kontras, n, np.random.default_rng(BENIH[0]))
    kovarians = np.cov(nol[FM], nol[JEPA])
    b = kovarians[0, 1] / kovarians[1, 1]
    sd_fm, sd_did = np.std(nol[FM], ddof=1), np.std(nol[FM] - nol[JEPA], ddof=1)
    return [tugas.nama, f"{sd_fm:.4f}", f"{sd_did:.4f}", f"{sd_did / sd_fm:.2f}",
            f"{np.corrcoef(nol[FM], nol[JEPA])[0, 1]:.2f}", f"{b:.2f}", f"{teramati[FM] - b * teramati[JEPA]:.3f}",
            f"{np.std(nol[FM] - b * nol[JEPA], ddof=1):.4f}"]


if __name__ == "__main__":
    main()
