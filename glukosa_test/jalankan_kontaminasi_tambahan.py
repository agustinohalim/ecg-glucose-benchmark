"""Analisis tambahan uji kontaminasi Glukosa atas telaah internal 2, protokol Artikel_Glukosa_Rencana.md §F langkah 10 (komit
270fbd1), PASCA-HASIL dan deskriptif.

Memakai prediksi tersimpan langkah 9 (`kontaminasi_*.npz`, baris dicek sejajar; tanpa melatih ulang). Mencetak agregat
saja: uji permutasi paparan per pasien, keseimbangan kelompok (SMD), DiD utama/sekunder/plasebo dengan bootstrap
bergugus pasien 10.000 ulangan, efek minimum terdeteksi, DiD skala logit, dan kepekaan definisi paparan.

Dari akar repo:
    python glukosa_test/jalankan_kontaminasi_tambahan.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

AKAR_EKG = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(AKAR_EKG / "glukosa_m3"), str(AKAR_EKG / "glukosa_m2"), str(AKAR_EKG / "glukosa_m1")]
from cetak import SEL_KECIL, dengan_selang, tabel, tampil_hitungan  # noqa: E402
from jalankan_m2 import pasang_gelombang, pengurai_dasar  # noqa: E402
from jalankan_test import KODE_TUGAS, NAMA_M1_GB, per_tiga_split  # noqa: E402
from keseimbangan_paparan import cetak_keseimbangan, cetak_status_pasien  # noqa: E402
from kontras_berbobot import (auroc_per_kelompok, buat_kontras, did_dari, logit, selisih_per_model,  # noqa: E402
                              ulangan_bootstrap)
from model_fondasi import ECGFM, ECGFOUNDER, ECGJEPA  # noqa: E402
from muat import BERKAS, TUGAS_BINER, muat_pasangan, pasangan_tugas  # noqa: E402
from pemodelan import BENIH  # noqa: E402
from prediksi import muat_prediksi  # noqa: E402
from singgahan import Singgahan  # noqa: E402

LABEL = {ECGFM.nama: "M3c ECG-FM", ECGJEPA.nama: "M3b ECG-JEPA", ECGFOUNDER.nama: "M3 ECGFounder",
         NAMA_M1_GB: "M1-GB"}
PASANGAN_DID = [(ECGFM.nama, ECGJEPA.nama, "utama"), (ECGFM.nama, ECGFOUNDER.nama, "sekunder"),
                (ECGFM.nama, NAMA_M1_GB, "sekunder"), (ECGFOUNDER.nama, ECGJEPA.nama, "plasebo"),
                (NAMA_M1_GB, ECGJEPA.nama, "plasebo"), (NAMA_M1_GB, ECGFOUNDER.nama, "plasebo")]
FAKTOR_MDE = 2.8
SETENGAH_EFEK_UTAMA = 0.033


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    argumen = baca_argumen()
    mulai = time.time()
    pasangan = pasang_gelombang(muat_pasangan(Path(argumen.folder) / BERKAS, ["waveform_path"]),
                                Singgahan.buka(argumen.singgahan))
    split = pd.read_csv(argumen.split, usecols=["source_path", "split"]).set_index("source_path").split
    pasangan["split_ecgfm"] = pasangan.waveform_path.map(split).values
    print(f"\nAnalisis tambahan kontaminasi, protokol §F langkah 10 (pasca-hasil); bootstrap {argumen.boot}, "
          f"permutasi {argumen.perm}.\n")
    for kode, tugas in zip(KODE_TUGAS, TUGAS_BINER):
        _, validasi, test = per_tiga_split(pasangan_tugas(pasangan[pasangan.bukan_exclude], tugas.kelas_positif))
        nilai = pd.concat([validasi, test])
        peluang = muat_prediksi(Path(argumen.prediksi) / f"kontaminasi_{kode}.npz", nilai)
        print(f"## {tugas.nama} — VALIDATION ∪ TEST (non-EXCLUDE)\n")
        analisis_tugas(nilai, peluang, argumen)
    print(f"\nWaktu jalan: {(time.time() - mulai) / 60:.1f} menit.")


def baca_argumen():
    pengurai = pengurai_dasar()
    pengurai.set_defaults(boot=10000)
    pengurai.add_argument("--perm", type=int, default=2000)
    pengurai.add_argument("--boot_kepekaan", type=int, default=2000)
    pengurai.add_argument("--prediksi", default="data/mimic-iv-ext-ecg-glucose/prediksi")
    pengurai.add_argument("--split", default="models/ECG-FM/meta_split_mimic_iv_ecg.csv")
    return pengurai.parse_args()


def analisis_tugas(nilai, peluang, argumen):
    tercantum = pd.notna(nilai.split_ecgfm.values)
    bagian = nilai[tercantum]
    peluang_tercantum = {m: p[tercantum] for m, p in peluang.items()}
    cetak_status_pasien(bagian)
    cetak_keseimbangan(bagian)
    kontras = buat_kontras(bagian, peluang_tercantum)
    cetak_bootstrap(kontras, argumen.boot)
    cetak_permutasi(kontras, argumen.perm)
    cetak_kepekaan(bagian, peluang_tercantum, argumen.boot_kepekaan)


def cetak_bootstrap(kontras, n):
    titik = auroc_per_kelompok(kontras, np.ones(len(kontras.y)))
    ulangan = ulangan_bootstrap(kontras, n, np.random.default_rng(BENIH[0]))
    print(f"### Bootstrap bergugus pasien, {len(ulangan)} ulangan terpakai\n")
    isi = []
    for a, b, jenis in PASANGAN_DID:
        sebaran = np.array([did_dari(u, a, b) for u in ulangan])
        sebaran_logit = np.array([did_dari(u, a, b, logit) for u in ulangan])
        isi.append([jenis, f"{LABEL[a]} − {LABEL[b]}",
                    dengan_selang(did_dari(titik, a, b), np.quantile(sebaran, [0.025, 0.975])),
                    f"{sebaran.std(ddof=1):.4f}", f"{FAKTOR_MDE * sebaran.std(ddof=1):.3f}",
                    dengan_selang(did_dari(titik, a, b, logit), np.quantile(sebaran_logit, [0.025, 0.975]))])
    tabel(["Jenis", "DiD", "Nilai [95 % CI]", "SD bootstrap", "Efek minimum terdeteksi (2,8 SD)",
           "DiD skala logit [95 % CI]"], isi)
    print(f"\nSetengah efek utama TEST: {SETENGAH_EFEK_UTAMA:.3f}.\n")


def cetak_permutasi(kontras, n):
    rng = np.random.default_rng(BENIH[0])
    dilihat_pasien = np.zeros(kontras.n_pasien, bool)
    dilihat_pasien[kontras.pasien[kontras.dilihat]] = True
    satu = np.ones(len(kontras.y))
    teramati = selisih_per_model(kontras, satu, kontras.dilihat)
    nol = [selisih_per_model(kontras, satu, rng.permutation(dilihat_pasien)[kontras.pasien]) for _ in range(n)]
    print(f"### Uji permutasi paparan per pasien ({n} permutasi)\n")
    isi = [baris_permutasi(f"selisih {LABEL[m]}", teramati[m], np.array([s[m] for s in nol])) for m in teramati]
    isi += [baris_permutasi(f"DiD {LABEL[a]} − {LABEL[b]}", teramati[a] - teramati[b],
                            np.array([s[a] - s[b] for s in nol])) for a, b, _ in PASANGAN_DID]
    tabel(["Besaran", "Teramati", "Rerata nol", "SD nol", "p dua sisi"], isi)
    print()


def baris_permutasi(nama, teramati, nol):
    p = (1 + np.sum(np.abs(nol - nol.mean()) >= abs(teramati - nol.mean()))) / (1 + len(nol))
    return [nama, f"{teramati:.3f}", f"{nol.mean():.4f}", f"{nol.std(ddof=1):.4f}", f"{p:.3f}"]


def cetak_kepekaan(bagian, peluang, n):
    print(f"### Kepekaan definisi paparan, DiD utama (bootstrap {n})\n")
    pilihan = {"tidak dilihat = `test` ECG-FM saja": bagian.split_ecgfm.ne("valid").values,
               "VALIDATION saja": bagian.split.eq("VALIDATION").values,
               "TEST saja": bagian.split.eq("TEST").values}
    isi = [baris_kepekaan(nama, bagian[pilih], {m: p[pilih] for m, p in peluang.items()}, n)
           for nama, pilih in pilihan.items()]
    tabel(["Himpunan", "Positif dilihat", "Positif tidak dilihat", "DiD M3c − M3b [95 % CI]"], isi)
    print()


def baris_kepekaan(nama, bagian, peluang, n):
    kontras = buat_kontras(bagian, peluang)
    positif = [kontras.y[kontras.dilihat].sum(), kontras.y[~kontras.dilihat].sum()]
    if min(positif) < SEL_KECIL:
        return [nama, tampil_hitungan(positif[0]), tampil_hitungan(positif[1]), "positif < 11"]
    titik = auroc_per_kelompok(kontras, np.ones(len(kontras.y)))
    ulangan = ulangan_bootstrap(kontras, n, np.random.default_rng(BENIH[0]))
    sebaran = [did_dari(u, ECGFM.nama, ECGJEPA.nama) for u in ulangan]
    return [nama, tampil_hitungan(positif[0]), tampil_hitungan(positif[1]),
            dengan_selang(did_dari(titik, ECGFM.nama, ECGJEPA.nama), np.quantile(sebaran, [0.025, 0.975]))]


if __name__ == "__main__":
    main()
