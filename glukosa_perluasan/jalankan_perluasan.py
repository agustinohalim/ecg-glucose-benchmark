"""Perluasan uji kontaminasi Glukosa menurut protokol Artikel_Glukosa_Rencana.md §F langkah 11 (komit f240ca7), dicatat
sebelum embedding ECG-FM fine-tuned dibangun dan sebelum tugas pernyataan atau evaluasi cross-fitting dinilai.

Tugas: hipoglikemia dan hiperglikemia berat (pasangan non-EXCLUDE dari ketiga split dataset) dan lima pernyataan mesin
MIMIC-IV-ECG (satu unit per EKG). Semua unit diprediksi di luar lipatan (5 lipatan per pasien). Kontras dilihat lawan
tidak dilihat menurut daftar split ECG-FM; DiD utama ECG-FM pretrained − ECG-JEPA, sekunder lawan ECGFounder, plasebo
ECGFounder − ECG-JEPA, kontrol positif ECG-FM fine-tuned − ECG-JEPA; vonis dengan margin ε = 0,01. Keluaran stdout hanya
agregat; prediksi disimpan di --prediksi (di luar repo). Bila terputus, jalankan ulang; prediksi tersimpan dipakai.

Pakai lingkungan ECG-FM (README), dari akar repo:
    python glukosa_perluasan/jalankan_perluasan.py [--boot 2000]
        [--perm 1000] [--boot_kepekaan 1000]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

AKAR_EKG = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(AKAR_EKG / d) for d in ("glukosa_test", "glukosa_m3", "glukosa_m2", "glukosa_m1")]
from cetak import dengan_selang, tabel, tampil_hitungan  # noqa: E402
from cetak_perluasan import FM, FT, JEPA, cetak_ringkasan, cetak_tugas  # noqa: E402
from jalankan_m2 import pasang_gelombang, pengurai_dasar  # noqa: E402
from jalankan_test import KODE_TUGAS  # noqa: E402
from kontras_berbobot import auroc_per_kelompok, buat_kontras, did_dari  # noqa: E402
from kontras_perluasan import sebaran_permutasi, selang, ulangan_bersama, vonis  # noqa: E402
from muat import BERKAS, TUGAS_BINER, muat_pasangan, pasangan_tugas  # noqa: E402
from pemodelan import BENIH  # noqa: E402
from prediksi_perluasan import TugasSilang, hitung_peluang  # noqa: E402
from silang_lipatan import lipatan_pasien  # noqa: E402
from singgahan import Singgahan  # noqa: E402
from tugas_pernyataan import NAMA_TUGAS, POLA_TUGAS, bagian_tugas, tabel_ekg, tambah_label  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    argumen = baca_argumen()
    mulai = time.time()
    pasangan = pasang_gelombang(muat_pasangan(Path(argumen.folder) / BERKAS, ["waveform_path"]),
                                Singgahan.buka(argumen.singgahan))
    split = pd.read_csv(argumen.split, usecols=["source_path", "split"]).set_index("source_path").split
    pasangan["split_ecgfm"] = pasangan.waveform_path.map(split).values
    glukosa, pernyataan = susun_tugas(pasangan, argumen.zip)
    peluang = hitung_peluang(glukosa + pernyataan, argumen.singgahan, argumen.prediksi)
    print(f"\nPerluasan uji kontaminasi, protokol §F langkah 11; bootstrap {argumen.boot}, permutasi {argumen.perm}.\n")
    print("## Tugas glukosa, cross-fitting atas ketiga split (non-EXCLUDE)\n")
    for tugas in glukosa:
        laporkan([tugas], peluang, argumen)
    print("## Tugas pernyataan mesin, cross-fitting atas semua EKG\n")
    laporkan(pernyataan, peluang, argumen)
    print(f"## Kepekaan: tidak dilihat = `test` ECG-FM saja (bootstrap {argumen.boot_kepekaan})\n")
    laporkan_kepekaan(glukosa, pernyataan, peluang, argumen)
    print(f"\nWaktu jalan: {(time.time() - mulai) / 60:.1f} menit.")


def baca_argumen():
    pengurai = pengurai_dasar()
    pengurai.set_defaults(boot=2000)
    pengurai.add_argument("--perm", type=int, default=1000)
    pengurai.add_argument("--boot_kepekaan", type=int, default=1000)
    pengurai.add_argument("--prediksi", default="data/mimic-iv-ext-ecg-glucose/prediksi")
    pengurai.add_argument("--split", default="models/ECG-FM/meta_split_mimic_iv_ecg.csv")
    pengurai.add_argument("--zip", default="data/mimic-iv-ecg-1.0.zip")
    return pengurai.parse_args()


def susun_tugas(pasangan, arsip):
    glukosa = []
    for kode, tugas in zip(KODE_TUGAS, TUGAS_BINER):
        bagian = pasangan_tugas(pasangan[pasangan.bukan_exclude], tugas.kelas_positif).reset_index(drop=True)
        glukosa.append(TugasSilang(kode, tugas.nama, bagian, lipatan_pasien(bagian.subject_id.values, BENIH[0])))
    ekg = tambah_label(tabel_ekg(pasangan), arsip)
    lipatan = lipatan_pasien(ekg.subject_id.values, BENIH[0])
    pernyataan = [TugasSilang(kode, NAMA_TUGAS[kode], bagian_tugas(ekg, kode), lipatan) for kode in POLA_TUGAS]
    print(f"\n- EKG unik: {tampil_hitungan(len(ekg))}; positif per pernyataan: "
          + ", ".join(f"{NAMA_TUGAS[k]} {tampil_hitungan(ekg[f'y_{k}'].sum())}" for k in POLA_TUGAS) + "\n")
    return glukosa, pernyataan


def kontras_tugas(tugas, peluang, buang_valid=False):
    pilih = tugas.bagian.split_ecgfm.notna().values
    if buang_valid:
        pilih &= tugas.bagian.split_ecgfm.ne("valid").values
    return buat_kontras(tugas.bagian[pilih], {m: p[pilih] for m, p in peluang[tugas.kode].items()})


def laporkan(daftar, peluang, argumen):
    kontras = [kontras_tugas(t, peluang) for t in daftar]
    ulangan = ulangan_bersama(kontras, argumen.boot, np.random.default_rng(BENIH[0]))
    for k, (tugas, satu) in enumerate(zip(daftar, kontras)):
        permutasi = sebaran_permutasi(satu, argumen.perm, np.random.default_rng(BENIH[0]))
        cetak_tugas(tugas.nama, satu, [u[k] for u in ulangan], permutasi)
    if len(daftar) > 1:
        cetak_ringkasan(kontras, ulangan, f"Ringkasan: rerata DiD {len(daftar)} tugas pernyataan (bobot sama)")


def laporkan_kepekaan(glukosa, pernyataan, peluang, argumen):
    isi = []
    for kelompok in [[t] for t in glukosa] + [pernyataan]:
        kontras = [kontras_tugas(t, peluang, buang_valid=True) for t in kelompok]
        ulangan = ulangan_bersama(kontras, argumen.boot_kepekaan, np.random.default_rng(BENIH[0]))
        for k, tugas in enumerate(kelompok):
            titik = auroc_per_kelompok(kontras[k], np.ones(len(kontras[k].y)))
            isi.append([tugas.nama, *(sel_kepekaan(titik, [u[k] for u in ulangan], a) for a in (FM, FT))])
    tabel(["Tugas", "DiD utama [95 % CI], vonis", "Kontrol positif [95 % CI], vonis"], isi)


def sel_kepekaan(titik, ulangan, model):
    batas = selang(ulangan, lambda u: did_dari(u, model, JEPA))
    return f"{dengan_selang(did_dari(titik, model, JEPA), batas)} {vonis(*batas)}"


if __name__ == "__main__":
    main()
