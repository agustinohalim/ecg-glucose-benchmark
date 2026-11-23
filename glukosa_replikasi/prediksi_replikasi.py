"""Unit dan prediksi di luar lipatan untuk replikasi Glukosa (protokol langkah 15): memuat unit_replikasi.csv dan tiga
embedding beku, membuang baris dengan embedding tak terhingga, lalu probe L2 cross-fitting per tugas dan model dengan
simpanan .npz di folder replikasi (di luar repo, tidak dibuka di sesi asisten)."""
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

AKAR_EKG = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(AKAR_EKG / d) for d in ("glukosa_perluasan", "glukosa_test", "glukosa_m3", "glukosa_m2",
                                            "glukosa_m1")]
from cetak import tampil_hitungan  # noqa: E402
from model_fondasi import ECGFM, ECGFM_FT, ECGJEPA  # noqa: E402
from silang_lipatan import DataLipatan, lipatan_pasien, luar_lipatan_probe, muat_simpanan, pilih_C_subsampel  # noqa: E402
from silang_lipatan import simpan  # noqa: E402

MODEL = (ECGJEPA, ECGFM, ECGFM_FT)
POTONGAN = 50000


@dataclass
class Konteks:
    unit: pd.DataFrame
    emb: dict
    folder: Path
    boot: int


def muat_konteks(folder, boot):
    unit = pd.read_csv(folder / "unit_replikasi.csv")
    emb = {}
    for spek in MODEL:
        if not np.load(folder / spek.berkas_selesai).all():
            raise SystemExit(f"Embedding {spek.kode} belum lengkap.")
        emb[spek.kode] = np.load(folder / spek.berkas_emb, mmap_mode="r")
    terhingga = np.ones(len(unit), bool)
    for matriks in emb.values():
        for awal in range(0, len(unit), POTONGAN):
            terhingga[awal:awal + POTONGAN] &= np.isfinite(matriks[awal:awal + POTONGAN]).all(axis=1)
    print(f"Unit {len(unit)}; dibuang karena embedding tak terhingga {tampil_hitungan((~terhingga).sum())}.")
    return Konteks(unit[terhingga].reset_index(drop=True), emb, folder, boot)


def prediksi_semua(konteks, daftar_tugas, benih):
    unit = konteks.unit
    lipatan = lipatan_pasien(unit.subject_id.values, benih)
    simpanan = konteks.folder / f"prediksi_b{benih}"
    simpanan.mkdir(exist_ok=True)
    peluang = {k: {} for k in daftar_tugas}
    for spek in MODEL:
        X = np.asarray(konteks.emb[spek.kode][unit.baris_gelombang.values])
        for kode in daftar_tugas:
            data = DataLipatan(X, unit[f"y_{kode}"].values, unit.subject_id.values, lipatan)
            peluang[kode][spek.kode] = prediksi_satu(data, simpanan / f"{kode}_{spek.kode}.npz", benih)
            print(f"{time.strftime('%H:%M')} benih {benih} {kode} {spek.kode}: selesai", flush=True)
        del X
    return peluang


def prediksi_satu(data, jalur, benih):
    bagian = pd.DataFrame({"y": data.y, "subject_id": data.subject_id})
    tersimpan = muat_simpanan(jalur, bagian)
    if tersimpan is not None:
        return tersimpan[0]
    C = pilih_C_subsampel(data, benih)
    peluang = luar_lipatan_probe(data, C)
    simpan(jalur, bagian, peluang, C)
    return peluang
