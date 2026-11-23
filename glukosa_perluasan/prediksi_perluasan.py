"""Prediksi di luar lipatan untuk protokol §F langkah 11: model ukuran (GB) dan probe L2 di atas embedding beku ECGFounder,
ECG-JEPA, ECG-FM pretrained dan ECG-FM fine-tuned, per tugas. C tugas glukosa = C terpilih di TRAIN (Hasil 11); model
atau tugas lain memilih C pada subsampel lipatan 1. Embedding dimuat satu model sekaligus."""
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from model_fondasi import ECGFM, ECGFM_FT, ECGFOUNDER, ECGJEPA
from muat import DEMOGRAFI, UKURAN_EKG
from pemodelan import BENIH
from silang_lipatan import (DataLipatan, jalur_simpanan, luar_lipatan_gb, luar_lipatan_probe, muat_simpanan,
                            pilih_C_subsampel, simpan)

GB = "gb"
C_TRAIN = {"hipo": {ECGFOUNDER.kode: 1e-3, ECGJEPA.kode: 1e-3, ECGFM.kode: 1e-3},
           "hiper_berat": {ECGFOUNDER.kode: 1e-3, ECGJEPA.kode: 1e-2, ECGFM.kode: 1e-3}}
PROBE = [ECGFOUNDER, ECGJEPA, ECGFM, ECGFM_FT]
KOLOM_GB = DEMOGRAFI + UKURAN_EKG


@dataclass
class TugasSilang:
    kode: str
    nama: str
    bagian: pd.DataFrame
    lipatan: np.ndarray

    def data(self, X):
        return DataLipatan(X, self.bagian.y.values, self.bagian.subject_id.values, self.lipatan)


def muat_embedding(spek, folder):
    if not np.load(folder / spek.berkas_selesai).all():
        raise SystemExit(f"Embedding {spek.kode} belum lengkap.")
    return np.load(folder / spek.berkas_emb, mmap_mode="r")


def prediksi_gb(tugas, folder):
    jalur = jalur_simpanan(folder, tugas.kode, GB)
    simpanan = muat_simpanan(jalur, tugas.bagian)
    if simpanan is not None:
        return simpanan[0]
    peluang = luar_lipatan_gb(tugas.data(tugas.bagian[KOLOM_GB].values), BENIH[0])
    simpan(jalur, tugas.bagian, peluang, np.nan)
    print(f"{time.strftime('%H:%M')} {tugas.kode} {GB}: selesai", flush=True)
    return peluang


def prediksi_probe(spek, emb, tugas, folder):
    jalur = jalur_simpanan(folder, tugas.kode, spek.kode)
    simpanan = muat_simpanan(jalur, tugas.bagian)
    if simpanan is not None:
        print(f"- {tugas.kode} {spek.kode}: tersimpan, C {simpanan[1]}", flush=True)
        return simpanan[0]
    data = tugas.data(np.asarray(emb[tugas.bagian.baris_gelombang.values]))
    C = C_TRAIN.get(tugas.kode, {}).get(spek.kode) or pilih_C_subsampel(data, BENIH[0])
    peluang = luar_lipatan_probe(data, C)
    simpan(jalur, tugas.bagian, peluang, C)
    print(f"{time.strftime('%H:%M')} {tugas.kode} {spek.kode}: C {C}", flush=True)
    return peluang


def hitung_peluang(daftar, folder_singgahan, folder_prediksi):
    peluang = {t.kode: {GB: prediksi_gb(t, folder_prediksi)} for t in daftar}
    for spek in PROBE:
        emb = muat_embedding(spek, Path(folder_singgahan))
        for t in daftar:
            peluang[t.kode][spek.kode] = prediksi_probe(spek, emb, t, folder_prediksi)
        del emb
    return peluang
