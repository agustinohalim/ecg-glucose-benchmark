"""Singgahan gelombang untuk M2: 8 sadapan independen (I, II, V1–V6), 500 → 250 Hz (rerata dua sampel), rerata per
sadapan dibuang, float16 µV, satu baris per `waveform_path` unik. Status per baris memungkinkan pembangunan dilanjutkan.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from zip_wfdb import baca_rekaman

SADAPAN = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
FS_ASAL, N_SAMPEL_ASAL = 500, 5000
FAKTOR_TURUN = 2
N_SAMPEL = N_SAMPEL_ASAL // FAKTOR_TURUN
BELUM, BAIK, TAK_ADA, CHECKSUM_SALAH, SAMPEL_HILANG, BENTUK_SALAH = -1, 0, 1, 2, 3, 4
NAMA_STATUS = {
    BAIK: "baik",
    SAMPEL_HILANG: "ada sampel hilang (diisi 0, tetap dipakai)",
    TAK_ADA: "tidak ada di ZIP",
    CHECKSUM_SALAH: "checksum WFDB tidak cocok",
    BENTUK_SALAH: "laju/panjang/sadapan tidak sesuai",
}
STATUS_DIPAKAI = (BAIK, SAMPEL_HILANG)
BERKAS_SINYAL, BERKAS_STATUS, BERKAS_JALUR = "gelombang_250hz.npy", "gelombang_status.npy", "gelombang_jalur.txt"


@dataclass(frozen=True)
class Singgahan:
    sinyal: np.ndarray
    status: np.ndarray
    baris: dict

    @classmethod
    def buka(cls, folder):
        folder = Path(folder)
        jalur = folder.joinpath(BERKAS_JALUR).read_text(encoding="ascii").split()
        return cls(np.load(folder / BERKAS_SINYAL, mmap_mode="r"), np.load(folder / BERKAS_STATUS),
                   {j: i for i, j in enumerate(jalur)})


@dataclass(frozen=True)
class Potongan:
    jalur_zip: str
    jalur_sinyal: str
    tugas: list


def siapkan_berkas(folder, jalur_rekaman):
    folder = Path(folder)
    berkas_jalur = folder / BERKAS_JALUR
    if berkas_jalur.exists():
        if berkas_jalur.read_text(encoding="ascii").split() != jalur_rekaman:
            raise SystemExit(f"{berkas_jalur} berisi daftar rekaman lain; hapus singgahan lama dulu.")
        return np.load(folder / BERKAS_STATUS)
    folder.mkdir(parents=True, exist_ok=True)
    np.lib.format.open_memmap(folder / BERKAS_SINYAL, mode="w+", dtype=np.float16,
                              shape=(len(jalur_rekaman), len(SADAPAN), N_SAMPEL)).flush()
    status = np.full(len(jalur_rekaman), BELUM, dtype=np.int8)
    np.save(folder / BERKAS_STATUS, status)
    berkas_jalur.write_text("\n".join(jalur_rekaman) + "\n", encoding="ascii")
    return status


def olah_rekaman(rekaman):
    header = rekaman.header
    if header.fs != FS_ASAL or header.n_sampel != N_SAMPEL_ASAL or not set(SADAPAN) <= set(header.sadapan):
        return None, BENTUK_SALAH
    if not rekaman.checksum_cocok:
        return None, CHECKSUM_SALAH
    x = rekaman.sinyal_uv[[header.sadapan.index(s) for s in SADAPAN]]
    x = x.reshape(len(SADAPAN), N_SAMPEL, FAKTOR_TURUN).mean(axis=2)
    ada_hilang = bool(np.isnan(x).any())
    x = np.nan_to_num(x - np.nanmean(x, axis=1, keepdims=True)) if ada_hilang else x - x.mean(axis=1, keepdims=True)
    return x.astype(np.float16), SAMPEL_HILANG if ada_hilang else BAIK


def kerjakan_potongan(potongan):
    sinyal = np.load(potongan.jalur_sinyal, mmap_mode="r+")
    hasil = []
    with open(potongan.jalur_zip, "rb") as berkas:
        for baris, jalur_rekaman, indeks in potongan.tugas:
            x, status = baca_baris(berkas, indeks, jalur_rekaman)
            if x is not None:
                sinyal[baris] = x
            hasil.append((baris, status))
    sinyal.flush()
    return hasil


def baca_baris(berkas, indeks, jalur_rekaman):
    if len(indeks) < 2:
        return None, TAK_ADA
    try:
        return olah_rekaman(baca_rekaman(berkas, indeks, jalur_rekaman))
    except ValueError:
        return None, BENTUK_SALAH
