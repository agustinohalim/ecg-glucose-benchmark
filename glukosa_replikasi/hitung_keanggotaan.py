"""Skor keanggotaan per rekaman terpilih (protokol langkah 16b): dibaca dari simpanan bila ada, bila tidak dihitung sekali
dengan ECG-FM pretrained (benih torch dan numpy 20261120 dipasang sekali sesudah model dimuat, urutan rekaman tetap)
lalu disimpan di folder replikasi di luar repo."""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from baca_keanggotaan import baca_berurutan
from skor_keanggotaan import PenilaiKeanggotaan, SkorRekaman
from statistik_keanggotaan import BENIH

BERKAS_SKOR = f"keanggotaan_b{BENIH}.npz"
KOLOM_SKOR = ("terbaca", "kehilangan", "kosinus")
GAGAL = SkorRekaman(np.nan, np.nan)
LAPOR_TIAP = 5000


def muat_atau_hitung(pilihan, argumen):
    jalur = Path(argumen.folder) / BERKAS_SKOR
    if not jalur.exists():
        sinyal = baca_berurutan(argumen.zip, list(pilihan.waveform_path), argumen.pekerja)
        simpan_skor(jalur, pilihan, hitung_skor(sinyal))
    return muat_skor(jalur, pilihan)


def hitung_skor(daftar_sinyal):
    import torch
    perangkat = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    penilai = PenilaiKeanggotaan(perangkat)
    torch.manual_seed(BENIH)
    np.random.seed(BENIH)
    hasil = []
    for nomor, sinyal in enumerate(daftar_sinyal, 1):
        hasil.append(GAGAL if sinyal is None else penilai.nilai(sinyal))
        lapor_kemajuan(nomor)
    return pd.DataFrame({"terbaca": [h is not GAGAL for h in hasil], "kehilangan": [h.kehilangan for h in hasil],
                         "kosinus": [h.kosinus for h in hasil]})


def lapor_kemajuan(nomor):
    if nomor % LAPOR_TIAP == 0:
        print(f"{time.strftime('%H:%M')} rekaman dinilai {nomor}", flush=True)


def simpan_skor(jalur, pilihan, skor):
    np.savez(jalur, baris_gelombang=pilihan.baris_gelombang.to_numpy(), **{k: skor[k].to_numpy() for k in KOLOM_SKOR})


def muat_skor(jalur, pilihan):
    tersimpan = np.load(jalur)
    if not np.array_equal(tersimpan["baris_gelombang"], pilihan.baris_gelombang.to_numpy()):
        raise SystemExit(f"Baris {jalur.name} tidak sejajar dengan pilihan unit sekarang.")
    return pd.DataFrame({k: tersimpan[k] for k in KOLOM_SKOR})
