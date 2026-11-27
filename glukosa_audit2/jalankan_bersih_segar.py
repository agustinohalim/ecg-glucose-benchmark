"""Kesesuaian segmen ECG-JEPA dan ECGFounder pada unit uji segar langkah 19a (protokol Artikel_Glukosa_Rencana.md §F
langkah 19b), dengan pengukuran yang sama seperti langkah 18b. Unit dibaca dari tabel yang disusun jalankan_fm_segar.py.
Simpanan di data/audit2, di luar repo dan tidak dibuka di sesi asisten; stdout hanya agregat.

Pakai venv-gpu (GPU), dari akar repo, sekali, sesudah jalankan_fm_segar.py:
    python -u glukosa_audit2/jalankan_bersih_segar.py [--pekerja 4]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AKAR_EKG = Path(__file__).resolve().parent.parent
FOLDER_PAKAI = ("glukosa_audit", "glukosa_kontrol", "glukosa_replikasi", "glukosa_perluasan", "glukosa_test",
                "glukosa_m3", "glukosa_m2", "glukosa_m1")
sys.path[:0] = [str(AKAR_EKG / d) for d in FOLDER_PAKAI]
from cetak import tampil_hitungan  # noqa: E402
from jalankan_kesesuaian import hitung  # noqa: E402
from kesesuaian_bersih import baca_dua_model  # noqa: E402
from statistik_keanggotaan import BENIH  # noqa: E402
from unit_segar import BERKAS_UNIT_SEGAR, FOLDER_AUDIT, unit_uji  # noqa: E402

BERKAS_KOSINUS_BERSIH = f"kesesuaian_bersih_segar_b{BENIH}.npz"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    argumen = baca_argumen()
    folder = Path(argumen.keluaran)
    jalur = folder / BERKAS_KOSINUS_BERSIH
    if jalur.exists():
        print(f"{jalur.name} sudah ada; tidak dihitung ulang.")
        return
    uji = unit_uji(pd.read_csv(folder / BERKAS_UNIT_SEGAR))
    print(f"Kesesuaian segmen model bersih langkah 19b; benih {BENIH}; rekaman {tampil_hitungan(len(uji))}.",
          flush=True)
    jepa, founder = hitung(baca_dua_model(argumen.zip, list(uji.waveform_path), argumen.pekerja), len(uji))
    np.savez(jalur, baris_gelombang=uji.baris_gelombang.to_numpy(), kosinus_ecgjepa=jepa, kosinus_ecgfounder=founder)
    terbaca = np.isfinite(jepa) & np.isfinite(founder)
    print(f"Selesai: terbaca {tampil_hitungan(terbaca.sum())} dari {tampil_hitungan(len(uji))}.")


def baca_argumen():
    pengurai = argparse.ArgumentParser()
    pengurai.add_argument("--keluaran", default=str(FOLDER_AUDIT))
    pengurai.add_argument("--zip", default="data/mimic-iv-ecg-1.0.zip")
    pengurai.add_argument("--pekerja", type=int, default=4)
    return pengurai.parse_args()


if __name__ == "__main__":
    main()
