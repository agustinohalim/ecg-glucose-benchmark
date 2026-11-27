"""Kesesuaian segmen ECG-JEPA dan ECGFounder untuk 44.116 rekaman uji keanggotaan langkah 16 (satu per pasien, urutan
`baris_gelombang`; protokol Artikel_Glukosa_Rencana.md §F langkah 18b). Simpanan `kesesuaian_bersih_b20261120.npz` di
data/kontrol, di luar repo dan tidak dibuka di sesi asisten; stdout hanya agregat. Bila simpanan sudah ada,
tidak dihitung ulang.

Pakai venv-gpu (GPU), dari akar repo, sekali:
    python -u glukosa_audit/jalankan_kesesuaian.py [--pekerja 4]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

AKAR_EKG = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(AKAR_EKG / d) for d in ("glukosa_kontrol", "glukosa_replikasi", "glukosa_perluasan", "glukosa_test",
                                            "glukosa_m3", "glukosa_m2", "glukosa_m1")]
from cetak import tampil_hitungan  # noqa: E402
from kesesuaian_bersih import PenilaiKesesuaian, baca_dua_model, tumpuk  # noqa: E402
from penugasan_kontrol import FOLDER_KONTROL, FOLDER_REPLIKASI, muat_unit, pilihan_satu_per_pasien  # noqa: E402
from statistik_keanggotaan import BENIH  # noqa: E402

BERKAS_KESESUAIAN = f"kesesuaian_bersih_b{BENIH}.npz"
BATCH_GPU = 64
LAPOR_TIAP = 10000


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    argumen = baca_argumen()
    jalur = Path(argumen.keluaran) / BERKAS_KESESUAIAN
    if jalur.exists():
        print(f"{jalur.name} sudah ada; tidak dihitung ulang.")
        return
    pilihan = pilihan_satu_per_pasien(muat_unit(argumen.folder))
    print(f"Kesesuaian segmen model bersih langkah 18b; benih {BENIH}; rekaman {tampil_hitungan(len(pilihan))}.",
          flush=True)
    jepa, founder = hitung(baca_dua_model(argumen.zip, list(pilihan.waveform_path), argumen.pekerja), len(pilihan))
    terbaca = np.isfinite(jepa) & np.isfinite(founder)
    np.savez(jalur, baris_gelombang=pilihan.baris_gelombang.to_numpy(), terbaca=terbaca,
             kosinus_ecgjepa=jepa, kosinus_ecgfounder=founder)
    print(f"Selesai: terbaca {tampil_hitungan(terbaca.sum())} dari {tampil_hitungan(len(pilihan))}.")


def baca_argumen():
    pengurai = argparse.ArgumentParser()
    pengurai.add_argument("--folder", default=str(FOLDER_REPLIKASI))
    pengurai.add_argument("--keluaran", default=str(FOLDER_KONTROL))
    pengurai.add_argument("--zip", default="data/mimic-iv-ecg-1.0.zip")
    pengurai.add_argument("--pekerja", type=int, default=4)
    return pengurai.parse_args()


def hitung(aliran, n_rekaman):
    import torch
    perangkat = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    penilai = PenilaiKesesuaian(perangkat)
    torch.manual_seed(BENIH)
    jepa, founder = np.full(n_rekaman, np.nan), np.full(n_rekaman, np.nan)
    antre = []
    for nomor, masukan in enumerate(aliran):
        if masukan is not None:
            antre.append((nomor, masukan))
        if len(antre) == BATCH_GPU or (nomor == n_rekaman - 1 and antre):
            isi_batch(penilai, antre, jepa, founder)
            antre = []
        if (nomor + 1) % LAPOR_TIAP == 0:
            print(f"{time.strftime('%H:%M')} rekaman {nomor + 1}", flush=True)
    return jepa, founder


def isi_batch(penilai, antre, jepa, founder):
    indeks = [nomor for nomor, _ in antre]
    hasil_jepa, hasil_founder = penilai.nilai(tumpuk([m[0] for _, m in antre]), tumpuk([m[1] for _, m in antre]))
    jepa[indeks], founder[indeks] = hasil_jepa, hasil_founder


if __name__ == "__main__":
    main()
