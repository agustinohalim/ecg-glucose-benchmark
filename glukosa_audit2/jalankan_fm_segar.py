"""Bacaan ECG-FM langkah 19 (protokol Artikel_Glukosa_Rencana.md §F langkah 19): menyusun unit segar 19a bila belum ada,
kesesuaian segmen ECG-FM dan ECG-FM-acak pada unit uji segar (19b), dan kehilangan bertopeng sama ECG-FM asli dan
ECG-FM+4 pada 22.058 pasien tidak dilihat langkah 17 (19d), dengan cek determinisme pada rekaman pertama. Simpanan di
data/audit2, di luar repo dan tidak dibuka di sesi asisten; stdout hanya agregat. Bacaan yang sudah tersimpan
dilewati.

Pakai venv-ecgfm (GPU), dari akar repo, sekali:
    python -u glukosa_audit2/jalankan_fm_segar.py [--pekerja 4]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

AKAR_EKG = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(AKAR_EKG / d) for d in ("glukosa_kontrol", "glukosa_replikasi", "glukosa_perluasan", "glukosa_test",
                                            "glukosa_m3", "glukosa_m2", "glukosa_m1")]
from baca_keanggotaan import baca_berurutan  # noqa: E402
from cetak import tabel, tampil_hitungan  # noqa: E402
from jalankan_bacaan import muat_bobot  # noqa: E402
from penilai_fm import kehilangan_bertopeng_tetap, kosinus_rekaman, penilai_acak, penilai_dengan_bobot  # noqa: E402
from penugasan_kontrol import FOLDER_KONTROL, muat_unit, pilihan_satu_per_pasien  # noqa: E402
from skor_keanggotaan import PenilaiKeanggotaan  # noqa: E402
from statistik_keanggotaan import BENIH  # noqa: E402
from unit_segar import muat_atau_susun_unit, tambah_argumen_unit, unit_uji  # noqa: E402

BERKAS_KOSINUS_FM = f"kesesuaian_fm_segar_b{BENIH}.npz"
BERKAS_TOPENG_SAMA = f"kehilangan_topeng_sama_b{BENIH}.npz"
N_CEK_DETERMINISME = 20
LAPOR_TIAP = 5000


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    argumen = baca_argumen()
    unit = muat_atau_susun_unit(argumen)
    cetak_unit(unit)
    folder = Path(argumen.keluaran)
    if not (folder / BERKAS_KOSINUS_FM).exists():
        hitung_kosinus_fm(unit_uji(unit), argumen, folder / BERKAS_KOSINUS_FM)
    if not (folder / BERKAS_TOPENG_SAMA).exists():
        hitung_topeng_sama(argumen, folder / BERKAS_TOPENG_SAMA)
    print("Selesai.")


def baca_argumen():
    pengurai = argparse.ArgumentParser()
    tambah_argumen_unit(pengurai)
    pengurai.add_argument("--kontrol", default=str(FOLDER_KONTROL))
    return pengurai.parse_args()


def cetak_unit(unit):
    print(f"\nUnit segar langkah 19a; benih {BENIH}.\n")
    tabel(["Peran", "Dilihat", "Tidak dilihat (valid)", "Tidak dilihat (test)"],
          [[peran, tampil_hitungan(bagian.dilihat.sum()), tampil_hitungan(bagian.split_ecgfm.eq("valid").sum()),
            tampil_hitungan(bagian.split_ecgfm.eq("test").sum())] for peran, bagian in unit.groupby("peran")])


def hitung_kosinus_fm(uji, argumen, jalur):
    import torch
    perangkat = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    penilai = {"ecgfm": PenilaiKeanggotaan(perangkat), "ecgfm_acak": penilai_acak(perangkat)}
    kosinus = {kode: np.full(len(uji), np.nan) for kode in penilai}
    for nomor, sinyal in enumerate(baca_berurutan(argumen.zip, list(uji.waveform_path), argumen.pekerja)):
        if sinyal is not None:
            for kode, satu in penilai.items():
                kosinus[kode][nomor] = kosinus_rekaman(satu, sinyal)
        lapor(nomor, "kosinus")
    np.savez(jalur, baris_gelombang=uji.baris_gelombang.to_numpy(), **{f"kosinus_{k}": v for k, v in kosinus.items()})
    print(f"Kosinus ECG-FM terbaca {tampil_hitungan(np.isfinite(kosinus['ecgfm']).sum())} dari "
          f"{tampil_hitungan(len(uji))}.", flush=True)


def hitung_topeng_sama(argumen, jalur):
    import torch
    perangkat = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pilihan = pilihan_satu_per_pasien(muat_unit(argumen.replikasi))
    tidak = pilihan[~pilihan.dilihat].reset_index(drop=True)
    asli = PenilaiKeanggotaan(perangkat)
    plus4 = penilai_dengan_bobot(perangkat, muat_bobot(Path(argumen.kontrol), 4))
    kehilangan = np.full((len(tidak), 2), np.nan)
    for nomor, sinyal in enumerate(baca_berurutan(argumen.zip, list(tidak.waveform_path), argumen.pekerja)):
        if sinyal is not None:
            kehilangan[nomor] = [kehilangan_bertopeng_tetap(m, sinyal, nomor) for m in (asli, plus4)]
            if nomor < N_CEK_DETERMINISME:
                pastikan_deterministik(asli, sinyal, nomor, kehilangan[nomor, 0])
        lapor(nomor, "kehilangan")
    np.savez(jalur, baris_gelombang=tidak.baris_gelombang.to_numpy(), kehilangan_asli=kehilangan[:, 0],
             kehilangan_plus4=kehilangan[:, 1])
    print(f"Kehilangan bertopeng sama terbaca {tampil_hitungan(np.isfinite(kehilangan).all(axis=1).sum())} dari "
          f"{tampil_hitungan(len(tidak))}; cek determinisme {N_CEK_DETERMINISME} rekaman pertama: identik.", flush=True)


def pastikan_deterministik(penilai, sinyal, nomor, pertama):
    if kehilangan_bertopeng_tetap(penilai, sinyal, nomor) != pertama:
        raise SystemExit(f"Skor rekaman ke-{nomor} berubah pada penilaian ulang; topeng tidak deterministik.")


def lapor(nomor, nama):
    if (nomor + 1) % LAPOR_TIAP == 0:
        print(f"{time.strftime('%H:%M')} {nama} rekaman {nomor + 1}", flush=True)


if __name__ == "__main__":
    main()
