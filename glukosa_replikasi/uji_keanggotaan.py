"""Uji keanggotaan pra-latih langsung untuk Glukosa menurut protokol Artikel_Glukosa_Rencana.md §F langkah 16 (komit
8d30713): satu rekaman per pasien dari unit replikasi langkah 15, semua pasien valid ∪ test lawan sampel acak pasien
train sama banyak; skor = kehilangan InfoNCE ECG-FM pretrained sendiri (skor_keanggotaan.py); aturan M1 atas
D = rerata tidak dilihat − rerata dilihat, kontrol negatif dua paruh pasien tidak dilihat, dan sekunder deskriptif
(AUROC keanggotaan, D valid dan test, kosinus segmen). Skor per rekaman disimpan di folder replikasi (di luar repo, tidak
dibuka di sesi asisten); stdout hanya agregat (hitungan 1–10 disamarkan).

Pakai venv-ecgfm (GPU), dari akar repo, sekali:
    python -u glukosa_replikasi/uji_keanggotaan.py [--boot 2000]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AKAR_EKG = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(AKAR_EKG / d) for d in ("glukosa_perluasan", "glukosa_test", "glukosa_m3", "glukosa_m2",
                                            "glukosa_m1")]
from cetak import tabel, tampil_hitungan  # noqa: E402
from hitung_keanggotaan import muat_atau_hitung  # noqa: E402
from skor_keanggotaan import K_UNDIAN  # noqa: E402
from statistik_keanggotaan import BENIH, bootstrap, buat_data, pasien_dua_kelompok, pilih_unit  # noqa: E402
from statistik_keanggotaan import statistik_uji  # noqa: E402

BERKAS_UNIT = "unit_replikasi.csv"
KOLOM_UNIT = ["subject_id", "waveform_path", "split_ecgfm", "baris_gelombang"]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    argumen = baca_argumen()
    unit = pd.read_csv(Path(argumen.folder) / BERKAS_UNIT, usecols=KOLOM_UNIT)
    pilihan = pilih_unit(unit)
    cetak_pilihan(unit, pilihan)
    skor = muat_atau_hitung(pilihan, argumen)
    bagian = buang_tak_terpakai(pd.concat([pilihan, skor], axis=1))
    hasil = bootstrap(statistik_uji(buat_data(bagian)), len(bagian), argumen.boot)
    cetak_hasil(hasil, argumen.boot)
    cetak_keputusan(hasil)


def baca_argumen():
    pengurai = argparse.ArgumentParser()
    pengurai.add_argument("--folder", default="data/replikasi")
    pengurai.add_argument("--zip", default="data/mimic-iv-ecg-1.0.zip")
    pengurai.add_argument("--boot", type=int, default=2000)
    pengurai.add_argument("--pekerja", type=int, default=6)
    return pengurai.parse_args()


def cetak_pilihan(unit, pilihan):
    print(f"\nUji keanggotaan langkah 16; benih {BENIH}; K = {K_UNDIAN} undian topeng.\n")
    print(f"Pasien dengan rekaman di train dan di valid/test: {tampil_hitungan(pasien_dua_kelompok(unit))}.\n")
    hitungan = pilihan.split_ecgfm.value_counts()
    tabel(["Split", "Pasien (satu rekaman per pasien)"], [[s, tampil_hitungan(hitungan.get(s, 0))]
                                                          for s in ("train", "valid", "test")])


def buang_tak_terpakai(gabungan):
    terhingga = np.isfinite(gabungan.kehilangan) & np.isfinite(gabungan.kosinus)
    baris = [[nama, tampil_hitungan((~gabungan.terbaca & pilih).sum()),
              tampil_hitungan((gabungan.terbaca & ~terhingga & pilih).sum())]
             for nama, pilih in (("dilihat", gabungan.dilihat), ("tidak dilihat", ~gabungan.dilihat))]
    print("\n## Rekaman dibuang\n")
    tabel(["Kelompok", "Gagal dibaca", "Skor tak terhingga"], baris)
    return gabungan[terhingga].reset_index(drop=True)


def cetak_hasil(hasil, n_ulangan):
    print(f"\n## Statistik (bootstrap {n_ulangan} ulangan atas pasien, benih {BENIH})\n")
    tabel(["Statistik", "Nilai [95 % CI]"], [[nama, sel(*nilai)] for nama, nilai in hasil.items()])


def sel(nilai, bawah, atas):
    return f"{nilai:.4f} [{bawah:.4f}, {atas:.4f}]"


def cetak_keputusan(hasil):
    _, bawah_plasebo, atas_plasebo = hasil["D plasebo"]
    plasebo_lolos = bawah_plasebo <= 0 <= atas_plasebo
    print(f"\n## Keputusan praregistrasi\n\n- Kontrol negatif (selang D plasebo memuat 0): "
          f"{'LOLOS' if plasebo_lolos else 'GAGAL; uji bermasalah, M1 tidak dibaca'}.")
    if plasebo_lolos:
        print(f"- M1 (batas bawah D > 0): {vonis_m1(hasil['D'][1])}.")


def vonis_m1(bawah):
    if bawah > 0:
        return f"keanggotaan pra-latih TERKONFIRMASI (batas bawah {bawah:.4f})"
    return f"keanggotaan TIDAK terkonfirmasi (batas bawah {bawah:.4f}); batas 0,002 bersyarat pada daftar split"


if __name__ == "__main__":
    main()
