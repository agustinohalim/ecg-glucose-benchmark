"""Vonis audit terkalibrasi Glukosa menurut protokol Artikel_Glukosa_Rencana.md §F langkah 18: K2 (batas bawah Dₚ
ECG-FM+4 > 0), C2 (batas bawah Cₚ ECG-FM+4 > 0), M2 (batas bawah Δ_c > 0 untuk ECG-JEPA dan ECGFounder), ditambah bacaan
deskriptif. Membaca skor langkah 16–17 dan simpanan jalankan_kesesuaian.py; stdout hanya agregat (hitungan 1–10
disamarkan).

Pakai lingkungan utama (README), dari akar repo, sekali, sesudah jalankan_kesesuaian.py:
    python -u glukosa_audit/uji_audit.py [--boot 2000]
"""
import argparse
import sys
from pathlib import Path

AKAR_EKG = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(AKAR_EKG / d) for d in ("glukosa_kontrol", "glukosa_replikasi", "glukosa_perluasan", "glukosa_test",
                                            "glukosa_m3", "glukosa_m2", "glukosa_m1")]
from cetak import tabel, tampil_hitungan  # noqa: E402
from jalankan_kesesuaian import BERKAS_KESESUAIAN  # noqa: E402
from penugasan_kontrol import FOLDER_KONTROL, FOLDER_REPLIKASI, muat_unit, pilihan_satu_per_pasien  # noqa: E402
from statistik_audit import muat_berpasangan, muat_plasebo, smd_kovariat  # noqa: E402
from statistik_audit import statistik_berpasangan, statistik_plasebo  # noqa: E402
from statistik_keanggotaan import BENIH, bootstrap  # noqa: E402

NAMA = {"ecgfm": "ECG-FM", "ecgjepa": "ECG-JEPA", "ecgfounder": "ECGFounder", "plus4": "ECG-FM+4", "plus1": "ECG-FM+1"}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    argumen = baca_argumen()
    unit = muat_unit(argumen.folder)
    pilihan = pilihan_satu_per_pasien(unit)
    berpasangan = muat_berpasangan(pilihan, argumen.folder, argumen.keluaran)
    plasebo = muat_plasebo(pilihan, unit, argumen.folder, Path(argumen.keluaran) / BERKAS_KESESUAIAN)
    cetak_hitungan(berpasangan, plasebo)
    hasil_a = bootstrap(statistik_berpasangan(berpasangan), len(berpasangan.kelompok), argumen.boot)
    hasil_b = bootstrap(statistik_plasebo(plasebo), len(plasebo.kelompok), argumen.boot)
    print(f"\n## Statistik (bootstrap {argumen.boot} ulangan atas pasien per populasi, benih {BENIH})\n")
    tabel(["(a) Berpasangan, disuntik lawan ditahan", "ECG-FM+4", "ECG-FM+1"],
          [[nama, sel(hasil_a[f"{kunci} plus4"]), sel(hasil_a[f"{kunci} plus1"])] for kunci, nama in (
              ("Dp", "Dp = rerata Δℓ ditahan − disuntik"), ("Cp", "Cp = rerata Δc disuntik − ditahan"),
              ("AUROC -dl", "AUROC keanggotaan dari −Δℓ"), ("AUROC dc", "AUROC keanggotaan dari Δc"),
              ("SMD dc", "SMD Δc disuntik − ditahan"))])
    print(f"\nSMD kosinus ECG-FM+4 disuntik − ditahan (tak berpasangan): {sel(hasil_a['SMD kosinus plus4'])}\n")
    tabel(["(b) Kesesuaian segmen, dilihat − tidak dilihat", "Nilai [95 % CI]"],
          [[f"SMD {NAMA[m]}", sel(hasil_b[f"SMD {m}"])] for m in ("ecgfm", "ecgjepa", "ecgfounder")]
          + [[f"Δ = SMD ECG-FM − SMD {NAMA[m]}", sel(hasil_b[f"Delta {m}"])] for m in ("ecgjepa", "ecgfounder")]
          + [[f"SMD ECG-FM, tidak dilihat = {b}", sel(hasil_b[f"SMD ecgfm {b}"])] for b in ("valid", "test")])
    print()
    kovariat = smd_kovariat(plasebo)
    tabel(["Kovariat (dilihat − tidak dilihat)", "SMD", "Hilang"],
          [[k, f"{r.smd:.3f}", tampil_hitungan(r.hilang)] for k, r in kovariat.iterrows()])
    cetak_keputusan(hasil_a, hasil_b)


def baca_argumen():
    pengurai = argparse.ArgumentParser()
    pengurai.add_argument("--folder", default=str(FOLDER_REPLIKASI))
    pengurai.add_argument("--keluaran", default=str(FOLDER_KONTROL))
    pengurai.add_argument("--boot", type=int, default=2000)
    return pengurai.parse_args()


def cetak_hitungan(berpasangan, plasebo):
    print(f"\nAudit terkalibrasi langkah 18; benih {BENIH}.\n")
    tabel(["Populasi", "Kelompok", "Pembanding", "Dibuang"],
          [["(a) tidak dilihat: disuntik / ditahan", tampil_hitungan(berpasangan.kelompok.sum()),
            tampil_hitungan((~berpasangan.kelompok).sum()), tampil_hitungan(berpasangan.dibuang)],
           ["(b) langkah 16: dilihat / tidak dilihat", tampil_hitungan(plasebo.kelompok.sum()),
            tampil_hitungan((~plasebo.kelompok).sum()), tampil_hitungan(plasebo.dibuang)]])


def sel(nilai, desimal=4):
    titik, bawah, atas = nilai
    return f"{titik:.{desimal}f} [{bawah:.{desimal}f}, {atas:.{desimal}f}]"


def cetak_keputusan(hasil_a, hasil_b):
    print("\n## Keputusan praregistrasi\n")
    for kode, nilai, arti in (("K2", hasil_a["Dp plus4"], "latih lanjut meninggalkan jejak pada kehilangan pasien disuntik"),
                              ("C2", hasil_a["Cp plus4"], "latih lanjut menaikkan kesesuaian segmen pasien disuntik")):
        lolos = nilai[1] > 0
        print(f"- {kode} (batas bawah > 0): {'LOLOS' if lolos else 'TIDAK LOLOS'} (batas bawah {nilai[1]:.4f}); "
              f"{arti if lolos else 'tidak terbukti: ' + arti}.")
    batas = {m: hasil_b[f"Delta {m}"][1] for m in ("ecgjepa", "ecgfounder")}
    lolos = all(b > 0 for b in batas.values())
    print(f"- M2 (batas bawah Δ > 0 untuk kedua model bersih): {'LOLOS' if lolos else 'TIDAK LOLOS'} "
          f"(ECG-JEPA {batas['ecgjepa']:.4f}, ECGFounder {batas['ecgfounder']:.4f}); "
          f"{'daftar didukung oleh jejak khas ECG-FM' if lolos else 'daftar tetap tidak terverifikasi'}.")


if __name__ == "__main__":
    main()
