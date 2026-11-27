"""Vonis langkah 19 Glukosa menurut protokol Artikel_Glukosa_Rencana.md §F langkah 19: M3 (batas bawah Δ_c > 0 untuk
ECG-JEPA, ECGFounder, dan ECG-FM-acak pada unit segar), D1 (batas bawah Δ_DI > 0 untuk ECG-JEPA dan ECGFounder), K3
(batas bawah Dₚ > 0 dengan topeng sama), ditambah bacaan deskriptif 19e. Membaca simpanan jalankan_fm_segar.py,
jalankan_bersih_segar.py, embedding kohort langkah 11, dan skor langkah 16–18; stdout hanya agregat (hitungan 1–10
disamarkan).

Pakai lingkungan utama (README), dari akar repo, sekali, sesudah kedua skrip bacaan:
    python -u glukosa_audit2/uji_audit2.py [--boot 2000]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

AKAR_EKG = Path(__file__).resolve().parent.parent
FOLDER_PAKAI = ("glukosa_audit", "glukosa_kontrol", "glukosa_replikasi", "glukosa_perluasan", "glukosa_test",
                "glukosa_m3", "glukosa_m2", "glukosa_m1")
sys.path[:0] = [str(AKAR_EKG / d) for d in FOLDER_PAKAI]
from cetak import tabel, tampil_hitungan  # noqa: E402
from jalankan_bersih_segar import BERKAS_KOSINUS_BERSIH  # noqa: E402
from jalankan_fm_segar import BERKAS_KOSINUS_FM, BERKAS_TOPENG_SAMA  # noqa: E402
from jalankan_kesesuaian import BERKAS_KESESUAIAN  # noqa: E402
from penugasan_kontrol import FOLDER_KONTROL, FOLDER_REPLIKASI, muat_unit, pilihan_satu_per_pasien  # noqa: E402
from statistik_audit import muat_plasebo, smd_kovariat  # noqa: E402
from statistik_audit2 import BERSIH, KOVARIAT_SEGAR, MODEL_DI, daya_pisah, muat_segar, muat_topeng_sama  # noqa: E402
from statistik_audit2 import statistik_segar, statistik_topeng_sama  # noqa: E402
from statistik_keanggotaan import BENIH, bootstrap  # noqa: E402
from unit_segar import BERKAS_UNIT_SEGAR, FOLDER_AUDIT  # noqa: E402

NAMA = {"ecgfm": "ECG-FM", "ecgfm_acak": "ECG-FM-acak", "ecgjepa": "ECG-JEPA", "ecgfounder": "ECGFounder"}
RUJUKAN_M3 = ("ecgjepa", "ecgfounder", "ecgfm_acak")
Z_SELANG = 3.92
FAKTOR_TERDETEKSI = 2.8


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    argumen = baca_argumen()
    folder = Path(argumen.keluaran)
    unit_lama = muat_unit(argumen.replikasi)
    pilihan = pilihan_satu_per_pasien(unit_lama)
    segar = muat_segar(pd.read_csv(folder / BERKAS_UNIT_SEGAR), folder / BERKAS_KOSINUS_FM,
                       folder / BERKAS_KOSINUS_BERSIH, argumen.singgahan)
    topeng = muat_topeng_sama(pilihan, folder / BERKAS_TOPENG_SAMA)
    lama = muat_plasebo(pilihan, unit_lama, argumen.replikasi, Path(argumen.kontrol) / BERKAS_KESESUAIAN)
    cetak_hitungan(segar, topeng, lama)
    hasil_s = bootstrap(statistik_segar(segar), len(segar.kelompok), argumen.boot)
    hasil_k = bootstrap(statistik_topeng_sama(topeng), len(topeng.kelompok), argumen.boot)
    cetak_statistik(hasil_s, hasil_k, argumen.boot)
    cetak_deskriptif(segar, lama)
    cetak_keputusan(hasil_s, hasil_k)


def baca_argumen():
    pengurai = argparse.ArgumentParser()
    pengurai.add_argument("--replikasi", default=str(FOLDER_REPLIKASI))
    pengurai.add_argument("--kontrol", default=str(FOLDER_KONTROL))
    pengurai.add_argument("--keluaran", default=str(FOLDER_AUDIT))
    pengurai.add_argument("--singgahan", default="data/mimic-iv-ext-ecg-glucose/gelombang")
    pengurai.add_argument("--boot", type=int, default=2000)
    return pengurai.parse_args()


def cetak_hitungan(segar, topeng, lama):
    print(f"\nLangkah 19; benih {BENIH}.\n")
    tabel(["Populasi", "Kelompok", "Pembanding", "Dibuang"],
          [["segar (b, c, e): dilihat / tidak dilihat", tampil_hitungan(segar.kelompok.sum()),
            tampil_hitungan((~segar.kelompok).sum()), tampil_hitungan(segar.dibuang)],
           ["topeng sama (d): disuntik / ditahan", tampil_hitungan(topeng.kelompok.sum()),
            tampil_hitungan((~topeng.kelompok).sum()), tampil_hitungan(topeng.dibuang)],
           ["langkah 16/18 (e): dilihat / tidak dilihat", tampil_hitungan(lama.kelompok.sum()),
            tampil_hitungan((~lama.kelompok).sum()), tampil_hitungan(lama.dibuang)]])


def cetak_statistik(hasil_s, hasil_k, n_boot):
    print(f"\n## Statistik (bootstrap {n_boot} ulangan atas pasien per populasi, benih {BENIH})\n")
    tabel(["(b) Kesesuaian segmen segar, dilihat − tidak dilihat", "Nilai [95 % CI]", "Efek terdeteksi minimum"],
          [[f"SMD {NAMA[m]}", sel(hasil_s[f"SMD {m}"]), ""] for m in ("ecgfm", *RUJUKAN_M3)]
          + [[f"Δ_c = SMD ECG-FM − SMD {NAMA[m]}", sel(hasil_s[f"Delta {m}"]), mde(hasil_s[f"Delta {m}"])]
             for m in RUJUKAN_M3]
          + [[f"SMD ECG-FM, tidak dilihat = {b}", sel(hasil_s[f"SMD ecgfm {b}"]), ""] for b in ("valid", "test")])
    print()
    tabel(["(c) Inferensi dataset, log-likelihood dilihat − tidak dilihat", "Nilai [95 % CI]",
           "Efek terdeteksi minimum"],
          [[f"SMD {NAMA[m]}", sel(hasil_s[f"SMD di {m}"]), ""] for m in MODEL_DI]
          + [[f"Δ_DI = SMD ECG-FM − SMD {NAMA[m]}", sel(hasil_s[f"Delta di {m}"]), mde(hasil_s[f"Delta di {m}"])]
             for m in BERSIH])
    print()
    tabel(["(d) Topeng sama, disuntik lawan ditahan", "Nilai [95 % CI]", "Efek terdeteksi minimum"],
          [["Dp = rerata Δℓ ditahan − disuntik", sel(hasil_k["Dp"]), mde(hasil_k["Dp"])],
           ["AUROC keanggotaan dari −Δℓ", sel(hasil_k["AUROC -dl"]), ""]])


def cetak_deskriptif(segar, lama):
    print("\n## Deskriptif 19e\n")
    for judul, pop, kode in (("segar, kosinus", segar, ("ecgfm", *RUJUKAN_M3)),
                             ("segar, log-likelihood", segar, tuple(f"di {m}" for m in MODEL_DI)),
                             ("langkah 16/18, kosinus", lama, ("ecgfm", *BERSIH))):
        daya = daya_pisah(pop, kode)
        tabel([f"Daya pisah per rekaman ({judul})", *daya.columns],
              [[k, *(f"{v:.4f}" for v in baris)] for k, baris in daya.iterrows()])
        print()
    kovariat = smd_kovariat(segar, KOVARIAT_SEGAR)
    tabel(["Kovariat segar (dilihat − tidak dilihat)", "SMD", "Hilang"],
          [[k, f"{r.smd:.3f}", tampil_hitungan(r.hilang)] for k, r in kovariat.iterrows()])


def sel(nilai, desimal=4):
    titik, bawah, atas = nilai
    return f"{titik:.{desimal}f} [{bawah:.{desimal}f}, {atas:.{desimal}f}]"


def mde(nilai):
    return f"{FAKTOR_TERDETEKSI * (nilai[2] - nilai[1]) / Z_SELANG:.4f}"


def cetak_keputusan(hasil_s, hasil_k):
    print("\n## Keputusan praregistrasi\n")
    for kode, rujukan, awalan, arti in (
            ("M3", RUJUKAN_M3, "Delta", "dukungan daftar direplikasi pada data segar melawan plasebo sepadan masukan"),
            ("D1", BERSIH, "Delta di", "inferensi dataset mendukung daftar")):
        batas = {m: hasil_s[f"{awalan} {m}"][1] for m in rujukan}
        lolos = all(b > 0 for b in batas.values())
        rinci = ", ".join(f"{NAMA[m]} {b:.4f}" for m, b in batas.items())
        print(f"- {kode} (batas bawah > 0 untuk semua rujukan): {'LOLOS' if lolos else 'TIDAK LOLOS'} ({rinci}); "
              f"{arti if lolos else 'tidak terbukti: ' + arti}.")
    lolos = hasil_k["Dp"][1] > 0
    arti = ("kehilangan mencatat suntikan bila derau topeng dihapus" if lolos
            else "kehilangan bertopeng ini tidak mencatat empat epoch suntikan")
    print(f"- K3 (batas bawah Dp > 0): {'LOLOS' if lolos else 'TIDAK LOLOS'} (batas bawah {hasil_k['Dp'][1]:.4f}); "
          f"{arti}.")


if __name__ == "__main__":
    main()
