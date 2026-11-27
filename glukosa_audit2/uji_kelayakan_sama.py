"""Langkah 20 Glukosa (protokol Artikel_Glukosa_Rencana.md §F langkah 20, PASCA-HASIL): uji ulang M3 dan D1 langkah 19
pada unit segar yang kelayakannya sama di kedua kelompok (kelayakan_sama.py), untuk dilihat, tidak dilihat, dan set
penyetelan kepadatan. Mencetak hitungan kelayakan, SMD kovariat dilihat tak layak − layak, cek keseimbangan (gerbang
tafsir |SMD| < 0,1), statistik M3* dan D1*, dan daya pisah per rekaman. Membaca simpanan langkah 19 dan embedding
kohort langkah 11; stdout hanya agregat (hitungan 1–10 disamarkan).

Pakai lingkungan utama (README), dari akar repo, sekali:
    python -u glukosa_audit2/uji_kelayakan_sama.py [--boot 2000]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from uji_audit2 import NAMA, RUJUKAN_M3, mde, sel
from cetak import tabel, tampil_hitungan  # noqa: E402
from jalankan_bersih_segar import BERKAS_KOSINUS_BERSIH  # noqa: E402
from jalankan_fm_segar import BERKAS_KOSINUS_FM  # noqa: E402
from kelayakan_sama import bagian, dilihat_menurut_kelayakan, rekaman_layak, tandai_layak, unit_untuk_muat  # noqa: E402
from penugasan_kontrol import BERKAS_UNIT, FOLDER_REPLIKASI  # noqa: E402
from statistik_audit import smd_kovariat  # noqa: E402
from statistik_audit2 import BERSIH, KOVARIAT_SEGAR, MODEL_DI, daya_pisah, muat_segar, statistik_segar  # noqa: E402
from statistik_keanggotaan import BENIH, bootstrap  # noqa: E402
from unit_segar import BERKAS_UNIT_SEGAR, FOLDER_AUDIT  # noqa: E402

AMBANG_SEIMBANG = 0.1


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    argumen = baca_argumen()
    folder = Path(argumen.keluaran)
    unit = tandai_layak(pd.read_csv(folder / BERKAS_UNIT_SEGAR), Path(argumen.replikasi) / BERKAS_UNIT)
    cetak_hitungan(unit)
    segar = muat_segar(unit_untuk_muat(unit), folder / BERKAS_KOSINUS_FM, folder / BERKAS_KOSINUS_BERSIH,
                       argumen.singgahan)
    layak = rekaman_layak(segar, unit)
    print()
    cetak_kovariat("Kovariat dilihat (tak layak − layak)",
                   smd_kovariat(dilihat_menurut_kelayakan(segar, layak), KOVARIAT_SEGAR))
    pop = bagian(segar, layak)
    sebanding = cetak_keseimbangan(pop)
    hasil = bootstrap(statistik_segar(pop), len(pop.kelompok), argumen.boot)
    cetak_statistik(hasil, argumen.boot)
    cetak_daya_pisah(pop)
    cetak_keputusan(hasil, sebanding)


def baca_argumen():
    pengurai = argparse.ArgumentParser()
    pengurai.add_argument("--replikasi", default=str(FOLDER_REPLIKASI))
    pengurai.add_argument("--keluaran", default=str(FOLDER_AUDIT))
    pengurai.add_argument("--singgahan", default="data/mimic-iv-ext-ecg-glucose/gelombang")
    pengurai.add_argument("--boot", type=int, default=2000)
    return pengurai.parse_args()


def cetak_hitungan(unit):
    print(f"\nLangkah 20 (pasca-hasil); benih {BENIH}.\n")
    baris = []
    for peran, dilihat, nama in (("uji", True, "uji, dilihat"), ("uji", False, "uji, tidak dilihat"),
                                 ("setel", True, "setel, dilihat")):
        kelompok = unit[unit.peran.eq(peran) & unit.dilihat.eq(dilihat)]
        baris.append([nama, tampil_hitungan(kelompok.layak.sum()), tampil_hitungan((~kelompok.layak).sum())])
    tabel(["Unit segar 19a", "Layak", "Tak layak (punya rekaman luar kohort)"], baris)


def cetak_kovariat(judul, kovariat):
    tabel([judul, "SMD", "Hilang"], [[k, f"{r.smd:.3f}", tampil_hitungan(r.hilang)] for k, r in kovariat.iterrows()])


def cetak_keseimbangan(pop):
    kovariat = smd_kovariat(pop, KOVARIAT_SEGAR)
    print(f"\n## (b) Keseimbangan unit uji layak: dilihat {tampil_hitungan(pop.kelompok.sum())}, "
          f"tidak dilihat {tampil_hitungan((~pop.kelompok).sum())}, dibuang (semua unit) {tampil_hitungan(pop.dibuang)}\n")
    cetak_kovariat("Kovariat layak (dilihat − tidak dilihat)", kovariat)
    sebanding = bool((kovariat.smd.abs() < AMBANG_SEIMBANG).all())
    print(f"\nSebanding (|SMD| < {AMBANG_SEIMBANG} untuk keempat kovariat): {'YA' if sebanding else 'TIDAK'}")
    return sebanding


def cetak_statistik(hasil, n_boot):
    print(f"\n## (c, d) Statistik (bootstrap {n_boot} ulangan atas pasien, benih {BENIH})\n")
    tabel(["M3*: kesesuaian segmen, dilihat − tidak dilihat", "Nilai [95 % CI]", "Efek terdeteksi minimum"],
          [[f"SMD {NAMA[m]}", sel(hasil[f"SMD {m}"]), ""] for m in ("ecgfm", *RUJUKAN_M3)]
          + [[f"Δ_c = SMD ECG-FM − SMD {NAMA[m]}", sel(hasil[f"Delta {m}"]), mde(hasil[f"Delta {m}"])]
             for m in RUJUKAN_M3]
          + [[f"SMD ECG-FM, tidak dilihat = {b}", sel(hasil[f"SMD ecgfm {b}"]), ""] for b in ("valid", "test")])
    print()
    tabel(["D1*: log-likelihood, dilihat − tidak dilihat", "Nilai [95 % CI]", "Efek terdeteksi minimum"],
          [[f"SMD {NAMA[m]}", sel(hasil[f"SMD di {m}"]), ""] for m in MODEL_DI]
          + [[f"Δ_DI = SMD ECG-FM − SMD {NAMA[m]}", sel(hasil[f"Delta di {m}"]), mde(hasil[f"Delta di {m}"])]
             for m in BERSIH])


def cetak_daya_pisah(pop):
    print("\n## Deskriptif: daya pisah per rekaman, unit uji layak\n")
    daya = daya_pisah(pop, ("ecgfm", *RUJUKAN_M3, *(f"di {m}" for m in MODEL_DI)))
    tabel(["Skor", *daya.columns], [[k, *(f"{v:.4f}" for v in baris)] for k, baris in daya.iterrows()])


def cetak_keputusan(hasil, sebanding):
    print("\n## Keputusan langkah 20(f)\n")
    gerbang = "sebanding" if sebanding else "TIDAK sebanding; uji ulang segar tak dapat ditafsirkan"
    print(f"- Gerbang tafsir: {gerbang}.")
    for kode, rujukan, awalan in (("M3*", RUJUKAN_M3, "Delta"), ("D1*", BERSIH, "Delta di")):
        batas = {m: hasil[f"{awalan} {m}"][1] for m in rujukan}
        lolos = all(b > 0 for b in batas.values())
        rinci = ", ".join(f"{NAMA[m]} {b:.4f}" for m, b in batas.items())
        print(f"- {kode} (batas bawah > 0 untuk semua rujukan): {'LOLOS' if lolos else 'TIDAK LOLOS'} ({rinci}).")


if __name__ == "__main__":
    main()
