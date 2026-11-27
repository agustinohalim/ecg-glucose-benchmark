"""Label tugas baru langkah 21 Glukosa (protokol Artikel_Glukosa_Rencana.md §F langkah 21a) pada unit replikasi
langkah 15: menulis unit_lepas_plafon.csv ke folder replikasi (di luar repo) dan mencetak agregat saja, yaitu prevalensi
pola lawan label ECG-FM (label_def.csv) per split dan bagian positif yang juga positif pada tugas lama.

Pakai lingkungan utama (README), dari akar repo:
    python glukosa_lepas_plafon/siapkan_lepas_plafon.py
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

AKAR_EKG = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(AKAR_EKG / d) for d in ("glukosa_m1",)]
from cetak import tabel, tampil_hitungan  # noqa: E402
from tugas_lepas_plafon import LABEL_ECGFM_BARU, POLA_BARU, TUGAS_LAMA, tambah_label_baru  # noqa: E402

BERKAS_UNIT_BARU = "unit_lepas_plafon.csv"
SPLIT = ("train", "valid", "test")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    argumen = baca_argumen()
    folder = Path(argumen.folder)
    unit = tambah_label_baru(pd.read_csv(folder / "unit_replikasi.csv"), argumen.zip)
    unit[["study_id", "baris_gelombang", *(f"y_{k}" for k in POLA_BARU)]].to_csv(folder / BERKAS_UNIT_BARU, index=False)
    label_def = pd.read_csv(argumen.label_def).set_index("name")
    print(f"\nUnit replikasi: {tampil_hitungan(len(unit))} rekaman.\n")
    tabel(["Tugas", "Positif", "Pola train / valid / test (%)", "Label ECG-FM train / valid / test (%)",
           "Selisih terbesar (poin persen)", "Positif yang juga positif tugas lama (%)"],
          [baris_tugas(unit, label_def, kode) for kode in POLA_BARU])


def baca_argumen():
    pengurai = argparse.ArgumentParser()
    pengurai.add_argument("--folder", default="data/replikasi")
    pengurai.add_argument("--zip", default="data/mimic-iv-ecg-1.0.zip")
    pengurai.add_argument("--label_def", default="models/ECG-FM/label_def_mimic_iv_ecg.csv")
    return pengurai.parse_args()


def baris_tugas(unit, label_def, kode):
    pola = [100 * unit.loc[unit.split_ecgfm.eq(s), f"y_{kode}"].mean() for s in SPLIT]
    acuan = [100 * label_def.loc[LABEL_ECGFM_BARU[kode], f"pos_percent_{s}"] for s in SPLIT]
    positif = unit[f"y_{kode}"].eq(1)
    tumpang = 100 * unit.loc[positif, [f"y_{k}" for k in TUGAS_LAMA]].any(axis=1).mean()
    return [kode, tampil_hitungan(positif.sum()), " / ".join(f"{p:.2f}" for p in pola),
            " / ".join(f"{a:.2f}" for a in acuan), f"{max(abs(p - a) for p, a in zip(pola, acuan)):.2f}",
            f"{tumpang:.1f}"]


if __name__ == "__main__":
    main()
