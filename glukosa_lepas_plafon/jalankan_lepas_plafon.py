"""Langkah 21 Glukosa (protokol Artikel_Glukosa_Rencana.md §F langkah 21): DiD pra-latih ECG-FM lawan ECG-JEPA pada
label ECG-FM fine-tuned yang belum dinilai dan setia polanya, di rekaman replikasi langkah 15. Keluarga utama = tugas
yang AUROC ECG-JEPA di luar lipatan atas semua unit ≤ 0,95 (lepas plafon). Gerbang G21: batas bawah rerata DiD
fine-tuned > 0. Uji H21: vonis rerata DiD pretrained dengan margin 0,01. stdout hanya agregat.

Pakai lingkungan utama (README), dari akar repo, sekali, sesudah siapkan_lepas_plafon.py:
    python -u glukosa_lepas_plafon/jalankan_lepas_plafon.py [--boot 2000]
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

AKAR_EKG = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(AKAR_EKG / "glukosa_replikasi")]
from prediksi_replikasi import muat_konteks, prediksi_semua  # noqa: E402
from statistik_replikasi import SKALA, rerata_did, sampel, statistik  # noqa: E402
from cetak import dengan_selang, tabel, tampil_hitungan  # noqa: E402
from kontras_perluasan import vonis  # noqa: E402
from model_fondasi import ECGFM, ECGFM_FT, ECGJEPA  # noqa: E402
from siapkan_lepas_plafon import BERKAS_UNIT_BARU  # noqa: E402

FM, FT, JEPA = ECGFM.kode, ECGFM_FT.kode, ECGJEPA.kode
SETIA = ("infark", "pvc", "flutter", "pacu", "mutu", "sinus")
AMBANG_PLAFON = 0.95
AMBANG_KEMURNIAN = (0.01, 0.005)
BENIH_LIPATAN = 20261120


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    argumen = baca_argumen()
    folder = Path(argumen.folder)
    konteks = muat_konteks(folder, argumen.boot)
    baru = pd.read_csv(folder / BERKAS_UNIT_BARU).drop(columns="study_id")
    konteks = replace(konteks, unit=konteks.unit.merge(baru, on="baris_gelombang", how="left", validate="one_to_one"))
    print(f"\nLangkah 21; bootstrap {argumen.boot}, benih lipatan {BENIH_LIPATAN}.\n")
    peluang = prediksi_semua(konteks, SETIA, BENIH_LIPATAN)
    keluarga = pilih_keluarga(konteks.unit, peluang)
    if not keluarga:
        print("\nTidak ada tugas lepas plafon: langkah 21 tidak informatif; H21 tidak diuji.")
        return
    unit = konteks.unit
    varian = {"Utama (tidak dilihat = valid ∪ test)": np.ones(len(unit), bool),
              "TEST saja": unit.split_ecgfm.isin(["train", "test"]).values,
              "Pasien di luar kohort": unit.lepas_pasien.values.astype(bool)}
    hasil = {judul: sampel(konteks, peluang, pilih, SETIA) for judul, pilih in varian.items()}
    for judul, sampel_ in hasil.items():
        cetak_varian(judul, sampel_, keluarga)
    cetak_keputusan(next(iter(hasil.values())), keluarga)


def baca_argumen():
    pengurai = argparse.ArgumentParser()
    pengurai.add_argument("--folder", default="data/replikasi")
    pengurai.add_argument("--boot", type=int, default=2000)
    return pengurai.parse_args()


def pilih_keluarga(unit, peluang):
    auroc = {k: roc_auc_score(unit[f"y_{k}"].values, peluang[k][JEPA]) for k in SETIA}
    keluarga = tuple(k for k in SETIA if auroc[k] <= AMBANG_PLAFON)
    print("## Seleksi lepas plafon (AUROC ECG-JEPA di luar lipatan, semua unit)\n")
    tabel(["Tugas", "Positif", "AUROC ECG-JEPA", f"Masuk keluarga utama (≤ {AMBANG_PLAFON})"],
          [[k, tampil_hitungan(unit[f"y_{k}"].sum()), f"{auroc[k]:.4f}", "ya" if k in keluarga else "tidak"]
           for k in SETIA])
    return keluarga


def cetak_varian(judul, sampel_, keluarga):
    k0 = next(iter(sampel_.kontras.values()))
    print(f"\n## {judul}: {tampil_hitungan(k0.dilihat.sum())} dilihat, {tampil_hitungan((~k0.dilihat).sum())} tidak "
          f"dilihat; ulangan terpakai {len(sampel_.ulangan)}\n")
    kelompok = [(k, (k,)) for k in SETIA] + [("Rerata keluarga utama", keluarga)]
    baris = []
    for nama, anggota in kelompok:
        for skala in SKALA:
            sel = [statistik(sampel_, rerata_did(anggota, SETIA, m, skala)) for m in (FM, FT)]
            baris.append([nama, skala, *(sel_did(n, b, a, skala) for n, b, a in sel)])
    tabel(["Tugas", "Skala", "DiD ECG-FM pretrained", "DiD ECG-FM fine-tuned (kontrol positif)"], baris)


def sel_did(nilai, bawah, atas, skala):
    teks = f"{nilai:.4f} [{bawah:.4f}, {atas:.4f}]"
    return f"{teks} {vonis(bawah, atas)}" if skala == "AUROC" else teks


def cetak_keputusan(utama, keluarga):
    _, bawah_ft, _ = statistik(utama, rerata_did(keluarga, SETIA, FT, "AUROC"))
    nilai, bawah, atas = statistik(utama, rerata_did(keluarga, SETIA, FM, "AUROC"))
    g21 = bawah_ft > 0
    print(f"\n## Keputusan langkah 21\n\n- Keluarga utama: {', '.join(keluarga)}.")
    print(f"- G21 (batas bawah rerata DiD fine-tuned > 0): {'LOLOS' if g21 else 'GAGAL'} ({bawah_ft:.4f}).")
    print(f"- H21 rerata DiD pretrained: {dengan_selang(nilai, (bawah, atas))}, vonis {vonis(bawah, atas)}"
          f"{'' if g21 else ' (tidak ditafsirkan: G21 gagal)'}.")
    for ambang in AMBANG_KEMURNIAN:
        batas = atas / ambang if atas > 0 else 0.0
        print(f"- Batas atas melewati {ambang} bila selisih laju keanggotaan sejati antarkelompok < {batas:.3f}.")


if __name__ == "__main__":
    main()
