"""Replikasi konfirmatori Glukosa menurut protokol Artikel_Glukosa_Rencana.md §F langkah 15 (komit d24c095): probe L2
cross-fitting di atas embedding beku ECG-JEPA, ECG-FM pretrained dan ECG-FM fine-tuned pada rekaman daftar split ECG-FM
di luar kohort glukosa; DiD AUROC dilihat − tidak dilihat lawan ECG-JEPA dengan CI bootstrap bergugus pasien; gerbang R1
(kontrol positif) dan uji konfirmatori R2 (batas atas rerata DiD pretrained empat tugas berlabel < 0,002); sekunder
deskriptif (lima pernyataan, per tugas, logit, TEST saja, pasien di luar kohort, benih lipatan, keseimbangan SMD).
stdout hanya agregat.

Pakai lingkungan utama (README), dari akar repo, sesudah siapkan_replikasi.py dan ketiga embedding selesai:
    python -u glukosa_replikasi/jalankan_replikasi.py [--boot 2000]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from prediksi_replikasi import muat_konteks, prediksi_semua
from statistik_replikasi import BENIH_BOOT, SKALA, rerata_did, sampel, statistik
from cetak import dengan_selang, tabel, tampil_hitungan
from keseimbangan_paparan import smd
from kontras_perluasan import vonis
from model_fondasi import ECGFM, ECGFM_FT
from tugas_pernyataan import LABEL_FINETUNED, NAMA_TUGAS, POLA_TUGAS

FM, FT = ECGFM.kode, ECGFM_FT.kode
SEMUA = tuple(POLA_TUGAS)
BERLABEL = tuple(k for k in SEMUA if k in LABEL_FINETUNED)
BATAS_R2 = 0.002


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    argumen = baca_argumen()
    konteks = muat_konteks(Path(argumen.folder), argumen.boot)
    print(f"\nReplikasi langkah 15; bootstrap {argumen.boot}, benih bootstrap {BENIH_BOOT}, benih lipatan "
          f"{argumen.benih}.\n")
    cetak_keseimbangan(konteks.unit)
    peluang = prediksi_semua(konteks, SEMUA, argumen.benih)
    utama = sampel(konteks, peluang, np.ones(len(konteks.unit), bool), SEMUA)
    cetak_varian("Utama (tidak dilihat = valid ∪ test)", utama)
    cetak_keputusan(utama)
    for judul, pilih in (("TEST saja", konteks.unit.split_ecgfm.isin(["train", "test"]).values),
                         ("Pasien di luar kohort", konteks.unit.lepas_pasien.values.astype(bool))):
        cetak_varian(judul, sampel(konteks, peluang, pilih, SEMUA))
    cetak_kepekaan_lipatan(konteks, argumen, utama)


def baca_argumen():
    pengurai = argparse.ArgumentParser()
    pengurai.add_argument("--folder", default="data/replikasi")
    pengurai.add_argument("--boot", type=int, default=2000)
    pengurai.add_argument("--benih", type=int, default=20261120)
    pengurai.add_argument("--benih_kepekaan", type=int, nargs="*", default=[20261121, 20261122])
    return pengurai.parse_args()


def cetak_keseimbangan(unit):
    dilihat = unit.split_ecgfm.eq("train").values
    peubah = {f"positif {NAMA_TUGAS[k]}": unit[f"y_{k}"] for k in SEMUA}
    peubah["unit per pasien"] = unit.groupby("subject_id").subject_id.transform("size")
    peubah["denyut jantung"], peubah["durasi QRS"] = unit.denyut, unit.qrs_ms
    baris = []
    for nama, nilai in peubah.items():
        v = np.asarray(nilai, dtype=float)
        a, b = v[dilihat & np.isfinite(v)], v[~dilihat & np.isfinite(v)]
        baris.append([nama, f"{a.mean():.3f}", f"{b.mean():.3f}", f"{smd(a, b):.3f}"])
    print("## Keseimbangan kelompok\n")
    tabel(["Peubah", "Dilihat", "Tidak dilihat", "SMD"], baris)


def cetak_varian(judul, sampel_):
    tugas = list(sampel_.kontras)
    k0 = next(iter(sampel_.kontras.values()))
    print(f"\n## {judul}: {tampil_hitungan(k0.dilihat.sum())} dilihat, {tampil_hitungan((~k0.dilihat).sum())} tidak "
          f"dilihat; ulangan terpakai {len(sampel_.ulangan)}\n")
    kelompok = [(NAMA_TUGAS[k], (k,)) for k in tugas] + [("Rerata empat tugas berlabel", BERLABEL),
                                                        ("Rerata lima pernyataan", SEMUA)]
    baris = []
    for nama, anggota in kelompok:
        for skala in SKALA:
            sel = [statistik(sampel_, rerata_did(anggota, tugas, m, skala)) for m in (FM, FT)]
            baris.append([nama, skala] + [sel_did(n, b, a, skala) for n, b, a in sel])
    tabel(["Kelompok", "Skala", "DiD ECG-FM pretrained", "DiD ECG-FM fine-tuned (kontrol positif)"], baris)


def sel_did(nilai, bawah, atas, skala):
    teks = dengan_selang(nilai, (bawah, atas))
    return f"{teks} {vonis(bawah, atas)}" if skala == "AUROC" else teks


def cetak_keputusan(utama):
    tugas = list(utama.kontras)
    _, bawah_ft, _ = statistik(utama, rerata_did(BERLABEL, tugas, FT, "AUROC"))
    nilai, bawah, atas = statistik(utama, rerata_did(BERLABEL, tugas, FM, "AUROC"))
    r1, r2 = bawah_ft > 0, atas < BATAS_R2
    print(f"\n## Keputusan praregistrasi\n\n- R1 (kontrol positif, batas bawah rerata DiD fine-tuned > 0): "
          f"{'LOLOS' if r1 else 'GAGAL'} (batas bawah {bawah_ft:.4f}).")
    print(f"- R2 (batas atas rerata DiD pretrained < {BATAS_R2}): {'LOLOS' if r2 else 'GAGAL'} "
          f"({nilai:.4f} [{bawah:.4f}, {atas:.4f}]).")
    print(f"- Inflasi oleh pra-latih terdeteksi (batas bawah > 0): {'ya' if bawah > 0 else 'tidak'}.")
    print(f"- Klaim terkonfirmasi: {'ya' if r1 and r2 else 'tidak'}.")


def cetak_kepekaan_lipatan(konteks, argumen, utama):
    baris = [baris_benih(argumen.benih, utama)]
    semua_baris = np.ones(len(konteks.unit), bool)
    for benih in argumen.benih_kepekaan:
        peluang = prediksi_semua(konteks, BERLABEL, benih)
        baris.append(baris_benih(benih, sampel(konteks, peluang, semua_baris, BERLABEL)))
    print("\n## Kepekaan benih lipatan (rerata empat tugas berlabel, AUROC)\n")
    tabel(["Benih lipatan", "DiD ECG-FM pretrained", "DiD ECG-FM fine-tuned"], baris)


def baris_benih(benih, sampel_):
    tugas = list(sampel_.kontras)
    sel = [statistik(sampel_, rerata_did(BERLABEL, tugas, m, "AUROC")) for m in (FM, FT)]
    return [benih] + [dengan_selang(n, (b, a)) for n, b, a in sel]


if __name__ == "__main__":
    main()
