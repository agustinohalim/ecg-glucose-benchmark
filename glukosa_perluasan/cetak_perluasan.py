"""Cetak kontras protokol §F langkah 11d–f per tugas dan ringkasan lima tugas pernyataan: AUROC per kelompok paparan,
DiD utama/sekunder/plasebo/kontrol positif dengan CI bootstrap, skala logit, p permutasi, dan vonis. Agregat saja."""
import numpy as np

from cetak import dengan_selang, tabel, tampil_hitungan
from kontras_berbobot import DILIHAT, TIDAK, auroc_per_kelompok, did_dari, logit
from kontras_perluasan import p_permutasi, positif_cukup, selang, vonis

PLAFON = 0.97
GB, FOUNDER, JEPA, FM, FT = "gb", "ecgfounder", "ecgjepa", "ecgfm", "ecgfm_ft"
LABEL = {GB: "Model ukuran (GB)", FOUNDER: "ECGFounder (bersih)", JEPA: "ECG-JEPA (pembanding)",
         FM: "ECG-FM pretrained (terpapar)", FT: "ECG-FM fine-tuned (kontrol positif)"}
KONTRAS_DID = [(FM, JEPA, "utama"), (FM, FOUNDER, "sekunder"), (FOUNDER, JEPA, "plasebo"), (FT, JEPA, "kontrol positif")]


def cetak_ukuran(kontras):
    isi = [[nama, tampil_hitungan(pilih.sum()), tampil_hitungan(len(np.unique(kontras.pasien[pilih]))),
            tampil_hitungan(kontras.y[pilih].sum())]
           for nama, pilih in (("dilihat", kontras.dilihat), ("tidak dilihat", ~kontras.dilihat))]
    tabel(["Kelompok", "Unit", "Pasien", "Positif"], isi)
    print()


def cetak_model(kontras, titik, ulangan, permutasi):
    teramati, nol = permutasi
    isi = []
    for m in kontras.peringkat:
        celah = lambda a, m=m: a[(m, DILIHAT)] - a[(m, TIDAK)]
        isi.append([LABEL[m], dengan_selang(titik[(m, DILIHAT)], selang(ulangan, lambda a, m=m: a[(m, DILIHAT)])),
                    dengan_selang(titik[(m, TIDAK)], selang(ulangan, lambda a, m=m: a[(m, TIDAK)])),
                    dengan_selang(celah(titik), selang(ulangan, celah)), f"{p_permutasi(teramati[m], nol[m]):.3f}"])
    tabel(["Model", "AUROC dilihat", "AUROC tidak dilihat", "Dilihat − tidak", "p permutasi"], isi)
    print()


def baris_did(titik, ulangan, permutasi, kontras_did):
    a, b, jenis = kontras_did
    teramati, nol = permutasi
    batas = selang(ulangan, lambda u: did_dari(u, a, b))
    return [jenis, f"{LABEL[a]} − {LABEL[b]}", dengan_selang(did_dari(titik, a, b), batas),
            dengan_selang(did_dari(titik, a, b, logit), selang(ulangan, lambda u: did_dari(u, a, b, logit))),
            f"{p_permutasi(teramati[a] - teramati[b], nol[a] - nol[b]):.3f}", vonis(*batas)]


def cetak_did(titik, ulangan, permutasi):
    model = {m for m, _ in titik}
    isi = [baris_did(titik, ulangan, permutasi, k) for k in KONTRAS_DID if k[0] in model]
    tabel(["Jenis", "DiD", "Nilai [95 % CI]", "Skala logit [95 % CI]", "p permutasi", "Vonis (ε = 0,01)"], isi)
    plafon = min(titik[(JEPA, DILIHAT)], titik[(JEPA, TIDAK)]) > PLAFON
    print(f"\nDekat plafon (AUROC ECG-JEPA kedua kelompok > {PLAFON}): {'ya' if plafon else 'tidak'}.\n")


def cetak_tugas(nama, kontras, ulangan, permutasi):
    print(f"### {nama}\n")
    cetak_ukuran(kontras)
    if not positif_cukup(kontras):
        print("Positif < 11 pada salah satu kelompok; kontras tidak dicetak.\n")
        return
    titik = auroc_per_kelompok(kontras, np.ones(len(kontras.y)))
    print(f"Ulangan bootstrap terpakai: {len(ulangan)}; permutasi: {len(next(iter(permutasi[1].values())))}.\n")
    cetak_model(kontras, titik, ulangan, permutasi)
    cetak_did(titik, ulangan, permutasi)


def cetak_ringkasan(daftar_kontras, ulangan, judul):
    titik = [auroc_per_kelompok(k, np.ones(len(k.y))) for k in daftar_kontras]
    isi = []
    for a, b, jenis in KONTRAS_DID:
        rerata = lambda daftar, a=a, b=b: np.mean([did_dari(t, a, b) for t in daftar])
        batas = selang(ulangan, rerata)
        isi.append([jenis, f"{LABEL[a]} − {LABEL[b]}", dengan_selang(rerata(titik), batas), vonis(*batas)])
    print(f"### {judul}\n")
    tabel(["Jenis", "DiD", "Rerata [95 % CI]", "Vonis (ε = 0,01)"], isi)
    print()
