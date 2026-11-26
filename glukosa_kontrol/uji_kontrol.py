"""Vonis kontrol positif swaawas Glukosa menurut protokol Artikel_Glukosa_Rencana.md §F langkah 17d–e: N0 (celah ECG-JEPA
pada LVH dan rerata lima pernyataan memuat nol), K1 (batas bawah Dₛ ECG-FM+4 > 0), P1 (batas bawah E LVH > 0), P2 (batas
bawah E rerata empat tugas berlabel > 0), ditambah bacaan sekunder deskriptif. Membaca embedding dan skor dari
jalankan_latih.py dan jalankan_bacaan.py; stdout hanya agregat (hitungan 1–10 disamarkan).

Pakai lingkungan utama (README), dari akar repo, sekali:
    python -u glukosa_kontrol/uji_kontrol.py [--boot 2000]
"""
import argparse
import sys
from pathlib import Path

AKAR_EKG = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(AKAR_EKG / d) for d in ("glukosa_replikasi", "glukosa_perluasan", "glukosa_test", "glukosa_m3",
                                            "glukosa_m2", "glukosa_m1")]
from cetak import tabel, tampil_hitungan  # noqa: E402
from penugasan_kontrol import FOLDER_KONTROL, FOLDER_REPLIKASI, muat_unit, pasien_disuntik  # noqa: E402
from penugasan_kontrol import pilihan_satu_per_pasien, unit_bacaan  # noqa: E402
from statistik_kontrol import KELOMPOK, MODEL, SEMUA, bootstrap, muat_data  # noqa: E402
from statistik_keanggotaan import BENIH  # noqa: E402
from tugas_pernyataan import NAMA_TUGAS  # noqa: E402

NAMA_MODEL = {"ecgjepa": "ECG-JEPA", "ecgfm": "ECG-FM asli", "plus1": "ECG-FM+1", "plus4": "ECG-FM+4"}
BARIS = [(t, NAMA_TUGAS[t]) for t in SEMUA] + [(k, k) for k in KELOMPOK]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    argumen = baca_argumen()
    unit = muat_unit(argumen.folder)
    pilihan = pilihan_satu_per_pasien(unit)
    bacaan = unit_bacaan(unit, pasien_disuntik(pilihan))
    tidak = pilihan[~pilihan.dilihat].reset_index(drop=True)
    data = muat_data(bacaan, tidak, argumen.folder, argumen.keluaran)
    cetak_unit(data)
    hasil, terpakai = bootstrap(data, argumen.boot)
    print(f"\n## Statistik (bootstrap {argumen.boot} ulangan atas pasien, terpakai {terpakai}, benih {BENIH})\n")
    tabel(["Tugas", *(f"Celah {NAMA_MODEL[m]}" for m in MODEL)],
          [[nama, *(sel(hasil[f"g {m} {k}"]) for m in MODEL)] for k, nama in BARIS])
    print()
    tabel(["Tugas", "E ECG-FM+4 (utama)", "E ECG-FM+1", "E ECG-FM+4 lawan ECG-JEPA"],
          [[nama, sel(hasil[f"E plus4 {k}"]), sel(hasil[f"E plus1 {k}"]), sel(hasil[f"E plus4 lawan ECG-JEPA {k}"])]
           for k, nama in BARIS])
    print()
    tabel(["Bacaan kehilangan", "Nilai [95 % CI]"],
          [[f"Ds {NAMA_MODEL[m]} (ditahan − disuntik)", sel(hasil[f"Ds {m}"], 4)] for m in ("ecgfm", "plus1", "plus4")]
          + [["AUROC keanggotaan ECG-FM+4", sel(hasil["AUROC keanggotaan plus4"], 4)],
             ["Kosinus segmen ECG-FM+4 (disuntik − ditahan)", sel(hasil["kosinus plus4"], 4)]])
    cetak_keputusan(hasil)


def baca_argumen():
    pengurai = argparse.ArgumentParser()
    pengurai.add_argument("--folder", default=str(FOLDER_REPLIKASI))
    pengurai.add_argument("--keluaran", default=str(FOLDER_KONTROL))
    pengurai.add_argument("--boot", type=int, default=2000)
    return pengurai.parse_args()


def cetak_unit(data):
    print(f"\nKontrol positif swaawas langkah 17; benih {BENIH}.\n")
    tabel(["Dibuang", "Rekaman"], [[k, tampil_hitungan(v)] for k, v in data.dibuang.items()])
    print()
    contoh = next(iter(data.kontras.values()))
    isi = [["rekaman", tampil_hitungan(contoh.dilihat.sum()), tampil_hitungan((~contoh.dilihat).sum())],
           ["pasien dengan skor kehilangan", tampil_hitungan(data.disuntik_skor.sum()),
            tampil_hitungan((~data.disuntik_skor).sum())]]
    isi += [[f"positif {NAMA_TUGAS[t]}", tampil_hitungan(k.y[k.dilihat].sum()), tampil_hitungan(k.y[~k.dilihat].sum())]
            for t, k in data.kontras.items()]
    tabel(["Hitungan", "Disuntik", "Ditahan"], isi)


def sel(nilai, desimal=3):
    titik, bawah, atas = nilai
    return f"{titik:.{desimal}f} [{bawah:.{desimal}f}, {atas:.{desimal}f}]"


def cetak_keputusan(hasil):
    n0 = all(hasil[k][1] <= 0 <= hasil[k][2] for k in ("g ecgjepa lvh", "g ecgjepa rerata lima pernyataan"))
    print("\n## Keputusan praregistrasi\n")
    print(f"- N0 (celah ECG-JEPA LVH dan rerata lima memuat 0): {'LOLOS' if n0 else 'TIDAK LOLOS'}.")
    for kode, kunci, arti in (("K1", "Ds plus4", "uji kehilangan peka pada anggota yang diketahui"),
                              ("P1", "E plus4 lvh", "paparan swaawas terdeteksi pada LVH"),
                              ("P2", "E plus4 rerata empat berlabel", "paparan swaawas terdeteksi di plafon")):
        bawah = hasil[kunci][1]
        print(f"- {kode} (batas bawah {kunci} > 0): {'LOLOS' if bawah > 0 else 'TIDAK LOLOS'} "
              f"(batas bawah {bawah:.4f}); {arti if bawah > 0 else 'tidak terbukti: ' + arti}.")


if __name__ == "__main__":
    main()
