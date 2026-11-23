"""Gambar S1 suplemen Glukosa (protokol pertama) dari tabel agregat Hasil_Glukosa_12.md (bootstrap 10.000, protokol §F langkah 10); tidak
membaca data atau prediksi. Selisih-dalam-selisih AUROC (DiD) per tugas: utama (ECG-FM lawan ECG-JEPA), sekunder, dan
plasebo antar model tanpa paparan, CI 95 % bootstrap bergugus pasien; garis putus = setengah efek utama TEST (0,033).
Keluaran: Gambar_Glukosa_S1.png (300 dpi) dan .pdf di .

Pakai, dari akar repo:
    python glukosa_test/gambar_kontaminasi.py
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AKAR_EKG = Path(__file__).resolve().parent.parent
SUMBER = AKAR_EKG / "Hasil_Glukosa_12.md"
KELUARAN = AKAR_EKG / "Gambar_Glukosa_S1"
TUGAS = [("Hipoglikemia", "Hypoglycemia", "#0072b2"), ("Hiperglikemia berat", "Severe hyperglycemia", "#d55e00")]
KONTRAS = [("M3c ECG-FM − M3b ECG-JEPA", "ECG-FM − ECG-JEPA (primary)"),
           ("M3c ECG-FM − M3 ECGFounder", "ECG-FM − ECGFounder"),
           ("M3c ECG-FM − M1-GB", "ECG-FM − measurement model"),
           ("M3 ECGFounder − M3b ECG-JEPA", "Placebo: ECGFounder − ECG-JEPA"),
           ("M1-GB − M3b ECG-JEPA", "Placebo: measurement − ECG-JEPA"),
           ("M1-GB − M3 ECGFounder", "Placebo: measurement − ECGFounder")]
NILAI_CI = re.compile(r"(-?\d+\.\d+) \[(-?\d+\.\d+), (-?\d+\.\d+)\]")
SETENGAH_EFEK_UTAMA = 0.033
GESER = 0.15


def bagian_tugas(teks, nama_tugas):
    awal = teks.index(f"### {nama_tugas} (")
    akhir = teks.find("\n### ", awal + 1)
    return teks[awal:akhir if akhir > 0 else None]


def baris_did(bagian):
    awal = bagian.index("| Jenis | DiD |")
    akhir = bagian.index("\n\n", awal)
    return [baris for baris in bagian[awal:akhir].splitlines() if baris.startswith("| ") and "---" not in baris][1:]


def nilai_kontras(baris, nama):
    for isi in baris:
        if isi.split("|")[2].strip() == nama:
            return tuple(float(angka) for angka in NILAI_CI.findall(isi)[0])
    raise SystemExit(f"kontras {nama!r} tidak ditemukan di {SUMBER.name}")


def baca_hasil():
    teks = SUMBER.read_text(encoding="utf-8")
    return {nama_tugas: [nilai_kontras(baris_did(bagian_tugas(teks, nama_tugas)), nama) for nama, _ in KONTRAS]
            for nama_tugas, _, _ in TUGAS}


def gambar(sumbu, hasil):
    for k, (nama_tugas, label_tugas, warna) in enumerate(TUGAS):
        for i, (titik, bawah, atas) in enumerate(hasil[nama_tugas]):
            y = i + (k - 0.5) * 2 * GESER
            sumbu.errorbar(titik, y, xerr=[[titik - bawah], [atas - titik]], fmt="o", color=warna, capsize=3,
                           label=label_tugas if i == 0 else None)
    sumbu.axvline(0, color="#555555", linewidth=0.8)
    sumbu.axvline(SETENGAH_EFEK_UTAMA, color="#555555", linewidth=0.8, linestyle="--")
    sumbu.axhline(2.5, color="#bbbbbb", linewidth=0.6)
    sumbu.set_yticks(range(len(KONTRAS)), [label for _, label in KONTRAS])
    sumbu.invert_yaxis()
    sumbu.grid(axis="x", color="#dddddd", linewidth=0.6)
    sumbu.set_xlabel("Difference in differences of AUROC, seen − unseen [95 % CI]")


def main():
    fig, sumbu = plt.subplots(figsize=(7.2, 3.0))
    gambar(sumbu, baca_hasil())
    fig.tight_layout()
    fig.legend(*sumbu.get_legend_handles_labels(), loc="upper center", ncol=2, fontsize=8, frameon=False,
               bbox_to_anchor=(0.6, 1.07))
    fig.savefig(KELUARAN.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(KELUARAN.with_suffix(".pdf"), bbox_inches="tight")
    print("ok:", KELUARAN.with_suffix(".png"))


if __name__ == "__main__":
    main()
