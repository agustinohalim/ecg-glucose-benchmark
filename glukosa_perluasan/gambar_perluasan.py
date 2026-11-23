"""Gambar 1 naskah Glukosa dari tabel agregat Hasil_Glukosa_13.md (protokol §F langkah 11); tidak membaca data atau
prediksi. Per tugas: DiD utama (ECG-FM pretrained − ECG-JEPA) dan kontrol positif (ECG-FM fine-tuned − ECG-JEPA), CI 95 %
bootstrap bergugus pasien, panel kiri skala AUROC dengan garis ε = 0,01, panel kanan skala logit.
Keluaran: Gambar_Glukosa_1.png (300 dpi) dan .pdf di .

Pakai, dari akar repo:
    python glukosa_perluasan/gambar_perluasan.py
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AKAR_EKG = Path(__file__).resolve().parent.parent
SUMBER = AKAR_EKG / "Hasil_Glukosa_13.md"
KELUARAN = AKAR_EKG / "Gambar_Glukosa_1"
TUGAS = [("Hipoglikemia", "Hypoglycemia"), ("Hiperglikemia berat", "Severe hyperglycemia"),
         ("Fibrilasi atrium", "Atrial fibrillation"), ("LBBB", "LBBB"), ("RBBB", "RBBB"),
         ("AV blok derajat 1", "First-degree AV block"), ("LVH", "LVH")]
SERI = [("utama", "Pretrained ECG-FM − ECG-JEPA (exposure)", "#0072b2", "o"),
        ("kontrol positif", "Fine-tuned ECG-FM − ECG-JEPA (positive control)", "#d55e00", "s")]
NILAI_CI = re.compile(r"(-?\d+\.\d+) \[(-?\d+\.\d+), (-?\d+\.\d+)\]")
EPSILON = 0.01
GESER = 0.17
PANEL = [(0, "AUROC scale", EPSILON), (1, "Logit scale", None)]


def bagian_tugas(teks, nama_tugas):
    awal = teks.index(f"#### {nama_tugas}")
    akhir = teks.find("\n###", awal + 1)
    return teks[awal:akhir if akhir > 0 else None]


def nilai_seri(bagian, jenis):
    for baris in bagian.splitlines():
        sel = baris.split("|")
        if len(sel) > 6 and sel[1].strip() == jenis:
            return [tuple(float(angka) for angka in nilai) for nilai in NILAI_CI.findall(baris)]
    raise SystemExit(f"baris {jenis!r} tidak ditemukan di {SUMBER.name}")


def baca_hasil():
    teks = SUMBER.read_text(encoding="utf-8")
    return {jenis: [nilai_seri(bagian_tugas(teks, nama), jenis) for nama, _ in TUGAS] for jenis, _, _, _ in SERI}


def gambar_panel(sumbu, hasil, panel):
    kolom, judul, garis = panel
    for k, (jenis, label, warna, penanda) in enumerate(SERI):
        for i, nilai in enumerate(hasil[jenis]):
            titik, bawah, atas = nilai[kolom]
            sumbu.errorbar(titik, i + (k - 0.5) * 2 * GESER, xerr=[[titik - bawah], [atas - titik]], fmt=penanda,
                           color=warna, markersize=4, capsize=2, label=label if i == 0 else None)
    sumbu.axvline(0, color="#555555", linewidth=0.8)
    if garis is not None:
        sumbu.axvline(garis, color="#555555", linewidth=0.8, linestyle="--")
        sumbu.text(garis, -0.75, " ε = 0.01", fontsize=7, color="#555555", va="center")
    sumbu.axhline(1.5, color="#bbbbbb", linewidth=0.6)
    sumbu.set_title(judul, fontsize=9)
    sumbu.grid(axis="x", color="#dddddd", linewidth=0.6)
    sumbu.set_xlabel("DiD, seen − unseen [95 % CI]", fontsize=8)
    sumbu.tick_params(labelsize=8)


def main():
    hasil = baca_hasil()
    fig, semua_sumbu = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    for sumbu, panel in zip(semua_sumbu, PANEL):
        gambar_panel(sumbu, hasil, panel)
    semua_sumbu[0].set_yticks(range(len(TUGAS)), [label for _, label in TUGAS])
    semua_sumbu[0].set_ylim(len(TUGAS) - 0.5, -1.0)
    fig.tight_layout()
    fig.legend(*semua_sumbu[0].get_legend_handles_labels(), loc="upper center", ncol=2, fontsize=8, frameon=False,
               bbox_to_anchor=(0.55, 1.07))
    fig.savefig(KELUARAN.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(KELUARAN.with_suffix(".pdf"), bbox_inches="tight")
    print("ok:", KELUARAN.with_suffix(".png"))


if __name__ == "__main__":
    main()
